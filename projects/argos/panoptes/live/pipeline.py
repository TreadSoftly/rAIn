# projects/argos/panoptes/live/pipeline.py
"""
LivePipeline: source -> infer -> annotate -> sink(s).
Keeps progress UX similar to other ARGOS tasks and returns the saved path (if any).
"""
from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING, Protocol, Mapping, cast

if TYPE_CHECKING:  # pragma: no cover
    from .tasks import TaskAdapter

from .camera import FrameSource, open_camera, synthetic_source
from .sinks import DisplaySink, VideoSink, MultiSink
from .overlay import hud
from . import tasks as live_tasks
from . import config as live_config
from .config import ModelSelection
from ._types import NDArrayU8
from panoptes.logging_config import bind_context, current_run_dir  # type: ignore[import]

# Live progress spinner (robust fallback)
class SpinnerLike(Protocol):
    def __enter__(self) -> "SpinnerLike": ...
    def __exit__(self, exc_type: Optional[type], exc: Optional[BaseException], tb: Optional[Any]) -> Optional[bool]: ...
    def update(self, **kwargs: object) -> None: ...

try:
    from panoptes.progress import percent_spinner as _spinner_impl  # type: ignore[import]
except Exception:  # pragma: no cover
    class _NullSpinner:
        def __enter__(self) -> "SpinnerLike":
            return self

        def __exit__(self, exc_type: Optional[type], exc: Optional[BaseException], tb: Optional[Any]) -> Optional[bool]:
            return False

        def update(self, **__: object) -> None:
            return None

    def _spinner_impl(*_a: object, **_k: object) -> SpinnerLike:
        return _NullSpinner()


def _progress_spinner(*args: object, **kwargs: object) -> SpinnerLike:
    return _spinner_impl(*args, **kwargs)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
LOGGER.setLevel(logging.ERROR)
_ERROR_TOKENS = ("error", "fail", "failed", "exception", "warning")
_ERROR_KEYS = ("error", "reason")

FramePacket = Tuple[NDArrayU8, float]

def _log(event: str, **info: object) -> None:
    event_lower = event.lower()
    should_emit = any(token in event_lower for token in _ERROR_TOKENS)
    if not should_emit:
        for key in _ERROR_KEYS:
            value = info.get(key)
            if isinstance(value, str):
                if value and value.strip().lower() not in {"ok", "success"}:
                    should_emit = True
                    break
            elif value not in (None, 0, False):
                should_emit = True
                break
    if not should_emit:
        return
    detail = " ".join(f"{key}={info[key]}" for key in sorted(info) if info[key] is not None)
    if detail:
        LOGGER.error("%s %s", event, detail)
    else:
        LOGGER.error("%s", event)


def _results_dir() -> Path:
    # Mirror the offline CLI layout: projects/argos/tests/results
    root = Path(__file__).resolve().parents[2]
    out = root / "tests" / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass
class LivePipeline:
    source: Union[str, int]
    task: str
    autosave: bool = True
    out_path: Optional[str] = None
    prefer_small: bool = True
    fps: Optional[int] = None
    size: Optional[Tuple[int, int]] = None
    headless: bool = False
    conf: float = 0.25
    iou: float = 0.45
    camera_auto_exposure: Optional[str] = None
    camera_exposure: Optional[float] = None
    duration: Optional[float] = None  # seconds; None = until user closes
    override: Optional[Path] = None
    display_name: Optional[str] = None
    preprocess_device: str = "auto"
    warmup: bool = True
    backend: str = "auto"
    ort_threads: Optional[int] = None
    ort_execution: Optional[str] = None
    nms_mode: str = "auto"
    resolution_schedule: Optional[Sequence[Tuple[int, int]]] = None
    capture_options: Optional[Dict[str, Any]] = None
    hm_smoothing: bool = True
    hm_decay: float = 0.3
    hm_history: int = 3
    hm_reset_frames: int = 5
    hm_alpha: float = 0.35
    clf_topk: int = 3
    clf_smooth_probs: bool = False
    clf_whitelist: Tuple[str, ...] = ()
    clf_blacklist: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self._hud_notice: Optional[str] = None
        self._hud_notice_until: float = 0.0
        self._user_size_override = self.size
        self.preprocess_device = (self.preprocess_device or "auto").strip().lower()
        self.warmup = bool(self.warmup)
        self.backend = (self.backend or "auto").strip().lower()
        self.nms_mode = (self.nms_mode or "auto").strip().lower()
        self.hm_smoothing = bool(self.hm_smoothing)
        try:
            self.hm_decay = float(self.hm_decay)
        except Exception:
            self.hm_decay = 0.3
        self.hm_decay = max(0.0, min(0.95, self.hm_decay))
        self.hm_history = max(1, int(self.hm_history))
        self.hm_reset_frames = max(1, int(self.hm_reset_frames))
        try:
            self.hm_alpha = float(self.hm_alpha)
        except Exception:
            self.hm_alpha = 0.35
        self.hm_alpha = max(0.0, min(1.0, self.hm_alpha))
        self.clf_topk = max(1, int(self.clf_topk))
        self.clf_smooth_probs = bool(self.clf_smooth_probs)
        self.clf_whitelist = tuple(token for token in (p.strip() for p in self.clf_whitelist) if token)
        self.clf_blacklist = tuple(token for token in (p.strip() for p in self.clf_blacklist) if token)
        if self.camera_auto_exposure is not None:
            auto_val = self.camera_auto_exposure.strip().lower()
            self.camera_auto_exposure = auto_val or None
        if self.nms_mode not in {"auto", "graph", "torch"}:
            self.nms_mode = "auto"
        exec_mode = (self.ort_execution or "").strip().lower()
        self.ort_execution = exec_mode or None
        if self.ort_threads is not None:
            try:
                self.ort_threads = max(1, int(self.ort_threads))
            except Exception:
                self.ort_threads = None
        self._last_logged_nms: Optional[str] = None
        self._nms_summary: Optional[dict[str, object]] = None
        healed_provider = os.environ.pop("PANOPTES_ORT_HEALED", "").strip()
        if healed_provider:
            provider_label = healed_provider
            if provider_label.endswith("ExecutionProvider"):
                provider_label = provider_label.replace("ExecutionProvider", "").strip()
            formatted = provider_label.upper() if provider_label else "CUDA"
            self._register_toast(f"{formatted} acceleration restored; ONNX Runtime is using the GPU provider.")
        self._hardware_info: Optional[live_config.HardwareInfo] = None
        self._resolution_schedule: List[Tuple[int, int]] = []
        self._auto_schedule_enabled = False
        self._current_size_index = 0
        self._fps_history: Deque[float] = deque(maxlen=30)
        self._frames_since_size_change = 0
        self._downscale_threshold = 30.0  # keep FPS comfortably above 30 (Optimize Research:90-104)
        self._downscale_warmup_frames = 60
        self._last_preprocess_device = "cpu"
        self._apply_backend_defaults()
        self._initialise_resolution_policy()

    def _initialise_resolution_policy(self) -> None:
        """
        Establish resolution defaults and whether adaptive resizing is enabled.

        Defaults follow the guidance from Optimize Research (640 as the balanced
        starting point, stepping down through 512/416/320 when FPS slips).
        """
        schedule = self._prepare_resolution_schedule()
        manual_tuple: Optional[Tuple[int, int]] = None
        if self._user_size_override is not None:
            try:
                manual_tuple = (
                    int(self._user_size_override[0]),
                    int(self._user_size_override[1]),
                )
            except Exception:
                manual_tuple = None

        if manual_tuple is not None and manual_tuple not in schedule:
            schedule.insert(0, manual_tuple)
        if not schedule:
            schedule = [(640, 640)]

        if manual_tuple is not None:
            size_tuple = manual_tuple
            self._auto_schedule_enabled = False
        else:
            candidate: Optional[Tuple[int, int]] = None
            if self.size is not None:
                try:
                    candidate = (int(self.size[0]), int(self.size[1]))  # type: ignore[index]
                except Exception:
                    candidate = None
            size_tuple = candidate or schedule[0]
            self._auto_schedule_enabled = len(schedule) > 1
        schedule[0] = size_tuple
        self.size = size_tuple

        self._resolution_schedule = schedule
        try:
            self._current_size_index = self._resolution_schedule.index(size_tuple)
        except ValueError:
            self._resolution_schedule.insert(0, size_tuple)
            self._current_size_index = 0
        if self._hardware_info is not None:
            try:
                self._hardware_info.input_size = (int(size_tuple[0]), int(size_tuple[1]))
            except Exception:
                pass

    def _prepare_resolution_schedule(self) -> List[Tuple[int, int]]:
        raw = self.resolution_schedule
        sizes: List[Tuple[int, int]] = []
        if raw is not None:
            for entry in raw:
                try:
                    w = int(entry[0])
                    h = int(entry[1])
                except Exception:
                    continue
                if w > 0 and h > 0:
                    sizes.append((w, h))
        if not sizes:
            sizes = [(640, 640), (512, 512), (416, 416), (320, 320)]
        unique: List[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()
        for size in sizes:
            if size not in seen:
                unique.append(size)
                seen.add(size)
        return unique

    def _maybe_adjust_resolution(
        self,
        task: "TaskAdapter",
        model_label: str,
        spinner: SpinnerLike,
    ) -> Tuple["TaskAdapter", str]:
        if not self._auto_schedule_enabled:
            return task, model_label
        next_index = self._current_size_index + 1
        if next_index >= len(self._resolution_schedule):
            self._auto_schedule_enabled = False
            return task, model_label
        if self._frames_since_size_change < self._downscale_warmup_frames:
            return task, model_label
        max_len = self._fps_history.maxlen or 0
        if max_len == 0 or len(self._fps_history) < max_len:
            return task, model_label

        avg_fps = sum(self._fps_history) / len(self._fps_history)
        if avg_fps >= self._downscale_threshold:
            return task, model_label

        self._current_size_index = next_index
        next_size = self._resolution_schedule[self._current_size_index]
        previous_size = self._resolution_schedule[self._current_size_index - 1] if self._current_size_index > 0 else self.size
        self.size = next_size
        try:
            spinner.update(job="resize", current=f"{next_size[0]}x{next_size[1]}")
        except Exception:
            pass
        self._register_toast(f"Auto resolution -> {next_size[0]}x{next_size[1]}")
        _log(
            "live.pipeline.autoscale",
            task=self.task,
            size=f"{next_size[0]}x{next_size[1]}",
            fps=f"{avg_fps:.2f}",
        )
        try:
            new_task = self._build_task()
        except Exception as exc:
            _log(
                "live.pipeline.autoscale.error",
                size=f"{next_size[0]}x{next_size[1]}",
                error=type(exc).__name__,
                message=str(exc),
            )
            if previous_size is not None:
                self.size = previous_size
            if self._current_size_index > 0:
                self._current_size_index -= 1
            self._auto_schedule_enabled = False
            return task, model_label
        label_new = str(getattr(new_task, "label", ""))
        if "no-ml" in label_new.lower():
            _log(
                "live.pipeline.autoscale.fallback_detected",
                size=f"{next_size[0]}x{next_size[1]}",
                label=label_new,
            )
            if previous_size is not None:
                self.size = previous_size
            if self._current_size_index > 0:
                self._current_size_index -= 1
            self._auto_schedule_enabled = False
            return task, model_label
        model_label = getattr(new_task, "label", model_label)
        self._fps_history.clear()
        self._frames_since_size_change = 0
        if self._hardware_info is not None:
            try:
                self._hardware_info.input_size = next_size  # type: ignore[attr-defined]
            except Exception:
                pass
        if self._current_size_index + 1 >= len(self._resolution_schedule):
            self._auto_schedule_enabled = False
        return new_task, model_label

    def _apply_backend_defaults(self) -> None:
        try:
            hw = live_config.probe_hardware()
        except Exception:
            hw = None
        self._hardware_info = hw
        if hw is None:
            return
        backend_hint = getattr(hw, "backend", None)
        if (self.backend == "auto" or not self.backend) and backend_hint and backend_hint != "auto":
            self.backend = backend_hint
        preprocess_hint = getattr(hw, "preprocess_device", None)
        if self.preprocess_device == "auto" and preprocess_hint and preprocess_hint != "auto":
            self.preprocess_device = preprocess_hint
        size_hint = getattr(hw, "input_size", None)
        if self.size is None and size_hint:
            try:
                self.size = (int(size_hint[0]), int(size_hint[1]))
            except Exception:
                self.size = None
        prefer_small_hint = getattr(hw, "prefer_small", None)
        if prefer_small_hint is not None and self.prefer_small == True:  # noqa: E712
            self.prefer_small = bool(prefer_small_hint)
        ort_threads_hint = getattr(hw, "ort_threads", None)
        if self.ort_threads is None and ort_threads_hint is not None:
            try:
                self.ort_threads = max(1, int(ort_threads_hint))
            except Exception:
                self.ort_threads = None
        ort_exec_hint = getattr(hw, "ort_execution", None)
        if self.ort_execution is None and ort_exec_hint:
            self.ort_execution = str(ort_exec_hint)
        nms_hint = getattr(hw, "nms_mode", None)
        if self.nms_mode == "auto" and nms_hint:
            self.nms_mode = str(nms_hint).strip().lower() or "auto"

    def _persist_nms_summary(self) -> None:
        if not self._nms_summary:
            return
        run_dir = current_run_dir()
        if run_dir is None:
            return
        safe_task = self.task.replace(" ", "_")
        filename = f"nms_strategy_{safe_task}_{os.getpid()}.json"
        path = Path(run_dir) / filename
        try:
            path.write_text(json.dumps(self._nms_summary, indent=2), encoding="utf-8")
        except Exception:
            LOGGER.debug("Failed to write NMS summary", exc_info=True)

    def _build_source(self) -> FrameSource:
        capture_opts = dict(self.capture_options or {})
        if capture_opts:
            _log(
                "live.pipeline.capture.options",
                options=",".join(sorted(str(k) for k in capture_opts.keys())),
            )
            # Future toggles (tiling, gamma, etc.) can be wired through capture_opts.
        if isinstance(self.source, str) and self.source.lower().startswith("synthetic"):
            return synthetic_source(size=self.size or (640, 480), fps=self.fps or 30)
        # camera/video path
        try:
            return open_camera(
                self.source,
                width=(self.size[0] if self.size else None),
                height=(self.size[1] if self.size else None),
                fps=self.fps,
                auto_exposure=self.camera_auto_exposure,
                exposure=self.camera_exposure,
            )
        except RuntimeError as exc:
            alt_source = self._autoprobe_camera(exc)
            if alt_source is not None:
                return alt_source
            if not self._should_use_synthetic_fallback():
                raise
            _log(
                "live.pipeline.source.fallback",
                original=str(self.source),
                fallback="synthetic",
                reason=str(exc),
            )
            try:
                synthetic = synthetic_source(size=self.size or (640, 480), fps=self.fps or 30)
            except Exception as fallback_exc:
                _log(
                    "live.pipeline.source.fallback.error",
                    original=str(self.source),
                    fallback="synthetic",
                    error=type(fallback_exc).__name__,
                )
                raise exc from fallback_exc
            self._register_toast("Camera unavailable; showing synthetic demo feed.")
            return synthetic

    def _should_use_synthetic_fallback(self) -> bool:
        numeric_source = self._numeric_source()
        if numeric_source is not None:
            return numeric_source == 0

        source_text = str(self.source).strip()
        if source_text.lower().startswith("synthetic"):
            return False
        if source_text.lstrip("+-").isdigit():
            try:
                return int(source_text) == 0
            except ValueError:
                return False
        return False

    def _numeric_source(self) -> Optional[int]:
        source = self.source
        if isinstance(source, int):
            return source
        token = str(source).strip()
        if token.lstrip("+-").isdigit():
            try:
                return int(token)
            except ValueError:
                return None
        return None

    def _autoprobe_camera(self, error: RuntimeError) -> Optional[FrameSource]:
        current_idx = self._numeric_source()
        if current_idx is None or current_idx < 0:
            return None

        probe_limit = getattr(live_config, "AUTO_CAMERA_PROBE_LIMIT", 4)
        attempted: list[int] = []
        for candidate in range(0, probe_limit + 1):
            if candidate == current_idx:
                continue
            attempted.append(candidate)
            try:
                camera = open_camera(
                    candidate,
                    width=(self.size[0] if self.size else None),
                    height=(self.size[1] if self.size else None),
                    fps=self.fps,
                )
            except RuntimeError:
                continue
            self.source = candidate
            _log(
                "live.pipeline.source.autodetected",
                original=str(current_idx),
                selected=str(candidate),
            )
            self._register_toast(f"Camera {candidate} detected automatically.")
            return camera

        if attempted:
            _log(
                "live.pipeline.source.autoprobe.failed",
                original=str(self.source),
                attempted=",".join(str(i) for i in attempted),
                reason=str(error),
            )
        return None

    def _register_toast(self, message: str) -> None:
        _log("live.model.toast", task=self.task, message=message)
        self._hud_notice = message
        self._hud_notice_until = time.time() + 3.0

    def _active_notice(self, now: float) -> Optional[str]:
        if self._hud_notice and now <= self._hud_notice_until:
            return self._hud_notice
        if self._hud_notice and now > self._hud_notice_until:
            self._hud_notice = None
        return None

    def _build_task(self) -> "TaskAdapter":
        t = self.task.lower()
        preprocess_device = self._resolve_preprocess_device()
        if t in ("d", "detect"):
            return live_tasks.build_detect(
                small=self.prefer_small,
                conf=self.conf,
                iou=self.iou,
                override=self.override if self.override is not None else live_tasks.LIVE_DETECT_OVERRIDE,
                input_size=self.size,
                preprocess_device=preprocess_device,
                warmup=self.warmup,
                backend=self.backend,
                ort_threads=self.ort_threads,
                ort_execution=self.ort_execution,
                nms_mode=self.nms_mode,
                hud_callback=self._register_toast,
            )
        if t in ("hm", "heatmap"):
            return live_tasks.build_heatmap(
                small=self.prefer_small,
                override=self.override if self.override is not None else live_tasks.LIVE_HEATMAP_OVERRIDE,
                input_size=self.size,
                preprocess_device=preprocess_device,
                warmup=self.warmup,
                backend=self.backend,
                ort_threads=self.ort_threads,
                ort_execution=self.ort_execution,
                hud_callback=self._register_toast,
                hm_smoothing=self.hm_smoothing,
                hm_decay=self.hm_decay,
                hm_history=self.hm_history,
                hm_reset_frames=self.hm_reset_frames,
                hm_alpha=self.hm_alpha,
            )
        if t in ("clf", "classify"):
            return live_tasks.build_classify(
                small=self.prefer_small,
                override=self.override if self.override is not None else live_tasks.LIVE_CLASSIFY_OVERRIDE,
                input_size=self.size,
                preprocess_device=preprocess_device,
                warmup=self.warmup,
                backend=self.backend,
                ort_threads=self.ort_threads,
                ort_execution=self.ort_execution,
                hud_callback=self._register_toast,
                topk=self.clf_topk,
                smooth_probs=self.clf_smooth_probs,
                class_whitelist=self.clf_whitelist,
                class_blacklist=self.clf_blacklist,
            )
        if t in ("pose", "pse"):
            return live_tasks.build_pose(
                small=self.prefer_small,
                conf=self.conf,
                override=self.override if self.override is not None else live_tasks.LIVE_POSE_OVERRIDE,
                input_size=self.size,
                preprocess_device=preprocess_device,
                warmup=self.warmup,
                backend=self.backend,
                ort_threads=self.ort_threads,
                ort_execution=self.ort_execution,
                hud_callback=self._register_toast,
            )
        if t in ("obb", "object"):
            return live_tasks.build_obb(
                small=self.prefer_small,
                conf=self.conf,
                iou=self.iou,
                override=self.override if self.override is not None else live_tasks.LIVE_OBB_OVERRIDE,
                input_size=self.size,
                preprocess_device=preprocess_device,
                warmup=self.warmup,
                backend=self.backend,
                ort_threads=self.ort_threads,
                ort_execution=self.ort_execution,
                hud_callback=self._register_toast,
            )
        raise ValueError(f"Unknown live task: {self.task}")

    def _resolve_preprocess_device(self) -> str:
        choice = (self.preprocess_device or "auto").strip().lower()
        if choice not in {"auto", "cpu", "gpu"}:
            choice = "auto"
        if choice == "cpu":
            self._last_preprocess_device = "cpu"
            return "cpu"
        if choice == "gpu":
            device = "gpu" if self._gpu_available() else "cpu"
            self._last_preprocess_device = device
            return device
        # auto
        hw = self._hardware_info
        if hw and getattr(hw, "preprocess_device", None) == "gpu":
            if self._gpu_available():
                self._last_preprocess_device = "gpu"
                return "gpu"
        if hw is None:
            try:
                hw = live_config.probe_hardware()
            except Exception:
                hw = None
            self._hardware_info = hw
        if hw and getattr(hw, "gpu", None):
            if self._gpu_available():
                self._last_preprocess_device = "gpu"
                return "gpu"
        self._last_preprocess_device = "cpu"
        return "cpu"

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return True
        except Exception:
            return False
        try:
            import cv2  # type: ignore

            if hasattr(cv2, "cuda"):
                count = cv2.cuda.getCudaEnabledDeviceCount()
                return bool(count and count > 0)
        except Exception:
            pass
        return False

    def _default_out_path(self) -> str:
        ts = time.strftime("%Y%m%d-%H%M%S")
        return str(_results_dir() / f"live_{self.task}_{ts}.mp4")



    def run(self) -> Optional[str]:
        """
        Run the pipeline. Returns path of saved video if a VideoSink was used,
        else None. Press 'q' or 'Esc' in the preview window to exit (when GUI available).
        """
        def _pretty_provider_label(raw: Optional[str]) -> Optional[str]:
            if not raw:
                return None
            token = raw.strip()
            lowered = token.lower()
            mapping = {
                "cudaexecutionprovider": "CUDA",
                "cudnnexecutionprovider": "CUDA",
                "cpuexecutionprovider": "CPU",
                "dmlexecutionprovider": "DirectML",
                "directmlexecutionprovider": "DirectML",
                "tensorrtexecutionprovider": "TensorRT",
            }
            if lowered in mapping:
                return mapping[lowered]
            if lowered.startswith("cuda:"):
                return "CUDA" + token[len("cuda"):]
            if lowered == "cuda":
                return "CUDA"
            if lowered == "cpu":
                return "CPU"
            return token

        with bind_context(live_task=self.task, source=str(self.source)):
            _log(
                "live.pipeline.start",
                task=self.task,
                source=str(self.source),
                autosave=self.autosave,
                out_path=self.out_path,
                prefer_small=self.prefer_small,
            )

        est_fps = float(self.fps or 30)
        total_frames = int(max(1.0, (self.duration or 1.0) * est_fps)) if self.duration is not None else 1

        video: Optional[VideoSink] = None
        saved_path: Optional[str] = None

        with _progress_spinner(prefix="LIVE") as sp:
            sp.update(total=total_frames, count=0, current="", job="init", model="")

            task: "TaskAdapter" = self._build_task()
            self._fps_history.clear()
            self._frames_since_size_change = 0
            hw = live_config.probe_hardware()
            _log("live.pipeline.hardware", arch=getattr(hw, "arch", None), gpu=getattr(hw, "gpu", None), ram=getattr(hw, "ram_gb", None))
            sel: ModelSelection = live_config.select_models_for_live(self.task, hw)
            model_label = getattr(task, "label", "") or str(sel.get("label", ""))
            _log("live.pipeline.models", task=self.task, label=model_label)
            sp.update(job="probe", current=hw.arch or "", model=model_label)
            size_label = f"{self.size[0]}x{self.size[1]}" if self.size else "auto"
            device_label = (self._last_preprocess_device or "cpu").upper()
            self._register_toast(f"{device_label} preprocess x {size_label}")

            src: FrameSource = self._build_source()
            sp.update(job="open-src", current=str(self.source))

            window_title = self.display_name or f"ARGOS Live ({self.task}:{self.source})"
            display: Optional[DisplaySink] = None if self.headless else DisplaySink(window_title, headless=False)
            saved_path = self.out_path or (self._default_out_path() if self.autosave and self.out_path is None else None)
            sinks: Optional[MultiSink] = None

            t0 = time.time()
            fps_est = 0.0
            last = t0
            frames_done = 0

            stop_event = threading.Event()
            frame_queue: Queue[Optional[FramePacket]] = Queue(maxsize=5)

            def _capture_loop() -> None:
                try:
                    for frame_bgr, ts in src.frames():
                        if stop_event.is_set():
                            break
                        try:
                            frame_queue.put((frame_bgr, ts), timeout=0.1)
                        except Full:
                            continue
                except Exception as exc:
                    _log("live.pipeline.capture.error", error=type(exc).__name__, message=str(exc))
                finally:
                    stop_event.set()
                    try:
                        frame_queue.put_nowait(None)
                    except Full:
                        pass

            capture_thread = threading.Thread(target=_capture_loop, name="argos-live-capture", daemon=True)
            capture_thread.start()

            try:
                while True:
                    try:
                        item = frame_queue.get(timeout=0.1)
                    except Empty:
                        if stop_event.is_set() and frame_queue.empty():
                            break
                        continue

                    if item is None:
                        break

                    frame_bgr, _ts = item
                    now = time.time()
                    dt = max(1e-6, now - last)
                    inst_fps = 1.0 / dt
                    fps_est = 0.9 * fps_est + 0.1 * inst_fps if fps_est > 0 else inst_fps
                    last = now
                    self._fps_history.append(inst_fps)
                    self._frames_since_size_change += 1
                    task, model_label = self._maybe_adjust_resolution(task, model_label, sp)

                    if sinks is None:
                        if saved_path:
                            Path(saved_path).parent.mkdir(parents=True, exist_ok=True)
                            try:
                                h_px, w_px = frame_bgr.shape[:2]
                            except Exception:
                                h_px = self.size[1] if self.size else 480
                                w_px = self.size[0] if self.size else 640
                            try:
                                video = VideoSink(saved_path, (w_px, h_px), float(self.fps or 30))
                                saved_path = video.output_path
                            except Exception as exc:
                                _log("live.pipeline.video_init.error", error=type(exc).__name__, message=str(exc))
                                raise
                        sinks = MultiSink(*(x for x in (display, video) if x is not None))
                        _log("live.pipeline.sinks", display=bool(display), video_path=saved_path if saved_path else None)
                        sp.update(
                            job="run",
                            current=(Path(saved_path).name if saved_path else "live-only"),
                            model=model_label,
                        )

                    result = task.infer(frame_bgr)
                    frame_anno = task.render(frame_bgr, result)

                    provider_label: Optional[str] = None
                    model_obj = getattr(task, "model", None)
                    if model_obj is not None:
                        try:
                            provider_seq = getattr(model_obj, "providers", None)
                        except Exception:
                            provider_seq = None
                        if provider_seq:
                            try:
                                provider_label = provider_seq[0]
                            except Exception:
                                provider_label = None
                    device = _pretty_provider_label(provider_label) or (hw.gpu or "CPU")
                    notice = self._active_notice(now)
                    dynamic_label = model_label
                    current_label_fn = getattr(task, "current_label", None)
                    if callable(current_label_fn):
                        try:
                            dynamic_label = str(current_label_fn())
                        except Exception:
                            dynamic_label = model_label
                    elif getattr(task, "label", None):
                        try:
                            dynamic_label = str(getattr(task, "label"))
                        except Exception:
                            dynamic_label = model_label
                    nms_mode_active: Optional[str] = None
                    get_nms_mode = getattr(task, "current_nms_mode", None)
                    if callable(get_nms_mode):
                        try:
                            nms_mode_active = str(get_nms_mode()).lower()
                        except Exception:
                            nms_mode_active = None
                    label_with_nms = dynamic_label
                    if nms_mode_active:
                        label_with_nms = f"{dynamic_label} | NMS:{nms_mode_active.upper()}"
                        if nms_mode_active != self._last_logged_nms:
                            _log("live.pipeline.nms", task=self.task, mode=nms_mode_active)
                            self._last_logged_nms = nms_mode_active
                            summary: dict[str, object] = {
                                "mode": nms_mode_active,
                                "task": self.task,
                                "backend": self.backend,
                                "override": self.nms_mode,
                            }
                            strategy_fn = getattr(task, "last_strategy", None)
                            if callable(strategy_fn):
                                try:
                                    detail = strategy_fn()
                                    if isinstance(detail, dict):
                                        detail_map = cast(Mapping[Any, Any], detail)
                                        for key_obj, value_obj in detail_map.items():
                                            summary[str(key_obj)] = value_obj
                                except Exception:
                                    pass
                            stats_fn = getattr(task, "nms_statistics", None)
                            if callable(stats_fn):
                                try:
                                    counts = stats_fn()
                                    suppressed: Dict[str, int] = {}
                                    if isinstance(counts, dict):
                                        counts_map = cast(Mapping[Any, Any], counts)
                                        for key_obj, value_obj in counts_map.items():
                                            try:
                                                suppressed[str(key_obj)] = int(value_obj)
                                            except Exception:
                                                continue
                                    summary["suppressed_counts"] = suppressed
                                except Exception:
                                    pass
                            self._nms_summary = summary
                            self._persist_nms_summary()
                    if label_with_nms != model_label:
                        model_label = label_with_nms
                        try:
                            sp.update(model=model_label)
                        except Exception:
                            pass
                    hud(
                        frame_anno,
                        fps=fps_est,
                        task=self.task,
                        model=model_label,
                        device=device,
                        notice=notice,
                    )

                    assert sinks is not None
                    sinks.write(frame_anno)

                    frames_done += 1
                    sp.update(count=frames_done)
                    if frames_done % max(1, int(self.fps or inst_fps or 30)) == 0:
                        _log("live.pipeline.fps", frames=frames_done, fps=f"{fps_est:.2f}", inst=f"{inst_fps:.2f}")

                    if self.duration is not None and (now - t0) >= self.duration:
                        _log("live.pipeline.stop", reason="duration", frames=frames_done)
                        stop_event.set()
                        break

                    if display is not None:
                        try:
                            if not display.is_open():
                                _log("live.pipeline.stop", reason="window-closed", frames=frames_done)
                                stop_event.set()
                                break
                        except Exception:
                            pass
                        try:
                            key = display.poll_key()
                            if key in (ord("q"), 27):
                                _log("live.pipeline.stop", reason="user-exit", frames=frames_done)
                                stop_event.set()
                                break
                        except Exception:
                            pass
            finally:
                stop_event.set()
                capture_thread.join(timeout=1.0)
                try:
                    src.release()
                except Exception:
                    pass
                if video is not None:
                    try:
                        video.close()
                    except Exception:
                        pass
                if display is not None:
                    try:
                        display.close()
                    except Exception:
                        pass

                _log("live.pipeline.cleanup", saved_path=saved_path, frames=frames_done)

        _log("live.pipeline.end", saved_path=saved_path)
        return saved_path
