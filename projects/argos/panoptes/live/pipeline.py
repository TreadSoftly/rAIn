# projects/argos/panoptes/live/pipeline.py
"""
LivePipeline: source → infer → annotate → sink(s).
Keeps progress UX similar to other ARGOS tasks and returns the saved path (if any).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import threading
from queue import Empty, Full, Queue
import time
from dataclasses import dataclass

from .camera import FrameSource, open_camera, synthetic_source
from .sinks import DisplaySink, VideoSink, MultiSink
from .overlay import hud
from . import tasks as live_tasks
from . import config as live_config
from .config import ModelSelection
from ._types import NDArrayU8
from panoptes.logging_config import bind_context, current_run_dir  # type: ignore[import]

# Live progress spinner (robust fallback)
try:
    from panoptes.progress import percent_spinner as _progress_percent_spinner  # type: ignore[import]
except Exception:  # pragma: no cover
    def _progress_percent_spinner(*_a: object, **_k: object):
        class _N:
            def __enter__(self): return self
            def __exit__(self, *_: object) -> bool: return False
            def update(self, **__: object): return self
        return _N()

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

    def __post_init__(self) -> None:
        self._hud_notice: Optional[str] = None
        self._hud_notice_until: float = 0.0
        self.preprocess_device = (self.preprocess_device or "auto").strip().lower()
        self.warmup = bool(self.warmup)
        self.backend = (self.backend or "auto").strip().lower()
        self.nms_mode = (self.nms_mode or "auto").strip().lower()
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
        self._apply_backend_defaults()

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

    def _build_task(self):
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
            return "cpu"
        if choice == "gpu":
            return "gpu" if self._cuda_available() else "cpu"
        # auto
        hw = self._hardware_info
        if hw and getattr(hw, "preprocess_device", None) == "gpu":
            if self._cuda_available():
                return "gpu"
        if hw is None:
            try:
                hw = live_config.probe_hardware()
            except Exception:
                hw = None
            self._hardware_info = hw
        if hw and getattr(hw, "gpu", None):
            if self._cuda_available():
                return "gpu"
        return "cpu"

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import cv2  # type: ignore

            if not hasattr(cv2, "cuda"):
                return False
            count = cv2.cuda.getCudaEnabledDeviceCount()
            return bool(count and count > 0)
        except Exception:
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

        with _progress_percent_spinner(prefix="LIVE") as sp:
            sp.update(total=total_frames, count=0, current="", job="init", model="")

            task = self._build_task()
            hw = live_config.probe_hardware()
            _log("live.pipeline.hardware", arch=getattr(hw, "arch", None), gpu=getattr(hw, "gpu", None), ram=getattr(hw, "ram_gb", None))
            sel: ModelSelection = live_config.select_models_for_live(self.task, hw)
            model_label = getattr(task, "label", "") or str(sel.get("label", ""))
            _log("live.pipeline.models", task=self.task, label=model_label)
            sp.update(job="probe", current=hw.arch or "", model=model_label)

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
                                        summary.update(detail)
                                except Exception:
                                    pass
                            stats_fn = getattr(task, "nms_statistics", None)
                            if callable(stats_fn):
                                try:
                                    counts = stats_fn()
                                    summary["suppressed_counts"] = {str(k): int(v) for k, v in counts.items()}
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
