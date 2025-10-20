# projects/argos/panoptes/live/pipeline.py
"""
LivePipeline: source → infer → annotate → sink(s).
Keeps progress UX similar to other ARGOS tasks and returns the saved path (if any).
"""
from __future__ import annotations

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
from panoptes.logging_config import bind_context # type: ignore[import]

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
TRACE_PIPELINE = os.getenv("PANOPTES_LIVE_TRACE_PIPELINE", "").strip().lower() in {"1", "true", "yes"}
LOG_DETAIL = os.getenv("PANOPTES_LOG_DETAIL", "").strip().lower() in {"1", "true", "yes"}
ESSENTIAL_EVENTS = {
    "live.pipeline.start",
    "live.pipeline.stop",
    "live.pipeline.end",
    "live.pipeline.capture.error",
    "live.pipeline.video_init.error",
    "live.pipeline.sinks",
}
BASIC_KEYS = ("task", "source", "reason", "error", "frames", "model", "video_path")

FramePacket = Tuple[NDArrayU8, float]

def _log(event: str, **info: object) -> None:
    if not LOGGER.isEnabledFor(logging.INFO):
        return
    if not TRACE_PIPELINE and event not in ESSENTIAL_EVENTS:
        return
    if info:
        if TRACE_PIPELINE or LOG_DETAIL:
            detail = " ".join(f"{k}={info[k]}" for k in sorted(info) if info[k] is not None)
        else:
            detail_parts = [f"{k}={info[k]}" for k in BASIC_KEYS if info.get(k) is not None]
            detail = " ".join(detail_parts)
        if detail:
            LOGGER.info("%s %s", event, detail)
        else:
            LOGGER.info(event)
    else:
        LOGGER.info(event)


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
    duration: Optional[float] = None  # seconds; None = until user closes
    override: Optional[Path] = None
    display_name: Optional[str] = None
    preprocess_device: str = "auto"

    def __post_init__(self) -> None:
        self._hud_notice: Optional[str] = None
        self._hud_notice_until: float = 0.0
        self.preprocess_device = (self.preprocess_device or "auto").strip().lower()

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
                hud_callback=self._register_toast,
            )
        if t in ("hm", "heatmap"):
            return live_tasks.build_heatmap(
                small=self.prefer_small,
                override=self.override if self.override is not None else live_tasks.LIVE_HEATMAP_OVERRIDE,
                input_size=self.size,
                preprocess_device=preprocess_device,
                hud_callback=self._register_toast,
            )
        if t in ("clf", "classify"):
            return live_tasks.build_classify(
                small=self.prefer_small,
                override=self.override if self.override is not None else live_tasks.LIVE_CLASSIFY_OVERRIDE,
                input_size=self.size,
                preprocess_device=preprocess_device,
                hud_callback=self._register_toast,
            )
        if t in ("pose", "pse"):
            return live_tasks.build_pose(
                small=self.prefer_small,
                conf=self.conf,
                override=self.override if self.override is not None else live_tasks.LIVE_POSE_OVERRIDE,
                input_size=self.size,
                preprocess_device=preprocess_device,
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
        try:
            hw = live_config.probe_hardware()
        except Exception:
            hw = None
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

                    device = hw.gpu or "CPU"
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
                    if dynamic_label != model_label:
                        model_label = dynamic_label
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
