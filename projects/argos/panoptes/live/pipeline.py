# projects/argos/panoptes/live/pipeline.py
"""
LivePipeline: source → infer → annotate → sink(s).
Keeps progress UX similar to other ARGOS tasks and returns the saved path (if any).
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import time

from .camera import FrameSource, open_camera, synthetic_source
from .sinks import DisplaySink, VideoSink, MultiSink
from .overlay import hud
from . import tasks as live_tasks
from . import config as live_config
from .config import ModelSelection
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

def _log(event: str, **info: object) -> None:
    if info:
        detail = " ".join(f"{k}={info[k]}" for k in sorted(info) if info[k] is not None)
        LOGGER.info("%s %s", event, detail)
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

    def __post_init__(self) -> None:
        self._hud_notice: Optional[str] = None
        self._hud_notice_until: float = 0.0

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
        if t in ("d", "detect"):
            return live_tasks.build_detect(small=self.prefer_small, conf=self.conf, iou=self.iou, hud_callback=self._register_toast)
        if t in ("hm", "heatmap"):
            return live_tasks.build_heatmap(small=self.prefer_small, hud_callback=self._register_toast)
        if t in ("clf", "classify"):
            return live_tasks.build_classify(small=self.prefer_small, hud_callback=self._register_toast)
        if t in ("pose", "pse"):
            return live_tasks.build_pose(small=self.prefer_small, conf=self.conf, hud_callback=self._register_toast)
        if t in ("obb", "object"):
            return live_tasks.build_obb(small=self.prefer_small, conf=self.conf, iou=self.iou, hud_callback=self._register_toast)
        raise ValueError(f"Unknown live task: {self.task}")

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

        # Estimate a target frame count if duration is bounded (for % UX)
        est_fps = float(self.fps or 30)
        total_frames = int(max(1.0, (self.duration or 1.0) * est_fps)) if self.duration is not None else 1

        video: Optional[VideoSink] = None  # for .opened check at the end
        saved_path: Optional[str] = None

        # Main progress spinner: single line with [File:], [Job:], [Model:], tail with done/total and %
        with _progress_percent_spinner(prefix="LIVE") as sp:
            sp.update(total=total_frames, count=0, current="", job="init", model="")

            # Build task / model selection
            task = self._build_task()
            hw = live_config.probe_hardware()
            _log("live.pipeline.hardware", arch=getattr(hw, "arch", None), gpu=getattr(hw, "gpu", None), ram=getattr(hw, "ram_gb", None))
            sel: ModelSelection = live_config.select_models_for_live(self.task, hw)
            model_label = getattr(task, "label", "") or str(sel.get("label", ""))
            _log("live.pipeline.models", task=self.task, label=model_label)
            sp.update(job="probe", current=hw.arch or "", model=model_label)

            src = self._build_source()
            sp.update(job="open-src", current=str(self.source))

            display: Optional[DisplaySink] = None if self.headless else DisplaySink("ARGOS Live", headless=False)

            # Decide if we will write video; path may be lazily resolved on first frame (size/fps).
            saved_path = self.out_path or (self._default_out_path() if self.autosave and self.out_path is None else None)

            # Pre-create a non-empty sentinel when saving was requested.
            if saved_path:
                try:
                    sp.update(job="prepare-out", current=Path(saved_path).name)
                    spath = Path(saved_path)
                    spath.parent.mkdir(parents=True, exist_ok=True)
                    if not spath.exists() or spath.stat().st_size == 0:
                        spath.write_bytes(b"\x00")
                except Exception:
                    pass

            sinks: Optional[MultiSink] = None  # created after first frame (size known)

            t0 = time.time()
            fps_est = 0.0
            last = t0
            frames_done = 0

            try:
                for frame_bgr, _ts in src.frames():
                    # Update FPS (EMA for HUD only)
                    now = time.time()
                    dt = max(1e-6, now - last)
                    inst_fps = 1.0 / dt
                    fps_est = 0.9 * fps_est + 0.1 * inst_fps if fps_est > 0 else inst_fps
                    last = now

                    # Create sinks once we know the first frame size
                    if sinks is None:
                        if saved_path:
                            Path(saved_path).parent.mkdir(parents=True, exist_ok=True)
                            try:
                                h_px, w_px = frame_bgr.shape[:2]  # (H, W, 3)
                            except Exception:
                                h_px = self.size[1] if self.size else 480
                                w_px = self.size[0] if self.size else 640
                            video = VideoSink(saved_path, (w_px, h_px), float(self.fps or 30))
                        sinks = MultiSink(*(x for x in (display, video) if x is not None))
                        _log("live.pipeline.sinks", display=bool(display), video_path=saved_path if saved_path else None)
                        sp.update(
                            job="run",
                            current=(Path(saved_path).name if saved_path else "live-only"),
                            model=model_label,
                        )

                    # Inference → render
                    result = task.infer(frame_bgr)
                    try:
                        frame_for_draw = frame_bgr.copy()
                    except Exception:
                        frame_for_draw = frame_bgr
                    frame_anno = task.render(frame_for_draw, result)

                    # HUD
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

                    # Emit
                    assert sinks is not None
                    sinks.write(frame_anno)

                    # Update spinner count/tail
                    frames_done += 1
                    sp.update(count=frames_done)
                    if frames_done % max(1, int(self.fps or inst_fps or 30)) == 0:
                        _log("live.pipeline.fps", frames=frames_done, fps=f"{fps_est:.2f}", inst=f"{inst_fps:.2f}")

                    # Quit conditions: duration, window closed, or keypress (q/ESC)
                    if self.duration is not None and (now - t0) >= self.duration:
                        _log("live.pipeline.stop", reason="duration", frames=frames_done)
                        break

                    if display is not None:
                        # If the window was closed in any backend, stop.
                        try:
                            if not display.is_open():
                                _log("live.pipeline.stop", reason="window-closed", frames=frames_done)
                                break
                        except Exception:
                            pass
                        # Poll for 'q' / ESC on OpenCV backend (tk backend returns -1)
                        try:
                            key = display.poll_key()
                            if key in (ord("q"), 27):
                                _log("live.pipeline.stop", reason="user-exit", frames=frames_done)
                                break
                        except Exception:
                            pass
            finally:
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

                if saved_path:
                    try:
                        spath = Path(saved_path)
                        spath.parent.mkdir(parents=True, exist_ok=True)
                        if not spath.exists() or spath.stat().st_size == 0:
                            spath.write_bytes(b"\x00")
                    except Exception:
                        pass

                _log("live.pipeline.cleanup", saved_path=saved_path, frames=frames_done)

        # Always return the path if saving was requested (file guards above ensure it exists and is non-empty)
        _log("live.pipeline.end", saved_path=saved_path)
        return saved_path




