# projects/argos/panoptes/live/camera.py
"""
Live capture sources for ARGOS.

Prefers OpenCV backends with OS hints (DSHOW/AVFOUNDATION/V4L2/MSMF) and falls back
quietly. Also provides a deterministic "synthetic" source for CI/headless runs.
"""

from __future__ import annotations

import contextlib
import math
import logging
import os
import sys
import time
from typing import Any, Iterator, Optional, Protocol, Tuple, Union, cast, Sequence

# Lightweight progress status for open/init phases
try:
    from panoptes.progress import simple_status as _progress_simple_status  # type: ignore[import]
except Exception:  # pragma: no cover
    def _progress_simple_status(*_a: object, **_k: object):
        class _N:
            def __enter__(self): return self
            def __exit__(self, *_: object) -> bool: return False
            def update(self, **__: object): return self
        return _N()

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

os.environ.setdefault("OPENCV_VIDEOIO_ENABLE_OBSENSOR", "0")

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore
else:
    try:
        if hasattr(cv2, "getLogLevel") and hasattr(cv2, "setLogLevel"):
            try:
                cv2.setLogLevel(0)
            except Exception:
                pass
    except Exception:
        pass
    try:
        cv2_utils = getattr(cv2, "utils", None)
        cv2_logging = getattr(cv2_utils, "logging", None) if cv2_utils else None
        if cv2_logging and hasattr(cv2_logging, "setLogLevel"):
            for level_name in ("LOG_LEVEL_SILENT", "LOG_LEVEL_FATAL", "LOG_LEVEL_ERROR"):
                if hasattr(cv2_logging, level_name):
                    cv2_logging.setLogLevel(getattr(cv2_logging, level_name))
                    break
    except Exception:
        pass

from ._types import NDArrayU8
from panoptes.logging_config import bind_context  # type: ignore[import]

TRACE_CAMERA = os.getenv("PANOPTES_LIVE_TRACE_CAMERA", "").strip().lower() in {"1", "true", "yes"}
ENABLE_DSHOW = os.getenv("PANOPTES_LIVE_ENABLE_DSHOW", "").strip().lower() in {"1", "true", "yes", "on"}


@contextlib.contextmanager
def _temporary_cv2_log_level() -> Iterator[None]:
    """Suppress overly noisy OpenCV logging while probing devices."""
    if cv2 is None:
        yield
        return
    utils = getattr(cv2, "utils", None)
    logging_mod = getattr(utils, "logging", None) if utils else None

    if logging_mod is not None and hasattr(logging_mod, "getLogLevel"):
        try:
            prev_level = logging_mod.getLogLevel()
        except Exception:
            prev_level = None

        target_level = None
        for candidate in ("LOG_LEVEL_SILENT", "LOG_LEVEL_FATAL", "LOG_LEVEL_ERROR"):
            if hasattr(logging_mod, candidate):
                target_level = getattr(logging_mod, candidate)
                break

        try:
            if target_level is not None:
                logging_mod.setLogLevel(target_level)
        except Exception:
            prev_level = None

        try:
            yield
        finally:
            if prev_level is not None:
                try:
                    logging_mod.setLogLevel(prev_level)
                except Exception:
                    pass
        return

    prev_global = None
    try:
        if hasattr(cv2, "getLogLevel") and hasattr(cv2, "setLogLevel"):
            prev_global = cv2.getLogLevel()
            cv2.setLogLevel(0)
    except Exception:
        prev_global = None

    try:
        yield
    finally:
        if prev_global is not None:
            try:
                cv2.setLogLevel(prev_global)
            except Exception:
                pass


LOGGER = logging.getLogger(__name__)


def _log(event: str, **info: object) -> None:
    if info:
        detail = " ".join(f"{k}={info[k]}" for k in sorted(info) if info[k] is not None)
        LOGGER.info("%s %s", event, detail)
    else:
        LOGGER.info(event)


class FrameSource(Protocol):
    def frames(self) -> Iterator[tuple[NDArrayU8, float]]: ...
    def release(self) -> None: ...


def _decode_frame_to_bgr(frame: Any, mode: Optional[str] = None) -> NDArrayU8:
    if np is None or cv2 is None:
        raise RuntimeError("OpenCV/numpy required for camera decoding")

    arr = np.asarray(frame)

    if mode == "yuy":
        if arr.ndim == 3 and arr.shape[2] == 3:
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
            return np.ascontiguousarray(arr[:, :, :3])
        if arr.ndim == 3 and arr.shape[2] == 2:
            arr2d = arr
        elif arr.ndim == 2:
            h, w = arr.shape
            if w % 2 == 0:
                arr2d = arr.reshape(h, w // 2, 2)
            else:
                arr2d = arr[:, :, None]
        else:
            arr2d = np.atleast_2d(arr)
        if arr2d.dtype != np.uint8:
            arr2d = np.clip(arr2d, 0, 255).astype(np.uint8, copy=False)
        return cast(NDArrayU8, cv2.cvtColor(np.ascontiguousarray(arr2d), cv2.COLOR_YUV2BGR_YUY2))
    if mode == "nv12":
        arr2d = np.atleast_2d(arr)
        if arr2d.dtype != np.uint8:
            arr2d = np.clip(arr2d, 0, 255).astype(np.uint8, copy=False)
        return cast(NDArrayU8, cv2.cvtColor(np.ascontiguousarray(arr2d), cv2.COLOR_YUV2BGR_NV12))
    if mode == "nv21":
        arr2d = np.atleast_2d(arr)
        if arr2d.dtype != np.uint8:
            arr2d = np.clip(arr2d, 0, 255).astype(np.uint8, copy=False)
        return cast(NDArrayU8, cv2.cvtColor(np.ascontiguousarray(arr2d), cv2.COLOR_YUV2BGR_NV21))

    # If OpenCV already delivered a standard 3-channel buffer, keep it.
    if arr.ndim == 3 and arr.shape[2] == 3:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
        return np.ascontiguousarray(arr[:, :, :3])
    if arr.ndim == 3 and arr.shape[2] == 4:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
        return cast(NDArrayU8, cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_BGRA2BGR))

    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
        return cast(NDArrayU8, cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_GRAY2BGR))

    if arr.ndim == 3 and arr.shape[2] == 1:
        flat = arr.reshape(arr.shape[0], arr.shape[1])
        if flat.dtype != np.uint8:
            flat = np.clip(flat, 0, 255).astype(np.uint8, copy=False)
        return cast(NDArrayU8, cv2.cvtColor(np.ascontiguousarray(flat), cv2.COLOR_GRAY2BGR))

    # Fallback: ensure contiguous uint8 and let OpenCV attempt a generic gray→BGR conversion.
    arr2d = np.atleast_2d(arr)
    if arr2d.dtype != np.uint8:
        arr2d = np.clip(arr2d, 0, 255).astype(np.uint8, copy=False)
    return cast(NDArrayU8, cv2.cvtColor(np.ascontiguousarray(arr2d), cv2.COLOR_GRAY2BGR))


def _is_convert_rgb_enabled(value: Optional[float]) -> bool:
    if value is None:
        return False
    try:
        val = float(value)
    except (TypeError, ValueError):
        return False
    return abs(val - 1.0) <= 1e-3


def _guess_backend_ids() -> list[int]:
    """Return a list of platform-appropriate OpenCV backend ids (try in order)."""
    ids: list[int] = []
    if cv2 is None:
        return ids
    plat = sys.platform
    if plat.startswith("win"):
        msmf = getattr(cv2, "CAP_MSMF", 0)
        if msmf:
            ids.append(msmf)
        if ENABLE_DSHOW:
            dshow = getattr(cv2, "CAP_DSHOW", 0)
            if dshow:
                ids.append(dshow)
        elif TRACE_CAMERA:
            _log("live.camera.backend.skip", backend="CAP_DSHOW")
    elif plat == "darwin":
        ids.append(getattr(cv2, "CAP_AVFOUNDATION", 0))
    elif "linux" in plat:
        ids.append(getattr(cv2, "CAP_V4L2", 0))
    # Always end with 0 → default/auto
    ids.append(0)
    # Deduplicate but keep order
    out: list[int] = []
    for i in ids:
        if i and i not in out:
            out.append(i)
    # Ensure default at end
    out.append(0)
    return out


class _CVCamera(FrameSource):
    # Explicit so Pyright knows the attribute exists
    cap: Any

    def __init__(
        self,
        source: Union[str, int],
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV not available for camera capture.")
        self._source = str(source)
        self._convert_rgb_prop: Optional[int] = None
        self._convert_rgb_forced = False
        self._convert_rgb_enabled = False
        self._backend: Optional[int] = None
        self._initial_decode_mode: Optional[str] = None
        self._flat_frame_streak = 0
        # Allow "0", "-1" strings, or file/rtsp paths
        if isinstance(source, str):
            try:
                src_val: Union[str, int] = int(source)
            except ValueError:
                src_val = source
        else:
            src_val = source

        # Try a few backends if index source; file/rtsp → default only
        backends = _guess_backend_ids() if isinstance(src_val, int) else [0]

        context = bind_context(component="camera", source=self._source)
        exc_info = (None, None, None)
        context.__enter__()
        selected_backend: Optional[int] = None
        try:
            _log(
                "live.camera.open.start",
                source=self._source,
                backends=",".join(str(b) for b in backends),
                width=width,
                height=height,
                fps=fps,
            )
            last_err: Optional[Exception] = None
            devnull = None
            with _progress_simple_status(f"Opening source {source!r}"):
                self.cap = None
                try:
                    devnull = open(os.devnull, "w")
                except Exception:
                    devnull = None
                stderr_ctx = contextlib.redirect_stderr(devnull) if devnull else contextlib.nullcontext()
                stdout_ctx = contextlib.redirect_stdout(devnull) if devnull else contextlib.nullcontext()
                with _temporary_cv2_log_level():
                    with stderr_ctx, stdout_ctx:
                        for be in backends:
                            if TRACE_CAMERA:
                                _log(
                                    "live.camera.open.try",
                                    source=self._source,
                                    backend=str(be),
                                )
                            try:
                                self.cap = cv2.VideoCapture(src_val, be) if be else cv2.VideoCapture(src_val)
                                if self.cap and self.cap.isOpened():
                                    selected_backend = be
                                    break
                                if self.cap:
                                    self.cap.release()
                                    self.cap = None
                            except Exception as e:
                                last_err = e
                                self.cap = None
                                if TRACE_CAMERA:
                                    _log(
                                        "live.camera.open.fail",
                                        source=self._source,
                                        backend=str(be),
                                        error=type(e).__name__,
                                    )
                if not self.cap or not self.cap.isOpened():
                    raise RuntimeError(f"Failed to open camera/source: {source}") from last_err
                if devnull is not None:
                    try:
                        devnull.close()
                    except Exception:
                        pass

                if width is not None:
                    self.cap.set(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3), float(width))
                if height is not None:
                    self.cap.set(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4), float(height))
                if fps is not None:
                    self.cap.set(getattr(cv2, "CAP_PROP_FPS", 5), float(fps))

            actual_w = actual_h = None
            actual_fps: Optional[float] = None
            try:
                actual_w = int(self.cap.get(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)))
                actual_h = int(self.cap.get(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)))
                actual_fps = float(self.cap.get(getattr(cv2, "CAP_PROP_FPS", 5)))
            except Exception:
                pass

            fourcc_val: Optional[int] = None
            try:
                fourcc_val = int(self.cap.get(getattr(cv2, "CAP_PROP_FOURCC", 6)))
            except Exception:
                fourcc_val = None

            fourcc = ""
            if fourcc_val:
                chars = [chr((fourcc_val >> (8 * i)) & 0xFF) for i in range(4)]
                fourcc = "".join(chars).strip()

            convert_rgb = None
            try:
                prop = getattr(cv2, "CAP_PROP_CONVERT_RGB", 37)
                self._convert_rgb_prop = prop
                convert_rgb = self.cap.get(prop)
            except Exception:
                self._convert_rgb_prop = None
                convert_rgb = None

            if _is_convert_rgb_enabled(convert_rgb) and self._convert_rgb_prop is not None:
                try:
                    self.cap.set(self._convert_rgb_prop, 0.0)
                    convert_rgb = self.cap.get(self._convert_rgb_prop)
                except Exception:
                    pass

            raw_format = None
            try:
                raw_format = self.cap.get(getattr(cv2, "CAP_PROP_FORMAT", 8))
            except Exception:
                raw_format = None

            self._convert_rgb_enabled = _is_convert_rgb_enabled(convert_rgb)
            self._convert_rgb_forced = not self._convert_rgb_enabled
            if TRACE_CAMERA:
                _log(
                    "live.camera.open.convert_rgb",
                    source=self._source,
                    initial=convert_rgb,
                    forced=self._convert_rgb_forced,
                )

            decode_mode: Optional[str] = None
            if not self._convert_rgb_enabled:
                fourcc_upper = fourcc.upper()
                if fourcc_upper.startswith(("YUY", "UYVY")):
                    decode_mode = "yuy"
                elif fourcc_upper.startswith("NV12"):
                    decode_mode = "nv12"
                elif fourcc_upper.startswith("NV21"):
                    decode_mode = "nv21"
                elif fourcc_upper in {"GREY", "GRAY"}:
                    decode_mode = "gray"

            self._decode_mode = decode_mode
            self._initial_decode_mode = decode_mode
            if TRACE_CAMERA:
                _log(
                    "live.camera.open.decode",
                    source=self._source,
                    mode=decode_mode,
                    convert_rgb=convert_rgb,
                    convert_rgb_enabled=self._convert_rgb_enabled,
                )

            if TRACE_CAMERA:
                _log(
                    "live.camera.open.trace",
                    source=self._source,
                    fourcc=fourcc or None,
                    convert_rgb=convert_rgb,
                    raw_format=raw_format,
                )

            _log(
                "live.camera.open.success",
                source=self._source,
                backend=str(selected_backend),
                width=actual_w,
                height=actual_h,
                fps=actual_fps,
                fourcc=fourcc or None,
                convert_rgb=convert_rgb,
                raw_format=raw_format,
                decode_mode=self._decode_mode,
            )
            self._backend = selected_backend
        except Exception as exc:
            exc_info = sys.exc_info()
            _log(
                "live.camera.open.error",
                source=self._source,
                error=type(exc).__name__,
            )
            raise
        finally:
            context.__exit__(*exc_info)

    def frames(self) -> Iterator[tuple[NDArrayU8, float]]:
        assert cv2 is not None and np is not None
        # Narrow Optional[Any] → Any for the loop
        cap = self.cap
        assert cap is not None
        # Some cameras need a couple of warm-up grabs
        warmups = 0
        frame_counter = 0
        start_time = time.time()
        next_report = start_time + 10.0
        reported_raw = False
        while True:
            ok, frame = cap.read()
            if not ok:
                warmups += 1
                if warmups <= 30:
                    # Give the device a moment and retry
                    time.sleep(0.01)
                    if TRACE_CAMERA and warmups in {1, 5, 10, 20, 30}:
                        _log(
                            "live.camera.read.retry",
                            source=getattr(self, "_source", "unknown"),
                            warmups=warmups,
                    )
                    continue
                _log("live.camera.read.error", source=getattr(self, "_source", "unknown"), warmups=warmups)
                break
            if not reported_raw:
                raw_shape = getattr(frame, "shape", None)
                raw_dtype = getattr(frame, "dtype", None)
                raw_str = str(raw_shape) if raw_shape is not None else None
                if np is None:  # type: ignore[comparison-overlap]
                    raw_min = raw_max = raw_mean = None
                else:
                    try:
                        frame_sample = np.asarray(frame)
                        raw_min = float(np.min(frame_sample))
                        raw_max = float(np.max(frame_sample))
                        raw_mean = float(np.mean(frame_sample))
                    except Exception:
                        raw_min = raw_max = raw_mean = None
                _log(
                    "live.camera.frame.raw",
                    source=getattr(self, "_source", "unknown"),
                    shape=raw_str,
                    dtype=str(raw_dtype) if raw_dtype is not None else None,
                    decode_mode=getattr(self, "_decode_mode", None),
                    convert_rgb_forced=self._convert_rgb_forced,
                    convert_prop=self._convert_rgb_prop,
                    convert_rgb_enabled=self._convert_rgb_enabled,
                    backend=str(self._backend) if self._backend is not None else None,
                    raw_min=raw_min,
                    raw_max=raw_max,
                    raw_mean=raw_mean,
                )
                reported_raw = True
            current_mode = getattr(self, "_decode_mode", None)
            arr = np.asarray(frame)
            channels = arr.shape[2] if arr.ndim == 3 else 1
            if frame_counter <= 5 or TRACE_CAMERA:
                try:
                    raw_min = float(arr.min())
                    raw_max = float(arr.max())
                    raw_mean = float(arr.mean())
                    raw_std = float(arr.std())
                except Exception:
                    raw_min = raw_max = raw_mean = raw_std = None
                _log(
                    "live.camera.frame.raw.stats",
                    source=getattr(self, "_source", "unknown"),
                    idx=frame_counter,
                    shape=str(arr.shape),
                    dtype=str(arr.dtype),
                    min=raw_min,
                    max=raw_max,
                    mean=raw_mean,
                    std=raw_std,
                    mode=current_mode,
                    convert_rgb=self._convert_rgb_enabled,
                    channels=channels,
                )
            if current_mode in {"yuy", "nv12", "nv21"} and channels >= 3:
                if TRACE_CAMERA:
                    _log(
                        "live.camera.decode.mode.clear",
                        source=getattr(self, "_source", "unknown"),
                        previous_mode=current_mode,
                        reason=f"{channels}ch",
                    )
                setattr(self, "_decode_mode", None)
                current_mode = None
                if self._convert_rgb_prop is not None:
                    try:
                        self.cap.set(self._convert_rgb_prop, 1.0)
                    except Exception:
                        pass
                self._convert_rgb_enabled = True
                self._convert_rgb_forced = False
                # Discard this frame and grab the next one with the updated setting.
                continue
            try:
                frame_bgr: NDArrayU8 = _decode_frame_to_bgr(frame, current_mode)
            except Exception as exc:
                _log(
                    "live.camera.decode.error",
                    source=getattr(self, "_source", "unknown"),
                    error=type(exc).__name__,
                    message=str(exc),
                    mode=current_mode,
                )
                # Fallback: restore OpenCV's RGB conversion and retry once.
                if current_mode is not None:
                    if self._convert_rgb_prop is not None and not self._convert_rgb_enabled:
                        try:
                            self.cap.set(self._convert_rgb_prop, 1.0)
                            self._convert_rgb_enabled = True
                            if TRACE_CAMERA:
                                _log(
                                    "live.camera.decode.fallback.convert_rgb",
                                    source=getattr(self, "_source", "unknown"),
                                    set_to=1.0,
                                )
                        except Exception:
                            pass
                    self._convert_rgb_forced = False
                    setattr(self, "_decode_mode", None)
                    _log(
                        "live.camera.decode.fallback",
                        source=getattr(self, "_source", "unknown"),
                        previous_mode=current_mode,
                    )
                    # Drop this frame and retry with new settings.
                    continue
                else:
                    continue
            frame_counter += 1
            now = time.time()
            if TRACE_CAMERA:
                if frame_counter == 1:
                    _log(
                        "live.camera.read.first",
                        source=getattr(self, "_source", "unknown"),
                        shape=f"{frame_bgr.shape[1]}x{frame_bgr.shape[0]}",
                        dtype=str(frame_bgr.dtype),
                        decode=getattr(self, "_decode_mode", None),
                        warmups=warmups,
                        convert_rgb=self._convert_rgb_enabled,
                    )
                elif now >= next_report:
                    elapsed = now - start_time
                    fps_est = frame_counter / elapsed if elapsed > 0 else None
                    _log(
                        "live.camera.read.stats",
                        source=getattr(self, "_source", "unknown"),
                        frames=frame_counter,
                        elapsed=f"{elapsed:.2f}",
                        fps=f"{fps_est:.2f}" if fps_est is not None else None,
                        warmups=warmups,
                    )
                    next_report = now + 10.0
            try:
                frame_min = float(frame_bgr.min())
                frame_max = float(frame_bgr.max())
                frame_mean = float(frame_bgr.mean())
                frame_std = float(frame_bgr.std())
            except Exception:
                frame_min = frame_max = frame_mean = frame_std = None

            if frame_counter <= 5 or TRACE_CAMERA:
                channels_out: Optional[int] = None
                try:
                    shape_seq = cast(Sequence[int], getattr(frame_bgr, "shape", ()))
                    if len(shape_seq) >= 3:
                        channels_out = int(shape_seq[2])
                except Exception:
                    channels_out = None
                _log(
                    "live.camera.frame.stats",
                    source=getattr(self, "_source", "unknown"),
                    idx=frame_counter,
                    min=frame_min,
                    max=frame_max,
                    mean=frame_mean,
                    std=frame_std,
                    decode=getattr(self, "_decode_mode", None),
                    convert_rgb=self._convert_rgb_enabled,
                    channels=channels_out,
                )

            if frame_std is not None and frame_std < 5.0:
                self._flat_frame_streak += 1
            else:
                self._flat_frame_streak = 0

            if self._flat_frame_streak >= 3 and self._convert_rgb_prop is not None:
                streak_reason = "low-variance"
                if not self._convert_rgb_enabled:
                    try:
                        self.cap.set(self._convert_rgb_prop, 1.0)
                        self._convert_rgb_enabled = True
                        self._convert_rgb_forced = False
                        setattr(self, "_decode_mode", None)
                        _log(
                            "live.camera.decode.convert_rgb.auto_enable",
                            source=getattr(self, "_source", "unknown"),
                            idx=frame_counter,
                            reason=streak_reason,
                        )
                        self._flat_frame_streak = 0
                        continue
                    except Exception as conv_exc:
                        _log(
                            "live.camera.decode.convert_rgb.auto_enable.error",
                            source=getattr(self, "_source", "unknown"),
                            idx=frame_counter,
                            reason=streak_reason,
                            error=type(conv_exc).__name__,
                        )
                        self._flat_frame_streak = 0
                else:
                    # Try toggling back to the original raw mode if available.
                    if self._initial_decode_mode:
                        try:
                            self.cap.set(self._convert_rgb_prop, 0.0)
                        except Exception:
                            pass
                        self._convert_rgb_enabled = False
                        self._convert_rgb_forced = True
                        setattr(self, "_decode_mode", self._initial_decode_mode)
                        _log(
                            "live.camera.decode.convert_rgb.auto_disable",
                            source=getattr(self, "_source", "unknown"),
                            idx=frame_counter,
                            reason=streak_reason,
                            mode=self._initial_decode_mode,
                        )
                        self._flat_frame_streak = 0
                        continue
                # Avoid rapid toggling if property unsupported.
                self._flat_frame_streak = 0

            yield frame_bgr, time.time()

    def release(self) -> None:
        cap = getattr(self, "cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


class _SyntheticSource(FrameSource):
    """Deterministic gradient + overlayed clock; great for CI/headless."""

    def __init__(self, size: Tuple[int, int] = (640, 480), fps: int = 30) -> None:
        if np is None:
            raise RuntimeError("numpy is required for the synthetic source.")
        with _progress_simple_status(f"Opening synthetic {size[0]}x{size[1]} @ {fps}fps"):
            self.w, self.h = int(size[0]), int(size[1])
            self.fps = max(1, int(fps))
            self._t0 = time.time()
            self._n = 0
        _log("live.camera.synthetic", size=f"{self.w}x{self.h}", fps=self.fps)

    def frames(self) -> Iterator[tuple[NDArrayU8, float]]:
        assert np is not None
        period = 1.0 / self.fps
        while True:
            now = time.time()
            phase = (now - self._t0)
            # vertical gradient
            y = np.linspace(0, 255, self.h, dtype=np.uint8)[:, None]
            x = np.linspace(0, 255, self.w, dtype=np.uint8)[None, :]
            base = ((y + x) // 2).astype(np.uint8)
            # BGR moving bands
            b = base
            g = ((base.astype(np.int16) + int((math.sin(phase) + 1) * 64)) % 256).astype(np.uint8)
            r = ((base.astype(np.int16) + int((math.cos(phase * 0.7) + 1) * 64)) % 256).astype(np.uint8)
            frame = np.dstack([b, g, r])
            self._n += 1
            yield frame, now
            # simple pacing
            delay = period - (time.time() - now)
            if delay > 0:
                time.sleep(delay)

    def release(self) -> None:
        return


def open_camera(
    source: Union[str, int],
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
) -> FrameSource:
    """
    Open a camera/video source, using OS-specific backend hints when available.
    If OpenCV is unavailable, raises. Use `synthetic_source()` for CI/headless.
    """
    _log("live.camera.request", source=str(source), width=width, height=height, fps=fps)
    if cv2 is None:
        raise RuntimeError("OpenCV not available. Install opencv-python (not headless) or use synthetic_source().")
    return _CVCamera(source, width=width, height=height, fps=fps)


def synthetic_source(size: Tuple[int, int] = (640, 480), fps: int = 30) -> FrameSource:
    """Synthetic source that always works (requires numpy)."""
    return _SyntheticSource(size=size, fps=fps)
