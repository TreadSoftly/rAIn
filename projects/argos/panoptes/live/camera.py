# projects/argos/panoptes/live/camera.py
"""
Live capture sources for ARGOS.

Prefers OpenCV backends with OS hints (DSHOW/AVFOUNDATION/V4L2/MSMF) and falls back
quietly. Also provides a deterministic "synthetic" source for CI/headless runs.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Iterator, Optional, Protocol, Tuple, Union, cast

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
except Exception as _e:  # pragma: no cover
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

from ._types import NDArrayU8


class FrameSource(Protocol):
    def frames(self) -> Iterator[tuple[NDArrayU8, float]]: ...
    def release(self) -> None: ...


def _guess_backend_ids() -> list[int]:
    """Return a list of platform-appropriate OpenCV backend ids (try in order)."""
    ids: list[int] = []
    if cv2 is None:
        return ids
    plat = sys.platform
    if plat.startswith("win"):
        # Try DSHOW, then MSMF, then any/default.
        for name in ("CAP_DSHOW", "CAP_MSMF"):
            ids.append(getattr(cv2, name, 0))
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

        last_err: Optional[Exception] = None
        with _progress_simple_status(f"Opening source {source!r}"):
            self.cap = None
            for be in backends:
                try:
                    self.cap = cv2.VideoCapture(src_val, be) if be else cv2.VideoCapture(src_val)
                    if self.cap and self.cap.isOpened():
                        break
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                except Exception as e:
                    last_err = e
                    self.cap = None
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera/source: {source}") from last_err

            # Try to set properties (best effort)
            if width is not None:
                self.cap.set(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3), float(width))
            if height is not None:
                self.cap.set(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4), float(height))
            if fps is not None:
                self.cap.set(getattr(cv2, "CAP_PROP_FPS", 5), float(fps))

    def frames(self) -> Iterator[tuple[NDArrayU8, float]]:
        assert cv2 is not None and np is not None
        # Narrow Optional[Any] → Any for the loop
        cap = self.cap
        assert cap is not None
        # Some cameras need a couple of warm-up grabs
        warmups = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                warmups += 1
                if warmups <= 30:
                    # Give the device a moment and retry
                    time.sleep(0.01)
                    continue
                break
            # Ensure dtype=uint8 and narrow the type for the checker
            if getattr(frame, "dtype", None) is not None and frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            yield cast(NDArrayU8, frame), time.time()

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
    if cv2 is None:
        raise RuntimeError("OpenCV not available. Install opencv-python (not headless) or use synthetic_source().")
    return _CVCamera(source, width=width, height=height, fps=fps)


def synthetic_source(size: Tuple[int, int] = (640, 480), fps: int = 30) -> FrameSource:
    """Synthetic source that always works (requires numpy)."""
    return _SyntheticSource(size=size, fps=fps)
