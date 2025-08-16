# projects/argos/panoptes/live/camera.py
"""
Live capture sources for ARGOS.

Prefers OpenCV backends with OS hints (DSHOW/AVFOUNDATION/V4L2) and falls back
quietly. Also provides a deterministic "synthetic" source for CI/headless runs.
"""

from __future__ import annotations

from typing import Iterator, Protocol, Tuple, Optional, Union, cast
import time
import sys
import math

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


def _guess_backend_id() -> int:
    """Pick a platform-appropriate OpenCV backend if available."""
    if cv2 is None:
        return 0
    plat = sys.platform
    if plat.startswith("win"):
        return getattr(cv2, "CAP_DSHOW", 0)
    if plat == "darwin":
        return getattr(cv2, "CAP_AVFOUNDATION", 0)
    if "linux" in plat:
        return getattr(cv2, "CAP_V4L2", 0)
    return 0


class _CVCamera(FrameSource):
    def __init__(
        self,
        source: Union[str, int],
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV not available for camera capture.")
        backend = _guess_backend_id()
        # Index vs path (support negative indices like -1)
        if isinstance(source, str):
            try:
                src_val: Union[str, int] = int(source)
            except ValueError:
                src_val = source
        else:
            src_val = source
        # Prefer backend only for index sources
        if isinstance(src_val, int) and backend:
            self.cap = cv2.VideoCapture(src_val, backend)
        else:
            self.cap = cv2.VideoCapture(src_val)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera/source: {source}")

        # Try to set properties (best effort)
        if width is not None:
            self.cap.set(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3), float(width))
        if height is not None:
            self.cap.set(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4), float(height))
        if fps is not None:
            self.cap.set(getattr(cv2, "CAP_PROP_FPS", 5), float(fps))

    def frames(self) -> Iterator[tuple[NDArrayU8, float]]:
        assert cv2 is not None and np is not None
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            # Ensure dtype=uint8 and narrow the type for the checker
            if getattr(frame, "dtype", None) is not None and frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            yield cast(NDArrayU8, frame), time.time()

    def release(self) -> None:
        if getattr(self, "cap", None) is not None:
            try:
                self.cap.release()
            except Exception:
                pass


class _SyntheticSource(FrameSource):
    """Deterministic gradient + overlayed clock; great for CI/headless."""

    def __init__(self, size: Tuple[int, int] = (640, 480), fps: int = 30) -> None:
        if np is None:
            raise RuntimeError("numpy is required for the synthetic source.")
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
        raise RuntimeError("OpenCV not available. Install opencv-python or use synthetic_source().")
    return _CVCamera(source, width=width, height=height, fps=fps)


def synthetic_source(size: Tuple[int, int] = (640, 480), fps: int = 30) -> FrameSource:
    """Synthetic source that always works (requires numpy)."""
    return _SyntheticSource(size=size, fps=fps)
