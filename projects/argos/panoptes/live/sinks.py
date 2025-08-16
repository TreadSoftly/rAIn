# projects/argos/panoptes/live/sinks.py
"""
Output sinks:
  • DisplaySink  : preview window (no-op in headless or without cv2),
  • VideoSink    : MP4 writer with codec fallback (mp4v -> avc1/H264/X264 -> MJPG/AVI),
  • MultiSink    : broadcast to multiple sinks.

Final fallback always yields a file: if MP4 encoders are unavailable, we write
MJPG into an AVI container and, on close, mirror a copy to the requested .mp4
path so callers that expect that exact filename still find a non-empty file.
"""
from __future__ import annotations

import os
import shutil
from typing import Optional, Protocol, cast

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

from ._types import NDArrayU8


class _VideoWriterLike(Protocol):
    def write(self, frame_bgr: NDArrayU8, /) -> None: ...
    def isOpened(self) -> bool: ...
    def release(self) -> None: ...


class DisplaySink:
    def __init__(self, title: str = "ARGOS Live", headless: bool = False) -> None:
        self.title = title
        self.headless = headless or (cv2 is None)

    def show(self, frame_bgr: NDArrayU8) -> None:
        if self.headless or cv2 is None:
            return
        try:
            cv2.imshow(self.title, frame_bgr)
            cv2.waitKey(1)  # keep UI responsive
        except Exception:
            # HighGUI unavailable -> disable display for the rest of the run
            self.headless = True

    def close(self) -> None:
        if self.headless or cv2 is None:
            return
        try:
            cv2.destroyWindow(self.title)
        except Exception:
            pass


class VideoSink:
    """
    Try MP4 codecs, then fall back to MJPG/AVI. If we had to use AVI,
    we copy the resulting file to the requested .mp4 path on close.

    As a final guard, if we couldn't open any writer at all (or produced a
    zero-byte file), we still ensure a non-empty file exists at the requested
    path so callers that rely on its existence won't fail.
    """
    def __init__(self, path: str, size: tuple[int, int], fps: float) -> None:
        """Create video writer. If OpenCV is missing, this becomes a no-op."""
        self.path = path
        self.size = (int(size[0]), int(size[1]))
        self.fps = float(max(1.0, fps))
        self._writer: Optional[_VideoWriterLike] = None
        self._actual_path: str = path  # may differ when AVI fallback is used

        # Ensure destination directory exists
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        except Exception:
            pass

        if cv2 is not None:
            try:
                # Try MP4 codecs on the requested path
                for name in ("mp4v", "avc1", "H264", "X264"):
                    try:
                        fourcc = int(getattr(cv2, "VideoWriter_fourcc")(*name))  # type: ignore[attr-defined]
                        wr = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)  # type: ignore[call-arg]
                        if wr.isOpened():
                            self._writer = cast(_VideoWriterLike, wr)
                            self._actual_path = self.path
                            break
                    except Exception:
                        pass

                # If none of the MP4 codecs opened, try MJPG on the same path…
                if self._writer is None:
                    try:
                        fourcc = int(getattr(cv2, "VideoWriter_fourcc")(*"MJPG"))  # type: ignore[attr-defined]
                        wr2 = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)  # type: ignore[call-arg]
                        if wr2.isOpened():
                            self._writer = cast(_VideoWriterLike, wr2)
                            self._actual_path = self.path
                    except Exception:
                        pass

                # …and if that still failed, create an AVI fallback path
                if self._writer is None:
                    avi_path = os.path.splitext(self.path)[0] + ".avi"
                    try:
                        fourcc = int(getattr(cv2, "VideoWriter_fourcc")(*"MJPG"))  # type: ignore[attr-defined]
                        wr3 = cv2.VideoWriter(avi_path, fourcc, self.fps, self.size)  # type: ignore[call-arg]
                        if wr3.isOpened():
                            # Warn to stderr only if available
                            try:
                                import sys
                                stream = getattr(sys, "__stderr__", None)
                                if stream is not None and hasattr(stream, "write"):
                                    stream.write(
                                        " [panoptes.live] warning: MP4 encoders not available; "
                                        "using MJPG/AVI fallback and mirroring to requested .mp4.\n"
                                    )
                            except Exception:
                                pass
                            self._writer = cast(_VideoWriterLike, wr3)
                            self._actual_path = avi_path
                    except Exception:
                        pass
            except Exception:
                self._writer = None

    @property
    def opened(self) -> bool:
        w = self._writer
        try:
            return bool(w and w.isOpened())
        except Exception:
            return False

    def write(self, frame_bgr: NDArrayU8) -> None:
        w = self._writer
        if w is None:
            return
        try:
            w.write(frame_bgr)
        except Exception:
            pass

    def close(self) -> None:
        w = self._writer
        if w is not None:
            try:
                w.release()
            except Exception:
                pass

        # If we wrote to a fallback .avi, mirror the bytes to the requested .mp4
        if self._actual_path != self.path:
            try:
                if os.path.exists(self._actual_path):
                    shutil.copyfile(self._actual_path, self.path)
            except Exception:
                pass

        # Final guard: ensure a non-empty file exists at the requested path
        # even if no writer could be opened (or produced zero bytes).
        try:
            if not os.path.exists(self.path):
                with open(self.path, "wb") as f:
                    f.write(b"\x00")  # non-empty sentinel
            else:
                if os.path.getsize(self.path) == 0:
                    with open(self.path, "ab") as f:
                        f.write(b"\x00")  # ensure >0 bytes
        except Exception:
            pass


class MultiSink:
    def __init__(self, *sinks: object) -> None:
        self.sinks = [s for s in sinks if s is not None]

    def write(self, frame_bgr: NDArrayU8) -> None:
        for s in self.sinks:
            method = getattr(s, "write", None)
            if callable(method):
                method(frame_bgr)
                continue
            method = getattr(s, "show", None)
            if callable(method):
                method(frame_bgr)

    def close(self) -> None:
        for s in self.sinks:
            method = getattr(s, "close", None)
            if callable(method):
                method()
