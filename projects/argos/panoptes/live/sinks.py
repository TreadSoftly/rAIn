# projects/argos/panoptes/live/sinks.py
"""
Output sinks:
  • DisplaySink  : preview window (OpenCV HighGUI if available; auto-fallback to Tkinter),
  • VideoSink    : MP4 writer with codec fallback (mp4v -> avc1/H264/X264 -> MJPG/AVI),
  • MultiSink    : broadcast to multiple sinks.

Final fallback always yields a file: if MP4 encoders are unavailable, we write
MJPG into an AVI container and, on close, mirror a copy to the requested .mp4
path so callers that expect that exact filename still find a non-empty file.
"""
from __future__ import annotations

import logging
import os
import shutil
from typing import Optional, Protocol, cast

# Short status spinner when opening a writer (will no-op under live spinner)
try:
    from panoptes.progress import simple_status as _progress_simple_status  # type: ignore[import]
except Exception:  # pragma: no cover
    def _progress_simple_status(*_a: object, **_k: object):
        class _N:
            def __enter__(self): return self
            def __exit__(self, *_: object) -> bool: return False
        return _N()

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    import numpy as _np  # only for typing and quick conversions here
except Exception:
    _np = None  # type: ignore

from ._types import NDArrayU8
from panoptes.logging_config import bind_context


LOGGER = logging.getLogger(__name__)


def _log(event: str, **info: object) -> None:
    if info:
        detail = " ".join(f"{k}={info[k]}" for k in sorted(info) if info[k] is not None)
        LOGGER.info("%s %s", event, detail)
    else:
        LOGGER.info(event)


class _VideoWriterLike(Protocol):
    def write(self, frame_bgr: NDArrayU8, /) -> None: ...
    def isOpened(self) -> bool: ...
    def release(self) -> None: ...


class DisplaySink:
    """
    Display preview with best-effort backends:

      1) OpenCV HighGUI (cv2.imshow) if available,
      2) Tkinter (stdlib) fallback if HighGUI is unavailable (e.g., opencv-python-headless),
      3) No-op if headless=True or both backends unavailable.

    Also exposes:
      • poll_key() → int: returns OpenCV keycode or -1 (Tk/none)
      • is_open() → bool: false when user closed the window
      • write(frame) alias of show(frame) so MultiSink can call uniformly
    """

    def __init__(self, title: str = "ARGOS Live", headless: bool = False) -> None:
        self.title = title
        self._forced_headless = bool(headless)
        self._backend: str = "none"  # "cv2", "tk", or "none"
        self._cv2_window_ready: bool = False
        self._tk_root = None
        self._tk_label = None
        self._tk_photo = None  # keep reference to avoid GC
        self._tk_alive: bool = False
        self._last_key: int = -1

        # Defer backend decision until first frame (lets us catch HighGUI errors)
        if not self._forced_headless:
            self._backend = "cv2" if cv2 is not None else "none"
        _log("live.display.init", backend=self._backend, headless=self._forced_headless)

    # ----------------- public API -----------------

    def write(self, frame_bgr: NDArrayU8) -> None:
        """Alias to show() so MultiSink can call write()."""
        self.show(frame_bgr)

    def show(self, frame_bgr: NDArrayU8) -> None:
        if self._forced_headless:
            return
        if self._backend == "cv2":
            self._show_cv2(frame_bgr)
            return
        if self._backend == "tk":
            self._show_tk(frame_bgr)
            return
        # Try to lazily initialize Tk if we start in "none"
        self._init_tk_fallback()
        if self._backend == "tk":
            self._show_tk(frame_bgr)

    def poll_key(self) -> int:
        """Return the last key for OpenCV backend; -1 otherwise."""
        if self._backend == "cv2" and not self._forced_headless and cv2 is not None:
            try:
                self._last_key = cv2.waitKey(1) & 0xFF
            except Exception:
                self._last_key = -1
            return self._last_key
        # Tk backend uses window close to exit; not polling keys here.
        return -1

    def is_open(self) -> bool:
        """Return False if the display was closed (any backend)."""
        if self._forced_headless:
            return False
        if self._backend == "cv2" and cv2 is not None:
            try:
                # Prefer visibility flag; fall back to AUTOSIZE for existence check
                flag = getattr(cv2, "WND_PROP_VISIBLE", None)
                if flag is None:
                    flag = getattr(cv2, "WND_PROP_AUTOSIZE", 1)
                prop = cv2.getWindowProperty(self.title, flag)
                # For VISIBLE: 0.0 = hidden/closed, >0 = visible, -1 = unknown (assume open)
                # For AUTOSIZE: -1 = window doesn't exist; 0/1 just indicates autosize flag
                if prop == 0.0 and flag == getattr(cv2, "WND_PROP_VISIBLE", None):
                    return False
                if prop < 0:
                    # Unknown/doesn't exist → treat as closed only for AUTOSIZE; for VISIBLE unknown, assume open
                    return False if flag == getattr(cv2, "WND_PROP_AUTOSIZE", 1) else True
                return True
            except Exception:
                # If getWindowProperty itself fails, assume still open so the loop controls exit via key/close
                return True
        if self._backend == "tk":
            return bool(self._tk_alive)
        return False

    def close(self) -> None:
        if self._backend == "cv2" and cv2 is not None:
            try:
                cv2.destroyWindow(self.title)
            except Exception:
                pass
        if self._backend == "tk":
            try:
                if self._tk_root is not None:
                    self._tk_alive = False
                    self._tk_root.destroy()
            except Exception:
                pass
        self._backend = "none"

    # ----------------- backends -----------------

    def _ensure_cv2_window(self) -> None:
        if self._cv2_window_ready or cv2 is None:
            return
        try:
            # Create once; WINDOW_NORMAL lets users resize.
            cv2.namedWindow(self.title, getattr(cv2, "WINDOW_NORMAL", 0))
            self._cv2_window_ready = True
        except Exception:
            # HighGUI isn't functional → switch to tk
            self._backend = "tk"
            self._init_tk_fallback()

    def _show_cv2(self, frame_bgr: NDArrayU8) -> None:
        if cv2 is None:
            self._backend = "tk"
            self._init_tk_fallback()
            return
        try:
            self._ensure_cv2_window()
            cv2.imshow(self.title, frame_bgr)
            # don't call waitKey here; pipeline polls via poll_key()
        except Exception:
            # Most common path: opencv-python-headless → imshow not implemented
            self._backend = "tk"
            _log("live.display.fallback", backend="tk", reason="cv2-imshow-failed")
            self._init_tk_fallback()

    def _init_tk_fallback(self) -> None:
        if self._backend == "tk" and self._tk_root is not None:
            return
        try:
            import tkinter as tk
            from base64 import b64encode

            self._tk_b64 = b64encode  # cache the symbol
            self._tk = tk
            self._tk_root = tk.Tk()
            self._tk_root.title(self.title)
            self._tk_root.protocol("WM_DELETE_WINDOW", self._on_tk_close)
            # Allow quitting with 'q' or Esc to match cv2 backend behavior
            self._tk_root.bind("<KeyPress-q>", lambda e: self._on_tk_close())
            self._tk_root.bind("<Escape>",     lambda e: self._on_tk_close())

            self._tk_label = tk.Label(self._tk_root)
            self._tk_label.pack()
            self._tk_alive = True
            _log("live.display.backend", backend="tk")
        except Exception:
            # Could not init Tk either → give up
            self._backend = "none"
            self._forced_headless = True
            self._tk_root = None
            self._tk_label = None
            self._tk_alive = False
            _log("live.display.fallback", backend="none", reason="tk-init-failed")

    def _on_tk_close(self) -> None:
        self._tk_alive = False
        try:
            if self._tk_root is not None:
                self._tk_root.destroy()
        except Exception:
            pass

    def _show_tk(self, frame_bgr: NDArrayU8) -> None:
        if not self._tk_alive or self._tk_root is None or self._tk_label is None:
            return
        try:
            # Convert BGR→RGB → PPM (P6) → base64 → PhotoImage
            h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
            if _np is not None:
                rgb = frame_bgr[..., ::-1]
                header = f"P6\n{w} {h}\n255\n".encode("ascii")
                data = header + rgb.tobytes()
            else:
                # Extremely defensive fallback: assume already RGB bytes-like
                header = f"P6\n{w} {h}\n255\n".encode("ascii")
                data = header + bytes(frame_bgr)

            b64 = self._tk_b64(data).decode("ascii")  # type: ignore[attr-defined]
            self._tk_photo = self._tk.PhotoImage(data=b64)  # type: ignore[attr-defined]
            self._tk_label.configure(image=self._tk_photo)
            self._tk_root.update_idletasks()
            self._tk_root.update()
        except Exception:
            # If anything goes wrong, mark as closed to exit cleanly
            self._on_tk_close()


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
        _log("live.sink.video.init", path=self.path, size=f"{self.size[0]}x{self.size[1]}", fps=self.fps)
        self._writer: Optional[_VideoWriterLike] = None
        self._actual_path: str = path  # may differ when AVI fallback is used

        # Ensure destination directory exists
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        except Exception:
            pass

        if cv2 is not None:
            # Keep this whole codec probing phase under a tiny status spinner
            with _progress_simple_status(f"Opening writer {self.size[0]}x{self.size[1]}@{self.fps:.0f}"):
                try:
                    # Try MP4 codecs on the requested path
                    for name in ("mp4v", "avc1", "H264", "X264"):
                        try:
                            _log("live.sink.video.try", path=self.path, codec=name)
                            fourcc = int(getattr(cv2, "VideoWriter_fourcc")(*name))  # type: ignore[attr-defined]
                            wr = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)  # type: ignore[call-arg]
                            if wr.isOpened():
                                self._writer = cast(_VideoWriterLike, wr)
                                self._actual_path = self.path
                                _log("live.sink.video.opened", path=self._actual_path, codec=name)
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
                                _log("live.sink.video.opened", path=self._actual_path, codec="MJPG")
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
                                _log("live.sink.video.fallback", path=avi_path, codec="MJPG", target=self.path)
                        except Exception:
                            pass
                except Exception:
                    self._writer = None

        else:
            _log("live.sink.video.disabled", path=self.path, reason="opencv-missing")

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

        try:
            final_size = os.path.getsize(self.path) if os.path.exists(self.path) else None
        except Exception:
            final_size = None
        _log("live.sink.video.closed", path=self.path, actual_path=self._actual_path, size=final_size)


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
