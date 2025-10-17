"""
Utility helpers for locating an ffmpeg executable.

We prefer a deterministic, cross-platform search that does not assume ffmpeg
is on PATH. The order mirrors the logic historically embedded in the CLI:

1. Honour the ``FFMPEG_BINARY`` environment variable when it points to a file.
2. Look for ``ffmpeg`` on PATH via ``shutil.which``.
3. Fall back to the bundled binary from ``imageio-ffmpeg`` when available.

The helpers return both the resolved executable path (or ``None``) and a short
string describing the source used, which allows callers to emit diagnostics.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple


def resolve_ffmpeg() -> Tuple[Optional[str], str]:
    """
    Return ``(path, source)`` where *path* is the ffmpeg executable to use.

    *source* is a short label among ``{"env", "path", "imageio", "missing"}``
    that describes where the executable came from. When no executable can be
    located, *path* is ``None`` and *source* is ``"missing"``.
    """
    env = os.environ.get("FFMPEG_BINARY", "").strip()
    if env:
        candidate = Path(env)
        if candidate.exists():
            return str(candidate), "env"

    exe = shutil.which("ffmpeg")
    if exe:
        return exe, "path"

    try:
        import imageio_ffmpeg  # type: ignore

        raw = imageio_ffmpeg.get_ffmpeg_exe()  # type: ignore[attr-defined]
        if isinstance(raw, str) and Path(raw).exists():
            return raw, "imageio"
    except Exception:
        pass

    return None, "missing"

