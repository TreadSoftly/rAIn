"""
Lightweight package init.

Avoid importing heavy optional deps (Ultralytics, Torch, OpenCV) at import time.
CLI and submodules should import what they need locally.
"""

from __future__ import annotations

try:
    from importlib.metadata import version as _pkg_version  # Python 3.8+
except Exception:  # pragma: no cover
    _pkg_version = None  # type: ignore[assignment]

__all__ = ["__version__"]

def _detect_version() -> str:
    if _pkg_version is None:
        return "0+unknown"
    try:
        return _pkg_version("Argos")
    except Exception:  # pragma: no cover
        return "0+unknown"

__version__ = _detect_version()
