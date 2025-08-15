# \rAIn\projects\argos\panoptes\__init__.py
"""
Lightweight package init.

Avoid importing heavy optional deps (Ultralytics, Torch, OpenCV) at import time.
CLI and submodules should import what they need locally.

Exports:
    __version__ : best-effort package version (falls back to "0+unknown")
    ROOT        : repository project root (…/projects/argos)
"""

from __future__ import annotations

from pathlib import Path

try:
    from importlib.metadata import version as _pkg_version  # Python 3.8+
except Exception:  # pragma: no cover
    _pkg_version = None  # type: ignore[assignment]

__all__ = ["__version__", "ROOT"]


def _detect_version() -> str:
    """
    Try both "Argos" and normalized "argos" distribution names,
    since metadata names are case-insensitive but may vary in CI.
    """
    if _pkg_version is None:
        return "0+unknown"
    for dist in ("Argos", "argos"):
        try:
            return _pkg_version(dist)
        except Exception:
            continue
    return "0+unknown"


# Export the project root for CLI modules that expect it.
# panoptes/__init__.py → parent is /panoptes; parent of that is the project root (/projects/argos)
ROOT = Path(__file__).resolve().parents[1]

__version__ = _detect_version()
