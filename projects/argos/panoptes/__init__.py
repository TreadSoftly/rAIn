# \rAIn\projects\argos\panoptes\__init__.py
"""
Lightweight package init.

Avoid importing heavy optional deps (Ultralytics, Torch, OpenCV) at import time.
CLI and submodules should import what they need locally.

Exports:
    __version__ : best-effort package version (falls back to "0+unknown")
    ROOT        : repository project root (./projects/argos)
    results_dir : helper that honours PANOPTES_RESULTS_DIR
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from importlib.metadata import version as _pkg_version  # Python 3.8+
except Exception:  # pragma: no cover - Python < 3.8
    _pkg_version = None  # type: ignore[assignment]

__all__ = ["__version__", "ROOT", "results_dir"]


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
ROOT = Path(__file__).resolve().parents[1]

__version__ = _detect_version()


def _compute_results_dir() -> Path:
    """
    Mirror CLI behaviour for locating the results directory while honouring
    PANOPTES_RESULTS_DIR when set.  Always attempts to create the directory.
    """
    override = os.getenv("PANOPTES_RESULTS_DIR")
    if override:
        base = Path(override).expanduser()
    else:
        base = ROOT / "tests" / "results"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        return base.resolve()
    except Exception:
        return base


def results_dir() -> Path:
    """
    Return the preferred results directory, ensuring it exists.
    """
    return _compute_results_dir()
