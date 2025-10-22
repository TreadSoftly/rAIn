"""Minimal diagnostics helpers for Argos.

Importing this module now only ensures that a lightweight file logger exists.
The previous behaviour that patched other modules and emitted extensive
snapshots has been removed to reduce overhead and noise.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

__all__ = ["logger", "log", "get_log_file"]

_LOG = logging.getLogger("panoptes.diagnostics")
_LOG.setLevel(logging.INFO)
_LOG.propagate = False

_log_path: Optional[Path] = None
_initialised = False


def _default_base_dir() -> Path:
    """Return a per-user writable directory for diagnostic artefacts."""
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        if base:
            return Path(base) / "rAIn"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "rAIn"
    xdg_home = os.getenv("XDG_DATA_HOME")
    if xdg_home:
        return Path(xdg_home) / "rAIn"
    return Path.home() / ".local" / "share" / "rAIn"


def _resolve_log_file() -> Path:
    """Derive the output log file path, honouring explicit overrides."""
    file_override = os.getenv("PANOPTES_DIAGNOSTICS_FILE")
    if file_override:
        return Path(file_override).expanduser()

    dir_override = os.getenv("PANOPTES_DIAGNOSTICS_DIR") or os.getenv("PANOPTES_RESULTS_DIR")
    if dir_override:
        return Path(dir_override).expanduser() / "argos_diagnostics.log"

    return _default_base_dir() / "logs" / "argos_diagnostics.log"


def _ensure_handler() -> None:
    """Attach a single file handler to the diagnostics logger."""
    global _initialised, _log_path
    if _initialised:
        return

    path = _resolve_log_file()
    handler: logging.Handler

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    except Exception:
        # Fall back to a guaranteed writable tmp location.
        fallback = Path(tempfile.gettempdir()) / "argos_diagnostics.log"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(fallback, encoding="utf-8")
        path = fallback

    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    _LOG.addHandler(handler)
    _log_path = path
    _initialised = True


def log(message: str, *, level: int = logging.INFO) -> None:
    """Log a message to the diagnostics file."""
    _ensure_handler()
    _LOG.log(level, message)


def get_log_file() -> Optional[Path]:
    """Return the fully resolved diagnostics log file path, if available."""
    _ensure_handler()
    return _log_path


# Public alias so callers can log directly via panoptes.diagnostics.logger.
logger = _LOG

_ensure_handler()
