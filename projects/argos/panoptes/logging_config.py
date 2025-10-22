"""Minimal logging helpers for Argos.

This module keeps only the essentials required by the rest of the codebase:

* ``setup_logging`` initialises a single error-level file handler.
* ``bind_context`` is a lightweight no-op context manager for compatibility.
* ``current_run_dir`` / ``current_run_id`` expose the location used for logs.

All rich formatting, environment snapshots, and dynamic context decoration have
been removed to minimise overhead and keep logging output to the bare minimum.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
os.environ.setdefault("ORT_LOGGING_SEVERITY_LEVEL", "3")
try:
    import onnxruntime as _ort  # type: ignore

    if hasattr(_ort, "set_default_logger_severity"):
        _ort.set_default_logger_severity(3)
except Exception:
    pass

__all__ = [
    "bind_context",
    "current_run_dir",
    "current_run_id",
    "setup_logging",
]

_CONFIGURED = False
_RUN_DIR: Optional[Path] = None
_RUN_ID: Optional[str] = None
_LOG_PATH: Optional[Path] = None


def _platform_data_dir() -> Path:
    """Return a per-user writable application data directory."""
    app = "rAIn"
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or (Path.home() / "AppData" / "Local"))
        return base / app
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app
    xdg_home = os.getenv("XDG_DATA_HOME")
    if xdg_home:
        return Path(xdg_home) / app
    return Path.home() / ".local" / "share" / app


def _default_logs_dir() -> Path:
    """Compute the default directory used for log artefacts."""
    return _platform_data_dir() / "logs"


def current_run_dir() -> Optional[Path]:
    """Return the directory that currently holds log artefacts."""
    return _RUN_DIR


def current_run_id() -> Optional[str]:
    """Return a simple identifier for the active logging run."""
    return _RUN_ID


def _resolve_log_path(file_env: str) -> Path:
    override = os.getenv(file_env)
    if override:
        path = Path(override).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    logs_dir = _default_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / "argos.log"


def setup_logging(
    *,
    level_env: str = "ARGOS_LOG_LEVEL",
    file_env: str = "ARGOS_LOG_FILE",
) -> Path:
    """
    Configure the root logger with a single file handler.

    The handler logs ERROR and higher messages to ``argos.log`` (or the path
    specified by ``ARGOS_LOG_FILE``). The function is idempotent: repeated calls
    return the previously configured log directory without reconfiguring.
    """
    global _CONFIGURED, _RUN_DIR, _RUN_ID, _LOG_PATH

    if _CONFIGURED:
        return _RUN_DIR if _RUN_DIR is not None else _default_logs_dir()

    log_path = _resolve_log_path(file_env)
    _LOG_PATH = log_path
    _RUN_DIR = log_path.parent
    _RUN_ID = "current"

    level_name = os.getenv(level_env, "ERROR").upper().strip()
    level = getattr(logging, level_name, logging.ERROR)
    if level < logging.ERROR:
        # Clamp to ERROR to keep output minimal even if env is misconfigured.
        level = logging.ERROR

    handler: logging.Handler
    try:
        handler = logging.FileHandler(log_path, encoding="utf-8")
    except Exception:
        fallback = Path(tempfile.gettempdir()) / "argos.log"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(fallback, encoding="utf-8")
        _LOG_PATH = fallback
        _RUN_DIR = fallback.parent

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # Silence Ultralytics logging noise (it tends to spam INFO banners on load).
    for name in (
        "ultralytics",
        "ultralytics.yolo",
        "ultralytics.yolo.engine.model",
        "ultralytics.yolo.utils",
        "ultralytics.nn.autobackend",
    ):
        try:
            lg = logging.getLogger(name)
            lg.handlers.clear()
            lg.setLevel(logging.ERROR)
            lg.propagate = False
        except Exception:
            pass
    try:
        from ultralytics.utils import LOGGER as _ULY_LOGGER  # type: ignore

        try:
            _ULY_LOGGER.remove()
        except Exception:
            for h in list(_ULY_LOGGER.handlers):
                _ULY_LOGGER.removeHandler(h)
        _ULY_LOGGER.setLevel(logging.ERROR)
    except Exception:
        pass

    _CONFIGURED = True
    return _RUN_DIR if _RUN_DIR is not None else _default_logs_dir()


@contextmanager
def bind_context(**_: object) -> Iterator[None]:
    """Compatibility no-op context manager used by existing call sites."""
    yield


def get_log_path() -> Optional[Path]:
    """Expose the resolved log file path for modules that need it."""
    return _LOG_PATH
