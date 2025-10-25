"""
Ensure Argos CLI entry points always run inside the managed virtual environment.

When users invoke ``python -m panoptes.live.cli`` (or any of the repo shims) the
sys.executable may still resolve to a system wide interpreter.  On clean
machines that interpreter will not have Torch / ONNX Runtime installed which in
turn breaks live video.  This helper re-execs the current module inside the
per-user Argos venv whenever it detects the mismatch.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Optional

_ENV_FLAG = "PANOPTES_ALREADY_REEXEC"


def _managed_python_path(expected_rel: str) -> Optional[Path]:
    """
    Resolve the managed venv python using %LOCALAPPDATA% as the root.

    Returns None if the path cannot be resolved or does not exist.
    """
    root = os.environ.get("LOCALAPPDATA")
    if not root:
        return None
    candidate = Path(root) / expected_rel
    try:
        candidate = candidate.expanduser().resolve()
    except Exception:
        return None
    return candidate if candidate.exists() else None


def _discover_entry_module() -> Optional[str]:
    """
    Best-effort detection of the module that should be relaunched with ``-m``.

    When a package is executed via ``python -m`` the ``__main__`` module retains
    a ``__spec__`` attribute that contains the fully-qualified name.
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return None

    spec = getattr(main_mod, "__spec__", None)
    if spec is not None:
        name = getattr(spec, "name", None)
        if name and name != "__main__":
            return name  # type: ignore[no-any-return]

    pkg = getattr(main_mod, "__package__", None)
    if pkg:
        name = getattr(main_mod, "__name__", None)
        if name and name != "__main__":
            return name
    return None


def maybe_reexec_into_managed_venv(
    expected_rel: str = r"rAIn\venvs\py312-argos\Scripts\python.exe",
) -> None:
    """
    If the current interpreter is not the Argos-managed venv, re-exec into it.

    The operation is idempotent thanks to the ``PANOPTES_ALREADY_REEXEC`` flag.
    Any failure to probe/re-exec is swallowed intentionally to avoid blocking
    startup when running in unconventional environments.
    """
    if os.environ.get("PANOPTES_DISABLE_REEXEC") == "1":
        return

    if os.environ.get(_ENV_FLAG):
        return

    # During pytest runs we want to stay inside the invoking interpreter so test
    # doubles and monkeypatches keep working predictably.  Allow opt-in via env.
    if (
        os.environ.get("PANOPTES_ENABLE_REEXEC_UNDER_PYTEST") != "1"
        and "pytest" in sys.modules
    ):
        return

    try:
        managed_python = _managed_python_path(expected_rel)
        if managed_python is None:
            return

        try:
            current = Path(sys.executable).resolve()
        except Exception:
            current = Path(sys.executable)

        if current == managed_python:
            return

        entry_module = _discover_entry_module()
        if not entry_module:
            return

        os.environ[_ENV_FLAG] = "1"
        argv = [str(managed_python), "-m", entry_module, *sys.argv[1:]]
        msg = f"Re-exec into Argos venv: {managed_python}"
        try:
            sys.stderr.write(msg + os.linesep)
        except Exception:
            pass

        os.execv(str(managed_python), argv)
    except Exception:
        # Never let bootstrap failures prevent the CLI from running.  Logging
        # would require the full logging stack; opt for a silent failure with a
        # hidden traceback to ease debugging when the environment variable is
        # toggled.
        if os.environ.get("PANOPTES_RUNTIME_DEBUG"):
            traceback.print_exc()
        return
