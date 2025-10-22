"""
Lightweight backend availability probes.

These utilities intentionally avoid importing heavy dependencies at module
import time. Callers can use the helpers to decide which model weights to use
without paying the cost of initialising ONNX Runtime sessions.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Tuple, Iterable, Sequence, Callable, cast

import importlib.util as importlib_util

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())
LOG.setLevel(logging.ERROR)


def _load_bootstrap_module() -> Optional[Any]:
    path = Path(__file__).resolve().parents[2] / "bootstrap.py"
    if not path.exists():
        return None
    try:
        spec = importlib_util.spec_from_file_location("argos_bootstrap", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[misc]
        return module
    except Exception:
        LOG.exception("onnx.bootstrap_import_failed")
        return None


_BOOTSTRAP = _load_bootstrap_module()


def _heal_lock_path() -> Path:
    base = Path(os.environ.get("PANOPTES_VENV_ROOT", tempfile.gettempdir()))
    try:
        lock_dir = base / "tmp"
        lock_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        lock_dir = Path(tempfile.gettempdir())
    return lock_dir / "onnx_heal.lock"


@contextmanager
def _heal_lock(lock_path: Path):
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        yield False
        return
    try:
        os.write(fd, str(os.getpid()).encode())
    finally:
        os.close(fd)
    try:
        yield True
    finally:
        try:
            lock_path.unlink()
        except Exception:
            pass


def _try_import_ort() -> Tuple[bool, Optional[str], Optional[list[str]], Optional[str]]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        return False, None, None, f"{type(exc).__name__}: {exc}"
    version = getattr(ort, "__version__", None)
    try:
        providers_getter = cast(Optional[Callable[[], Sequence[str]]], getattr(ort, "get_available_providers", None))
        if providers_getter is None:
            return True, version, None, "providers unavailable: missing get_available_providers"
        providers_iter = cast(Iterable[str], providers_getter())
        providers: list[str] = list(providers_iter)
    except Exception as exc:
        return True, version, None, f"providers unavailable: {exc}"
    return True, version, providers, None


def torch_available() -> bool:
    """Return True if ``torch`` can be imported."""
    try:
        import torch  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def ort_available() -> Tuple[bool, Optional[str], Optional[list[str]], Optional[str]]:
    """Attempt to import ONNX Runtime and report availability details."""
    if os.environ.get("ARGOS_DISABLE_ONNX"):
        return False, None, None, "disabled via ARGOS_DISABLE_ONNX"

    ok, version, providers, reason = _try_import_ort()
    if ok:
        return True, version, providers, reason

    heal_reason = reason
    summary: Optional[dict[str, Any]] = None

    if os.name == "nt" and _BOOTSTRAP is not None and hasattr(_BOOTSTRAP, "ensure_onnxruntime"):
        lock_path = _heal_lock_path()
        with _heal_lock(lock_path) as acquired:
            if acquired:
                try:
                    ensure = getattr(_BOOTSTRAP, "ensure_onnxruntime")
                    venv_py = None
                    if hasattr(_BOOTSTRAP, "venv_python"):
                        try:
                            venv_py = _BOOTSTRAP.venv_python()
                        except Exception as exc:
                            LOG.exception("onnx.heal.venv_python_failed: %s", exc)
                    def _heal_log(msg: str) -> None:
                        msg_lower = msg.lower()
                        if "error" in msg_lower or "fail" in msg_lower:
                            LOG.error("onnx.heal %s", msg)
                    summary = ensure(venv_py, log=_heal_log)
                except Exception as exc:
                    LOG.exception("onnx.heal.ensure_exception: %s", exc)
                    summary = {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
            else:
                summary = {"installed": False, "error": f"heal already in progress ({lock_path})"}

        if summary:
            if summary.get("error"):
                try:
                    LOG.error("onnx.heal.summary %s", json.dumps(summary, default=str))
                except Exception:
                    LOG.error("onnx.heal.summary %s", summary)
            if summary.get("installed"):
                ok, version, providers, reason = _try_import_ort()
                if ok:
                    return True, version, providers, reason
            heal_reason = summary.get("error") or heal_reason
            if summary.get("providers") and not providers:
                providers = list(summary.get("providers") or [])
            if summary.get("ort_version") and not version:
                version = str(summary.get("ort_version"))

    return False, version, providers, heal_reason
