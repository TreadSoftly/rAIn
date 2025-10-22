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
from typing import Any, Dict, Optional, Tuple, Iterable, Sequence, Callable, cast

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
_cap_cache: Optional[Dict[str, Any]] = None


def _get_capabilities() -> Optional[Dict[str, Any]]:
    global _cap_cache
    if _cap_cache is not None:
        return _cap_cache
    if _BOOTSTRAP is None or not hasattr(_BOOTSTRAP, "read_capabilities"):
        return None
    try:
        data = _BOOTSTRAP.read_capabilities()  # type: ignore[call-arg]
        if isinstance(data, dict):
            _cap_cache = cast(Dict[str, Any], data)
            return _cap_cache
    except Exception:
        LOG.exception("capabilities.read_failed")
    return None


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


def _log_ort_status(
    ok: bool,
    version: Optional[str],
    providers: Optional[Sequence[str]],
    reason: Optional[str],
    accelerator_hint: str,
) -> None:
    providers_list = list(providers or [])
    if ok:
        if LOG.isEnabledFor(logging.INFO):
            LOG.info("onnxruntime available (v%s, providers=%s)", version or "?", providers_list)
        if accelerator_hint in {"cuda", "directml", "tensorrt"}:
            lowered = [p.lower() for p in providers_list]
            expected = False
            if accelerator_hint == "cuda":
                expected = any("cuda" in p for p in lowered)
            elif accelerator_hint == "directml":
                expected = any("dml" in p or "directml" in p for p in lowered)
            elif accelerator_hint == "tensorrt":
                expected = any("tensorrt" in p for p in lowered)
            if not expected:
                LOG.error(
                    "onnxruntime providers missing expected accelerator (wanted %s, providers=%s)",
                    accelerator_hint,
                    providers_list,
                )
    else:
        LOG.error("onnxruntime unavailable: %s", reason or "unknown")


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

    accelerator_hint = os.environ.get("ARGOS_ACCELERATOR", "").strip().lower()
    if not accelerator_hint:
        caps = _get_capabilities()
        if caps:
            pref = caps.get("preferred_accelerator")
            if isinstance(pref, str):
                accelerator_hint = pref.strip().lower()
            if accelerator_hint and not os.environ.get("ARGOS_ACCELERATOR"):
                os.environ["ARGOS_ACCELERATOR"] = accelerator_hint

    ok, version, providers, reason = _try_import_ort()
    if ok:
        _log_ort_status(True, version, providers, reason, accelerator_hint)
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
                    _log_ort_status(True, version, providers, reason, accelerator_hint)
                    return True, version, providers, reason
            heal_reason = summary.get("error") or heal_reason
            if summary.get("providers") and not providers:
                providers = list(summary.get("providers") or [])
            if summary.get("ort_version") and not version:
                version = str(summary.get("ort_version"))

    _log_ort_status(False, version, providers, heal_reason, accelerator_hint)
    return False, version, providers, heal_reason
