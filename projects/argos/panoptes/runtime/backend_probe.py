"""
Lightweight backend availability probes.

These utilities intentionally avoid importing heavy dependencies at module
import time. Callers can use the helpers to decide which model weights to use
without paying the cost of initialising ONNX Runtime sessions.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Iterable, Sequence, Callable, cast, NamedTuple, List, Mapping

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


class OrtProbeStatus(NamedTuple):
    ok: bool
    version: Optional[str]
    providers: Optional[List[str]]
    reason: Optional[str]
    providers_ok: bool
    expected_provider: Optional[str]
    healed: bool
    summary: Optional[Dict[str, Any]]


def _coerce_optional_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _normalize_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, Mapping):
        mapping_view = cast(Mapping[Any, Any], obj)
        return {str(key): value for key, value in mapping_view.items()}
    return {}


def _normalize_providers(providers: Optional[Iterable[str]]) -> Optional[List[str]]:
    if not providers:
        return None
    return [str(provider) for provider in providers]


def _expected_provider_for_accelerator(accelerator: str) -> Optional[str]:
    accel = accelerator.strip().lower()
    if accel == "cuda":
        return "CUDAExecutionProvider"
    if accel == "directml":
        return "DmlExecutionProvider"
    if accel == "tensorrt":
        return "TensorrtExecutionProvider"
    return None


def _package_to_expected_provider(package: Optional[str]) -> Optional[str]:
    if not package:
        return None
    name = package.strip().lower()
    if name == "onnxruntime-gpu":
        return "CUDAExecutionProvider"
    if name == "onnxruntime-directml":
        return "DmlExecutionProvider"
    return None


def _providers_match_expected(providers: Optional[Sequence[str]], expected: Optional[str]) -> bool:
    if not expected:
        return True
    if not providers:
        return False
    lowered = [str(provider).lower() for provider in providers]
    expected_lower = expected.lower()
    if expected_lower in lowered:
        return True
    if expected_lower.startswith("cuda"):
        return any("cuda" in provider for provider in lowered)
    if expected_lower.startswith(("dml", "directml")):
        return any("dml" in provider or "directml" in provider for provider in lowered)
    if expected_lower.startswith("tensorrt"):
        return any("tensorrt" in provider for provider in lowered)
    return False


def _resolve_expectations() -> tuple[str, Optional[str], Dict[str, Any]]:
    accelerator_hint = _coerce_optional_str(os.environ.get("ARGOS_ACCELERATOR", "")) or ""
    caps_raw = _get_capabilities()
    caps = _normalize_dict(caps_raw)
    if not accelerator_hint:
        pref = _coerce_optional_str(caps.get("preferred_accelerator"))
        if pref:
            accelerator_hint = pref
            if not os.environ.get("ARGOS_ACCELERATOR"):
                os.environ["ARGOS_ACCELERATOR"] = pref
    ort_meta = _normalize_dict(caps.get("onnxruntime"))
    expected_provider = _coerce_optional_str(ort_meta.get("expected_provider"))
    if not expected_provider:
        package_obj = ort_meta.get("package")
        package_name = _coerce_optional_str(package_obj)
        if package_name:
            expected_provider = _package_to_expected_provider(package_name)
        else:
            package_meta = _normalize_dict(package_obj)
            name_obj = _coerce_optional_str(package_meta.get("name"))
            if name_obj:
                expected_provider = _package_to_expected_provider(name_obj)
    if not expected_provider:
        expected_provider = _expected_provider_for_accelerator(accelerator_hint)
    return accelerator_hint, expected_provider, ort_meta


def _log_probe_status(status: OrtProbeStatus, accelerator_hint: str) -> None:
    providers_list = status.providers or []
    expected_display = status.expected_provider or (accelerator_hint or "cpu")
    if status.ok:
        if LOG.isEnabledFor(logging.INFO):
            LOG.info(
                "onnxruntime ready expected=%s providers=%s healed=%s",
                expected_display,
                providers_list,
                status.healed,
            )
    else:
        LOG.error(
            "onnxruntime not-ready reason=%s expected=%s providers=%s healed=%s",
            status.reason or "unknown",
            expected_display,
            providers_list,
            status.healed,
        )

def _get_capabilities() -> Optional[Dict[str, Any]]:
    global _cap_cache
    if _cap_cache is not None:
        return _cap_cache
    if _BOOTSTRAP is None or not hasattr(_BOOTSTRAP, "read_capabilities"):
        return None
    try:
        data = _BOOTSTRAP.read_capabilities()  # type: ignore[call-arg]
        if isinstance(data, Mapping):
            mapping_view = cast(Mapping[Any, Any], data)
            _cap_cache = {str(key): value for key, value in mapping_view.items()}
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



def torch_available() -> bool:
    """Return True if ``torch`` can be imported."""
    try:
        import torch  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def ort_available() -> OrtProbeStatus:
    """Attempt to import ONNX Runtime and report availability details."""
    if os.environ.get("ARGOS_DISABLE_ONNX"):
        status = OrtProbeStatus(False, None, None, "disabled via ARGOS_DISABLE_ONNX", False, None, False, None)
        _log_probe_status(status, accelerator_hint="")
        return status

    accelerator_hint, expected_provider, _ = _resolve_expectations()

    import_ok, version, providers_raw, reason = _try_import_ort()
    providers = _normalize_providers(providers_raw)
    providers_ok = _providers_match_expected(providers or [], expected_provider)
    overall_ok = import_ok and providers_ok
    reason_final: Optional[str] = None
    if not import_ok:
        reason_final = reason
    elif not providers_ok and expected_provider:
        reason_final = f"missing_expected_provider:{expected_provider}"

    if overall_ok:
        status = OrtProbeStatus(True, version, providers, None, True, expected_provider, False, None)
        _log_probe_status(status, accelerator_hint)
        return status

    heal_summary: Optional[Dict[str, Any]] = None
    healed = False
    providers_after = providers
    version_after = version
    heal_reason = reason_final or reason

    if _BOOTSTRAP is not None and hasattr(_BOOTSTRAP, "ensure_onnxruntime"):
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
                    heal_summary = ensure(venv_py, log=_heal_log)
                except Exception as exc:
                    LOG.exception("onnx.heal.ensure_exception: %s", exc)
                    heal_summary = {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
            else:
                heal_summary = {"installed": False, "error": f"heal already in progress ({lock_path})"}

        if heal_summary:
            healed = bool(heal_summary.get("installed"))
            heal_reason = heal_summary.get("error") or heal_reason
            summary_expected = heal_summary.get("expected_provider")
            if isinstance(summary_expected, str) and summary_expected.strip():
                expected_provider = summary_expected.strip()
            summary_providers = heal_summary.get("providers")
            if (
                providers_after is None
                and isinstance(summary_providers, Sequence)
                and not isinstance(summary_providers, (str, bytes))
            ):
                providers_after = [str(p) for p in cast(Sequence[Any], summary_providers)]
            summary_version = heal_summary.get("ort_version")
            if isinstance(summary_version, str) and summary_version.strip():
                version_after = summary_version.strip()
            summary_accel = _coerce_optional_str(heal_summary.get("accelerator"))
            if summary_accel:
                accelerator_hint = summary_accel.lower()
                os.environ["ARGOS_ACCELERATOR"] = summary_accel

        import_ok, version_post, providers_post_raw, reason_post = _try_import_ort()
        if import_ok:
            version_after = version_post or version_after
            providers_after = _normalize_providers(providers_post_raw) or providers_after
        else:
            heal_reason = reason_post or heal_reason
        providers_ok = _providers_match_expected(providers_after or [], expected_provider)
        overall_ok = import_ok and providers_ok
        if overall_ok:
            reason_final = None
        else:
            reason_final = heal_reason or reason_post or reason_final
    else:
        reason_final = reason_final or reason or "onnxruntime unavailable"

    providers_ok = _providers_match_expected(providers_after or [], expected_provider)
    overall_ok = import_ok and providers_ok

    status = OrtProbeStatus(
        overall_ok,
        version_after,
        providers_after,
        reason_final,
        providers_ok,
        expected_provider,
        healed,
        heal_summary,
    )
    _log_probe_status(status, accelerator_hint)
    return status
