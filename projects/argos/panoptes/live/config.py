"""
Hardware probe + centralized small-first model selection for live.

We intentionally keep this lightweight. Any heavier logic can be delegated to a
central model registry in your repo once it’s available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict, Dict, Tuple, Any, Protocol, cast

import platform

# Lightweight progress note during hardware probing
try:
    from panoptes.progress import simple_status as _progress_simple_status  # type: ignore[import]
except Exception:  # pragma: no cover
    def _progress_simple_status(*_a: object, **_k: object):
        class _N:
            def __enter__(self): return self
            def __exit__(self, *_: object) -> bool: return False
        return _N()

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

class BackendDecisionProto(Protocol):
    preferred_backend: str
    preprocess_device: str
    prefer_small: bool
    input_size: Tuple[int, int]
    ort_threads: Optional[int]
    ort_execution: Optional[str]
    nms_mode: Optional[str]
    fingerprint: Dict[str, Any]

try:
    from .backend_controller import ensure_backend_decision as _ensure_backend_decision, load_backend_decision as _load_backend_decision  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    def _ensure_backend_decision(force: bool = False) -> Optional[Any]:
        return None

    def _load_backend_decision() -> Optional[Any]:
        return None


ensure_backend_decision = _ensure_backend_decision
load_backend_decision = _load_backend_decision


def _coerce_backend_decision(obj: Optional[Any]) -> Optional[BackendDecisionProto]:
    if obj is None:
        return None
    return cast(BackendDecisionProto, obj)


@dataclass
class HardwareInfo:
    gpu: Optional[str]
    backend: str
    arch: str
    preprocess_device: str = "auto"
    prefer_small: Optional[bool] = None
    input_size: Optional[Tuple[int, int]] = None
    ort_threads: Optional[int] = None
    ort_execution: Optional[str] = None
    nms_mode: Optional[str] = None
    fingerprint: Optional[Dict[str, Any]] = None
    ram_gb: Optional[float] = None


def probe_hardware() -> HardwareInfo:
    """Detect very basic device/backends; keep it fast and robust."""
    gpu: Optional[str] = None
    backend: str = "auto"
    arch: str = "unknown"
    preprocess_device = "auto"
    prefer_small: Optional[bool] = None
    input_size: Optional[Tuple[int, int]] = None
    ort_threads: Optional[int] = None
    ort_execution: Optional[str] = None
    nms_mode: Optional[str] = None
    fingerprint: Optional[Dict[str, Any]] = None
    with _progress_simple_status("Probing hardware"):
        if torch is not None and torch.cuda.is_available():  # pragma: no cover
            try:
                idx = torch.cuda.current_device()
                gpu = torch.cuda.get_device_name(idx)
            except Exception:
                gpu = "CUDA"
        arch = platform.machine() or "unknown"
    decision_obj = _coerce_backend_decision(load_backend_decision())
    if decision_obj is None:
        decision_obj = _coerce_backend_decision(ensure_backend_decision(force=False))
    if decision_obj is not None:
        if decision_obj.preferred_backend:
            backend = decision_obj.preferred_backend
        if decision_obj.preprocess_device:
            preprocess_device = decision_obj.preprocess_device
        prefer_small = decision_obj.prefer_small
        size_hint = decision_obj.input_size
        try:
            input_size = (int(size_hint[0]), int(size_hint[1]))
        except Exception:
            pass
        if decision_obj.ort_threads is not None:
            ort_threads = max(1, int(decision_obj.ort_threads))
        if decision_obj.ort_execution is not None:
            ort_execution = decision_obj.ort_execution
        if decision_obj.nms_mode:
            nms_mode = decision_obj.nms_mode
        fingerprint = {str(key): value for key, value in decision_obj.fingerprint.items()}
    return HardwareInfo(
        gpu=gpu,
        backend=backend,
        arch=arch,
        preprocess_device=preprocess_device,
        prefer_small=prefer_small,
        input_size=input_size,
        ort_threads=ort_threads,
        ort_execution=ort_execution,
        nms_mode=nms_mode,
        fingerprint=fingerprint,
    )


class ModelSelection(TypedDict, total=False):
    label: str
    names: Dict[int, str]


def select_models_for_live(task: str, hw: HardwareInfo) -> ModelSelection:
    """
    Pick small/fast defaults. This is a *placeholder* that you can later route
    through your central panoptes model registry. We do not download weights.
    """
    if task.lower() in ("detect", "d"):
        return {"label": "fast-contour (no-ML)", "names": {}}
    if task.lower() in ("heatmap", "hm"):
        return {"label": "laplacian-heatmap (no-ML)", "names": {}}
    if task.lower() in ("classify", "clf"):
        return {"label": "simple-classify (no-ML)", "names": {}}
    if task.lower() in ("pose", "pse"):
        return {"label": "simple-pose (no-ML)", "names": {}}
    if task.lower() in ("obb", "object"):
        return {"label": "simple-obb (no-ML)", "names": {}}
    return {"label": "custom", "names": {}}
