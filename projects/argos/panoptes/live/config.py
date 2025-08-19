"""
Hardware probe + centralized small-first model selection for live.

We intentionally keep this lightweight. Any heavier logic can be delegated to a
central model registry in your repo once it’s available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict, Dict

import platform

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


@dataclass
class HardwareInfo:
    gpu: Optional[str]
    backend: str
    arch: str


def probe_hardware() -> HardwareInfo:
    """Detect very basic device/backends; keep it fast and robust."""
    gpu = None
    if torch is not None and torch.cuda.is_available():  # pragma: no cover
        try:
            idx = torch.cuda.current_device()
            gpu = torch.cuda.get_device_name(idx)
        except Exception:
            gpu = "CUDA"
    backend = "auto"
    arch = platform.machine() or "unknown"
    return HardwareInfo(gpu=gpu, backend=backend, arch=arch)


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
    if task.lower() in ("pose",):
        return {"label": "simple-pose (no-ML)", "names": {}}
    if task.lower() in ("pse",):
        return {"label": "simple-pse (no-ML)", "names": {}}
    if task.lower() in ("obb", "object"):
        return {"label": "simple-obb (no-ML)", "names": {}}
    return {"label": "custom", "names": {}}
