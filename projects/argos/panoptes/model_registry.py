"""
panoptes.model_registry
--------------------------
*Single* source-of-truth for every weight-selection decision.

Hard-locking rules (2025-08-07)
────────────────────────────────────────────────────────────────────────
1. **Only** the paths listed in ``WEIGHT_PRIORITY`` are ever considered.
2. No directory scans, no implicit “-seg” discovery, no env-variable
   overrides (`panoptes_*`).  The *override=* argument in
   ``load_detector/segmenter`` is retained exclusively for unit-tests or
   ad-hoc scripts that call those helpers **directly**.
3. If a required weight is missing we raise **RuntimeError** immediately
   – downstream code must catch or let the program abort.
"""

from __future__ import annotations

import functools
import logging
import importlib
import sys
from pathlib import Path
from typing import Final, Literal, Optional
# ────────────────────────────────────────────────────────────────
#  Ultralytics import (kept soft so linting works without it)
# ────────────────────────────────────────────────────────────────
try:
    _ultra_mod = importlib.import_module("ultralytics")
    _yolo_cls = getattr(_ultra_mod, "YOLO", None)
except Exception:  # pragma: no cover
    _yolo_cls = None
    YOLO = None

# ────────────────────────────────────────────────────────────────
#  logging (explicit, human-friendly, no stack noise)
# ────────────────────────────────────────────────────────────────
_LOG = logging.getLogger("panoptes.model_registry")
if not _LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
_LOG.setLevel(logging.INFO)


def _say(msg: str) -> None:
    _LOG.info(f"[panoptes] {msg}")


# ────────────────────────────────────────────────────────────────
#  Canonical model folders
# ────────────────────────────────────────────────────────────────
_ROOT: Final[Path] = Path(__file__).resolve().parent  # …/panoptes
_MODEL_DIR_A = _ROOT / "model"  # packaged weights (preferred)
_MODEL_DIR_B = _ROOT.parent / "model"  # legacy path (fallback)

# Prefer packaged dir; only fall back to legacy if it already exists
MODEL_DIR: Final[Path] = (
    _MODEL_DIR_A if _MODEL_DIR_A.exists() or not _MODEL_DIR_B.exists() else _MODEL_DIR_B
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────
#  *** WEIGHT SELECTION TABLE ***
#  Edit this ONE place if you add/rename checkpoints
#
#  Only list **real** upstream files you actually keep in panoptes/model.
#  (No placeholders. No names that don’t exist.)
# ────────────────────────────────────────────────────────────────
_DETECT_LIST: list[Path] = [
    MODEL_DIR / "yolov8x.pt",    # MAIN
    MODEL_DIR / "yolo11x.pt",    # BACKUP
    MODEL_DIR / "yolov12x.onnx", # light/fast dev
    MODEL_DIR / "yolov12x.pt",
    MODEL_DIR / "yolo11x.onnx",
    MODEL_DIR / "yolov8x.onnx",
    MODEL_DIR / "yolo11s.pt",
    MODEL_DIR / "yolo11s.onnx",
    MODEL_DIR / "yolov8s.pt",
    MODEL_DIR / "yolov8s.onnx",
    MODEL_DIR / "yolov12n.onnx",
    MODEL_DIR / "yolov12n.pt",
    MODEL_DIR / "yolo11n.pt",
    MODEL_DIR / "yolo11n.onnx",
    MODEL_DIR / "yolov8n.pt",
    MODEL_DIR / "yolov8n.onnx",
]

_HEATMAP_LIST: list[Path] = [
    MODEL_DIR / "yolo11x-seg.pt",  # MAIN
    MODEL_DIR / "yolo11x-seg.onnx",
    MODEL_DIR / "yolo11s-seg.pt",
    MODEL_DIR / "yolo11s-seg.onnx",
    MODEL_DIR / "yolov8x-seg.pt",
    MODEL_DIR / "yolov8x-seg.onnx",
    MODEL_DIR / "yolov12s-seg.onnx",
    MODEL_DIR / "yolov12s-seg.pt",
    MODEL_DIR / "yolo11n-seg.pt",
    MODEL_DIR / "yolo11n-seg.onnx",
    MODEL_DIR / "yolov8n-seg.pt",
    MODEL_DIR / "yolov8n-seg.onnx",
]

_DETECT_SMALL_LIST: list[Path] = [
    MODEL_DIR / "yolov12n.onnx",
    MODEL_DIR / "yolov8n.pt",
    MODEL_DIR / "yolo11n.pt",
    MODEL_DIR / "yolov8n.onnx",
    MODEL_DIR / "yolo11n.onnx",
]

_HEATMAP_SMALL_LIST: list[Path] = [
    MODEL_DIR / "yolov12s-seg.onnx",
    MODEL_DIR / "yolo11n-seg.onnx",
    MODEL_DIR / "yolov8n-seg.onnx",
    MODEL_DIR / "yolov12s-seg.pt",
    MODEL_DIR / "yolo11n-seg.pt",
    MODEL_DIR / "yolov8n-seg.pt",
]

WEIGHT_PRIORITY: dict[str, list[Path]] = {
    # Object detection
    "detect": _DETECT_LIST,
    # GeoJSON uses detection picks (copy to keep lists independent)
    "geojson": list(_DETECT_LIST),
    # Instance segmentation (heatmaps)
    "heatmap": _HEATMAP_LIST,
    # “small / fast” (live video / tiny devices)
    "detect_small": _DETECT_SMALL_LIST,
    "heatmap_small": _HEATMAP_SMALL_LIST,
}

# ────────────────────────────────────────────────────────────────
#  Internal helpers
# ────────────────────────────────────────────────────────────────
def _first_existing(paths: list[Path]) -> Optional[Path]:
    """Return the first path that exists on disk or *None*."""
    return next((p for p in paths if p.exists()), None)
@functools.lru_cache(maxsize=None)
def _load(weight: Optional[Path], *, task: Literal["detect", "segment"]) -> object | None:
    """
    Cached wrapper around ``YOLO(path, task=...)`` to avoid re-inits,
    and to kill the “Unable to automatically guess model task” warning.

    Some Ultralytics builds don't accept the ``task=`` kwarg; fall back
    to plain ``YOLO(path)`` in that case.
    """
    if _yolo_cls is None or weight is None:
        return None
    _say(f"init YOLO: task={task} path={weight}")
    try:
        return _yolo_cls(str(weight), task=task)
    except TypeError:
        return _yolo_cls(str(weight))


def _require(model: object | None, task: str) -> object:
    """Abort loudly when the chosen weight is missing."""
    if model is None:
        raise RuntimeError(f"[model_registry] no weight configured for task “{task}”")
    return model


# ────────────────────────────────────────────────────────────────
#  Public helpers
# ────────────────────────────────────────────────────────────────
def pick_weight(
    task: Literal["detect", "heatmap", "geojson"],
    *,
    small: bool = False,
) -> Optional[Path]:
    """
    Return the path to the **first** existing weight for *task* or *None*.

    When *small=True* the *_small* key is used (falls back to normal key
    if not defined).
    """
    prefer_key = f"{task}_small" if small else task
    paths = WEIGHT_PRIORITY.get(prefer_key)
    if not paths:
        paths = WEIGHT_PRIORITY.get(task, [])
    return _first_existing(paths)


def load_detector(*, small: bool = False, override: str | Path | None = None) -> object:
    """Return a *YOLO* detector – honouring the override when provided."""
    if override is not None:
        chosen = Path(override).expanduser()
        _say(f"task=detect small={small} override={chosen}")
    else:
        chosen = pick_weight("detect", small=small)
        _say(f"task=detect small={small} weight={chosen}")
    model = _load(chosen, task="detect")
    return _require(model, "detect")


def load_segmenter(*, small: bool = False, override: str | Path | None = None) -> object:
    """Return a *YOLO-Seg* model – abort if no suitable weight is found."""
    if override is not None:
        chosen = Path(override).expanduser()
        _say(f"task=heatmap small={small} override={chosen}")
    else:
        chosen = pick_weight("heatmap", small=small)
        _say(f"task=heatmap small={small} weight={chosen}")
    model = _load(chosen, task="segment")
    return _require(model, "heatmap")


# Re-export for “from panoptes import *”
__all__ = [
    "MODEL_DIR",
    "WEIGHT_PRIORITY",
    "pick_weight",
    "load_detector",
    "load_segmenter",
]
