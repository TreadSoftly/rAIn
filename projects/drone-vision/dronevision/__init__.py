from __future__ import annotations

import importlib.util as _iu
import os
import sys as _sys
from pathlib import Path

from . import geo_sink as geo_sink  # re-export
from .heatmap import heatmap_overlay

# ────────────────────────────────────────────────────────────
#  Canonical project locations
# ────────────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent          # …/dronevision
ROOT = _BASE.parent                              # …/projects/drone-vision

# Single, authoritative model directory (env-var overrideable)
MODEL_DIR: Path = Path(os.getenv("DRONEVISION_MODEL_PATH", ROOT / "model")).resolve()
os.environ.setdefault("DRONEVISION_MODEL_PATH", str(MODEL_DIR))      # propagate downstream

# Primary checkpoints: unified YOLOv8 models for detection & segmentation
MODELS = {
    "primary": MODEL_DIR / "yolov8x.pt",
}

# Weight-selection tables (first existing file wins)
WEIGHT_PRIORITY: dict[str, list[Path]] = {
    # ── DETECTION ─────────────────────────────────────────────
    "detect": [
        MODEL_DIR / "yolov8x.pt",
        MODEL_DIR / "yolo11x.pt",
        MODEL_DIR / "yolov12x.pt",
        MODEL_DIR / "yolov8x.onnx",
        MODEL_DIR / "yolo11x.onnx",
        MODEL_DIR / "yolov12x.onnx",
        MODEL_DIR / "yolov12n.pt",
        MODEL_DIR / "yolov12n.onnx",
    ],
    # ── SEGMENTATION / HEAT-MAP ───────────────────────────────
    "heatmap": [
        MODEL_DIR / "yolov8x-seg.pt",
        MODEL_DIR / "yolov8x-seg.onnx",
        MODEL_DIR / "yolo11x-seg.pt",
        MODEL_DIR / "yolo11x-seg.onnx",
        MODEL_DIR / "yolo11m-seg.pt",
        MODEL_DIR / "yolo11m-seg.onnx",
        MODEL_DIR / "yolov12x-seg.pt",
        MODEL_DIR / "yolov12x-seg.onnx",
        MODEL_DIR / "yolov8n-seg.pt",
        MODEL_DIR / "yolov8n-seg.onnx",
        MODEL_DIR / "yolo11n-seg.pt",
        MODEL_DIR / "yolo11n-seg.onnx",
        MODEL_DIR / "yolov12n-seg.pt",
        MODEL_DIR / "yolov12n-seg.onnx",
    ],
    "geojson": [
        MODEL_DIR / "yolov8x.pt",
        MODEL_DIR / "yolo11x.pt",
    ],
    # ── SMALL / FAST VARIANTS ─────────────────────────────────
    "detect_small": [
        MODEL_DIR / "yolov8n.pt",
        MODEL_DIR / "yolo11n.pt",
        MODEL_DIR / "yolov12n.pt",
        MODEL_DIR / "yolov8n.onnx",
        MODEL_DIR / "yolo11n.onnx",
        MODEL_DIR / "yolov12n.onnx",
    ],
    "heatmap_small": [
        MODEL_DIR / "yolov8n-seg.pt",
        MODEL_DIR / "yolov8n-seg.onnx",
        MODEL_DIR / "yolo11n-seg.pt",
        MODEL_DIR / "yolo11n-seg.onnx",
        MODEL_DIR / "yolov12n-seg.pt",
        MODEL_DIR / "yolov12n-seg.onnx",
    ],
}
# ────────────────────────────────────────────────────────────


__all__ = [
    "MODEL_DIR",
    "MODELS",
    "WEIGHT_PRIORITY",
    "geo_sink",
    "heatmap_overlay",
]

_tests_path = ROOT / "tests" / "unit-tests" / "test_tasks.py"

if _tests_path.exists():
    _spec = _iu.spec_from_file_location("dronevision.test_tasks", _tests_path)
    if _spec and _spec.loader:
        _mod = _iu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _sys.modules["dronevision.test_tasks"] = _mod
