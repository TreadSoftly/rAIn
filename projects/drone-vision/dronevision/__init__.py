"""dronevision – package initialisation
Centralises model discovery so every module uses *one* source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── canonical project locations ──────────────────────────────────────────
_BASE = Path(__file__).resolve().parent           # …/dronevision
ROOT  = _BASE.parent                              # …/projects/drone-vision

# single, authoritative model directory (env-var overrideable)
MODEL_DIR: Path = Path(os.getenv("DRONEVISION_MODEL_PATH", ROOT / "model")).resolve()
os.environ.setdefault("DRONEVISION_MODEL_PATH", str(MODEL_DIR))  # propagate downstream

# primary checkpoints: unified YOLOv8 models for detection and segmentation
MODELS = {
    "drone":    MODEL_DIR / "yolov8x.pt",   # unified detection model (YOLOv8x)
    "airplane": MODEL_DIR / "yolov8x.pt",   # alias to unified detection model
}

# weight-selection priority tables (first existing file wins)
WEIGHT_PRIORITY: dict[str, list[Path]] = {
    # full-scale object detection (images/videos)
    "detect": [
        MODEL_DIR / "yolov8x.pt",    # main unified detector
        MODEL_DIR / "yolo11x.pt",    # backup custom detector (if available)
        # (Legacy ONNX models removed)
    ],
    # segmentation-based overlays (heatmaps via instance masks)
    "heatmap": [
        MODEL_DIR / "yolo11x-seg.pt",  # main segmentation model
        MODEL_DIR / "yolo11m-seg.pt",  # backup segmentation model
        MODEL_DIR / "yolov8n-seg.pt",  # alternative lightweight model
    ],
    # geojson detection (uses same unified detector)
    "geojson": [
        MODEL_DIR / "yolov8x.pt",    # main unified detector
        MODEL_DIR / "yolo11x.pt",    # backup custom detector
    ],
    # small / fast variants for live video & IoT
    "detect_small": [
        MODEL_DIR / "yolov8n.pt",    # main small detector
        MODEL_DIR / "yolo11n.pt",    # backup small detector
    ],
    "heatmap_small": [
        MODEL_DIR / "yolov8n-seg.pt",  # main small segmentation model
        MODEL_DIR / "yolo11n-seg.pt",  # backup small segmentation model
    ],
}

__all__ = ["MODEL_DIR", "MODELS", "WEIGHT_PRIORITY"]
