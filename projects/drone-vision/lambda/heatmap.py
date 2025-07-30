"""
Lambda helper – overlay YOLO segmentation masks + labels on an image.

Changes from original:
• Looks in MODEL_DIR first, then repo-root, for *-seg.pt weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO  # type: ignore
    _has_yolo = True
except Exception:                 # pragma: no cover
    YOLO = None  # type: ignore
    _has_yolo = False

_MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
_ROOT      = _MODEL_DIR.parent

_seg_candidates = [
    _MODEL_DIR / "yolo11x-seg.pt",
    _MODEL_DIR / "yolo11m-seg.pt",
    _MODEL_DIR / "yolov8x-seg.pt",
    _MODEL_DIR / "yolov8s-seg.pt",
    _MODEL_DIR / "yolov8n-seg.pt",
    _ROOT      / "yolo11x-seg.pt",
    _ROOT      / "yolo11m-seg.pt",
    _ROOT      / "yolov8x-seg.pt",
    _ROOT      / "yolov8s-seg.pt",
    _ROOT      / "yolov8n-seg.pt",
]

_seg_model = None
if _has_yolo and YOLO is not None:
    for cand in _seg_candidates:
        if cand.exists():
            _seg_model = YOLO(str(cand))  # type: ignore[arg-type]
            break

__all__ = ["heatmap_overlay"]

def heatmap_overlay(
    img: Image.Image | tuple[int, int] | np.ndarray[Any, Any],
    *,
    boxes: np.ndarray[Any, Any] | None = None,
    masks: Iterable[np.ndarray[Any, Any]] | None = None,
    **kw: Any,
) -> np.ndarray[Any, Any]:
    """
    Return a BGR ndarray with segmentation masks + labels, or the original
    image (BGR) if no segmentation model is available.
    """
    if _seg_model is None:
        if isinstance(img, np.ndarray):
            return img.copy()
        if isinstance(img, Image.Image):
            return np.asarray(img.convert("RGB"))[:, :, ::-1]
        if len(img) == 2:
            w, h = img
            return np.zeros((h, w, 3), dtype=np.uint8)
        raise TypeError(f"Unsupported image type: {type(img)}")

    # normalise input to BGR ndarray
    if isinstance(img, np.ndarray):
        bgr = img
    elif isinstance(img, Image.Image):
        bgr = np.asarray(img.convert("RGB"))[:, :, ::-1]
    elif len(img) == 2:
        w, h = img
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
    else:  # str / Path
        bgr = np.asarray(Image.open(img).convert("RGB"))[:, :, ::-1]

    result = _seg_model(bgr)[0]       # type: ignore[index]
    return result.plot()  # type: ignore[attr-defined]  # already BGR ndarray
