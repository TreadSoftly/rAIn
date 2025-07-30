"""
Toy “heat‑map” overlay – converts YOLO detections into a pseudo‑thermal view.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from ultralytics import YOLO  # type: ignore[import]

# ─── tiny 5‑colour “jet” palette ───────────────────────────────
_CMAP = np.asarray(
    [
        [0,   0,   0],
        [0,   0, 255],
        [0, 255, 255],
        [255, 255,   0],
        [255,   0,   0],
    ],
    dtype=np.uint8,
)

# always download / search for the seg‑weights **inside the repo**
_SEG_WEIGHTS = Path(__file__).resolve().parents[1] / "yolov8n-seg.pt"


def _to_heat(mask: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    idx = np.clip((mask * (_CMAP.shape[0] - 1)).astype(int), 0, _CMAP.shape[0] - 1)
    return _CMAP[idx].astype(np.float32)


def heatmap_overlay(
    img: Image.Image,
    masks: Iterable[NDArray[Any]] | None = None,
) -> Image.Image:
    """
    Quick‑and‑dirty visualisation: blend every mask into a single heat‑map.
    If *masks* is None we run a YOLOv8‑seg model and build masks on‑the‑fly.
    """
    img_arr = np.asarray(img).astype(np.float32) / 255.0

    # ── generate segmentation masks on demand ─────────────────
    if masks is None:
        mdl = YOLO(str(_SEG_WEIGHTS))
        pred = mdl.predict(  # type: ignore[attr-defined]
            source=img, imgsz=640, conf=0.25, retina_masks=True, verbose=False
        )[0]

        if pred.masks is not None:
            data: torch.Tensor | np.ndarray[Any, Any] = pred.masks.data  # type: ignore[type-annotation]
            arr = data.cpu().numpy() if isinstance(data, torch.Tensor) else data # type: ignore[assignment]
            arr: NDArray[np.floating[Any]] = arr  # type: ignore[assignment]
            masks = [m.astype(np.float32) for m in arr]
        else:
            masks = []

    masks = list(masks)
    if not masks:
        return img

    heat = np.maximum.reduce(np.array(masks))   # union of objects
    heat = _to_heat(heat) / 255.0               # colour‑ise
    blended = np.clip((0.6 * img_arr + 0.4 * heat) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)
