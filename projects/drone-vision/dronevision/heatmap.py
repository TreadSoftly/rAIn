"""
Toy “heat-map” overlay – converts YOLO detections into a pseudo-thermal view.
"""
from __future__ import annotations

from typing import Iterable, Any
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

_CMAP = np.asarray(
    [
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 0],
    ],
    dtype=np.uint8,
)

def _to_heat(mask: NDArray[np.floating]) -> NDArray[np.floating]:  # noqa: D401
    idx = np.clip((mask * (_CMAP.shape[0] - 1)).astype(int), 0, _CMAP.shape[0] - 1)
    return _CMAP[idx].astype(np.float32)


def heatmap_overlay(img: Image.Image, masks: Iterable[NDArray[Any]] | None = None) -> Image.Image:
    """
    Quick-and-dirty visualisation: blend every mask into a single heat-map.

    *If* the caller passes masks they are used; otherwise YOLOv8 segmentation
    masks are generated on-the-fly.
    """
    from ultralytics import YOLO # type: ignore  # heavy import kept local

    img_arr = np.asarray(img).astype(np.float32) / 255

    if masks is None:
        mdl = YOLO("yolov8n-seg.pt")
        pred = mdl.predict(  # type: ignore
            source=img, imgsz=640, conf=0.25, retina_masks=True, verbose=False
        )[0]


        if pred.masks is not None:
            data: torch.Tensor | np.ndarray[Any, Any] = pred.masks.data  # type: ignore[type-annotation]
            if isinstance(data, torch.Tensor):
                masks = list(data.cpu().numpy().astype(np.float32))
            else:
                masks = list(np.array(data, dtype=np.float32))
        else:
            masks = []

    if not masks:
        return img

    masks_arr = np.array(list(masks))
    heat = np.maximum.reduce(masks_arr)  # union of every object
    heat = _to_heat(heat) / 255.0

    blended = (0.6 * img_arr + 0.4 * heat) * 255
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
