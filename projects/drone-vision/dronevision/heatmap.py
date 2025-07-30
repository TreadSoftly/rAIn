"""
Toy “heat‑map” overlay – converts YOLO detections into a pseudo‑thermal view.

Falls back to a no‑op overlay when Ultralytics / Torch / weights
are unavailable (so the Lambda heat‑map route never 500s in CI).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

# ─── try to import Ultralytics / Torch only if present ────────────────
try:
    from ultralytics import YOLO  # type: ignore
    import torch                  # type: ignore
    _has_yolo = True
except Exception:                 # missing wheels or no GPU etc.
    YOLO = None                   # type: ignore
    torch = None                  # type: ignore
    _has_yolo = False

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

_SEG_WEIGHTS = Path(__file__).resolve().parents[1] / "yolov8n-seg.pt"


def _to_heat(mask: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    idx = np.clip((mask * (_CMAP.shape[0] - 1)).astype(int), 0, _CMAP.shape[0] - 1)
    return _CMAP[idx].astype(np.float32)


def heatmap_overlay(
    img: Image.Image,
    masks: Iterable[NDArray[Any]] | None = None,
) -> Image.Image:
    """
    Blend every mask into a single heat‑map.

    • If *masks* is provided we trust the caller.    • If *masks* is None we *attempt* YOLO‑seg → masks, but silently fall back
      to the original image on any failure (no weights / no torch / net‑block).
    """
    img_arr = np.asarray(img).astype(np.float32) / 255.0

    # ── generate segmentation masks on‑demand ─────────────────────────
    mask_results: list[NDArray[np.floating[Any]]] = []
    if masks is None:
        if not _has_yolo or not _SEG_WEIGHTS.exists():
            return img                           # graceful degradation

        try:
            mdl = YOLO(str(_SEG_WEIGHTS))        # type: ignore[call-arg]
            pred = mdl.predict(                  # type: ignore[attr-defined]
                source=img,
                imgsz=640,
                conf=0.25,
                retina_masks=True,
                verbose=False,
            )[0]

            if pred.masks is not None:
                data: torch.Tensor | np.ndarray[Any, Any] = pred.masks.data  # type: ignore[type-annotation]
                arr = data.cpu().numpy() if torch is not None else data      # type: ignore[arg-type]
                generated_masks: list[NDArray[np.floating[Any]]] = [np.asarray(m, dtype=np.float32) for m in arr]  # type: ignore
            else:
                generated_masks: list[NDArray[np.floating[Any]]] = []
        except Exception:
            return img                            # any YOLO failure → no‑op

        mask_results = generated_masks

    mask_list: list[NDArray[np.floating[Any]]] = list(mask_results if masks is None else masks)
    if not mask_list:
        return img

    heat = np.maximum.reduce(np.array(mask_list))     # union of objects
    heat = _to_heat(heat) / 255.0                     # colour‑ise
    blended = np.clip((0.6 * img_arr + 0.4 * heat) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)
