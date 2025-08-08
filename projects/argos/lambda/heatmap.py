
# projects/argos/lambda/heatmap.py
"""
lambda.heatmap  – server‑side segmentation overlay helper
=========================================================

Lock‑down (2025‑08‑07)
────────────────────────────────────────────────────────────────────────
* **Strict weights** – only the path(s) declared in
  `panoptes.model_registry.WEIGHT_PRIORITY["heatmap"]` are consulted.
* A segmentation model is loaded once, at import‑time, via
  `load_segmenter()`; if the weight is missing the **module import aborts**
  with *RuntimeError*, surfacing a clear cold‑start error in Lambda.
* Public API is unchanged:

      heatmap_overlay(img, *, boxes=None, masks=None, **kw) → np.ndarray[BGR]

  Callers therefore do **not** need to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from panoptes.model_registry import load_segmenter

# ───────────────────────── single, authoritative model ──────────────────────────
# Any weight issue will raise *RuntimeError* here, stopping import immediately.
_seg_model = load_segmenter()          # ← hard‑fails if weight/YOLO missing

__all__ = ["heatmap_overlay"]

# ───────────────────────── public helper ────────────────────────────────────────
def heatmap_overlay(
    img: Image.Image | tuple[int, int] | np.ndarray[Any, Any] | str | Path,
    *,
    boxes: np.ndarray[Any, Any] | None = None,          # kept for signature parity
    masks: Iterable[np.ndarray[Any, Any]] | None = None,  # ditto (currently unused)
    **kw: Any,
) -> np.ndarray[Any, Any]:
    """
    Overlay YOLO‑Seg instance masks + labels on *img* and return a **BGR**
    ndarray ready for OpenCV / video encoding.

    Parameters
    ----------
    img
        • `PIL.Image.Image`
        • path‑like object
        • raw **BGR** `np.ndarray`
        • placeholder *(w, h)* tuple (creates a blank canvas – useful in tests)
    boxes, masks
        Accepted for forward‑compatibility but ignored – the segmentation model
        is fully responsible for mask generation.
    **kw
        Ignored; accepted to preserve call‑sites that may pass extra kwargs.

    Notes
    -----
    *Any* failure inside Ultralytics (e.g. unsupported image size) is trapped
    and the function falls back to returning the **unmodified** BGR image so
    downstream pipelines remain robust.
    """
    # ── normalise *img* to a BGR ndarray we can always return ────────────────
    if isinstance(img, np.ndarray):
        bgr = img
    elif isinstance(img, Image.Image):
        bgr = np.asarray(img.convert("RGB"))[:, :, ::-1]
    elif isinstance(img, (str, Path)):
        bgr = np.asarray(Image.open(img).convert("RGB"))[:, :, ::-1]
    elif len(img) == 2:  # (w, h) tuple placeholder
        w, h = img
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    # ── run segmentation & render (Ultralytics returns BGR already) ──────────
    try:
        res = _seg_model.predict(bgr, imgsz=640, conf=0.25, verbose=False)[0]  # type: ignore[index]
        return res.plot()  # type: ignore[no-any-return]
    except Exception:
        # Any runtime glitch → return the original image unchanged
        return bgr.copy()
