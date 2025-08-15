# \rAIn\projects\argos\lambda\heatmap.py
"""
lambda.heatmap  - server-side segmentation overlay helper
=========================================================

Lock-down (2025-08-07)
────────────────────────────────────────────────────────────────────────
* **Strict weights** - only the path(s) declared in
  `panoptes.model_registry.WEIGHT_PRIORITY["heatmap"]` are consulted.
* A segmentation model is loaded once, at import-time, via
  `load_segmenter()`; if the weight is missing the **module import aborts**
  with *RuntimeError*, surfacing a clear cold-start error in Lambda.
* Public API is unchanged:

      heatmap_overlay(img, *, boxes=None, masks=None, **kw) → np.ndarray[BGR]

Progress
────────
* Short `simple_status` around model init & predict (no-op if absent).
"""
from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from panoptes.model_registry import load_segmenter  # type: ignore

# optional progress
try:
    from panoptes.progress.progress_ux import simple_status  # type: ignore
except Exception:  # pragma: no cover
    simple_status = None  # type: ignore

# ───────────────────────── logging ─────────────────────────────────────
_LOG = logging.getLogger("lambda.heatmap")
if not _LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
_LOG.setLevel(logging.INFO)


def _say(msg: str) -> None:
    _LOG.info(f"[panoptes] {msg}")


# ───────────────────────── single, authoritative model ──────────────────────────
with (simple_status("init segmenter (lambda)") if simple_status is not None else contextlib.nullcontext()):
    _seg_model = load_segmenter(small=True)  # ← ONNX-first in Lambda; hard-fail if missing
_say("lambda.heatmap init: segmenter small=True (see registry log for weight)")

__all__ = ["heatmap_overlay"]

# ───────────────────────── public helper ────────────────────────────────────────
def heatmap_overlay(
    img: Image.Image | tuple[int, int] | np.ndarray[Any, Any] | str | Path,
    *,
    boxes: np.ndarray[Any, Any] | None = None,            # kept for signature parity
    masks: Iterable[np.ndarray[Any, Any]] | None = None,  # ditto (currently unused)
    **kw: Any,
) -> NDArray[np.uint8]:
    """
    Overlay YOLO-Seg instance masks + labels on *img* and return a **BGR**
    ndarray ready for OpenCV / video encoding.
    """
    # ── normalise *img* to a BGR ndarray we can always return ────────────────
    if isinstance(img, np.ndarray):
        bgr = img
    elif isinstance(img, Image.Image):
        bgr = np.asarray(img.convert("RGB"))[:, :, ::-1]
    elif isinstance(img, (str, Path)):
        bgr = np.asarray(Image.open(img).convert("RGB"))[:, :, ::-1]
    else:
        # (w, h) tuple placeholder
        w, h = img  # type: ignore[misc]
        bgr = np.zeros((h, w, 3), dtype=np.uint8)

    # ── run segmentation & render (Ultralytics returns BGR already) ──────────
    try:
        if simple_status is not None:
            ctx = simple_status("segment")
        else:
            class _Null:
                def __enter__(self) -> None:
                    return None

                def __exit__(
                    self,
                    exc_type: type[BaseException] | None,
                    exc: BaseException | None,
                    tb: TracebackType | None,
                ) -> bool:
                    return False
            ctx = _Null()
        with ctx:
            res = _seg_model.predict(bgr, imgsz=640, conf=0.25, verbose=False)[0]  # type: ignore[index]
            out: NDArray[np.uint8] = np.asarray(res.plot(), dtype=np.uint8)        # type: ignore[arg-type]
        return out
    except Exception:
        # Any runtime glitch → return the original image unchanged
        _say("lambda.heatmap: segmentation failed → returning original frame")
        return bgr.copy()
