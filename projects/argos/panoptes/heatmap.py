"""
panoptes.heatmap
───────────────────
Instance‑segmentation / heat‑map overlay helper.

Hard‑lock refactor (2025‑08‑07)
────────────────────────────────
* **No** path or environment probing – the *only* weight ever loaded is
  the one returned by `panoptes.model_registry.load_segmenter()`.
* Import‑time hard‑failure: if the segmentation weight is absent (or
  Ultralytics is not installed) the *module import* raises immediately,
  surfacing the problem just like in `lambda_like.py` and the Lambda
  handler.
* Public signature and behaviour are unchanged:
        heatmap_overlay(img, *, boxes=None, alpha=0.4, return_mask=False,…)
  – it still returns either a Pillow `Image` *or* a `np.ndarray` (BGR),
  and the `return_mask` flag keeps working.
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Union
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .model_registry import load_segmenter

# ─────────────────────────── model (hard‑fail) ─────────────────────────
_seg_model = load_segmenter()  # may raise RuntimeError – by design

__all__ = ["heatmap_overlay"]

# ---------------------------------------------------------------------- #
# helper – normalise *img* into a BGR ndarray we can paint on            #
# ---------------------------------------------------------------------- #
def _to_bgr(img: Image.Image | np.ndarray[Any, Any] | str | Path | tuple[int, int]) -> NDArray[np.uint8]:
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"))[:, :, ::-1]
    if isinstance(img, (str, Path)):
        return np.asarray(Image.open(img).convert("RGB"))[:, :, ::-1]
    if len(img) == 2:          # (w, h) placeholder
        w, h = img
        return np.zeros((h, w, 3), np.uint8)
    raise TypeError(f"Unsupported image type: {type(img)}")


# ---------------------------------------------------------------------- #
# public API                                                             #
# ---------------------------------------------------------------------- #
def heatmap_overlay(                               # noqa: C901  (visual‑logic)
    img: Image.Image | np.ndarray[Any, Any] | str | Path | tuple[int, int],
    *,
    boxes: Optional[np.ndarray[Any, Any]] = None,
    masks: Iterable[np.ndarray[Any, Any]] | None = None,   # accepted, ignored
    alpha: float = 0.4,
    return_mask: bool = False,
    cmap: str = "COLORMAP_JET",                    # accepted for CLI parity
    kernel_scale: float = 5.0,                    # ditto – currently unused
    **_: Any,
) -> Union[Image.Image, NDArray[np.uint8]]:
    """
    Overlay YOLO‑Seg instance masks (if available) on *img*.

    Parameters
    ----------
    img          : Pillow image | ndarray BGR | file path | (w, h) tuple
    boxes        : Detector boxes – only used for the *fallback* rectangle path
    alpha        : Blend factor for coloured masks
    return_mask  : If *True* return a binary mask instead of a coloured image
    cmap, kernel_scale, **kw : kept for CLI/api compatibility – currently ignored
    """
    bgr: NDArray[np.uint8] = _to_bgr(img)
    h, w = bgr.shape[:2]

    # ─────────────────────── SEGMENTATION PATH ────────────────────────
    try:
        res: Any = _seg_model.predict(bgr, imgsz=640, conf=0.25, verbose=False)[0]  # type: ignore[index]
        mdat: Optional[Any] = getattr(getattr(res if isinstance(res, object) else None, "masks", None), "data", None)
    except Exception:
        res: Any = None
        mdat: Optional[Any] = None

    if mdat is not None and res is not None:
        # ▸ (N,H,W) → np.float32
        try:
            import torch  # type: ignore
            masks_np: NDArray[np.float32] = (
                mdat.cpu().numpy().astype(np.float32)     # type: ignore[attr-defined]
                if isinstance(mdat, torch.Tensor)
                else np.asarray(mdat, dtype=np.float32)
            )
        except ImportError:
            masks_np = np.asarray(mdat, dtype=np.float32)

        overlay: NDArray[np.float32] = bgr.astype(np.float32)
        names: Any = getattr(_seg_model, "names", {})
        if not isinstance(names, dict):
            names = {}
        names = cast(dict[str, Any], names)  # type: ignore[import]

        for idx, m in enumerate(masks_np):
            if m.shape != (h, w):
                m: NDArray[np.float32] = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)

            mask_bool: np.ndarray[np.bool_, Any] = m >= 0.5

            # unique colour via golden‑ratio hue rotation
            hue: float = (idx * 0.61803398875) % 1.0
            r, g, b = (np.array(colorsys.hsv_to_rgb(hue, 1.0, 1.0)) * 255)
            colour: NDArray[np.float32]  = np.array([b, g, r], dtype=np.float32)

            overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + colour * alpha

            # ▸ draw label (class + conf + ID)
            boxes_obj: Any = getattr(res if isinstance(res, object) else object(), "boxes", None)
            if boxes_obj is not None and getattr(boxes_obj, "data", None) is not None:
                xyxy: NDArray[np.float32] = boxes_obj.xyxy[idx].cpu().numpy().astype(float)        # type: ignore[attr-defined]
                conf: float = float(boxes_obj.conf[idx])                           # type: ignore[attr-defined]
                cls: int  = int(boxes_obj.cls[idx])                              # type: ignore[attr-defined]
                text: str = f"{names.get(str(cls), str(cls))} {conf:.2f}  ID {idx+1}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1) # type: ignore[arg-type]
                x1: int = int(xyxy[0])
                y1_raw: float = float(xyxy[1])
                try:
                    y1_val: int = int(y1_raw)
                except (TypeError, ValueError):
                    y1_val = 0
                th_val: int = int(th)
                y1: int = int(max(0, y1_val - th_val - 4))
                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (int(x1 + tw + 2), int(y1 + th_val + 4)),
                    colour.tolist(),
                    -1
                )
                cv2.putText(
                    overlay,
                    text,
                    (int(x1 + 1), int(y1 + th + 2)),
                    font,
                    0.5,
                    (255, 255, 255),
                    1
                )

        if return_mask:
            return ((masks_np.max(axis=0) >= 0.5).astype(np.uint8) * 255)

        return Image.fromarray(overlay.astype(np.uint8)[:, :, ::-1])

    # ────────────────────── FALLBACK RECTANGLES ───────────────────────
    if boxes is None or (hasattr(boxes, "size") and boxes.size == 0):
        return np.zeros((h, w), np.uint8) if return_mask else Image.fromarray(bgr[:, :, ::-1])

    out_pil = Image.fromarray(bgr[:, :, ::-1])
    for bx in np.asarray(boxes).reshape(-1, boxes.shape[-1] if hasattr(boxes, "shape") else 5):
        if bx.shape[0] < 4:
            continue
        x1, y1, x2, y2 = map(int, bx[:4])
        rect = Image.new("RGBA", (x2 - x1, y2 - y1), (255, 0, 0, int(alpha * 255)))
        out_pil.paste(rect, (x1, y1), rect)

    if return_mask:
        msk = np.zeros((h, w), np.uint8)
        for bx in np.asarray(boxes).reshape(-1, boxes.shape[-1] if hasattr(boxes, "shape") else 5):
            if bx.shape[0] >= 4:
                x1, y1, x2, y2 = map(int, bx[:4])
                msk[y1:y2, x1:x2] = 255
        return msk

    return out_pil
