# projects/argos/panoptes/heatmap.py
"""
panoptes.heatmap
───────────────────
Instance‑segmentation / heat‑map overlay helper.

Lock‑down (2025‑08‑07)
────────────────────────────────
* No path/env probing — only the registry’s `load_segmenter()` is used.
* Import‑time hard‑failure if weights are missing (mirrors lambda_like).

Progress policy (single‑line UX)
────────────────────────────────
* The CLI owns the single, pinned spinner.
* This module is QUIET by default (no prints, no nested spinners).
* If you want per‑step animation here (e.g., debugging), opt‑in with:
      PANOPTES_NESTED_PROGRESS=1

API
───
    heatmap_overlay(img, *, boxes=None, alpha=0.4, return_mask=False, ...)
    → PIL.Image (RGB) or ndarray (BGR if return_mask=False; 8‑bit mask if True)
"""
from __future__ import annotations

import colorsys
import contextlib
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ContextManager, Iterable, Optional, Protocol, Union, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .model_registry import load_segmenter


# ─────────────────────────── progress (opt‑in only) ───────────────────────────
class ProgressLike(Protocol):
    def set_total(self, total_units: float) -> None: ...
    def set_current(self, label: str) -> None: ...
    def add(self, units: float, *, current_item: str | None = None) -> None: ...

# Nested spinners are disabled unless PANOPTES_NESTED_PROGRESS is truthy.
_ENABLE_NESTED = os.getenv("PANOPTES_NESTED_PROGRESS", "").strip().lower() in {
    "1", "true", "yes", "on"
}
try:  # local import so this file still loads if progress deps are missing
    from .progress import ProgressEngine  # type: ignore
    from .progress.bridges import live_percent  # type: ignore
except Exception:  # pragma: no cover
    ProgressEngine = None  # type: ignore
    live_percent = None  # type: ignore

# ─────────────────────────── logging ────────────────────────────
_LOG = logging.getLogger("panoptes.heatmap")
if not _LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
# QUIET BY DEFAULT so we don’t break the single-line spinner UX
_LOG.setLevel(logging.WARNING)

def _dbg(msg: str) -> None:
    _LOG.debug(f"[panoptes] {msg}")

# ─────────────────────────── model (hard‑fail) ─────────────────────────
_seg_model = load_segmenter()  # may raise RuntimeError — by design
_dbg("heatmap module init: segmenter ready (see registry log for weight)")

__all__ = ["heatmap_overlay"]

# ---------------------------------------------------------------------- #
# helper – normalise *img* into a BGR ndarray we can paint on            #
# ---------------------------------------------------------------------- #
def _to_bgr(img: Image.Image | np.ndarray[Any, Any] | str | Path | tuple[int, int]) -> NDArray[np.uint8]:
    if isinstance(img, np.ndarray):
        # Assume caller already gave us BGR uint8
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return img
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"))[:, :, ::-1]
    if isinstance(img, (str, Path)):
        return np.asarray(Image.open(img).convert("RGB"))[:, :, ::-1]
    if len(img) == 2:  # (w, h) placeholder
        w, h = img
        return np.zeros((h, w, 3), np.uint8)
    raise TypeError(f"Unsupported image type: {type(img)}")


# ---------------------------------------------------------------------- #
# public API                                                             #
# ---------------------------------------------------------------------- #
def heatmap_overlay(  # noqa: C901  (visual-logic)
    img: Image.Image | np.ndarray[Any, Any] | str | Path | tuple[int, int],
    *,
    boxes: Optional[np.ndarray[Any, Any]] = None,
    masks: Iterable[np.ndarray[Any, Any]] | None = None,  # accepted, ignored
    alpha: float = 0.4,
    return_mask: bool = False,
    cmap: str = "COLORMAP_JET",  # accepted for CLI parity
    kernel_scale: float = 5.0,  # ditto – currently unused
    **_: Any,
) -> Union[Image.Image, NDArray[np.uint8]]:
    """
    Overlay YOLO‑Seg instance masks (if available) on *img*.

    Returns PIL.Image (RGB) or ndarray (BGR if return_mask=False, or 8‑bit mask if True).
    """
    bgr: NDArray[np.uint8] = _to_bgr(img)
    h, w = bgr.shape[:2]

    # progress wiring (best‑effort; **disabled** unless user opts in)
    eng: Optional[ProgressLike]
    cm: ContextManager[Any]
    if _ENABLE_NESTED and ProgressEngine is not None and live_percent is not None:  # type: ignore[truthy-bool]
        eng_any = ProgressEngine()  # type: ignore[call-arg]
        eng = cast(ProgressLike, eng_any)
        cm = cast(ContextManager[Any], live_percent(eng, prefix="HEATMAP"))  # type: ignore[misc]
    else:
        eng = None
        cm = cast(ContextManager[Any], contextlib.nullcontext())

    with cm:
        if eng is not None:
            eng.set_total(4.0)
            eng.set_current("segment")

        # ─────────────────────── SEGMENTATION PATH ────────────────────────
        try:
            res: Any = _seg_model.predict(bgr, imgsz=640, conf=0.25, verbose=False)[0]  # type: ignore[index]
            mdat: Optional[Any] = getattr(getattr(res if isinstance(res, object) else None, "masks", None), "data", None)
        except Exception:
            res = None
            mdat = None

        if mdat is not None and res is not None:
            if eng is not None:
                eng.add(1.0, current_item="prepare masks")

            # ▸ (N,H,W) → np.float32
            try:
                import torch  # type: ignore
                masks_np: NDArray[np.float32] = (
                    mdat.cpu().numpy().astype(np.float32)  # type: ignore[attr-defined]
                    if isinstance(mdat, torch.Tensor)
                    else np.asarray(mdat, dtype=np.float32)
                )
            except ImportError:  # pragma: no cover
                masks_np = np.asarray(mdat, dtype=np.float32)

            overlay: NDArray[np.float32] = bgr.astype(np.float32)
            names_obj: object = getattr(_seg_model, "names", {})
            # Ultralytics may provide a dict[int,str] or list[str]; normalise with safe typing
            if isinstance(names_obj, Mapping):
                names_map = cast(Mapping[Union[int, str], object], names_obj)
                def _name_for(idx: int) -> str:
                    v = names_map.get(idx)
                    if v is None:
                        v = names_map.get(str(idx))
                    return str(v) if v is not None else str(idx)
            elif isinstance(names_obj, Sequence) and not isinstance(names_obj, (str, bytes)):
                names_seq = cast(Sequence[object], names_obj)
                def _name_for(idx: int) -> str:
                    return str(names_seq[idx]) if 0 <= idx < len(names_seq) else str(idx)
            else:
                def _name_for(idx: int) -> str:
                    return str(idx)

            mcount = int(getattr(masks_np, "shape", (0,))[0] or 0)
            if eng is not None:
                # recalibrate total to include per‑mask work (min 4 so % is sane)
                eng.set_total(max(4.0, float(mcount + 3)))
                eng.set_current(f"overlay {mcount} mask(s)")

            for idx, m_raw in enumerate(masks_np):
                m: NDArray[np.float32] = cast(NDArray[np.float32], m_raw)
                if m.shape != (h, w):
                    m = np.asarray(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST), dtype=np.float32)

                mask_bool: NDArray[np.bool_] = m >= 0.5

                # unique colour via golden‑ratio hue rotation
                hue: float = (idx * 0.61803398875) % 1.0
                r, g, b = (np.array(colorsys.hsv_to_rgb(hue, 1.0, 1.0)) * 255.0)
                colour_f32: NDArray[np.float32] = np.array([b, g, r], dtype=np.float32)
                colour_u8 = tuple(int(x) for x in (b, g, r))

                overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + colour_f32 * alpha

                if eng is not None:
                    eng.add(1.0, current_item=f"mask {idx + 1}/{mcount}")

                # optional label (class + conf + ID) if boxes are present
                boxes_obj: Any = getattr(res if isinstance(res, object) else object(), "boxes", None)
                if boxes_obj is not None and getattr(boxes_obj, "data", None) is not None:
                    try:
                        xyxy = boxes_obj.xyxy[idx]  # type: ignore[attr-defined]
                        conf = float(boxes_obj.conf[idx])  # type: ignore[attr-defined]
                        cls_id = int(boxes_obj.cls[idx])  # type: ignore[attr-defined]
                        try:
                            import torch  # type: ignore
                            if hasattr(xyxy, "cpu"):
                                xyxy = xyxy.cpu().numpy()
                        except ImportError:  # pragma: no cover
                            pass
                        xyxy = np.asarray(xyxy, dtype=float).reshape(-1)
                        text: str = f"{_name_for(cls_id)} {conf:.2f}"

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)  # type: ignore[arg-type]
                        x1: int = int(xyxy[0])
                        y1: int = max(0, int(xyxy[1]) - int(th) - 4)
                        cv2.rectangle(overlay, (x1, y1), (x1 + int(tw) + 2, y1 + int(th) + 4), colour_u8, -1)
                        cv2.putText(overlay, text, (x1 + 1, y1 + int(th) + 2), font, 0.5, (255, 255, 255), 1)
                    except Exception:
                        pass

            if return_mask:
                if eng is not None:
                    eng.add(1.0, current_item="merge to 8‑bit mask")
                return ((masks_np.max(axis=0) >= 0.5).astype(np.uint8) * 255)

            if eng is not None:
                eng.add(1.0, current_item="compose RGB")
            return Image.fromarray(overlay.astype(np.uint8)[:, :, ::-1])

        # ────────────────────── FALLBACK RECTANGLES ───────────────────────
        if boxes is None or (hasattr(boxes, "size") and getattr(boxes, "size") == 0):
            # quiet fallback to keep the single‑line UX clean
            if return_mask:
                if eng is not None:
                    eng.add(3.0, current_item="fallback mask (empty)")
                return np.zeros((h, w), np.uint8)
            if eng is not None:
                eng.add(3.0, current_item="fallback (original)")
            return Image.fromarray(bgr[:, :, ::-1])

        if eng is not None:
            eng.add(1.0, current_item="paint fallback rects")

        out_pil = Image.fromarray(bgr[:, :, ::-1])
        boxes_arr = np.asarray(boxes)
        last_dim = boxes_arr.shape[-1] if boxes_arr.ndim >= 2 else 5
        for bx in boxes_arr.reshape(-1, last_dim):
            if bx.shape[0] < 4:
                continue
            x1, y1, x2, y2 = map(int, bx[:4])
            rect = Image.new("RGBA", (max(0, x2 - x1), max(0, y2 - y1)), (255, 0, 0, int(alpha * 255)))
            out_pil.paste(rect, (x1, y1), rect)

        if return_mask:
            if eng is not None:
                eng.add(1.0, current_item="compose rect mask")
            msk = np.zeros((h, w), np.uint8)
            for bx in boxes_arr.reshape(-1, last_dim):
                if bx.shape[0] >= 4:
                    x1, y1, x2, y2 = map(int, bx[:4])
                    msk[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 255
            return msk

        if eng is not None:
            eng.add(1.0, current_item="done")
        return out_pil
