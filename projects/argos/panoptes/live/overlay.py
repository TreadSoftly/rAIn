"""
BGR drawing helpers for live:
  • detection boxes/labels,
  • segmentation heatmap compositing consistent with offline output,
  • tiny HUD: "FPS | TASK | MODEL | DEVICE" (top-left).

If OpenCV is missing, these become no-ops and simply return the input frame.
"""

from __future__ import annotations

from typing import Optional, Mapping

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

from ._types import NDArrayU8, Boxes, Names


def _ensure_np(frame: NDArrayU8) -> NDArrayU8:
    if np is None:
        raise RuntimeError("numpy required")
    return frame


def draw_boxes_bgr(
    frame: NDArrayU8,
    boxes: Boxes,
    names: Optional[Names] = None,
) -> NDArrayU8:
    """
    Draw boxes on BGR frame.
      box tuple: (x1, y1, x2, y2, conf, cls_id or None)

    Returns the same frame for chaining.
    """
    frame = _ensure_np(frame)
    if cv2 is None:
        return frame  # no-op without OpenCV

    nm: Mapping[int, str] = (names or {})
    for (x1, y1, x2, y2, conf, cls_id) in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 255), 2)
        label = nm.get(int(cls_id), str(cls_id)) if cls_id is not None else ""
        if conf is not None:
            if label:
                label = f"{label} {conf:.2f}"
            else:
                label = f"{conf:.2f}"
        if label:
            tl = max(1, int(0.5 + 0.002 * (frame.shape[0] + frame.shape[1])))
            cv2.putText(frame, label, (x1 + 2, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), tl)
    return frame


def draw_heatmap_bgr(frame: NDArrayU8, mask: NDArrayU8) -> NDArrayU8:
    """
    Composite a single-channel mask (0..255) over the frame as a heatmap.

    Only blend where mask > 0. Background remains the original BGR frame.
    """
    frame = _ensure_np(frame)

    if np is None:
        return frame

    # OpenCV path: colorize mask with JET and do per-pixel alpha blend
    if cv2 is not None:
        mask_u8 = mask.astype("uint8")

        # Colorize the mask (BGR heatmap)
        hm = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)

        # Ensure shapes match the frame
        H, W = int(frame.shape[0]), int(frame.shape[1])
        if hm.shape[:2] != (H, W):
            hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_NEAREST)

        # Per-pixel alpha scaled by mask intensity (0..alpha_max)
        alpha_max = 0.4
        a = (mask_u8.astype("float32") / 255.0) * alpha_max  # (H, W)
        a3 = a[..., None]  # (H, W, 1)

        # Blend only where mask > 0
        frame_f = frame.astype("float32")
        hm_f = hm.astype("float32")
        out_f = frame_f * (1.0 - a3) + hm_f * a3
        # Guarantee uint8 dtype for the return type
        out_u8: NDArrayU8 = np.clip(out_f, 0, 255).astype(np.uint8)
        return out_u8

    # NumPy fallback (no cv2): red-ish tint, masked only
    alpha_max = 0.4
    a = (mask.astype("float32") / 255.0) * alpha_max
    a3 = a[..., None]
    tint = frame.copy()
    tint[..., 2] = 255  # boost red channel
    out_float = (frame.astype("float32") * (1.0 - a3) + tint.astype("float32") * a3)
    out_u8: NDArrayU8 = out_float.astype(np.uint8)
    return out_u8


def hud(
    frame: NDArrayU8,
    *,
    fps: float,
    task: str,
    model: str,
    device: str,
) -> None:
    """Draw a tiny one-line HUD in the top-left corner."""
    frame = _ensure_np(frame)
    if cv2 is None:
        return
    text = f"{fps:5.1f} FPS | {task} | {model} | {device}"
    cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 245, 200), 2)
