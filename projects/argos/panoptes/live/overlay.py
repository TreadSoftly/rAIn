"""
BGR drawing helpers for live:
  • detection boxes/labels,
  • segmentation heatmap compositing consistent with offline output,
  • tiny HUD: "FPS | TASK | MODEL | DEVICE" (top-left).

If OpenCV is missing, these become no-ops and simply return the input frame.
"""

from __future__ import annotations

from typing import Optional, Mapping, cast

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
    Composite a single-channel mask (0..255) over frame as a heatmap.
    """
    frame = _ensure_np(frame)
    if cv2 is None:
        # simple alpha blend if OpenCV is absent
        alpha = 0.4
        # normalize mask to 0..1
        m = (mask.astype("float32") / 255.0)[..., None]
        # tint red-ish
        tint = frame.copy()
        tint[..., 2] = 255
        return (frame * (1 - alpha * m) + tint * (alpha * m)).astype(frame.dtype)

    hm = cv2.applyColorMap(mask.astype("uint8"), cv2.COLORMAP_JET)
    out = cv2.addWeighted(frame, 0.6, hm, 0.4, 0.0)
    return cast(NDArrayU8, out)


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
