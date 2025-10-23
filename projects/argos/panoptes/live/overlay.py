"""
BGR drawing helpers for live:
  • detection boxes/labels,
  • segmentation heatmap compositing (HM) and PSE mask compositing,
  • classification card (top-K),
  • pose skeletons (keypoints + edges),
  • OBB polygons,
  • tiny HUD: "FPS | TASK | MODEL | DEVICE" (top-left).

If OpenCV is missing, drawing falls back or becomes a no-op while returning the
input frame (never crashes the pipeline).
"""

from __future__ import annotations

from typing import Optional, Mapping, Iterable, List, Tuple, Dict, Sequence

# Touch progress module lightly so this file participates in UX integration.
# (No spinners here to avoid per-frame overhead.)
try:
    from panoptes.progress import should_enable_spinners as _progress_spinners_enabled  # type: ignore[import]
except Exception:  # pragma: no cover
    def _progress_spinners_enabled(*_a: object, **_k: object) -> bool: return False

# Reference once to avoid "unused import" warnings in some linters (and avoid ALL_CAPS redefinition)
try:
    _progress_available = bool(_progress_spinners_enabled())
except Exception:
    _progress_available = False

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


# ─────────────────────────────────────────────────────────────────────
# DETECT: axis-aligned boxes
# ─────────────────────────────────────────────────────────────────────

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
        label = nm.get(int(cls_id), str(cls_id)) if (cls_id is not None and nm) else ""
        if label:
            label = f"{label} {conf:.2f}"
        else:
            label = f"{conf:.2f}"
        if label:
            tl = max(1, int(0.5 + 0.002 * (frame.shape[0] + frame.shape[1])))
            cv2.putText(frame, label, (x1 + 2, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), tl)
    return frame


# ─────────────────────────────────────────────────────────────────────
# HEATMAP: intensity → LUT overlay (JET)
# ─────────────────────────────────────────────────────────────────────

def draw_heatmap_bgr(frame: NDArrayU8, mask: NDArrayU8) -> NDArrayU8:
    """
    Composite a single-channel mask (0..255) over the frame as a heatmap.
    Only blend where mask > 0. Background remains the original BGR frame.
    """
    frame = _ensure_np(frame)
    assert np is not None

    # OpenCV path: colorize mask with JET and do per-pixel alpha blend
    if cv2 is not None:
        mask_u8 = mask.astype("uint8")
        hm = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)

        H, W = int(frame.shape[0]), int(frame.shape[1])
        if hm.shape[:2] != (H, W):
            hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_NEAREST)

        alpha_max = 0.4
        a = (mask_u8.astype("float32") / 255.0) * alpha_max  # (H, W)
        a3 = a[..., None]  # (H, W, 1)

        frame_f = frame.astype("float32")
        hm_f = hm.astype("float32")
        out_f = frame_f * (1.0 - a3) + hm_f * a3
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


# ─────────────────────────────────────────────────────────────────────
# PSE: semantic/instance masks (solid/alpha compositing)
# ─────────────────────────────────────────────────────────────────────

def draw_masks_bgr(
    frame: NDArrayU8,
    instances: Iterable[Tuple[NDArrayU8, float, Optional[int]]],
    names: Optional[Names] = None,
    alpha: float = 0.35,
) -> NDArrayU8:
    """
    Draw instance masks (each mask uint8 0/255) with per-class color and label.

    instances: iterable of (mask_u8(H,W), conf, cls_id or None)
    """
    frame = _ensure_np(frame)
    assert np is not None

    H, W = int(frame.shape[0]), int(frame.shape[1])
    out = frame.copy()

    # Simple color palette
    palette = [
        (0, 197, 255), (255, 178, 29), (23, 204, 146), (255, 105, 97),
        (52, 148, 230), (222, 98, 98), (141, 218, 139), (255, 160, 122),
    ]

    alpha = float(max(0.0, min(1.0, alpha)))
    inv_alpha = 1.0 - alpha
    idx = 0
    for m_u8, conf, cls_id in instances:
        try:
            if m_u8.shape[:2] != (H, W):
                if cv2 is not None:
                    m_u8 = cv2.resize(m_u8, (W, H), interpolation=cv2.INTER_NEAREST)
                else:
                    # nearest-neighbor via NumPy
                    y_idx = np.round(np.linspace(0, m_u8.shape[0] - 1, H)).astype(int)
                    x_idx = np.round(np.linspace(0, m_u8.shape[1] - 1, W)).astype(int)
                    m_u8 = m_u8[y_idx[:, None], x_idx[None, :]]
            color = palette[idx % len(palette)]
            idx += 1

            mask_bool = np.greater(m_u8, 0)
            if not np.any(mask_bool):
                continue

            # Blend
            if cv2 is not None:
                color_vec = np.array(color, dtype=np.float32)
                region = out[mask_bool].astype(np.float32, copy=False)
                blended = region * inv_alpha + color_vec * alpha
                out[mask_bool] = blended.astype(np.uint8, copy=False)
                mask_u8 = mask_bool.astype(np.uint8) * 255
                # Optional thin outline for instance boundary
                edges = cv2.Canny(mask_u8, 50, 150)
                if edges.size:
                    out[edges > 0] = color
            else:
                # Minimal NumPy-only overlay
                color_vec = np.array(color, dtype=np.float32)
                region = out[mask_bool].astype(np.float32, copy=False)
                blended = region * inv_alpha + color_vec * alpha
                out[mask_bool] = blended.astype(np.uint8, copy=False)

            if cls_id is not None and names:
                label = names.get(int(cls_id), str(cls_id))
            else:
                label = ""
            if label and cv2 is not None:
                # find a spot to draw label: first pixel of mask bounding box
                ys, xs = np.nonzero(mask_bool)
                if ys.size > 0:
                    y0, x0 = int(ys.min()), int(xs.min())
                    txt = f"{label} {conf:.2f}"
                    cv2.putText(out, txt, (x0 + 2, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 245, 200), 2)
        except Exception:
            continue
    return out


# ─────────────────────────────────────────────────────────────────────
# CLASSIFY: top-K card
# ─────────────────────────────────────────────────────────────────────

def draw_classify_card_bgr(
    frame: NDArrayU8,
    topk: List[Tuple[str, float]],
) -> NDArrayU8:
    """
    Draw top-K predictions as a small card in the top-left (under the HUD).
    """
    frame = _ensure_np(frame)
    if cv2 is None:
        return frame

    x, y, pad = 8, 32, 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2

    # compute card size
    lines = [f"{lbl}: {p*100:.1f}%" for (lbl, p) in topk]
    w = 0
    for t in lines:
        (tw, _), _ = cv2.getTextSize(t, font, scale, thickness)
        w = max(w, tw)
    # derive a stable text line height from a sample string
    line_h = cv2.getTextSize("Ag", font, scale, thickness)[0][1] + 6
    h = line_h * len(lines) + pad * 2

    # background
    cv2.rectangle(frame, (x, y), (x + w + pad * 2, y + h), (20, 20, 20), -1)
    cv2.rectangle(frame, (x, y), (x + w + pad * 2, y + h), (80, 80, 80), 1)

    # text
    ty = y + pad + 14
    for t in lines:
        cv2.putText(frame, t, (x + pad, ty), font, scale, (255, 245, 200), thickness)
        ty += line_h

    return frame


# ─────────────────────────────────────────────────────────────────────
# POSE: keypoints + skeleton
# ─────────────────────────────────────────────────────────────────────

_COCO_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),     # arms
    (5, 6), (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (5, 1), (6, 2), (1, 3), (2, 4), (3, 0), (4, 0)  # head/eyes/ears (if present)
]

def draw_pose_bgr(
    frame: NDArrayU8,
    people: Sequence[Sequence[Sequence[float]]],  # each: sequence of (x, y, score)
    *,
    skeleton: Optional[List[Tuple[int, int]]] = None,
) -> NDArrayU8:
    frame = _ensure_np(frame)
    if cv2 is None:
        return frame

    edges = skeleton or _COCO_EDGES
    for kp in people:
        try:
            # points
            for item in kp:
                x, y, s = float(item[0]), float(item[1]), float(item[2])
                if s < 0.05:
                    continue
                cv2.circle(frame, (int(x), int(y)), 3, (0, 210, 255), -1)

            # edges
            n = len(kp)
            for (i, j) in edges:
                if i < 0 or j < 0 or i >= n or j >= n:
                    continue
                xi, yi, si = float(kp[i][0]), float(kp[i][1]), float(kp[i][2])
                xj, yj, sj = float(kp[j][0]), float(kp[j][1]), float(kp[j][2])
                if si < 0.05 or sj < 0.05:
                    continue
                cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), (52, 148, 230), 2)
        except Exception:
            continue
    return frame


# ─────────────────────────────────────────────────────────────────────
# OBB: rotated rectangles / polygons
# ─────────────────────────────────────────────────────────────────────

def draw_obb_bgr(
    frame: NDArrayU8,
    obbs: Iterable[Tuple[List[Tuple[int, int]], float, Optional[int]]],  # (pts4, conf, cls_id)
    names: Optional[Dict[int, str]] = None,
) -> NDArrayU8:
    frame = _ensure_np(frame)
    assert np is not None
    if cv2 is None:
        return frame

    for pts4, conf, cls_id in obbs:
        try:
            pts = np.array(pts4, dtype=np.int32).reshape((-1, 1, 2))  # type: ignore
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 178, 29), thickness=2)
            if cls_id is not None and names:
                label = names.get(int(cls_id), str(cls_id))
            else:
                label = ""
            if label:
                x0, y0 = int(pts4[0][0]), int(pts4[0][1])
                txt = f"{label} {conf:.2f}"
                cv2.putText(frame, txt, (x0 + 2, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 245, 200), 2)
        except Exception:
            continue
    return frame


# ─────────────────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────────────────

def hud(
    frame: NDArrayU8,
    *,
    fps: float,
    task: str,
    model: str,
    device: str,
    notice: Optional[str] = None,
) -> None:
    """Draw a tiny HUD in the top-left corner (optionally with a toast line)."""
    frame = _ensure_np(frame)
    if cv2 is None:
        return
    text = f"{fps:5.1f} FPS | {task} | {model} | {device}"
    cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 245, 200), 2)
    if notice:
        cv2.putText(frame, notice, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 200), 2)
