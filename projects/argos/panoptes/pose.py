# projects/argos/panoptes/pose.py
"""
panoptes.pose — image pose overlay

Contract
────────
run_image(src_img, *, out_dir, model=None, conf=None, iou=None, progress=None) -> Path

• Loads pose model STRICTLY via panoptes.model_registry when *model* is None.
• Draws keypoints (+ skeleton when topology is COCO‑like with 17 points).
• Writes <stem>_pose.<ext> under *out_dir* (uses source extension if jpg/png, else .jpg).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple, Union, cast

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import numpy as _np
    import numpy.typing as npt

    NDArrayU8 = npt.NDArray[_np.uint8]
    NDArrayF32 = npt.NDArray[_np.float32]
    NDArrayAny = npt.NDArray[Any]
else:
    NDArrayU8 = Any   # type: ignore[assignment]
    NDArrayF32 = Any  # type: ignore[assignment]
    NDArrayAny = Any  # type: ignore[assignment]

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from .model_registry import load_pose  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    load_pose = None  # type: ignore

_LOG = logging.getLogger("panoptes.pose")
if not _LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
_LOG.setLevel(logging.WARNING)


def _ensure_bgr_u8_writable(arr: NDArrayAny) -> NDArrayU8:
    """
    Ensure BGR uint8, C-contiguous, writable (copy if needed).
    Accepts HxW (grayscale) or HxWxC arrays of any dtype; returns HxWx3 BGR uint8.
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        # Grayscale → 3-channel
        a3 = np.stack([a, a, a], axis=2)
    else:
        if a.ndim != 3 or a.shape[2] < 3:
            raise TypeError(f"Unsupported ndarray shape for image: {a.shape!r}")
        a3 = a[:, :, :3]

    if a3.dtype != np.uint8:
        a3 = a3.astype(np.uint8, copy=False)

    bgr = np.ascontiguousarray(a3)
    bgr.setflags(write=True)
    return bgr


def _as_bgr(img: Union[Image.Image, NDArrayAny, str, Path]) -> Tuple[NDArrayU8, str, str]:
    """
    Return (BGR uint8 writable array, stem, suffix).
    For PIL inputs or file paths, convert RGB→BGR and force a real copy.
    For ndarray inputs, assume it is BGR and just enforce contiguity/writability.
    """
    stem = "image"
    suffix = ".jpg"

    if isinstance(img, Image.Image):
        rgb = np.array(img.convert("RGB"), dtype=np.uint8, copy=True)
        bgr = rgb[:, :, ::-1].copy()  # writable + contiguous
        return bgr, stem, suffix

    if isinstance(img, (str, Path)):
        p = Path(img)
        stem = p.stem or stem
        suffix = p.suffix.lower() or suffix
        with Image.open(p) as im:
            rgb = np.array(im.convert("RGB"), dtype=np.uint8, copy=True)
        bgr = rgb[:, :, ::-1].copy()  # writable + contiguous
        return bgr, stem, suffix

    arr = np.asarray(img)
    return _ensure_bgr_u8_writable(arr), stem, suffix


# COCO‑like skeleton (17 keypoints). If different K, we only draw points.
_COCO_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 3), (0, 2), (2, 4),     # head/ears/eyes
    (5, 7), (7, 9), (6, 8), (8, 10),    # arms
    (5, 6), (5, 11), (6, 12),           # shoulders→hips
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]


def run_image(
    src_img: Union[Image.Image, NDArrayAny, str, Path],
    *,
    out_dir: Path,
    model: Any | None = None,
    conf: float | None = None,
    iou: float | None = None,
    progress: Any = None,
) -> Path:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for pose drawing.")

    if progress:
        try:
            progress.update(current="pose")
        except Exception:
            pass

    bgr, stem, suf = _as_bgr(src_img)

    # Acquire pose model
    if model is None:
        if load_pose is None:
            raise RuntimeError("Pose loader is unavailable (model_registry missing).")
        loaded_model = load_pose()  # type: ignore[call-arg]
        if loaded_model is None:
            raise RuntimeError("Pose loader returned None (no model).")
        model = loaded_model

    # Inference
    assert model is not None, "Pose model is None"
    res_list: list[Any] = cast(
        list[Any],
        model.predict(
            bgr,
            imgsz=640,
            conf=(conf or 0.25),
            iou=(iou or 0.45),
            verbose=False,
        ),  # type: ignore[call-arg]
    )
    res = res_list[0] if res_list else None

    if res is not None and getattr(res, "keypoints", None) is not None:
        kp_obj = res.keypoints
        # Normalize to ndarray [N,K,2]
        try:
            pts = getattr(kp_obj, "xy", None)
            if pts is None:
                pts = getattr(kp_obj, "data", None)
            if pts is not None and hasattr(pts, "cpu"):
                try:
                    pts = pts.cpu().numpy()
                except Exception:
                    pass
            pts_np: NDArrayF32 = np.asarray(pts if pts is not None else [], dtype=np.float32)
        except Exception:
            pts_np = np.zeros((0, 0, 2), dtype=np.float32)

        if pts_np.size and pts_np.ndim == 3 and pts_np.shape[2] >= 2:
            n, k, _ = pts_np.shape
            color_pt = (0, 255, 255)
            color_ln = (0, 210, 255)
            for i in range(n):
                p = pts_np[i, :, :2]
                # Draw edges if COCO topology
                if k == 17:
                    for a, b in _COCO_EDGES:
                        ax, ay = int(p[a, 0]), int(p[a, 1])
                        bx, by = int(p[b, 0]), int(p[b, 1])
                        cv2.line(bgr, (ax, ay), (bx, by), color_ln, 2)
                # Draw points
                for j in range(k):
                    x, y = int(p[j, 0]), int(p[j, 1])
                    cv2.circle(bgr, (x, y), 3, color_pt, -1)

    # Save
    if progress:
        try:
            progress.update(current="write result")
        except Exception:
            pass

    out_ext = suf if suf in {".jpg", ".jpeg", ".png"} else ".jpg"
    out_path = (Path(out_dir).expanduser().resolve() / f"{stem}_pose{out_ext}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bgr[:, :, ::-1]).save(out_path)

    if progress:
        try:
            progress.update(current="done")
        except Exception:
            pass

    return out_path
