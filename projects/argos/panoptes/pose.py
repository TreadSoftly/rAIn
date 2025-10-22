# projects/argos/panoptes/pose.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple, Union, cast

import numpy as np
from PIL import Image

from .model_registry import load_pose  # strict registry

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

# OpenCV optional at import; required at call sites
try:
    import cv2 as _cv2_mod  # type: ignore
except Exception:  # pragma: no cover
    _cv2_mod = None  # type: ignore

def _require_cv2() -> Any:
    if _cv2_mod is None:
        raise RuntimeError("OpenCV is required for pose drawing.")
    return _cv2_mod  # Any

# ─────────────────────────── logging ────────────────────────────
_LOG = logging.getLogger("panoptes.pose")
_LOG.addHandler(logging.NullHandler())
_LOG.setLevel(logging.ERROR)


def _pupdate(progress: Any | None, **kwargs: Any) -> None:
    """Best-effort progress.update(**kwargs) if a parent spinner is provided."""
    if progress is None:
        return
    try:
        progress.update(**kwargs)
    except Exception:
        pass


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


def _model_label(m: Any) -> str:
    try:
        for attr in ("ckpt_path", "weights", "weight", "model", "name"):
            val = getattr(m, attr, None)
            if isinstance(val, (str, Path)):
                return Path(str(val)).stem
            if val:
                sv = getattr(val, "name", None)
                if isinstance(sv, (str, Path)):
                    return Path(str(sv)).stem
        return m.__class__.__name__
    except Exception:  # pragma: no cover
        return "model"


def run_image(
    src_img: Union[Image.Image, NDArrayAny, str, Path],
    *,
    out_dir: Path,
    model: Any | None = None,
    conf: float | None = None,
    iou: float | None = None,
    progress: Any = None,
) -> Path:
    cv = _require_cv2()

    bgr, stem, suf = _as_bgr(src_img)
    basename = f"{stem}{suf}"

    # Parent progress provided → only update context segments
    _pupdate(progress, item=basename, job="pose")

    # Acquire pose model
    if model is None:
        loaded_model = load_pose()  # strict loader; may raise
        if loaded_model is None:
            raise RuntimeError("Pose loader returned None (no model).")
        model = loaded_model

    # Update model label
    _pupdate(progress, model=_model_label(model))

    # Inference
    _pupdate(progress, job="infer")

    res_list: list[Any] = cast(
        list[Any],
        model.predict(  # type: ignore[call-arg]
            bgr,
            imgsz=640,
            conf=(conf or 0.25),
            iou=(iou or 0.45),
            verbose=False,
        ),
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
                        cv.line(bgr, (ax, ay), (bx, by), color_ln, 2)
                # Draw points
                for j in range(k):
                    x, y = int(p[j, 0]), int(p[j, 1])
                    cv.circle(bgr, (x, y), 3, color_pt, -1)

    # Save
    _pupdate(progress, job="write result")

    out_ext = suf if suf in {".jpg", ".jpeg", ".png"} else ".jpg"
    out_path = (Path(out_dir).expanduser().resolve() / f"{stem}_pose{out_ext}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bgr[:, :, ::-1]).save(out_path)

    _pupdate(progress, job="done")
    return out_path
