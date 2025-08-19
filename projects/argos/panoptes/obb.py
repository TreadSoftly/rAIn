# projects/argos/panoptes/obb.py
"""
panoptes.obb — oriented bounding boxes overlay (image)

Contract
────────
run_image(src_img, *, out_dir, model=None, conf=None, iou=None, progress=None) -> Path

• Loads OBB model STRICTLY via panoptes.model_registry when *model* is None.
• Prefers true oriented polygons from results. Falls back to axis‑aligned boxes if needed.
• Writes <stem>_obb.<ext> under *out_dir* (uses source extension if jpg/png, else .jpg).
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
    PolyF32 = npt.NDArray[_np.float32]   # shape (4, 2)
    NDArrayAny = npt.NDArray[Any]
else:
    NDArrayU8 = Any   # type: ignore[assignment]
    PolyF32 = Any     # type: ignore[assignment]
    NDArrayAny = Any  # type: ignore[assignment]

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from .model_registry import load_obb  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    load_obb = None  # type: ignore

_LOG = logging.getLogger("panoptes.obb")
if not _LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
_LOG.setLevel(logging.WARNING)


def _ensure_bgr_u8_writable(arr: NDArrayAny) -> NDArrayU8:
    """Ensure BGR uint8, C-contiguous, writable (copy if needed)."""
    a = np.asarray(arr)
    if a.ndim == 2:
        a3 = np.stack([a, a, a], axis=2)
    else:
        if a.ndim != 3 or a.shape[2] < 3:
            raise TypeError(f"Unsupported ndarray shape: {a.shape!r}")
        a3 = a[:, :, :3]

    if a3.dtype != np.uint8:
        a3 = a3.astype(np.uint8, copy=False)

    bgr = np.ascontiguousarray(a3)
    bgr.setflags(write=True)
    return bgr


def _as_bgr(img: Union[Image.Image, NDArrayAny, str, Path]) -> Tuple[NDArrayU8, str, str]:
    """
    Return (BGR uint8 writable array, stem, suffix).
    For PIL/paths we convert RGB→BGR and force a copy.
    For ndarray inputs we assume it is already BGR.
    """
    stem = "image"
    suffix = ".jpg"

    if isinstance(img, Image.Image):
        rgb = np.array(img.convert("RGB"), dtype=np.uint8, copy=True)
        bgr = rgb[:, :, ::-1].copy()
        return bgr, stem, suffix

    if isinstance(img, (str, Path)):
        p = Path(img)
        stem = p.stem or stem
        suffix = p.suffix.lower() or suffix
        with Image.open(p) as im:
            rgb = np.array(im.convert("RGB"), dtype=np.uint8, copy=True)
        bgr = rgb[:, :, ::-1].copy()
        return bgr, stem, suffix

    arr = np.asarray(img)
    return _ensure_bgr_u8_writable(arr), stem, suffix


def _extract_polys(res: Any) -> List[PolyF32]:
    """
    Return list of (4,2) float32 polygons if available; else [].
    Supports Ultralytics OBB results (xyxyxyxy) and a few common variants.
    """
    polys: List[PolyF32] = []

    obb = getattr(res, "obb", None)
    if obb is not None:
        for key in ("xyxyxyxy", "xyxy", "xy"):
            pts = getattr(obb, key, None)
            if pts is None:
                continue
            if hasattr(pts, "cpu"):
                try:
                    pts = pts.cpu().numpy()
                except Exception:
                    pass
            arr = np.asarray(pts, dtype=np.float32)
            if arr.size == 0:
                continue
            # Normalize to (N, 8) then to (N, 4, 2)
            arr = arr.reshape(-1, 8).reshape(-1, 4, 2)
            for poly in arr:
                polys.append(np.asarray(poly, dtype=np.float32))
            return polys  # prefer first successful representation

    # Fallback: axis-aligned boxes
    boxes = getattr(getattr(res, "boxes", None), "xyxy", None)
    if boxes is not None:
        if hasattr(boxes, "cpu"):
            try:
                boxes = boxes.cpu().numpy()
            except Exception:
                pass
        arr2 = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        for x1, y1, x2, y2 in arr2:
            polys.append(
                np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            )
    return polys


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
        raise RuntimeError("OpenCV is required for OBB drawing.")

    if progress:
        try:
            progress.update(current="obb")
        except Exception:
            pass

    bgr, stem, suf = _as_bgr(src_img)

    # Acquire OBB model
    mdl: Any
    if model is not None:
        mdl = model
    else:
        if load_obb is None:
            raise RuntimeError("OBB loader is unavailable (model_registry missing).")
        loaded = load_obb()  # type: ignore[call-arg]
        if loaded is None:
            raise RuntimeError("OBB loader returned None (failed to load model).")
        mdl = cast(Any, loaded)

    # Inference
    res_list: list[Any] = cast(
        list[Any],
        mdl.predict(
            bgr,
            imgsz=640,
            conf=(conf or 0.25),
            iou=(iou or 0.45),
            verbose=False,
        ),  # type: ignore[call-arg]
    )
    res = res_list[0] if res_list else None

    # Draw
    color = (0, 210, 255)
    if res is not None:
        polys = _extract_polys(res)
        for poly in polys:
            pts = np.asarray(poly, dtype=np.float32).reshape(4, 2).astype(np.int32)
            cv2.polylines(bgr, [pts], isClosed=True, color=color, thickness=2)

    # Save
    if progress:
        try:
            progress.update(current="write result")
        except Exception:
            pass

    out_ext = suf if suf in {".jpg", ".jpeg", ".png"} else ".jpg"
    out_path = (Path(out_dir).expanduser().resolve() / f"{stem}_obb{out_ext}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bgr[:, :, ::-1]).save(out_path)

    if progress:
        try:
            progress.update(current="done")
        except Exception:
            pass

    return out_path
