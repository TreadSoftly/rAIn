# projects/argos/panoptes/obb.py
"""
panoptes.obb — oriented bounding boxes overlay (image)

Contract
────────
run_image(src_img, *, out_dir, model=None, conf=None, iou=None, progress=None) -> Path

• Loads OBB model STRICTLY via panoptes.model_registry when *model* is None.
• Prefers true oriented polygons from results. Falls back to axis‑aligned boxes if needed.
• Writes <stem>_obb.<ext> under *out_dir* (uses source extension if jpg/png, else .jpg).

Progress policy (single‑line UX)
────────────────────────────────
* This module NEVER creates its own spinner and NEVER changes totals/counters.
* If a parent `progress` handle is supplied (the Halo/Rich spinner), we only update:
      item=[File basename], job in {"object","infer","write result","done"},
      model=<model-label>.
* If `progress` is None, run completely quiet (no prints, no nested spinners).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple, Union, cast

import numpy as np
from PIL import Image

from .model_registry import load_obb  # strict registry

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

# OpenCV is optional at import; we require it at call sites.
try:
    import cv2 as _cv2_mod  # type: ignore
except Exception:  # pragma: no cover
    _cv2_mod = None  # type: ignore

def _require_cv2() -> Any:
    if _cv2_mod is None:
        raise RuntimeError("OpenCV is required for OBB drawing.")
    return _cv2_mod  # typed as Any to silence 'attribute of None' linters

# ─────────────────────────── logging ────────────────────────────
_LOG = logging.getLogger("panoptes.obb")
if not _LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
# QUIET BY DEFAULT
_LOG.setLevel(logging.WARNING)


def _pupdate(progress: Any | None, **kwargs: Any) -> None:
    """Best-effort progress.update(**kwargs) if a parent spinner is provided."""
    if progress is None:
        return
    try:
        progress.update(**kwargs)
    except Exception:
        pass


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
    cv = _require_cv2()

    bgr, stem, suf = _as_bgr(src_img)
    basename = f"{stem}{suf}"

    # Parent progress provided → only update context segments
    _pupdate(progress, item=basename, job="object")

    # Acquire OBB model
    mdl: Any
    if model is not None:
        mdl = model
    else:
        loaded = load_obb()  # strict loader; may raise
        if loaded is None:
            raise RuntimeError("OBB loader returned None (failed to load model).")
        mdl = loaded

    # Update model label
    _pupdate(progress, model=_model_label(mdl))

    # Inference
    _pupdate(progress, job="infer")

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
            cv.polylines(bgr, [pts], isClosed=True, color=color, thickness=2)

    # Save
    _pupdate(progress, job="write result")

    out_ext = suf if suf in {".jpg", ".jpeg", ".png"} else ".jpg"
    out_path = (Path(out_dir).expanduser().resolve() / f"{stem}_obb{out_ext}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bgr[:, :, ::-1]).save(out_path)

    _pupdate(progress, job="done")
    return out_path
