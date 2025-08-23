# projects/argos/panoptes/classify.py
"""
panoptes.classify — image classification overlay

Contract
────────
run_image(src_img, *, out_dir, model=None, conf=None, topk=1, progress=None) -> Path

• Loads a classifier model STRICTLY via panoptes.model_registry when *model* is None.
• Renders a small top‑K label card in the top‑left of the image.
• Writes <stem>_cls.<ext> under *out_dir* (uses source extension if jpg/png, else .jpg).

Progress policy (single‑line UX)
────────────────────────────────
* This module NEVER creates its own spinner and NEVER changes totals/counters.
* If a parent `progress` handle is supplied (the Halo/Rich spinner), we only update:
      item=[File basename], job in {"classify","infer","write result","done"},
      model=<model-label>.
* If `progress` is None, run completely quiet (no prints, no nested spinners).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union, cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .model_registry import load_classifier  # strict registry

if TYPE_CHECKING:
    import numpy as _np
    import numpy.typing as npt

    NDArrayU8 = npt.NDArray[_np.uint8]
else:
    NDArrayU8 = Any  # type: ignore[assignment]

# ─────────────────────────── logging ────────────────────────────
_LOG = logging.getLogger("panoptes.classify")
if not _LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
# QUIET BY DEFAULT so we don’t break the single‑line spinner UX
_LOG.setLevel(logging.WARNING)


# ------------------------------ helpers ---------------------------------------
def _pupdate(progress: Any | None, **kwargs: Any) -> None:
    """Best-effort progress.update(**kwargs) if a parent spinner is provided."""
    if progress is None:
        return
    try:
        progress.update(**kwargs)
    except Exception:
        # Never let progress plumbing break the task
        pass


def _as_pil(img: Union[Image.Image, NDArrayU8, str, Path]) -> Tuple[Image.Image, str, str]:
    """
    Return (PIL.Image(RGB), source_stem, source_suffix_lower).
    If given an ndarray, assume BGR and convert to RGB.
    """
    stem = "image"
    suffix = ".jpg"
    if isinstance(img, Image.Image):
        pil = img.convert("RGB")
    elif isinstance(img, (str, Path)):
        p = Path(img)
        stem = p.stem or stem
        suffix = p.suffix.lower() or suffix
        with Image.open(p) as im:
            pil = im.convert("RGB")
    else:
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[2] == 3:
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            pil = Image.fromarray(arr[:, :, ::-1])  # BGR → RGB
        else:
            raise TypeError(f"Unsupported ndarray shape for image: {arr.shape!r}")
    return pil, stem, suffix


def _topk_from_probs(
    probs_obj: Any,
    *,
    names: dict[int, str] | Sequence[str] | None,
    k: int,
) -> List[Tuple[str, float]]:
    """
    Extract top‑K (label, prob) robustly from a YOLO‑like classification result.

    Accepts:
      - probs.data / probs (array‑like or torch.Tensor)
      - names as dict[int,str] or list[str]
    """
    def _name_for(i: int) -> str:
        if isinstance(names, dict):
            return str(cast(dict[int, str], names).get(i, i))  # type: ignore[arg-type]
        if isinstance(names, (list, tuple)):
            seq = cast(Sequence[str], names)
            return str(seq[i]) if 0 <= i < len(seq) else str(i)
        return str(i)

    # Normalize to a 1‑D float vector
    v = getattr(probs_obj, "data", probs_obj)
    if hasattr(v, "cpu"):
        try:
            v = v.cpu().numpy()  # torch.Tensor → ndarray
        except Exception:
            pass
    vec = np.asarray(v, dtype=np.float32).reshape(-1)

    idx = np.argsort(-vec)[: max(1, int(k))]
    out: List[Tuple[str, float]] = [(_name_for(int(i)), float(vec[int(i)])) for i in idx]
    return out


def _draw_topk_card(pil_img: Image.Image, items: Sequence[Tuple[str, float]]) -> Image.Image:
    """
    Overlay a semi‑transparent card listing top‑K classes with probabilities.
    """
    draw = ImageDraw.Draw(pil_img, "RGBA")
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None  # type: ignore[assignment]

    pad = 8
    gap = 4
    lines = [f"{i+1}. {name}  {prob:.2f}" for i, (name, prob) in enumerate(items)]

    # Measure
    w = 0
    h = 0
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        w = max(w, tw)
        h += th + gap
    h = max(0, h - gap)

    box = (pad, pad, pad + w + 2 * pad, pad + h + 2 * pad)
    draw.rectangle(box, fill=(0, 0, 0, 168))

    # Draw lines
    y = pad + pad
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        th = bbox[3] - bbox[1]
        draw.text((pad + pad, y), ln, fill=(255, 255, 255, 255), font=font)
        y += th + gap
    return pil_img


def _model_label(m: Any) -> str:
    """Best-effort label for the model to show in [Model:]."""
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


# ------------------------------- main API ------------------------------------
def run_image(
    src_img: Union[Image.Image, NDArrayU8, str, Path],
    *,
    out_dir: Path,
    model: Any | None = None,
    conf: float | None = None,   # accepted; most classifiers don't use it
    topk: int = 1,
    progress: Any = None,
) -> Path:
    """
    Draws top‑K class labels onto image and writes to <stem>_cls.<ext>.
    """
    # Normalize inputs and derive a stable basename for the spinner
    pil, stem, suf = _as_pil(src_img)
    basename = f"{stem}{suf}"

    # Parent progress provided → only update context segments
    _pupdate(progress, item=basename, job="classify")

    # Acquire classifier (strictly via registry if None)
    if model is None:
        model = load_classifier()  # may raise

    # Update model label
    _pupdate(progress, model=_model_label(model))

    # Inference
    _pupdate(progress, job="infer")

    res_list: list[Any] = cast(list[Any], model.predict(pil, imgsz=640, conf=(conf or 0.0), verbose=False))  # type: ignore[call-arg]
    res = res_list[0] if res_list else None
    if res is None or getattr(res, "probs", None) is None:
        items: List[Tuple[str, float]] = []
    else:
        names = getattr(model, "names", None) or getattr(res, "names", None)
        items = _topk_from_probs(res.probs, names=names, k=max(1, int(topk)))

    if items:
        pil = _draw_topk_card(pil, items)

    # Save
    _pupdate(progress, job="write result")

    out_ext = suf if suf in {".jpg", ".jpeg", ".png"} else ".jpg"
    out_path = (Path(out_dir).expanduser().resolve() / f"{stem}_cls{out_ext}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)

    _pupdate(progress, job="done")
    return out_path
