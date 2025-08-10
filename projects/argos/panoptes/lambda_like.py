"""
Lock-down (2025-08-07)
────────────────────────────────────────────────────────────────────
* Uses the hard-coded weights declared in *panoptes.model_registry*
  only – **no** environment-variable or directory scanning.
* One detector + one segmenter are initialised at import-time; if either
  weight is missing the module raises *RuntimeError* immediately.
* A tiny re-init helper lets the CLI steer --small per command without
  breaking the hard-locking rules.
"""

from __future__ import annotations

# ── stdlib ───────────────────────────────────────────────────────────────
import base64
import io
import json
import logging
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal, Optional, Union

# ── third-party ──────────────────────────────────────────────────────────
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

# ── internal project imports ─────────────────────────────────────────────
from .geo_sink import to_geojson
from .heatmap import heatmap_overlay
from .model_registry import load_detector, load_segmenter  # strict, hard-coded

# ─────────────────────────── logging ─────────────────────────────────────
_LOG = logging.getLogger("panoptes.lambda_like")
if not _LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
_LOG.setLevel(logging.INFO)

def _say(msg: str) -> None:
    _LOG.info(f"[panoptes] {msg}")

# ─────────────────────────── paths & helpers ─────────────────────────────
_BASE = Path(__file__).resolve().parent          # …/panoptes
ROOT  = _BASE.parent                             # …/projects/argos

# Whether Ultralytics is actually available in this environment
try:
    from ultralytics import YOLO  # type: ignore
    _has_yolo = True
except ImportError:                # pragma: no cover
    YOLO      = None               # type: ignore
    _has_yolo = False

# ─────────────────────────── model initialisation ────────────────────────
# Default boot: fast seg (small=True), normal det
_det_model: Any = load_detector(small=False)
_seg_model: Any = load_segmenter(small=True)

def reinit_models(
    *,
    detect_small: Optional[bool] = None,
    segment_small: Optional[bool] = None,
    det_override: Optional[Union[str, Path]] = None,
    seg_override: Optional[Union[str, Path]] = None,
) -> tuple[Any, Any]:
    """
    Re-initialize one or both models under hard-locked registry rules.

    Used by the CLI to make `--small` apply to still-image flows too.
    Returns (det_model, seg_model).
    """
    global _det_model, _seg_model
    if detect_small is not None or det_override is not None:
        _det_model = load_detector(small=bool(detect_small), override=det_override)
        _say(f"reinit detector small={bool(detect_small)}")
    if segment_small is not None or seg_override is not None:
        _seg_model = load_segmenter(small=bool(segment_small), override=seg_override)
        _say(f"reinit segmenter small={bool(segment_small)}")
    return _det_model, _seg_model

# ─────────────────────────── inference helpers ──────────────────────────
def run_inference(
    img: Image.Image,
    model: str = "primary",        # kept for CLI parity only
    *,
    conf_thr: float = 0.40,
) -> NDArray[np.float32]:
    """
    Run object detection on *img* using the global `_det_model`.

    Returns
    -------
    ndarray float32  [N,6] -> [x1, y1, x2, y2, conf, class_idx]
    """
    if not _has_yolo:
        raise RuntimeError("Ultralytics YOLO is not installed in this environment.")
    res_list = _det_model.predict(img, imgsz=640, conf=conf_thr, verbose=False)  # type: ignore
    if not res_list:
        return np.empty((0, 6), dtype=np.float32)

    res: Any = res_list[0]  # type: ignore[assignment]
    boxes = getattr(res, "boxes", None)  # type: ignore
    if boxes is None or not hasattr(boxes, "data"):
        return np.empty((0, 6), dtype=np.float32)

    boxes_data = res.boxes.data  # type: ignore
    try:
        import torch             # type: ignore
        if isinstance(boxes_data, torch.Tensor):
            boxes_arr = boxes_data.cpu().numpy().astype(np.float32)  # type: ignore
        else:
            boxes_arr = np.asarray(boxes_data, dtype=np.float32)
    except ImportError:           # pragma: no cover – torch absent
        boxes_arr = np.asarray(boxes_data, dtype=np.float32)

    return boxes_arr.reshape(-1, 6) if boxes_arr.size else np.empty((0, 6), dtype=np.float32)


def _draw_boxes(  # type: ignore[too-many-locals]
    img: Image.Image,
    boxes: np.ndarray[Any, Any],
    label: Optional[str] = None,
) -> Image.Image:
    """Draw bounding boxes (red) with optional labels on *img*."""
    boxes_arr = np.asarray(boxes, dtype=float).reshape(-1, boxes.shape[-1] if boxes.size else 5)
    draw      = ImageDraw.Draw(img)

    for box in boxes_arr:
        if box.shape[0] < 4:
            continue

        x1, y1, x2, y2   = map(int, box[:4])
        conf             = float(box[4]) if box.shape[0] > 4 else None

        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)

        text = ""
        if conf is not None:
            text = f"{conf:.2f}"

        if box.shape[0] >= 6:
            class_id = int(box[5])
            names = getattr(_det_model, "names", None)
            if isinstance(names, dict):
                names_dict: dict[int, str] = names  # type: ignore
                class_nm = names_dict.get(class_id, str(class_id))
            else:
                class_nm = str(class_id)
            if class_nm:
                text = f"{class_nm} {conf:.2f}" if conf is not None else class_nm
        elif label:
            text = f"{label} {text}" if text else label

        if text:
            font      = ImageFont.load_default()
            text_bbox = draw.textbbox((0, 0), text, font=font)
            tx, ty    = float(x1), max(0.0, float(y1) - (text_bbox[3] - text_bbox[1]) - 2.0)
            draw.rectangle(
                (int(round(tx)), int(round(ty)),
                 int(round(tx + (text_bbox[2] - text_bbox[0]) + 2)),
                 int(round(ty + (text_bbox[3] - text_bbox[1]) + 2)) ),
                fill="black",
            )
            draw.text((int(tx + 1), int(ty + 1)), text, fill="red", font=font)

    return img


def _is_remote(src: str) -> bool:
    """True if *src* is an HTTP URL or a data: URI."""
    return src.startswith(("http://", "https://", "data:"))


# ─────────────────────────── public entry-point ─────────────────────────
def run_single(  # noqa: C901 – core worker
    src: Union[str, os.PathLike[str]],
    *,
    model: str = "primary",              # accepted but unused (CLI legacy)
    task: Literal["detect", "heatmap", "geojson"] = "detect",
    **hm_kwargs: Any,
) -> None:
    """
    Process a single image/URL/BASE64 input for *task* and either
    write a file under tests/results/ or stream to stdout (for URLs).
    """
    src_str  = str(src)
    conf_thr = float(hm_kwargs.get("conf", 0.40))
    stem     = Path(src_str).stem or "image"

    # ---- load image from disk / URL / base64 --------------------------------
    if src_str.startswith("data:"):
        _, b64data = src_str.split(",", 1)
        pil_img    = Image.open(io.BytesIO(base64.b64decode(b64data))).convert("RGB")
    elif src_str.lower().startswith(("http://", "https://")):
        req = urllib.request.Request(src_str, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            pil_img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    else:
        pil_img = Image.open(src_str).convert("RGB")

    # ---- inference ----------------------------------------------------------
    boxes = run_inference(pil_img, model=model, conf_thr=conf_thr)

    # ── GEOJSON mode ─────────────────────────────────────────────────────────
    if task == "geojson":
        try:
            if (
                src_str.lower().startswith(("http://", "https://"))
                and re.search(r"lat-?\d+(?:\.\d+)?_lon-?\d+(?:\.\d+)?", urllib.parse.unquote(src_str))
            ):
                geo = __import__("lambda.geo_sink", fromlist=["to_geojson"]).to_geojson(
                    src_str,
                    [list(b)[:5] for b in boxes.tolist()] if boxes.size else None,
                )
            else:
                geo = to_geojson(
                    src_str,
                    [list(b)[:5] for b in boxes.tolist()] if boxes.size else None,
                )
        except Exception:
            geo = to_geojson("", [list(b)[:5] for b in boxes.tolist()] if boxes.size else None)

        import datetime as _dt
        geo["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

        if _is_remote(src_str):
            json.dump(geo, sys.stdout, separators=(",", ":"))
            sys.stdout.write("\n")
        else:
            out_path = ROOT / "tests" / "results" / f"{stem}.geojson"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(geo, indent=2))
            _say(f"wrote {out_path}")
        return

    # ── HEAT-MAP mode ────────────────────────────────────────────────────────
    if task == "heatmap":
        alpha_val = float(hm_kwargs.get("alpha", 0.4))
        overlay_result = heatmap_overlay(
            pil_img,
            boxes=boxes if boxes.size else None,
            alpha=alpha_val,
        )

        if isinstance(overlay_result, np.ndarray):
            out_img = Image.fromarray(
                overlay_result.astype(np.uint8) if overlay_result.ndim == 2
                else overlay_result[:, :, ::-1].astype(np.uint8)
            )
        else:
            out_img = overlay_result

        out_ext  = Path(src_str).suffix.lower() if Path(src_str).suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
        out_path = ROOT / "tests" / "results" / f"{stem}_heat{out_ext}"

    # ── DETECT mode (default) ────────────────────────────────────────────────
    else:
        res     = _det_model.predict(pil_img, imgsz=640, conf=conf_thr, verbose=False)[0]  # type: ignore
        plotted: np.ndarray = res.plot()                                                   # type: ignore
        arr = np.asarray(plotted, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]
        out_img = Image.fromarray(arr)

        out_ext  = Path(src_str).suffix.lower() if Path(src_str).suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
        out_path = ROOT / "tests" / "results" / f"{stem}_boxes{out_ext}"

    # ---- write / stream result ---------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    _say(f"wrote {out_path}")
