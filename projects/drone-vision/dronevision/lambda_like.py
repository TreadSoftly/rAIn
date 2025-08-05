from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from .geo_sink import to_geojson
from .heatmap import heatmap_overlay  # segmentation overlay for heatmap tasks

# ─────────────────────────── paths & model initialization ──────────────────────
_BASE = Path(__file__).resolve().parent           # …/dronevision
ROOT = _BASE.parent                               # …/projects/drone-vision

try:
    from ultralytics import YOLO  # type: ignore[import]
    _has_yolo = True
except ImportError:
    YOLO = None  # type: ignore
    _has_yolo = False

MODEL_DIR = Path(os.getenv("DRONEVISION_MODEL_PATH", ROOT / "model"))
# Define model references (drone and airplane both use YOLOv8x)
MODELS = {
    "drone": MODEL_DIR / "yolov8x.pt",
    "airplane": MODEL_DIR / "yolov8x.pt",
}

_det_model = None
_seg_model = None

if _has_yolo and YOLO is not None:
    # Load detection model (YOLOv8x or fallback)
    for cand in (MODEL_DIR / "yolov8x.pt", MODEL_DIR / "yolo11x.pt", MODEL_DIR / "yolov8n.pt", MODEL_DIR / "yolo11n.pt"):
        if cand.exists():
            _det_model = YOLO(str(cand))
            break
    # Load segmentation model (YOLOv8-seg or fallback)
    for cand in (MODEL_DIR / "yolo11x-seg.pt", MODEL_DIR / "yolo11m-seg.pt", MODEL_DIR / "yolov8x-seg.pt", MODEL_DIR / "yolov8s-seg.pt", MODEL_DIR / "yolov8n-seg.pt"):
        if cand.exists():
            _seg_model = YOLO(str(cand))
            break

# ─────────────────────────── inference and drawing helpers ─────────────────────
def run_inference(img: Image.Image, model: str = "drone", *, conf_thr: float = 0.40) -> NDArray[np.float32]:
    """Run object detection using the YOLO model (if available).
    Returns ndarray [N,6] -> [x1, y1, x2, y2, conf, class_idx]."""
    if not _has_yolo or _det_model is None:
        return np.empty((0, 6), dtype=np.float32)
    res_list = _det_model.predict(img, imgsz=640, conf=conf_thr, verbose=False)  # type: ignore
    if not res_list:
        return np.empty((0, 6), dtype=np.float32)
    res = res_list[0]
    if not hasattr(res, "boxes") or res.boxes is None or not hasattr(res.boxes, "data"):
        return np.empty((0, 6), dtype=np.float32)
    boxes_data = res.boxes.data  # type: ignore
    try:
        import torch  # type: ignore[import]
        if isinstance(boxes_data, torch.Tensor):
            boxes_arr: NDArray[np.float32] = boxes_data.cpu().numpy().astype(np.float32)  # type: ignore
        else:
            boxes_arr: NDArray[np.float32] = np.asarray(boxes_data, dtype=np.float32)
    except ImportError:
        boxes_arr: NDArray[np.float32] = np.asarray(boxes_data, dtype=np.float32)
    if boxes_arr.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    return boxes_arr.reshape(-1, 6)

def _draw_boxes(img: Image.Image, boxes: np.ndarray[Any, Any], label: Optional[str] = None) -> Image.Image:
    """Draw bounding boxes with class labels and confidence scores on the image."""
    boxes_arr = np.asarray(boxes, dtype=float).reshape(-1, boxes.shape[-1] if boxes.size else 5)
    draw = ImageDraw.Draw(img)
    for box in boxes_arr:
        if box.shape[0] < 4:
            continue
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        conf = float(box[4]) if box.shape[0] > 4 else None
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        # Prepare label text
        text = ""
        if conf is not None:
            text = f"{conf:.2f}"
        if box.shape[0] >= 6:
            class_id = int(box[5])
            if _has_yolo and _det_model is not None:
                class_name = _det_model.names.get(class_id, str(class_id))
            else:
                class_name = str(class_id)
            if class_name:
                text = f"{class_name} {conf:.2f}" if conf is not None else class_name
        elif label:
            text = f"{label} {text}" if text else label
        if text:
            font = ImageFont.load_default()
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            tx = max(0.0, float(x1))
            ty = max(0.0, float(y1) - float(text_h) - 2.0)
            draw.rectangle(
                (
                    int(round(tx)),
                    int(round(ty)),
                    int(round(tx + text_w + 2)),
                    int(round(ty + text_h + 2)),
                ),
                fill="black",
            )
            draw.text((int(tx + 1), int(ty + 1)), text, fill="red", font=font)
    return img

def _is_remote(src: str) -> bool:
    return src.startswith(("http://", "https://", "data:"))

# ────────────────────────── public worker (main entry) ─────────────────────────
def run_single(
    src: Union[str, os.PathLike[str]],
    *,
    model: Literal["drone", "airplane"] = "drone",
    task: Literal["detect", "heatmap", "geojson"] = "detect",
    **hm_kwargs: Any,
) -> None:
    """
    Process a single input for the specified task and output result to disk or stdout.
    - Local file inputs -> writes output file under tests/results/
    - Remote/URL inputs -> writes Base64/JSON to STDOUT.
    """
    src_str = str(src)
    # Load image (file path, URL, or base64 data URI)
    if src_str.startswith("data:"):
        _, b64data = src_str.split(",", 1)
        pil_img = Image.open(io.BytesIO(base64.b64decode(b64data))).convert("RGB")
    elif src_str.lower().startswith(("http://", "https://")):
        req = urllib.request.Request(src_str, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            pil_img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    else:
        pil_img = Image.open(src_str).convert("RGB")
    # Run detection model (if available)
    conf_thr = float(hm_kwargs.get("conf", 0.40))
    boxes = run_inference(pil_img, model=model, conf_thr=conf_thr)
    stem = Path(src_str).stem or "image"

    if task == "geojson":
        try:
            if src_str.lower().startswith(("http://", "https://")) and re.search(r"lat-?\d+(?:\.\d+)?_lon-?\d+(?:\.\d+)?", urllib.parse.unquote(src_str)):
                geo = import_module("lambda.geo_sink").to_geojson(src_str, [list(b)[:5] for b in boxes.tolist()] if boxes.size else None)
            else:
                geo = to_geojson(src_str, [list(b)[:5] for b in boxes.tolist()] if boxes.size else None)
        except Exception:
            geo = to_geojson("", [list(b)[:5] for b in boxes.tolist()] if boxes.size else None)
        # Add timestamp to GeoJSON output
        import datetime as _dt
        geo["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        if _is_remote(src_str):
            json.dump(geo, sys.stdout, separators=(",", ":"))
            sys.stdout.write("\n")
        else:
            out_path = ROOT / "tests" / "results" / f"{stem}.geojson"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(geo, indent=2))
            print(f"★ wrote {out_path}")
        return

    if task == "heatmap":
        out_img: Optional[Image.Image] = None
        try:
            alpha_val = float(hm_kwargs.get("alpha", 0.4))
            overlay_result = heatmap_overlay(pil_img, boxes=boxes if boxes.size else None, alpha=alpha_val)
            if isinstance(overlay_result, np.ndarray):
                if overlay_result.ndim == 3:
                    out_img = Image.fromarray(overlay_result.astype(np.uint8)[:, :, ::-1])
                else:
                    out_img = Image.fromarray(overlay_result.astype(np.uint8))
            else:
                out_img = overlay_result
        except Exception:
            out_img = None
        if out_img is None:
            out_img = _draw_boxes(pil_img.copy(), boxes, label=model)
        suffix = Path(src_str).suffix.lower()
        out_ext = suffix if suffix in {".jpg", ".jpeg", ".png"} else ".jpg"
        out_path = ROOT / "tests" / "results" / f"{stem}_heat{out_ext}"
    else:
        out_img = None
        if _has_yolo and _det_model is not None:
            try:
                res = _det_model.predict(pil_img, imgsz=640, conf=conf_thr, verbose=False)[0]  # type: ignore
                plotted: np.ndarray = res.plot()  # type: ignore
                out_img = Image.fromarray(plotted[:, :, ::-1].astype(np.uint8))
            except Exception:
                out_img = None
        if out_img is None:
            out_img = _draw_boxes(pil_img.copy(), boxes, label=model)
        suffix = Path(src_str).suffix.lower()
        out_ext = suffix if suffix in {".jpg", ".jpeg", ".png"} else ".jpg"
        out_path = ROOT / "tests" / "results" / f"{stem}_boxes{out_ext}"

    if isinstance(out_img, np.ndarray):
        out_img = Image.fromarray(out_img.astype(np.uint8))
    if _is_remote(src_str):
        buf = io.BytesIO()
        out_img.save(buf, format="JPEG")  # type: ignore
        sys.stdout.write(base64.b64encode(buf.getvalue()).decode() + "\n")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path)  # type: ignore
        print(f"★ wrote {out_path}")
