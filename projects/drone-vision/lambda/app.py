# projects/drone-vision/lambda/app.py
"""
Standalone version of the production Lambda. This mirrors the dronevision package
logic without requiring ONNX weights.
"""
from __future__ import annotations

import base64
import datetime as _dt
import io
import json
import os
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Sequence

import boto3  # AWS SDK for S3 interactions
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

# Import YOLO model class if available
try:
    from ultralytics import YOLO  # type: ignore
    _has_yolo = True
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore
    _has_yolo = False

from dronevision.heatmap import heatmap_overlay  # type: ignore[import]

# Local modules (packaged with Lambda)
from .geo_sink import to_geojson  # type: ignore[import]

# ───────────────────────── Load models once (cold start) ─────────────────────
_BASE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("DRONEVISION_MODEL_PATH", _BASE / "model"))

_det_model = None
_seg_model = None
if _has_yolo and YOLO is not None:
    # Load primary detection model (preferring YOLOv8x, fallback to others)
    for cand in (
        MODEL_DIR / "yolov8x.pt",
        MODEL_DIR / "yolo11x.pt",
        MODEL_DIR / "yolov8n.pt",
        MODEL_DIR / "yolo11n.pt",
    ):
        if cand.exists():
            _det_model = YOLO(str(cand))
            break
    # Load primary segmentation model if available
    for cand in (
        MODEL_DIR / "yolo11x-seg.pt",
        MODEL_DIR / "yolo11m-seg.pt",
        MODEL_DIR / "yolov8x-seg.pt",
        MODEL_DIR / "yolov8s-seg.pt",
        MODEL_DIR / "yolov8n-seg.pt",
    ):
        if cand.exists():
            _seg_model = YOLO(str(cand))
            break


# ───────────────────────── Inference and drawing helpers ────────────────────
def _fetch_image(url: str, timeout: int = 10) -> Image.Image:
    """Fetch an image from URL or data URI and return as PIL Image."""
    if url.startswith("data:"):
        _, b64data = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64data))).convert("RGB")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


def run_inference(img: Image.Image, model: str = "drone") -> NDArray[np.float32]:
    """
    Run object detection on the image using the YOLO model.
    Returns an array of detections [N,6] -> [x1, y1, x2, y2, conf, class_idx].
    """
    if not _has_yolo or _det_model is None:
        return np.empty((0, 6), dtype=np.float32)
    res = _det_model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]  # type: ignore
    boxes_data = getattr(getattr(res, "boxes", None), "data", None)
    if boxes_data is None:
        return np.empty((0, 6), dtype=np.float32)
    try:
        import torch  # type: ignore

        if isinstance(boxes_data, torch.Tensor):
            arr: NDArray[np.float32] = boxes_data.cpu().numpy().astype(np.float32)  # type: ignore
        else:
            arr: NDArray[np.float32] = np.asarray(boxes_data, dtype=np.float32)
    except ImportError:  # pragma: no cover
        arr = np.asarray(boxes_data, dtype=np.float32)
    return arr.reshape(-1, 6) if arr.size else np.empty((0, 6), dtype=np.float32)


def _draw_boxes(img: Image.Image, boxes: Sequence[Sequence[Any]]) -> Image.Image:
    """Draw red bounding boxes with class labels and confidence on the image."""
    draw = ImageDraw.Draw(img)
    for det in boxes:
        x1, y1, x2, y2, *rest = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = rest[0] if rest else None
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        if conf is not None:
            label = f"{float(conf):.2f}"
            if len(rest) > 1 and _det_model is not None:
                class_id = int(rest[1])
                class_name = _det_model.names.get(class_id, "")
                if class_name:
                    label = f"{class_name} {float(conf):.2f}"
            font = ImageFont.load_default()
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            tx = max(0, x1)
            ty = max(0, y1 - text_h - 2)
            draw.rectangle((tx, ty, tx + text_w + 2, ty + text_h + 2), fill="black")
            draw.text((tx + 1, ty + 1), label, fill="red", font=font)
    return img


# ───────────────────────── Lambda handler ─────────────────────────────────
def handler(event: Dict[str, Any], _context: Any) -> Dict[str, Any]:
    """AWS Lambda entry-point compatible with the unit-tests."""
    try:
        body = json.loads(event["body"])
        image_url = body["image_url"]
        task = body.get("task", "detect").lower()
        model = body.get("model", "drone").lower()

        # ── GeoJSON: no need to download or infer ────────────────────────
        if task == "geojson":
            geo = to_geojson(image_url, None)  # no boxes necessary
            geo["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            )

            s3_bucket = os.getenv("GEO_BUCKET", "out")
            key = f"detections/{_dt.date.today()}/{uuid.uuid4()}.geojson"

            # Explicitly type the S3 client for better type checking
            s3_client = boto3.client("s3") # type: ignore
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=key,
                Body=json.dumps(geo).encode(),
                ContentType="application/geo+json",
            )
            return {"statusCode": 201, "body": json.dumps({"s3_key": key})}

        # ── Detect / Heatmap: need the image (and maybe boxes) ───────────
        img = _fetch_image(image_url)

        # Allow tests to stub run_inference(img) (single-arg)
        try:
            boxes = run_inference(img, model)
        except TypeError:
            boxes = run_inference(img)

        if task == "heatmap":
            overlay = heatmap_overlay(img, boxes=boxes if boxes.size else None)
            out_img = (
                overlay
                if isinstance(overlay, Image.Image)
                else Image.fromarray(np.uint8(overlay))
            )
            buf = io.BytesIO()
            out_img.save(buf, format="JPEG")
            return {
                "statusCode": 200,
                "isBase64Encoded": True,
                "headers": {"Content-Type": "image/jpeg"},
                "body": base64.b64encode(buf.getvalue()).decode(),
            }

        # Default: task == "detect"
        boxes_list = boxes.tolist() if hasattr(boxes, "tolist") else list(boxes or [])
        annotated_img = _draw_boxes(img.copy(), boxes_list)
        buf = io.BytesIO()
        annotated_img.save(buf, format="JPEG")
        return {
            "statusCode": 200,
            "isBase64Encoded": True,
            "headers": {"Content-Type": "image/jpeg"},
            "body": base64.b64encode(buf.getvalue()).decode(),
        }

    except Exception as exc:  # pragma: no cover
        return {"statusCode": 500, "body": f"{exc.__class__.__name__}: {exc}"}
