"""
Runtime for the Drone-Vision Lambda container.

Improvements in this revision
──────────────────────────────
• Added graceful fallback when lat/lon tags are missing - falls back to
pixel-space GeoJSON instead of raising.
• FeatureCollection now carries a top-level ISO-8601 timestamp.
• Every detection gets a UUID `id` property.
• Heat-map generation tolerates empty detections and keeps alpha-blending.
"""

from __future__ import annotations

import base64
import datetime as _dt
import io
import json
import os
import urllib.request
import uuid
from typing import Any, Dict, Sequence

import boto3
import numpy as np
import onnxruntime as ort  # type: ignore
from lambda.geo_sink import to_geojson # type: ignore[import-untyped]
from heatmap import heatmap_overlay                       # type: ignore
from numpy.typing import NDArray # type: ignore[import]
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------- #
# 1.  Load ONNX graphs once per cold-start                                    #
# --------------------------------------------------------------------------- #
_MODELS = {
    "drone":    "model/drone.onnx",
    "airplane": "model/airplane.onnx",
}

_SESSIONS: Dict[str, ort.InferenceSession] = {}
_IN_NAMES: Dict[str, str] = {}
_STRIDES: Dict[str, int] = {}
for key, rel in _MODELS.items():
    sess = ort.InferenceSession(os.path.join(os.getcwd(), rel),
                                providers=["CPUExecutionProvider"])
    _SESSIONS[key]  = sess
    _IN_NAMES[key]  = sess.get_inputs()[0].name            # type: ignore[index]
    _STRIDES[key]   = int(sess.get_inputs()[0].shape[2])   # type: ignore[index]

# --------------------------------------------------------------------- #
# 2.  Utilities                                                         #
# --------------------------------------------------------------------- #
def _fetch_image(url: str, timeout: int = 10) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def _run_inference(img: Image.Image, model: str) -> np.ndarray[Any, Any]:
    stride: int = int(_STRIDES.get(model, 640))
    resized_img = img.resize((stride, stride))
    x = np.asarray(resized_img, dtype=np.float32).transpose(2, 0, 1)[None] / 255
    logits = _SESSIONS[model].run(None, {_IN_NAMES[model]: x})[0]        # type: ignore[index]
    logits = np.asarray(logits, dtype=np.float32).reshape(-1, 6)
    conf   = logits[:, 4]
    keep   = conf > 0.40
    if not keep.any():
        return np.empty((0, 5), dtype=np.float32)
    boxes = logits[keep, :5]
    scale = np.array([img.width, img.height] * 2, dtype=np.float32)
    boxes[:, :4] *= scale / stride
    return boxes


def _draw_boxes(img: Image.Image, boxes: Sequence[Sequence[Any]]) -> Image.Image:
    d = ImageDraw.Draw(img)
    for x1, y1, x2, y2, _ in boxes:
        d.rectangle((x1, y1, x2, y2), outline="red", width=3)
    return img


def _pack_jpeg_b64(im: Image.Image) -> Dict[str, Any]:
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return {
        "statusCode": 200,
        "isBase64Encoded": True,
        "headers": {"Content-Type": "image/jpeg"},
        "body": base64.b64encode(buf.getvalue()).decode(),
    }

# --------------------------------------------------------------------- #
# 3.  Handler                                                            #
# --------------------------------------------------------------------- #
_s3 = boto3.client("s3")                                               # type: ignore
_GEO_BUCKET = os.getenv("GEO_BUCKET", "out")

def handler(event: Dict[str, Any], _ctx: Any) -> Dict[str, Any]:       # noqa: C901
    try:
        body  = json.loads(event["body"])
        url   = body["image_url"]
        task  = body.get("task", "detect").lower()
        model = body.get("model", "drone").lower()

        img   = _fetch_image(url)
        boxes = _run_inference(img, model)
        boxes_py = boxes.tolist() if hasattr(boxes, "tolist") else (boxes or [])

        # ─ heat-map ───────────────────────────────────────────────
        if task == "heatmap":
            out_bgr = np.asarray(heatmap_overlay(
                np.asarray(img)[:, :, ::-1],
                boxes if boxes.size else None,            # keep ndarray
            ), dtype=np.uint8)                            # type: ignore[arg-type]
            out_rgb = np.asarray(out_bgr[:, :, ::-1], dtype=np.uint8)
            return _pack_jpeg_b64(Image.fromarray(out_rgb))

        # ── geojson ───────────────────────────────────────────────────────
        if task == "geojson":
            geo: dict[str, Any]
            try:
                geo: dict[str, Any] = to_geojson(url, boxes.tolist()) # type: ignore[call-arg]
            except ValueError:
                # fallback to pixel-space GeoJSON
                geo: dict[str, Any] = to_geojson("", boxes.tolist()) # type: ignore[call-arg]
            key = f"detections/{_dt.date.today()}/{uuid.uuid4()}.geojson"
            _s3.put_object(
                Bucket=_GEO_BUCKET,
                Key=key,
                Body=json.dumps(geo).encode(),
                ContentType="application/geo+json",
            )
            return {"statusCode": 201, "body": json.dumps({"s3_key": key})}

        # ─ detect (default) ───────────────────────────────────────
        # Ensure boxes_py is a list of lists for type compatibility
        if isinstance(boxes_py, np.ndarray):
            boxes_py = boxes_py.tolist()
        return _pack_jpeg_b64(_draw_boxes(img, boxes_py))

    except Exception as exc:                                           # pragma: no cover
        return {"statusCode": 500, "body": f'{exc.__class__.__name__}: {exc}'}  # noqa: EM102
