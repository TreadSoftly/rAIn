# projects/drone-vision/lambda/app.py
import base64
import io
import json
import os
import urllib.request
import uuid
import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort
import boto3                       # only used when task == "geojson"
import cv2                         # required for heat‑map overlay

from heatmap import heatmap_overlay
from geo_sink import to_geojson

# ---------------------------------------------------------------------------#
# 1.  load both ONNX graphs once per container                               #
# ---------------------------------------------------------------------------#
MODELS = {
    "drone":    "model/drone.onnx",        # VisDrone   (imgsz 640)
    "airplane": "model/airplane.onnx"      # HRPlanes   (imgsz 960)
}

SESSIONS, IN_NAMES, STRIDES = {}, {}, {}

for key, path in MODELS.items():
    session = ort.InferenceSession(
        os.path.join(os.getcwd(), path),
        providers=["CPUExecutionProvider"]
    )
    SESSIONS[key] = session
    IN_NAMES[key] = session.get_inputs()[0].name
    # yolov8 ONNX exports are square; we take the width as stride
    STRIDES[key] = session.get_inputs()[0].shape[2]

# ---------------------------------------------------------------------------#
# 2.  utilities                                                              #
# ---------------------------------------------------------------------------#
def fetch_image(url: str, timeout: int = 10) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")

def run_inference(img: Image.Image, model_key: str) -> np.ndarray:
    """
    Run the specified model and return ndarray[N,5] with
    (x1, y1, x2, y2, confidence) in *pixel* coordinates.
    """
    stride = STRIDES[model_key]
    img_rs = img.resize((stride, stride))
    x = np.asarray(img_rs).transpose(2, 0, 1)[None].astype(np.float32) / 255
    logits = SESSIONS[model_key].run(None, {IN_NAMES[model_key]: x})[0][0]  # (84,8400)
    conf   = logits[4]
    keep   = conf > 0.40
    if not keep.any():
        return np.empty((0, 5))

    boxes = np.vstack((logits[:4, keep].T, conf[keep])).T  # shape (N,5)
    scale = np.array([img.width, img.height] * 2)
    out   = []
    for x1, y1, x2, y2, p in boxes:
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append([*(np.array([x1, y1, x2, y2]) * scale / stride), p])
    return np.array(out, dtype=float)

def draw_boxes(img: Image.Image, boxes: np.ndarray) -> Image.Image:
    d = ImageDraw.Draw(img)
    for x1, y1, x2, y2, _ in boxes:
        d.rectangle((x1, y1, x2, y2), outline="red", width=3)
    return img

def pack_jpeg_base64(pil_img: Image.Image) -> dict:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return {
        "statusCode": 200,
        "isBase64Encoded": True,
        "headers": {"Content-Type": "image/jpeg"},
        "body": base64.b64encode(buf.getvalue()).decode()
    }

# ---------------------------------------------------------------------------#
# 3.  Lambda handler                                                         #
# ---------------------------------------------------------------------------#
s3      = boto3.client("s3")
BUCKET  = os.environ.get("GEO_BUCKET")      # set in your CDK / stack

def handler(event, _context):
    try:
        body = json.loads(event["body"])
        img  = fetch_image(body["image_url"])
        mdl  = body.get("model", "drone")
        task = body.get("task",  "detect")

        boxes = run_inference(img, mdl)

        # ---------- task router ------------------------------------------- #
        if task == "heatmap":
            out = heatmap_overlay(np.asarray(img)[:, :, ::-1], boxes)  # BGR → heatmap
            return pack_jpeg_base64(Image.fromarray(out[:, :, ::-1]))

        if task == "geojson":
            geo = to_geojson(body["image_url"], boxes)
            key = f"detections/{datetime.date.today()}/{uuid.uuid4()}.geojson"
            s3.put_object(
                Bucket=BUCKET,
                Key=key,
                Body=json.dumps(geo).encode(),
                ContentType="application/geo+json"
            )
            return {"statusCode": 201, "body": json.dumps({"s3_key": key})}

        # default: draw rectangles
        out = draw_boxes(img, boxes)
        return pack_jpeg_base64(out)

    except Exception as exc:   # surfacing every error helps debugging
        return {"statusCode": 500, "body": str(exc)}
