# projects/drone-vision/lambda/app.py
"""
Standalone version of the production Lambda so the local test-suite
(`tests/test_tasks.py`) can import it.
"""

from __future__ import annotations

import base64
import datetime
import io
import json
import os
import urllib.request
import uuid
from pathlib import Path
from typing import Any # type: ignore[import]

import numpy as np
import numpy.typing as npt
try:
    import cv2                      # OpenCV is optional on CI builders
except ModuleNotFoundError:         # pragma: no cover
    cv2 = None                      # type: ignore

import boto3
import onnxruntime as ort           # type: ignore
from PIL import Image, ImageDraw

# ───── optional mypy-boto3 type stubs ───────────────────────────────────────
try:
    from mypy_boto3_s3 import S3Client # type: ignore  # noqa: WPS433  (runtime-optional import)
except ModuleNotFoundError:             # pragma: no cover – stubs not installed
    S3Client = object  # type: ignore[assignment]

# local helpers
from .heatmap import heatmap_overlay    # type: ignore
from .geo_sink import to_geojson        # type: ignore

# ────────────────────────────────────────────────────────────
# 1 · load the ONNX graphs once per process
# ────────────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent
MODELS = {
    "drone":    _BASE / "model" / "drone.onnx",     # VisDrone  (imgsz 640)
    "airplane": _BASE / "model" / "airplane.onnx",  # HRPlanes (imgsz 960)
}
_SESSIONS: dict[str, ort.InferenceSession] = {}
_IN_NAMES: dict[str, str] = {}
_STRIDES: dict[str, int] = {}

for name, path in MODELS.items():
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    _SESSIONS[name] = sess
    _IN_NAMES[name] = sess.get_inputs()[0].name      # type: ignore[index]
    _STRIDES[name]  = sess.get_inputs()[0].shape[2]  # type: ignore[index]

# ────────────────────────────────────────────────────────────
# 2 · helpers
# ────────────────────────────────────────────────────────────
def _fetch_image(url: str, timeout: int = 10) -> Image.Image:
    """HTTP fetch or base-64 data-URL → ``PIL.Image``."""
    if url.startswith("data:"):
        _, b64 = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


def run_inference(
    img: Image.Image,
    model: str = "drone",
) -> npt.NDArray[np.float_]:
    """Return ``array[N,5]`` → (x1 y1 x2 y2 conf) in *pixel* coordinates."""
    stride = _STRIDES[model]
    x: npt.NDArray[np.float32] = (
        np.asarray(img.resize((stride, stride)))
        .transpose(2, 0, 1)[None]
        .astype(np.float32)
        / 255
    )
    logits = np.asarray(_SESSIONS[model].run( # type: ignore[no-untyped-call]
        None, {_IN_NAMES[model]: x},
    )[0][0], dtype=np.float_)  # type: ignore[index]
    conf = logits[4].astype(np.float_)
    keep = np.array(conf > 0.40, dtype=bool)
    if not keep.any():
        return np.empty((0, 5), dtype=np.float_)

    boxes = np.vstack((logits[:4, keep].T, conf[keep])).T #type: ignore[no-redef]
    scale = np.array([img.width, img.height] * 2, dtype=np.float_)
    out: list[list[float]] = []
    for x1, y1, x2, y2, p in boxes:
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:      # filter degenerate boxes
            continue
        out.append([*(np.array([x1, y1, x2, y2]) * scale / stride), p])
    return np.asarray(out, dtype=np.float_)


def _draw_boxes(
    img: Image.Image,
    boxes: npt.NDArray[np.float_],
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2, _ in boxes:
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
    return img


def _pack_jpeg_base64(pil_img: Image.Image) -> dict[str, object]:
    """Convert an image to the Lambda proxy-integration JSON shape."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return {
        "statusCode": 200,
        "isBase64Encoded": True,
        "headers": {"Content-Type": "image/jpeg"},
        "body": base64.b64encode(buf.getvalue()).decode(),
    }

# --------------------------------------------------------------------------- #
# 3 · Lambda handler                                                          #
# --------------------------------------------------------------------------- #
BUCKET = os.environ.get("OUT_BUCKET", "out")         # moto creates this in tests


def handler(event: dict[str, Any], _ctx: Any) -> dict[str, object]:   # noqa: D401
    """
    Entry-point for AWS Lambda **and** the local unit-tests.

    • **detect**   – rectangle overlay JPEG
    • **heatmap**  – pseudo-thermal JPEG (falls back to raw image if OpenCV
                    isn’t present)
    • **geojson**  – writes a GeoJSON point/collection to S3 **without**
                    downloading the image (network disabled in tests)
    """
    try:
        body  = json.loads(event["body"])
        task  = body.get("task", "detect")
        model = body.get("model", "drone")
        url   = body["image_url"]

        # ── geojson: no image fetch required ──────────────────────────────
        if task == "geojson":
            geo = to_geojson(url)                        # centre-point only
            key = f"detections/{datetime.date.today()}/{uuid.uuid4()}.geojson"
            s3 = boto3.client("s3")  # type: ignore  # noqa: PGH003
            s3.put_object(  # type: ignore[attr-defined]
                Bucket=BUCKET,
                Key=key,
                Body=json.dumps(geo).encode(),
                ContentType="application/geo+json",
            )
            return {"statusCode": 201, "body": json.dumps({"s3_key": key})}

        # ── detect / heatmap need the pixels ──────────────────────────────
        img   = _fetch_image(url)
        boxes = run_inference(img, model)

        if task == "heatmap":
            if cv2 is None:                              # OpenCV not avail.
                return _pack_jpeg_base64(img)
            bgr = heatmap_overlay(                       # expects BGR
                np.asarray(img)[:, :, ::-1],
                boxes.tolist(),
            )
            return _pack_jpeg_base64(Image.fromarray(bgr[:, :, ::-1]))

        # default → detect
        return _pack_jpeg_base64(_draw_boxes(img, boxes))

    except Exception as exc:                             # surface in tests
        return {"statusCode": 500, "body": str(exc)}
