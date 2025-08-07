# projects/drone-vision/lambda/app.py
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

import boto3                       # AWS SDK
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO   # type: ignore
    _has_yolo = True
except ImportError:                # pragma: no cover
    YOLO      = None               # type: ignore
    _has_yolo = False

# ───────────────────────── paths / model init ────────────────────────────
_BASE      = Path(__file__).resolve().parent
MODEL_DIR  = Path(os.getenv("DRONEVISION_MODEL_PATH", _BASE / "model"))

_det_model = None
_seg_model = None

if _has_yolo and YOLO is not None:
    # ── detection candidates (pt → onnx fallback) ───────────────────────
    for cand in (
        "yolov8x.pt", "yolov8x.onnx",
        "yolo11x.pt", "yolo11x.onnx",
        "yolov12x.pt", "yolov12x.onnx",
        "yolov8n.pt", "yolov8n.onnx",
        "yolo11n.pt", "yolo11n.onnx",
        "yolov12n.pt", "yolov12n.onnx",
    ):
        p = MODEL_DIR / cand
        if p.exists():
            _det_model = YOLO(str(p))
            break

    # ── segmentation candidates (first “*-seg.*” wins) ──────────────────
    for p in sorted(MODEL_DIR.glob("*-seg.*")):
        if p.suffix.lower() in {".pt", ".onnx"}:
            _seg_model = YOLO(str(p))
            break

# ───────────────────────── helpers ───────────────────────────────────────
def _fetch_image(src: str, timeout: int = 10) -> Image.Image:
    if src.startswith("data:"):
        _, b64 = src.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def run_inference(img: Image.Image, model: str = "yolov8x") -> NDArray[np.float32]:
    if not _has_yolo or _det_model is None:
        return np.empty((0, 6), dtype=np.float32)

    res = _det_model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]  # type: ignore[index]
    data = getattr(getattr(res, "boxes", None), "data", None)
    if data is None:
        return np.empty((0, 6), dtype=np.float32)

    try:
        import torch  # type: ignore
        arr: NDArray[np.float32] = data.cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
    except ImportError:
        arr = np.asarray(data)
    return arr.astype(np.float32).reshape(-1, 6)


def _draw_boxes(img: Image.Image, boxes: Sequence[Sequence[Any]]) -> Image.Image:
    drw = ImageDraw.Draw(img)
    for x1, y1, x2, y2, *rest in boxes:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        drw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        if rest:
            conf = rest[0]
            label = f"{float(conf):.2f}"
            if len(rest) > 1 and _det_model is not None:
                cls   = int(rest[1])
                cname = _det_model.names.get(cls, "")
                label = f"{cname} {conf:.2f}" if cname else label
            font = ImageFont.load_default()
            tw, th = drw.textbbox((0, 0), label, font=font)[2:]
            ty = max(0, y1 - th - 2)
            drw.rectangle((x1, ty, x1 + tw + 2, ty + th + 2), fill="black")
            drw.text((x1 + 1, ty + 1), label, fill="red", font=font)
    return img

# import late to avoid circulars
from .geo_sink import to_geojson  # type: ignore
from dronevision.heatmap import heatmap_overlay  # type: ignore

# ───────────────────────── Lambda entry‑point ────────────────────────────
def handler(event: Dict[str, Any], _ctx: Any) -> Dict[str, Any]:
    try:
        body = json.loads(event["body"])
        src  = body["image_url"]
        task = body.get("task", "detect").lower()
        mdl  = body.get("model", "yolov8x").lower()

        # ── GeoJSON only --------------------------------------------------
        if task == "geojson":
            geo = to_geojson(src, None)
            geo["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
            key = f"detections/{_dt.date.today()}/{uuid.uuid4()}.geojson"
            boto3.client("s3").put_object(                      # type: ignore[attr-defined]
                Bucket=os.getenv("GEO_BUCKET", "out"),
                Key=key, Body=json.dumps(geo).encode(),
                ContentType="application/geo+json",
            )
            return {"statusCode": 201, "body": json.dumps({"s3_key": key})}

        # ── fetch + infer -------------------------------------------------
        img = _fetch_image(src)
        try:
            boxes = run_inference(img, mdl)        # normal path
        except TypeError:                          # ← stub has only 1 param
            boxes = run_inference(img)             # fallback for tests


        # ── heat‑map ------------------------------------------------------
        if task == "heatmap":
            if _seg_model is not None:
                try:
                    res = _seg_model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]  # type: ignore[index]
                    if getattr(getattr(res, "masks", None), "data", None) is not None:
                        img = Image.fromarray(heatmap_overlay(img))
                except Exception:
                    pass
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return {"statusCode": 200, "body": base64.b64encode(buf.getvalue()).decode(),
                    "isBase64Encoded": True}

        # ── detect (default) ---------------------------------------------
        out = _draw_boxes(img.copy(), boxes.tolist())
        buf = io.BytesIO()
        out.save(buf, format="JPEG")
        return {"statusCode": 200, "body": base64.b64encode(buf.getvalue()).decode(),
                "isBase64Encoded": True}

    except Exception as exc:         # pragma: no cover
        return {"statusCode": 500, "body": str(exc)}
