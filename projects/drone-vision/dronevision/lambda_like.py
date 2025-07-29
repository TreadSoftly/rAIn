from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from .geo_sink import to_geojson
from .heatmap import heatmap_overlay

# --------------------------------------------------------------------------- #
# 1 · Load ONNX graphs once per process                                       #
# --------------------------------------------------------------------------- #
_BASE = Path(__file__).resolve().parent
MODELS = {
    "drone":    _BASE / "model/drone.onnx",
    "airplane": _BASE / "model/airplane.onnx",
}

_SESSIONS, _IN_NAMES, _STRIDES = {}, {}, {}
for k, p in MODELS.items():
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    _SESSIONS[k] = s
    _IN_NAMES[k] = s.get_inputs()[0].name
    _STRIDES[k]  = s.get_inputs()[0].shape[2]      # square export → width


# --------------------------------------------------------------------------- #
# 2 · Utilities                                                               #
# --------------------------------------------------------------------------- #
def _fetch_image(src: str, timeout=10) -> Image.Image:
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    # local path
    return Image.open(src).convert("RGB")


def _run_inference(img: Image.Image, model_key: str) -> np.ndarray:
    """Return ndarray[N,5]  (x1,y1,x2,y2,conf) in **pixel** coords."""
    stride = _STRIDES[model_key]
    x = (
        np.asarray(img.resize((stride, stride)))
        .transpose(2, 0, 1)[None]
        .astype(np.float32)
        / 255
    )
    logits = _SESSIONS[model_key].run(None, {_IN_NAMES[model_key]: x})[0][0]
    conf   = logits[4]
    keep   = conf > 0.40
    if not keep.any():
        return np.empty((0, 5))

    # coords   (N,4)   | scores (N,1)
    boxes = np.hstack((logits[:4, keep].T, conf[keep][:, None]))
    scale = np.array([img.width, img.height] * 2)

    out = []
    for x1, y1, x2, y2, p in boxes:
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append([*(np.array([x1, y1, x2, y2]) * scale / stride), p])
    return np.asarray(out, float)


def _draw_boxes(img: Image.Image, boxes: np.ndarray) -> Image.Image:
    d = ImageDraw.Draw(img)
    for x1, y1, x2, y2, _ in boxes:
        d.rectangle((x1, y1, x2, y2), outline="red", width=3)
    return img


# --------------------------------------------------------------------------- #
# 3 · Public entry - used by the CLI                                          #
# --------------------------------------------------------------------------- #
def run_single(
    src: str | os.PathLike,
    model: Literal["drone", "airplane"] = "drone",
    task:  Literal["detect", "heatmap", "geojson"] = "detect",
) -> None:
    """
    • For *local images*  → writes a file next to the original.
    • For *URL images*    → prints base64 JPEG (detect/heatmap) or GeoJSON (geojson).
    """
    pil = _fetch_image(str(src))
    boxes = _run_inference(pil, model)

    # ── dispatch ----------------------------------------------------------- #
    if task == "heatmap":
        bgr = heatmap_overlay(np.asarray(pil)[:, :, ::-1], boxes)
        out_img = Image.fromarray(bgr[:, :, ::-1])
    elif task == "geojson":
        geo = to_geojson(str(src), boxes)
        if src.startswith(("http://", "https://")):      # remote → print
            import json
            import sys
            json.dump(geo, sys.stdout)
            sys.stdout.write("\n")
            return
        else:                                            # local → write file
            geo_path = Path(src).with_suffix(".geojson")
            geo_path.write_text(json.dumps(geo, indent=2))
            print(f"★ wrote {geo_path}")
            return
    else:  # detect
        out_img = _draw_boxes(pil, boxes)

    # ── output ------------------------------------------------------------- #
    if src.startswith(("http://", "https://")):
        import base64
        import sys
        buf = io.BytesIO()
        out_img.save(buf, format="JPEG")
        sys.stdout.write(base64.b64encode(buf.getvalue()).decode() + "\n")
    else:
        dst = Path(src).with_name(f"{Path(src).stem}_boxes.jpg")
        out_img.save(dst, "JPEG")
        print(f"★ wrote {dst}")
