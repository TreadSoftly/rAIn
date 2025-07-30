"""
Local‑inference helper shared by the Typer CLI.

… (original doc‑string unchanged) …
"""
from __future__ import annotations

import base64
import io
import os
import re
import urllib.request
import contextlib
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort  # type: ignore
from PIL import Image, ImageDraw

from .geo_sink import to_geojson           # polygon sink (local images)
from .heatmap import heatmap_overlay       # type: ignore[import‑untyped]

# ─────────────────────────────────────────────────────────────
# 1 · Paths & models
# ─────────────────────────────────────────────────────────────
_BASE       = Path(__file__).resolve().parent            # …/dronevision
ROOT        = _BASE.parent                               # …/projects/drone‑vision
RESULTS_DIR = ROOT / "tests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# allow test‑suites / CI to inject a different model location
_MODEL_DIR = Path(os.getenv("DRONEVISION_MODEL_PATH", _BASE / "model"))

MODELS = {
    "drone":    _MODEL_DIR / "drone.onnx",
    "airplane": _MODEL_DIR / "airplane.onnx",
}

_SESSIONS: dict[str, ort.InferenceSession | None] = {}
_IN_NAMES: dict[str, str] = {}
_STRIDES:  dict[str, int] = {}

for key, onnx_path in MODELS.items():
    if onnx_path.exists():
        with contextlib.suppress(Exception):
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            _SESSIONS[key] = sess
            inp            = sess.get_inputs()[0]            # type: ignore[index]
            _IN_NAMES[key] = inp.name                       # type: ignore[attr-defined]
            _STRIDES[key]  = int(inp.shape[2])              # type: ignore[attr-defined]
            continue

    # ── stub‑out when weights are absent ───────────────────────
    _SESSIONS[key] = None
    _IN_NAMES[key] = ""
    _STRIDES[key]  = 640            # sensible default so resize() still works

# ─────────────────────────────────────────────────────────────
# 2 · Helpers
# ─────────────────────────────────────────────────────────────
_DATA_RE    = re.compile(r"^data:image/[^;]+;base64,(.+)", re.I | re.S)
_LATLON_TAG = re.compile(r"#lat-?\d+(?:\.\d+)?_lon-?\d+(?:\.\d+)?", re.I)


def _fetch_image(src: str, timeout: int = 10) -> Image.Image:
    """Return a PIL‑RGB image from http(s), data‑URL or local path."""
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")

    if m := _DATA_RE.match(src):
        raw = base64.b64decode(m.group(1))
        return Image.open(io.BytesIO(raw)).convert("RGB")

    return Image.open(Path(src)).convert("RGB")            # local file


def _normalise_logits(arr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Accept common YOLO‑ONNX layouts and return (N, 6)."""
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[1] >= 6:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 6:
        return arr.T
    raise ValueError(arr.shape)


def _run_inference(img: Image.Image, key: str) -> np.ndarray[Any, Any]:
    """
    Return ndarray[N, 5] := (x1, y1, x2, y2, conf) in pixels.

    If weights are missing we short‑circuit to “no detections”.
    """
    if _SESSIONS.get(key) is None:              # stub mode
        return np.empty((0, 5), float)

    stride = _STRIDES[key]
    x = (
        np.asarray(img.resize((stride, stride)))
        .transpose(2, 0, 1)[None]
        .astype(np.float32)
        / 255.0
    )
    raw    = np.asarray(_SESSIONS[key].run(None, {_IN_NAMES[key]: x})[0])  # type: ignore
    logits = _normalise_logits(raw)
    keep   = logits[:, 4] > 0.40
    if not keep.any():
        return np.empty((0, 5), float)
    boxes  = logits[keep, :5]
    scale  = np.array([img.width, img.height] * 2, float)
    boxes[:, :4] *= scale / stride
    return boxes.astype(float)


def _draw_boxes(img: Image.Image, boxes: np.ndarray[Any, Any]) -> Image.Image:
    d = ImageDraw.Draw(img)
    for x1, y1, x2, y2, _ in boxes:
        d.rectangle((x1, y1, x2, y2), outline="red", width=3)
    return img


def _is_remote(src: str) -> bool:
    return src.startswith(("http://", "https://", "data:"))

# ─────────────────────────────────────────────────────────────
# 3 · Public entry – used by `target`
# ─────────────────────────────────────────────────────────────
def run_single(
    src:   str | os.PathLike[str],
    model: Literal["drone", "airplane"] = "drone",
    task:  Literal["detect", "heatmap", "geojson"] = "detect",
) -> None:
    """
    • Local images → write to tests/results/
    • Remote (http/https/data‑URI) → stream base‑64 JPEG / GeoJSON to stdout.
    """
    import json  # local import for faster module load

    src_str = str(src)
    pil     = _fetch_image(src_str)
    boxes   = _run_inference(pil, model)
    stem    = Path(src_str).stem or "image"

    # ── task dispatch ─────────────────────────────────────────
    if task == "heatmap":
        out_img  = heatmap_overlay(pil)
        out_path = RESULTS_DIR / f"{stem}_heat.jpg"

    elif task == "geojson":
        if _is_remote(src_str) and _LATLON_TAG.search(src_str):
            url_geojson = import_module("lambda.geo_sink").to_geojson  # type: ignore
            geo = url_geojson(src_str, boxes.tolist() if len(boxes) else None)  # type: ignore[arg-type]
        else:
            geo = to_geojson(src_str, boxes)

        if _is_remote(src_str):
            import sys
            json.dump(geo, sys.stdout)
            sys.stdout.write("\n")
            return

        (RESULTS_DIR / f"{stem}.geojson").write_text(json.dumps(geo, indent=2))
        print(f"★ wrote {RESULTS_DIR / f'{stem}.geojson'}")
        return

    else:                                              # detect
        try:
            from ultralytics import YOLO  # type: ignore
            yolo = YOLO(str(ROOT / "yolov8n.pt"))
            res  = yolo.predict(pil, imgsz=640, conf=0.25, verbose=False)[0]  # type: ignore
            bgr  = res.plot()                                                 # type: ignore[no-untyped-call]
            out_img = Image.fromarray(bgr[:, :, ::-1].astype(np.uint8))
        except Exception:
            out_img = _draw_boxes(pil.copy(), boxes)
        out_path = RESULTS_DIR / f"{stem}_boxes.jpg"

    # ── write / stream ───────────────────────────────────────
    if _is_remote(src_str):
        import sys
        buf = io.BytesIO()
        out_img.save(buf, "JPEG")
        sys.stdout.write(base64.b64encode(buf.getvalue()).decode() + "\n")
    else:
        out_img.save(out_path, "JPEG")
        print(f"★ wrote {out_path}")
