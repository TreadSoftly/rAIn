"""
Local-inference helper shared by the Typer CLI.

* Loads ONNX weights (tiny, CPU-only).
* Writes every artefact into **tests/results/** – keeps tests/raw/ clean.
* Absolutely NO torch / cv2 imports → works fine on NumPy ≥ 2.
"""
from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort  # type: ignore
from PIL import Image, ImageDraw

from .geo_sink import to_geojson
from .heatmap import heatmap_overlay  # type: ignore[import-untyped]

# --------------------------------------------------------------------------- #
# 1 · Paths & models                                                          #
# --------------------------------------------------------------------------- #
_BASE = Path(__file__).resolve().parent
ROOT = _BASE.parent
RESULTS_DIR = ROOT / "tests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "drone": _BASE / "model/drone.onnx",
    "airplane": _BASE / "model/airplane.onnx",
}

_SESSIONS: dict[str, ort.InferenceSession] = {}
_IN_NAMES: dict[str, str] = {}
_STRIDES: dict[str, int] = {}

for key, model_path in MODELS.items():
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    _SESSIONS[key] = sess
    inputs: list[Any] = sess.get_inputs()  # type: ignore
    inp = inputs[0]
    _IN_NAMES[key] = inp.name  # type: ignore[attr-defined]
    _STRIDES[key] = int(inp.shape[2])  # type: ignore[attr-defined]  # square → width/height

# --------------------------------------------------------------------------- #
# 2 · Helpers                                                                 #
# --------------------------------------------------------------------------- #
def _fetch_image(src: str, timeout: int = 10) -> Image.Image:
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    return Image.open(src).convert("RGB")


def _run_inference(img: Image.Image, key: str) -> np.ndarray[Any, Any]:
    """Return ndarray[N,5]  (x1,y1,x2,y2,conf) in **pixels**."""
    stride = _STRIDES[key]
    x = (
        np.asarray(img.resize((stride, stride)))
        .transpose(2, 0, 1)[None]
        .astype(np.float32)
        / 255
    )
    output: np.ndarray[Any, Any] = _SESSIONS[key].run(None, {_IN_NAMES[key]: x})[0]  # type: ignore
    logits = output[0]
    conf = logits[4]
    keep = conf > 0.40
    if not keep.any():
        return np.empty((0, 5), dtype=float)

    boxes = np.hstack((logits[:4, keep].T, conf[keep, None])).astype(float)
    scale = np.array([img.width, img.height] * 2)
    pix = np.empty((0, 5), dtype=float)

    outs: list[list[float]] = []
    for x1, y1, x2, y2, p in boxes:
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if (x2 - x1) < 1 or (y2 - y1) < 1:
            continue
        outs.append([*(np.array([x1, y1, x2, y2]) * scale / stride), p])
    if outs:
        pix = np.asarray(outs, dtype=float)
    return pix


def _draw_boxes(img: Image.Image, boxes: np.ndarray[Any, Any]) -> Image.Image:
    d = ImageDraw.Draw(img)
    for x1, y1, x2, y2, _ in boxes:
        d.rectangle((x1, y1, x2, y2), outline="red", width=3)
    return img

# --------------------------------------------------------------------------- #
# 3 · Public entry – used by `target`                                         #
# --------------------------------------------------------------------------- #
def run_single(
    src: str | os.PathLike[str],
    model: Literal["drone", "airplane"] = "drone",
    task: Literal["detect", "heatmap", "geojson"] = "detect",
) -> None:
    """
    • Local images   → writes output file under tests/results/.
    • HTTP/S images  → prints base64 JPEG or GeoJSON to stdout.
    """
    pil = _fetch_image(str(src))
    boxes = _run_inference(pil, model)
    stem = Path(src).stem

    # ── task dispatch ────────────────────────────────────────────────────
    if task == "heatmap":
        bgr_img = np.asarray(pil)[:, :, ::-1]  # RGB → BGR
        bgr_pil = Image.fromarray(bgr_img)
        heat_bgr = np.asarray(heatmap_overlay(bgr_pil, boxes))
        out_img = Image.fromarray(heat_bgr[:, :, ::-1])  # back to RGB
        out_path = RESULTS_DIR / f"{stem}_heat.jpg"

    elif task == "geojson":
        geo = to_geojson(str(src), boxes)
        if str(src).startswith(("http://", "https://")):
            import json  # noqa: 402
            import sys  # noqa: 402
            json.dump(geo, sys.stdout)
            sys.stdout.write("\n")
            return
        out_path = RESULTS_DIR / f"{stem}.geojson"
        import json

        out_path.write_text(json.dumps(geo, indent=2))
        print(f"★ wrote {out_path}")
        return

    else:  # detect
        out_img = _draw_boxes(pil, boxes)
        out_path = RESULTS_DIR / f"{stem}_boxes.jpg"

    # ── output for image branches ───────────────────────────────────────
    if str(src).startswith(("http://", "https://")):
        import base64  # noqa: 402
        import sys  # noqa: 402
        buf = io.BytesIO()
        out_img.save(buf, format="JPEG")
        sys.stdout.write(base64.b64encode(buf.getvalue()).decode() + "\n")
    else:
        out_img.save(out_path, "JPEG")
        print(f"★ wrote {out_path}")
