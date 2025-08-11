"""
AWS-Lambda entry-point for the Argos demo
===============================================

Lock-down (2025-08-07)
────────────────────────────────────────────────────────────────────
* **Strict weights** – the only files ever consulted are those
  referenced in `panoptes.model_registry.WEIGHT_PRIORITY`.
* No `panoptes_*` environment variables, no directory walks.
* Prefer small/fast weights in Lambda for cold-starts.
* Public request/response JSON schema remains unchanged.

Pragmatic init (for tests & robustness)
────────────────────────────────────────────────────────────────────
We initialise one detector and one segmenter at import-time with a
graceful fallback chain:

  detector:  small → full-size → Dummy (returns 0 boxes)
  segmenter: small → full-size → None  (handler falls back to boxes)

This keeps the module importable even if some weights are missing,
so unit tests can monkey-patch inference without crashing process
startup. Only files listed in the registry are ever considered.
"""

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

# ── stdlib ─────────────────────────────────────────────────────────────
import base64
import datetime as _dt
import io
import json
import logging
import os
import urllib.request
import uuid
from typing import Any, Dict, Sequence

# ── third-party ────────────────────────────────────────────────────────
import boto3  # type: ignore
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

# Heatmap compositor (segmentation masks or box fallback)
from panoptes.heatmap import heatmap_overlay  # type: ignore

# ── internal project imports ──────────────────────────────────────────
from panoptes.model_registry import (  # type: ignore
    load_detector,  # single source-of-truth
    load_segmenter,
)

# If present, GeoJSON sink (kept as in original)
from .geo_sink import to_geojson  # type: ignore

# ───────────────────────── logging ────────────────────────────────────
_LOG = logging.getLogger("panoptes.lambda.app")
if not _LOG.handlers:
    import sys

    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
_LOG.setLevel(logging.INFO)


def _say(msg: str) -> None:
    _LOG.info(f"[panoptes] {msg}")


# ───────────────────────── init helpers (robust) ──────────────────────
class _DummyDetector:
    """Minimal stand-in so handlers can run even if no detect weight is present."""

    names: Dict[int, str] = {}

    class _Res:
        class _Boxes:
            def __init__(self) -> None:
                self.data = np.empty((0, 6), dtype=np.float32)

        def __init__(self) -> None:
            self.boxes = self._Boxes()

    def predict(self, *_args: Any, **_kw: Any) -> list["_DummyDetector._Res"]:
        return [self._Res()]


def _init_detector() -> Any:
    # Prefer small/fast for Lambda cold starts
    try:
        m = load_detector(small=True)
        _say("lambda init: detector small=True")
        return m
    except Exception as e_small:
        _say(f"lambda init: detector small=True unavailable ({e_small!s}); trying full-size")
        try:
            m = load_detector(small=False)
            _say("lambda init: detector small=False")
            return m
        except Exception as e_full:
            _say(f"lambda init: detector full-size unavailable ({e_full!s}); using DummyDetector")
            return _DummyDetector()


def _init_segmenter() -> Any | None:
    try:
        m = load_segmenter(small=True)
        _say("lambda init: segmenter small=True")
        return m
    except Exception as e_small:
        _say(f"lambda init: segmenter small=True unavailable ({e_small!s}); trying full-size")
        try:
            m = load_segmenter(small=False)
            _say("lambda init: segmenter small=False")
            return m
        except Exception as e_full:
            _say(f"lambda init: segmenter unavailable ({e_full!s}); heatmaps will fall back to boxes")
            return None


# ───────────────────────── model initialisation ───────────────────────
_det_model: Any = _init_detector()
_seg_model: Any | None = _init_segmenter()

# ───────────────────────── helpers ────────────────────────────────────
def _fetch_image(src: str, timeout: int = 10) -> Image.Image:
    """Download data-URI or remote image → RGB Pillow image."""
    if src.startswith("data:"):
        _, b64 = src.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def _run_inference(img: Image.Image) -> NDArray[np.float32]:
    """
    Run object detection – returns ndarray [N,6] (x1,y1,x2,y2,conf,cls).
    *Never* returns None; if a dummy detector is in use this yields (0,6).
    """
    res: Any = _det_model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]  # type: ignore[index]
    data = getattr(getattr(res, "boxes", None), "data", None)  # type: ignore[arg-type]
    if data is None:
        return np.empty((0, 6), dtype=np.float32)

    try:
        import torch  # type: ignore

        arr: NDArray[np.float32] = data.cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
    except ImportError:  # pragma: no cover
        arr = np.asarray(data)

    return arr.astype(np.float32).reshape(-1, 6)


# ✨ PUBLIC alias – unit-tests monkey-patch this symbol directly
run_inference = _run_inference  # noqa: E305


def _draw_boxes(img: Image.Image, boxes: Sequence[Sequence[Any]]) -> Image.Image:
    """Red rectangles + optional class/score labels (fallback when no seg masks)."""
    drw = ImageDraw.Draw(img)
    for x1, y1, x2, y2, *rest in boxes:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        drw.rectangle((x1, y1, x2, y2), outline="red", width=3)

        if not rest:
            continue
        conf = float(rest[0])
        label = f"{conf:.2f}"
        if len(rest) > 1:
            cls_id = int(rest[1])
            names = getattr(_det_model, "names", {})
            cname = names.get(cls_id, "") if hasattr(names, "get") else ""
            label = f"{cname} {conf:.2f}" if cname else label

        font = ImageFont.load_default()
        tw, th = drw.textbbox((0, 0), label, font=font)[2:]
        ty = max(0, y1 - th - 2)
        drw.rectangle((x1, ty, x1 + tw + 2, ty + th + 2), fill="black")
        drw.text((x1 + 1, ty + 1), label, fill="red", font=font)
    return img


# ───────────────────────── Lambda handler ─────────────────────────────
def handler(event: Dict[str, Any], _ctx: Any) -> Dict[str, Any]:
    try:
        body = json.loads(event["body"])
        src = body["image_url"]
        task = body.get("task", "detect").lower()
        _say(f"lambda request: task={task}")

        # ── GeoJSON only ───────────────────────────────────────────────
        if task == "geojson":
            geo = to_geojson(src, None)  # type: ignore[arg-type]
            geo["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

            key = f"detections/{_dt.date.today()}/{uuid.uuid4()}.geojson"
            boto3.client("s3").put_object(  # type: ignore[attr-defined]
                Bucket=os.getenv("GEO_BUCKET", "out"),
                Key=key,
                Body=json.dumps(geo).encode(),
                ContentType="application/geo+json",
            )
            return {"statusCode": 201, "body": json.dumps({"s3_key": key})}

        # ── fetch + detect ────────────────────────────────────────────
        img = _fetch_image(src)
        boxes = run_inference(img)

        # ── heat-map (true masks) ─────────────────────────────────────
        if task == "heatmap":
            try:
                if _seg_model is not None:
                    res = _seg_model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]  # type: ignore[index]
                    if getattr(getattr(res, "masks", None), "data", None) is not None:  # type: ignore[attr-defined]
                        img = Image.fromarray(heatmap_overlay(img))
                    else:
                        img = _draw_boxes(img, boxes.tolist())
                else:
                    img = _draw_boxes(img, boxes.tolist())
            except Exception:  # pragma: no cover
                img = _draw_boxes(img, boxes.tolist())

            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return {
                "statusCode": 200,
                "body": base64.b64encode(buf.getvalue()).decode(),
                "isBase64Encoded": True,
            }

        # ── detect (default) ─────────────────────────────────────────
        out = _draw_boxes(img.copy(), boxes.tolist())
        buf = io.BytesIO()
        out.save(buf, format="JPEG")
        return {
            "statusCode": 200,
            "body": base64.b64encode(buf.getvalue()).decode(),
            "isBase64Encoded": True,
        }

    except Exception as exc:  # pragma: no cover
        return {"statusCode": 500, "body": str(exc)}
