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
We initialise one detector at import-time with a graceful fallback:

  detector:  small → full-size → Dummy (returns 0 boxes)

This keeps the module importable even if some weights are missing, so
unit tests can monkey-patch inference without crashing process startup.
Only files listed in the registry are ever considered.

Progress instrumentation
────────────────────────────────────────────────────────────────────
* Uses `panoptes.progress` if available:
  - `live_percent` for coarse request steps (no-ops off-TTY / in Lambda)
  - `simple_status` for short one-off phases (model init, fetch, S3 put)
* Never throws; silently no-ops if progress layer isn’t present.
"""

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

# ── stdlib ─────────────────────────────────────────────────────────────
import base64
import contextlib
import datetime as _dt
import io
import json
import logging
import os
import urllib.request
import uuid
from types import TracebackType
from typing import Any, Dict, Sequence

# ── third-party ────────────────────────────────────────────────────────
import boto3  # type: ignore
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

# Local Lambda heatmap helper (segmentation overlay; returns **BGR** ndarray)
from .heatmap import heatmap_overlay  # type: ignore

# ── internal project imports ──────────────────────────────────────────
from panoptes.model_registry import (  # type: ignore
    load_detector,  # single source-of-truth
)

# If present, GeoJSON sink (kept as in original)
from .geo_sink import to_geojson  # type: ignore

# Optional progress layer
try:
    from panoptes.progress import ProgressEngine  # type: ignore
    from panoptes.progress.bridges import live_percent  # type: ignore
    from panoptes.progress.progress_ux import simple_status  # type: ignore
except Exception:  # pragma: no cover
    ProgressEngine = None  # type: ignore
    live_percent = None  # type: ignore
    simple_status = None  # type: ignore

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


# tiny no-op ctx for when progress is unavailable
class _Null:
    def __enter__(self) -> None:  # noqa: D401
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False


def _lp_ctx(prefix: str):
    if live_percent is not None and ProgressEngine is not None:  # type: ignore
        return live_percent(ProgressEngine(), prefix=prefix)  # type: ignore
    return _Null()


def _status_ctx(label: str):
    if simple_status is not None:
        return simple_status(label)  # type: ignore
    return contextlib.nullcontext()


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
    with _status_ctx("init detector (small)"):
        try:
            m = load_detector(small=True)
            _say("lambda init: detector small=True")
            return m
        except Exception as e_small:
            _say(f"lambda init: detector small=True unavailable ({e_small!s}); trying full-size")
    with _status_ctx("init detector (full)"):
        try:
            m = load_detector(small=False)
            _say("lambda init: detector small=False")
            return m
        except Exception as e_full:
            _say(f"lambda init: detector full-size unavailable ({e_full!s}); using DummyDetector")
            return _DummyDetector()


# ───────────────────────── model initialisation ───────────────────────
_det_model: Any = _init_detector()

# ───────────────────────── helpers ────────────────────────────────────
def _fetch_image(src: str, timeout: int = 10) -> Image.Image:
    """Download data-URI or remote image → RGB Pillow image."""
    with _status_ctx("fetch image"):
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
    with _status_ctx("detect"):
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
    # coarse request-level progress (safe no-op in Lambda)
    lp = _lp_ctx("LAMBDA")
    # try to access engine to set totals if available
    eng = getattr(lp, "engine", None) if hasattr(lp, "engine") else None  # type: ignore[attr-defined]

    with lp:
        try:
            if eng:
                eng.set_total(5.0)
                eng.set_current("parse body")
            body = json.loads(event["body"])
            src = body["image_url"]
            task = body.get("task", "detect").lower()
            _say(f"lambda request: task={task}")

            # ── GeoJSON only ───────────────────────────────────────────────
            if task == "geojson":
                if eng:
                    eng.set_total(3.0)
                    eng.set_current("compose geojson")
                geo = to_geojson(src, None)  # type: ignore[arg-type]
                geo["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

                if eng:
                    eng.add(1.0, current_item="upload s3")
                with _status_ctx("s3 put geojson"):
                    key = f"detections/{_dt.date.today()}/{uuid.uuid4()}.geojson"
                    boto3.client("s3").put_object(  # type: ignore[attr-defined]
                        Bucket=os.getenv("GEO_BUCKET", "out"),
                        Key=key,
                        Body=json.dumps(geo).encode(),
                        ContentType="application/geo+json",
                    )
                if eng:
                    eng.add(1.0, current_item="done")
                return {"statusCode": 201, "body": json.dumps({"s3_key": key})}

            # ── fetch + detect ────────────────────────────────────────────
            if eng:
                eng.set_current("fetch image")
            img = _fetch_image(src)

            if eng:
                eng.add(1.0, current_item="detect")
            boxes = run_inference(img)

            # ── heat-map (segmentation overlay) ───────────────────────────
            if task == "heatmap":
                if eng:
                    eng.set_current("render heatmap")
                # heatmap_overlay returns BGR ndarray; convert → RGB PIL for JPEG
                bgr = heatmap_overlay(img)
                rgb = bgr[:, :, ::-1]  # BGR → RGB
                if eng:
                    eng.add(1.0, current_item="encode")
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG")
                if eng:
                    eng.add(1.0, current_item="done")
                return {
                    "statusCode": 200,
                    "body": base64.b64encode(buf.getvalue()).decode(),
                    "isBase64Encoded": True,
                }

            # ── detect (default) ─────────────────────────────────────────
            if eng:
                eng.set_current("draw boxes")
            out = _draw_boxes(img.copy(), boxes.tolist())
            if eng:
                eng.add(1.0, current_item="encode")
            buf = io.BytesIO()
            out.save(buf, format="JPEG")
            if eng:
                eng.add(1.0, current_item="done")
            return {
                "statusCode": 200,
                "body": base64.b64encode(buf.getvalue()).decode(),
                "isBase64Encoded": True,
            }

        except Exception as exc:  # pragma: no cover
            return {"statusCode": 500, "body": str(exc)}
