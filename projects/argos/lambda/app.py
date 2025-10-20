"""
AWS-Lambda entry-point for the Argos demo
===============================================

Lock-down (2025-08-07)
────────────────────────────────────────────────────────────────────
* **Strict weights** - the only files ever consulted are those
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
import time
from typing import Any, Dict, Sequence, Optional

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
from panoptes.logging_config import bind_context, setup_logging # type: ignore

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
setup_logging()
LOGGER = logging.getLogger(__name__)
TRACE_LAMBDA = os.getenv("PANOPTES_LAMBDA_TRACE", "").strip().lower() in {"1", "true", "yes"}
LOG_DETAIL = os.getenv("PANOPTES_LOG_DETAIL", "").strip().lower() in {"1", "true", "yes"}
ESSENTIAL_LAMBDA_EVENTS = {
    "lambda.model.init.start",
    "lambda.model.init.success",
    "lambda.model.init.dummy",
    "lambda.request.start",
    "lambda.request.complete",
    "lambda.phase.inference.success",
    "lambda.phase.fetch",
    "lambda.phase.parse",
}
BASIC_KEYS = ("task", "status", "ms", "reason", "detections", "model", "small")


class _Null:
    """No-op context manager used when progress hooks are unavailable."""

    def __enter__(self) -> "_Null":
        return self

    def __exit__(self, *_: object) -> bool:
        return False


def _log(event: str, **info: object) -> None:
    if not LOGGER.isEnabledFor(logging.INFO):
        return
    if not TRACE_LAMBDA and event not in ESSENTIAL_LAMBDA_EVENTS:
        return
    if info:
        if TRACE_LAMBDA or LOG_DETAIL:
            detail = " ".join(f"{k}={info[k]}" for k in sorted(info) if info[k] is not None)
        else:
            detail_parts = [f"{k}={info[k]}" for k in BASIC_KEYS if info.get(k) is not None]
            detail = " ".join(detail_parts)
        if detail:
            LOGGER.info("%s %s", event, detail)
        else:
            LOGGER.info(event)
    else:
        LOGGER.info(event)

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
            _log("lambda.model.init.start", small=True)
            m = load_detector(small=True)
            _log("lambda.model.init.success", small=True, model=type(m).__name__)
            return m
        except Exception as e_small:
            _log(
                "lambda.model.init.failure",
                small=True,
                error=type(e_small).__name__,
                message=str(e_small),
            )
    with _status_ctx("init detector (full)"):
        try:
            _log("lambda.model.init.start", small=False)
            m = load_detector(small=False)
            _log("lambda.model.init.success", small=False, model=type(m).__name__)
            return m
        except Exception as e_full:
            _log(
                "lambda.model.init.failure",
                small=False,
                error=type(e_full).__name__,
                message=str(e_full),
            )
    _log("lambda.model.init.dummy", reason="no-weights")
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
    Run object detection - returns ndarray [N,6] (x1,y1,x2,y2,conf,cls).
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


# ✨ PUBLIC alias - unit-tests monkey-patch this symbol directly
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
    request_id: str | None = None
    rc = event.get("requestContext")
    if isinstance(rc, dict):
        request_id = rc.get("requestId") or rc.get("request_id")
    if not request_id:
        headers = event.get("headers") or {}
        if isinstance(headers, dict):
            request_id = (
                headers.get("x-request-id")
                or headers.get("x-amzn-trace-id")
                or headers.get("x-amzn-requestid")
            )
    if not isinstance(request_id, str) or not request_id:
        request_id_str = str(uuid.uuid4())
    else:
        request_id_str = request_id

    with bind_context(lambda_request_id=request_id_str):
        _log("lambda.request.start")
        # coarse request-level progress (safe no-op in Lambda)
        lp = _lp_ctx("LAMBDA")
        eng = getattr(lp, "engine", None) if hasattr(lp, "engine") else None  # type: ignore[attr-defined]

        with lp:
            try:
                if eng:
                    eng.set_total(5.0)
                    eng.set_current("parse body")

                body_raw = event.get("body")
                if body_raw is None:
                    raise ValueError("missing body")
                parse_start = time.perf_counter()
                body = json.loads(body_raw)
                task = body.get("task", "detect").lower()
                src = body["image_url"]
                parse_ms = (time.perf_counter() - parse_start) * 1000.0
                _log("lambda.phase.parse", task=task, source=src, ms=f"{parse_ms:.1f}")

                # ── GeoJSON only ───────────────────────────────────────────
                if task == "geojson":
                    if eng:
                        eng.set_total(3.0)
                        eng.set_current("compose geojson")
                    compose_start = time.perf_counter()
                    geo = to_geojson(src, None)  # type: ignore[arg-type]
                    geo["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
                    _log("lambda.phase.geojson.compose", ms=f"{(time.perf_counter() - compose_start) * 1000.0:.1f}")

                    bucket = os.getenv("GEO_BUCKET", "out")
                    key = f"detections/{_dt.date.today()}/{uuid.uuid4()}.geojson"
                    if eng:
                        eng.add(1.0, current_item="upload s3")
                    put_start = time.perf_counter()
                    with _status_ctx("s3 put geojson"):
                        boto3.client("s3").put_object(  # type: ignore[attr-defined]
                            Bucket=bucket,
                            Key=key,
                            Body=json.dumps(geo).encode(),
                            ContentType="application/geo+json",
                        )
                    _log(
                        "lambda.phase.s3.put",
                        bucket=bucket,
                        key=key,
                        ms=f"{(time.perf_counter() - put_start) * 1000.0:.1f}",
                    )
                    if eng:
                        eng.add(1.0, current_item="done")
                    _log("lambda.request.complete", status=201, task=task)
                    return {"statusCode": 201, "body": json.dumps({"s3_key": key})}

                # ── fetch image ────────────────────────────────────────────
                if eng:
                    eng.set_current("fetch image")
                fetch_start = time.perf_counter()
                img = _fetch_image(src)
                fetch_ms = (time.perf_counter() - fetch_start) * 1000.0
                try:
                    w, h = img.size  # type: ignore[attr-defined]
                except Exception:
                    w = h = None
                _log("lambda.phase.fetch", ms=f"{fetch_ms:.1f}", width=w, height=h)

                # ── inference ──────────────────────────────────────────────
                if eng:
                    eng.add(1.0, current_item="detect")
                infer_start = time.perf_counter()
                boxes = run_inference(img)
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
                det_count: Optional[int]
                try:
                    det_count = int(getattr(boxes, "shape", [0])[0])
                except Exception:
                    try:
                        det_count = len(boxes)  # type: ignore[arg-type]
                    except Exception:
                        det_count = None
                _log("lambda.phase.inference.success", ms=f"{infer_ms:.1f}", detections=det_count, task=task)

                # ── heat-map (segmentation overlay) ───────────────────────
                if task == "heatmap":
                    if eng:
                        eng.set_current("render heatmap")
                    render_start = time.perf_counter()
                    # heatmap_overlay returns BGR ndarray; convert → RGB PIL for JPEG
                    bgr = heatmap_overlay(img)
                    rgb = bgr[:, :, ::-1]  # BGR → RGB
                    render_ms = (time.perf_counter() - render_start) * 1000.0
                    if eng:
                        eng.add(1.0, current_item="encode")
                    encode_start = time.perf_counter()
                    buf = io.BytesIO()
                    Image.fromarray(rgb).save(buf, format="JPEG")
                    payload = buf.getvalue()
                    encode_ms = (time.perf_counter() - encode_start) * 1000.0
                    _log(
                        "lambda.phase.encode",
                        task=task,
                        render_ms=f"{render_ms:.1f}",
                        ms=f"{encode_ms:.1f}",
                        bytes=len(payload),
                    )
                    if eng:
                        eng.add(1.0, current_item="done")
                    _log("lambda.request.complete", status=200, task=task)
                    return {
                        "statusCode": 200,
                        "body": base64.b64encode(payload).decode(),
                        "isBase64Encoded": True,
                    }

                # ── detect (default) ──────────────────────────────────────
                if eng:
                    eng.set_current("draw boxes")
                draw_start = time.perf_counter()
                out = _draw_boxes(img.copy(), boxes.tolist())
                draw_ms = (time.perf_counter() - draw_start) * 1000.0
                if eng:
                    eng.add(1.0, current_item="encode")
                encode_start = time.perf_counter()
                buf = io.BytesIO()
                out.save(buf, format="JPEG")
                payload = buf.getvalue()
                encode_ms = (time.perf_counter() - encode_start) * 1000.0
                _log(
                    "lambda.phase.encode",
                    task=task,
                    render_ms=f"{draw_ms:.1f}",
                    ms=f"{encode_ms:.1f}",
                    bytes=len(payload),
                )
                if eng:
                    eng.add(1.0, current_item="done")
                _log("lambda.request.complete", status=200, task=task)
                return {
                    "statusCode": 200,
                    "body": base64.b64encode(payload).decode(),
                    "isBase64Encoded": True,
                }

            except Exception as exc:  # pragma: no cover
                _log(
                    "lambda.request.error",
                    error=type(exc).__name__,
                    message=str(exc),
                )
                return {"statusCode": 500, "body": str(exc)}
