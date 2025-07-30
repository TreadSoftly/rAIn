# projects/drone-vision/tests/unit-tests/test_tasks.py
"""
Lambda handler tests via Moto-mocked AWS services.

• `detect`  → returns JPEG with rectangles (base64, HTTP 200)
• `heatmap` → returns JPEG heat-map (HTTP 200)
• `geojson` → puts a GeoJSON object to S3 and replies HTTP 201
"""
from __future__ import annotations

import base64
import io
import json
from typing import Any
from typing import Dict

import boto3
import numpy as np
from PIL import Image

# Lambda entry-point - import via shim so local env-vars are pre-set
from lambda_function import app as lam  # type: ignore

try:
    from moto import mock_s3  # Moto ≥ 4 / 5
except ImportError:
    from moto import mock_s3  # pragma: no cover  - fallback


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────
def _evt(img: Image.Image, task: str = "detect") -> Dict[str, str]:
    """Return a Lambda-style event with base-64 JPEG in ``body``."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    body = {
        "image_url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
        "task": task,
    }
    return {"body": json.dumps(body)}


def _stub_boxes(monkeypatch: Any, *boxes: list[float]) -> None:
    """Monkey-patch lam.run_inference → returns our synthetic boxes."""
    monkeypatch.setattr(
        lam,
        "run_inference",
        lambda _img: np.array(boxes, dtype=float) if boxes else np.empty((0, 5), float),  # type: ignore
    )


# ────────────────────────────────────────────────────────────────────────────
#  Tests
# ────────────────────────────────────────────────────────────────────────────
def test_detect_draws_rectangle(monkeypatch: Any) -> None:
    img = Image.new("RGB", (640, 640), "white")
    _stub_boxes(monkeypatch, [10, 10, 100, 100, 0.9])

    rsp = lam.handler(_evt(img, "detect"), None)
    body = base64.b64decode(rsp["body"])

    assert rsp["statusCode"] == 200
    assert rsp["isBase64Encoded"] is True
    # sanity - JPEG magic bytes
    assert body.startswith(b"\xFF\xD8")


def test_heatmap_200(monkeypatch: Any) -> None:
    img = Image.new("RGB", (640, 640), "white")
    _stub_boxes(monkeypatch, [320, 320, 330, 330, 0.8])  # centre dot

    rsp = lam.handler(_evt(img, "heatmap"), None)
    assert rsp["statusCode"] == 200
    assert rsp["isBase64Encoded"] is True


def test_geojson_puts_to_s3(monkeypatch: Any) -> None:
    with mock_s3():
        s3 = boto3.client("s3", region_name="us-east-1")  # type: ignore[attr-defined]
        s3.create_bucket(Bucket="out")

        url = "https://example/#lat37.0_lon-122.0.jpg"
        evt = {"body": json.dumps({"image_url": url, "task": "geojson"})}

        rsp = lam.handler(evt, None)
        assert rsp["statusCode"] == 201

        key = json.loads(rsp["body"])["s3_key"]
        geo = json.loads(s3.get_object(Bucket="out", Key=key)["Body"].read())

        # new mandatory fields
        assert geo["type"] == "FeatureCollection"
        assert "timestamp" in geo
        assert all("id" in f["properties"] for f in geo["features"])
