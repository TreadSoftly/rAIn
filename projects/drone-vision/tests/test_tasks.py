import base64
import io
import json

import boto3
import numpy as np
from PIL import Image

# Lambda entry-point – import via shim so local env-vars are pre-set
from lambda_function import app as lam

# Moto renamed the decorator in v5 – support both trains
try:
    from moto import mock_s3            # Moto < 5
except ImportError:
    from moto import mock_aws as mock_s3  # Moto ≥ 5


# ---------------------------------------------------------------------------#
# helpers                                                                    #
# ---------------------------------------------------------------------------#
def _evt(img, task: str = "detect") -> dict[str, str]:
    """Serialize *img* into a data-URL so we never hit the network."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    body = {
        "image_url": "data:image/jpeg;base64,"
        + base64.b64encode(buf.getvalue()).decode(),
        "task": task,
    }
    return {"body": json.dumps(body)}


def _stub_boxes(monkeypatch, *boxes) -> None:
    """Force run_inference() to return our fake detections."""
    monkeypatch.setattr(
        lam,
        "run_inference",
        lambda _img, _mdl: np.array(boxes, float),
    )


# ---------------------------------------------------------------------------#
# tests                                                                      #
# ---------------------------------------------------------------------------#
def test_detect_draws_rectangle(monkeypatch):
    img = Image.new("RGB", (640, 640), "white")
    _stub_boxes(monkeypatch, [10, 10, 100, 100, 0.9])
    rsp = lam.handler(_evt(img, "detect"), None)
    assert rsp["statusCode"] == 200 and rsp["isBase64Encoded"]


def test_heatmap_200(monkeypatch):
    img = Image.new("RGB", (640, 640), "white")
    _stub_boxes(monkeypatch, [320, 320, 330, 330, 0.8])  # centre dot
    rsp = lam.handler(_evt(img, "heatmap"), None)
    assert rsp["statusCode"] == 200


def test_geojson_puts_to_s3(monkeypatch):
    with mock_s3():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket="out")

        url = "https://example/lat37.0_lon-122.0.jpg"
        evt = {"body": json.dumps({"image_url": url, "task": "geojson"})}

        rsp = lam.handler(evt, None)
        assert rsp["statusCode"] == 201
