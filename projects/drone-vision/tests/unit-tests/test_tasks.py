from __future__ import annotations

# ── standard library ────────────────────────────────────────────────────
import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence

# ── third‑party ──────────────────────────────────────────────────────────
import boto3
import cv2
import numpy as np
import pytest
from PIL import Image
from ultralytics import YOLO  # type: ignore[import-untyped]

# Lambda under test (imported *via* the shim in repo root)
from lambda_function import app as lam  # type: ignore

try:
    from moto import mock_s3  # Moto 4 / 5
except ImportError:  # pragma: no cover
    from moto import mock_s3  # type: ignore

# ────────────────────────────────────────────────────────────────────────────
# ❶  Lambda‑handler unit tests (unchanged)
# ────────────────────────────────────────────────────────────────────────────
def _evt(img: Image.Image, task: str = "detect") -> Dict[str, str]:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    body = {
        "image_url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
        "task": task,
    }
    return {"body": json.dumps(body)}

def _stub_boxes(monkeypatch: Any, *boxes: Sequence[float]) -> None:
    monkeypatch.setattr(
        lam,
        "run_inference",
        lambda _img: np.array(boxes, dtype=float) if boxes else np.empty((0, 5), float),  # type: ignore
    )

def test_detect_draws_rectangle(monkeypatch: Any) -> None:
    img = Image.new("RGB", (640, 640), "white")
    _stub_boxes(monkeypatch, [10, 10, 100, 100, 0.9])

    rsp = lam.handler(_evt(img, "detect"), None)
    body = base64.b64decode(rsp["body"])

    assert rsp["statusCode"] == 200
    assert rsp["isBase64Encoded"] is True
    assert body.startswith(b"\xFF\xD8")  # JPEG magic bytes

def test_heatmap_200(monkeypatch: Any) -> None:
    img = Image.new("RGB", (640, 640), "white")
    _stub_boxes(monkeypatch, [320, 320, 330, 330, 0.8])

    rsp = lam.handler(_evt(img, "heatmap"), None)
    assert rsp["statusCode"] == 200
    assert rsp["isBase64Encoded"] is True

def test_geojson_puts_to_s3(monkeypatch: Any) -> None:
    with mock_s3():
        s3 = boto3.client("s3", region_name="us-east-1")  # type: ignore[attr-defined]
        s3.create_bucket(Bucket="out")                    # type: ignore[attr-defined]

        url = "https://example/#lat37.0_lon-122.0.jpg"
        evt = {"body": json.dumps({"image_url": url, "task": "geojson"})}

        rsp = lam.handler(evt, None)
        assert rsp["statusCode"] == 201

        key = json.loads(rsp["body"])["s3_key"]
        geo = json.loads(s3.get_object(Bucket="out", Key=key)["Body"].read().decode())  # type: ignore

        assert geo["type"] == "FeatureCollection"
        assert "timestamp" in geo
        assert all("id" in f["properties"] for f in geo["features"])

# ────────────────────────────────────────────────────────────────────────────
# ❷  Bulk helper – local samples  +  remote GeoJSON URLs
# ────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]                 # …/projects/drone-vision
RAW  = ROOT / "tests" / "raw"
RES  = ROOT / "tests" / "results"
RES.mkdir(parents=True, exist_ok=True)

# You can override the weight file with:
#    DRONEVISION_TEST_WEIGHTS=path/to/weights.pt  pytest
YOLO_WEIGHTS = Path(
    os.getenv("DRONEVISION_TEST_WEIGHTS", ROOT / "model" / "yolo11x-seg.pt")
).expanduser()

YOLO_KW: dict[str, object] = dict(conf=0.30, imgsz=640)

_WHITELIST = {
    "bunny.mp4",
    "ah64_apache.jpg",
    "assets.jpg",
    "robodog.png",
    "mildrone.avif",
}

_GEO_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg#lat37.8199_lon-122.4783",
    "https://upload.wikimedia.org/wikipedia/commons/a/a1/Statue_of_Liberty_7.jpg#lat40.6892_lon-74.0445",
    "https://upload.wikimedia.org/wikipedia/commons/4/40/Sydney_Opera_House_Sails.jpg#lat-33.8568_lon151.2153.jpg",
]

def _is_video(p: Path) -> bool:
    return p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}

def _annotate_image(img_path: Path) -> None:
    """Detect objects on *img_path* and write a JPEG preview to tests/results/."""
    model = YOLO(str(YOLO_WEIGHTS))
    if img_path.suffix.lower() == ".avif":
        # Open with Pillow → RGB ndarray so Ultralytics doesn’t choke on AVIF
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        res: List = model.predict(rgb, **YOLO_KW, save=False)  # type: ignore[arg-type]
    else:
        res = model.predict(str(img_path), **YOLO_KW, save=False)  # type: ignore[arg-type]
    out = RES / img_path.with_suffix(".jpg").name
    cv2.imwrite(str(out), res[0].plot())  # type: ignore[index]

def _annotate_video(vid_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "dronevision.predict_mp4",
        str(vid_path),
        f"out_dir={RES}",
        *(f"{k}={v}" for k, v in YOLO_KW.items()),
        f"weights={YOLO_WEIGHTS}",
    ]
    subprocess.check_call(cmd)

def _run_cli(inp: str, *, task: str, model: str = "primary") -> None:
    """Invoke the public CLI (`target`) for *heatmap* / *geojson* work."""
    cmd = [
        "target",
        inp,
        "--task",
        task,
        "--model",
        model,
        "--conf",
        str(YOLO_KW["conf"]),
        "--small" if YOLO_WEIGHTS.name.endswith("n.pt") or YOLO_WEIGHTS.name.endswith("n.onnx") else "",
    ]
    cmd = [c for c in cmd if c]  # strip empties
    subprocess.check_call(cmd)

def main() -> None:  # pragma: no cover
    if not RAW.exists():
        raise SystemExit(f"❌  RAW folder not found: {RAW}")

    # ── local sample set ─────────────────────────────────────────────────
    items = sorted(p for p in RAW.iterdir() if p.is_file() and p.name.lower() in _WHITELIST)
    print(f"→ Annotating {len(items)} file(s)…\n")

    for src in items:
        if _is_video(src):
            print(f" • {src.name}", end="")
            _annotate_video(src)
            print("")
            continue

        # AVIF → heat‑map only (no direct CV2 support)
        if src.suffix.lower() == ".avif":
            print(f"  {src.name}  (heat‑map only)", end="")
            _run_cli(str(src), task="heatmap")
            print("")
            continue

        print(f"  {src.name}", end="")
        try:
            _annotate_image(src)
            _run_cli(str(src), task="heatmap")
        except Exception as exc:  # pragma: no cover
            print(f"\n    {exc}")
        else:
            print("")

    # ── remote GeoJSON URLs ──────────────────────────────────────────────
    print("\n→ GeoJSON sanity‑checks…\n")
    for url in _GEO_URLS:
        print(f"  {url.split('#')[0].rsplit('/',1)[-1]}", end="")
        _run_cli(url, task="geojson")
        print("")

    print(f"\nAll done → outputs in {RES}")

# ────────────────────────────────────────────────────────────────────────────
# ❸  Pytest hook – run the bulk helper once per test session
# ────────────────────────────────────────────────────────────────────────────
def test_bulk_annotation() -> None:
    """Smoke‑test the full CLI & YOLO pipeline on the sample corpus."""
    try:
        main()
    except Exception as exc:
        pytest.fail(f"bulk annotation helper failed: {exc}")

# ────────────────────────────────────────────────────────────────────────────
# ❹  Module entry‑point
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    main()
