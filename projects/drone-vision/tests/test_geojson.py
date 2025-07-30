import base64
import io
import json
import subprocess
from pathlib import Path

from PIL import Image

BUS = Path("projects/drone-vision/tests/raw/bus.jpg")

def test_local_geojson(tmp_path: Path):
    # run cli
    subprocess.check_call(["target", str(BUS), "--task", "geojson"])
    f = Path("projects/drone-vision/tests/results") / "bus.geojson"
    geo = json.loads(f.read_text())
    assert geo["type"] == "FeatureCollection"

def test_url_geojson():
    # tiny 1×1 white JPEG with lat/lon in data‑URL
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), "white").save(buf, "JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    url = f"data:image/jpeg;base64,{b64}#lat37_lon-122"
    out = subprocess.check_output(["target", url, "--task", "geojson"])
    geo = json.loads(out)
    assert geo["features"][0]["geometry"]["type"] == "Point"
