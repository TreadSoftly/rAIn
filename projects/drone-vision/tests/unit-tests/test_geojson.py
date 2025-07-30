# projects/drone-vision/tests/unit-tests/test_geojson.py
"""
Smoke-test the GeoJSON path of the Drone-Vision CLI (“target”) against three
public-domain photos whose URLs embed a ‘#lat…_lon…’ fragment.

For every URL we:
1. call the CLI with ``--task geojson``;
2. load the JSON it prints to *stdout*;
3. write that JSON to tests/results/<name>.geojson for manual inspection;
4. assert that basic structure & coordinate accuracy are correct.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import pytest

# ──────────────────────────────────────────────────────────────
#  Test cases  (url, expected_latitude, expected_longitude)
# ──────────────────────────────────────────────────────────────
CASES = [
    (
        "https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg"
        "#lat37.8199_lon-122.4783",
        37.8199,
        -122.4783,
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/a/a1/Statue_of_Liberty_7.jpg"
        "#lat40.6892_lon-74.0445",
        40.6892,
        -74.0445,
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/4/40/Sydney_Opera_House_Sails.jpg"
        "#lat-33.8568_lon151.2153.jpg",
        -33.8568,
        151.2153,
    ),
]

# folder used by the rest of the suite
RES_DIR = Path("projects/drone-vision/tests/results")
RES_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────
def _basename(url: str) -> str:
    """
    Return the stem of the final path component **without** any #fragment.

    Example
    -------
    >>> _basename('…/GoldenGateBridge-001.jpg#lat37_lon-122')
    'GoldenGateBridge-001'
    """
    tail = urlparse(url).path.split("/")[-1]       # e.g. GoldenGateBridge-001.jpg
    return Path(tail).stem                         # → GoldenGateBridge-001


# ──────────────────────────────────────────────────────────────
#  Parametrised test
# ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "url,exp_lat,exp_lon",
    CASES,
    ids=["golden_gate", "liberty", "sydney"],
)
def test_remote_geojson(url: str, exp_lat: float, exp_lon: float) -> None:
    """Ensure CLI returns valid GeoJSON and write it to tests/results/."""

    # target prints raw GeoJSON to stdout for remote URLs
    raw = subprocess.check_output(["target", url, "--task", "geojson"])
    geo = json.loads(raw)

    # 1️⃣  basic structure + mandatory fields ------------------------------
    assert geo["type"] == "FeatureCollection"
    assert "timestamp" in geo
    assert geo["features"], "no features returned"

    f0 = geo["features"][0]
    assert f0["geometry"]["type"] == "Point"
    assert "id" in f0["properties"]

    # 2️⃣  coordinate sanity (GeoJSON = [lon, lat]) -------------------------
    lon, lat = f0["geometry"]["coordinates"]
    # allow ~0.5° slack for fragment rounding / precision loss
    assert math.isclose(lat, exp_lat, abs_tol=0.5)
    assert math.isclose(lon, exp_lon, abs_tol=0.5)

    # 3️⃣  save to results folder ------------------------------------------
    out_file = RES_DIR / f"{_basename(url)}.geojson"
    out_file.write_text(json.dumps(geo, indent=2))
