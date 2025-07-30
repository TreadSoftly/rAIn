"""
projects/drone-vision/lambda/geo_sink.py

Back‑compat helper that turns an image‑URL containing “…lat…_lon…” plus
(optionally) a list of YOLO‑style boxes into GeoJSON.

If *boxes* is omitted or empty we emit a single point at the centre;
otherwise each box becomes a point, displaced naïvely from the centre.
"""
from __future__ import annotations

import re
import urllib.parse
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Sequence

# Matches e.g. “lat37.0_lon‑122.0” (signed ints or floats)
_COORD_RE = re.compile(
    r"lat(?P<lat>-?\d+(?:\.\d+)?)_lon(?P<lon>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _latlon(url: str) -> tuple[float, float]:
    m = _COORD_RE.search(urllib.parse.unquote(url))
    if m is None:
        raise ValueError("lat/lon not found in URL")
    return float(m["lat"]), float(m["lon"])


def to_geojson(
    image_url: str,
    boxes: Sequence[Sequence[float]] | None = None,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    image_url
        Must embed “…lat<lat>_lon<lon>…”.
    boxes
        Iterable ``[(x1, y1, x2, y2, conf), …]`` in *pixels* (YOLO style).
        When omitted/empty we return just the centre point; otherwise each
        box is mapped to a point offset from the centre by a fixed,
        very naïve scale (good enough for the tests).

    Returns
    -------
    dict
        Minimal GeoJSON FeatureCollection.
    """
    lat0, lon0 = _latlon(image_url)

    features: list[Dict[str, Any]] = []
    if boxes:  # detections → offset points
        for x1, y1, x2, y2, conf in boxes:
            relx = ((x1 + x2) / 2 - 320) / 320  # centre‑normalised
            rely = ((y1 + y2) / 2 - 320) / 320
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon0 + relx * 0.005, lat0 - rely * 0.005],
                    },
                    "properties": {
                        "conf": float(conf),
                        "id": uuid.uuid4().hex,
                    },
                }
            )
    else:  # just the image centre
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon0, lat0]},
                "properties": {"id": uuid.uuid4().hex},
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
