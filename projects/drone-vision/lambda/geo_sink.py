"""
projects/drone-vision/lambda/geo_sink.py

Helper that converts an *image URL* containing “…lat…_lon…” plus optional
YOLO-style boxes into a GeoJSON FeatureCollection.

* If *boxes* is omitted / empty → a single point at the photo centre.
* Otherwise → one point per box, naïvely displaced from the centre.

This file purposefully stays *very* light-weight - no heavy deps.
"""
from __future__ import annotations

import re
import urllib.parse
import uuid
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple

# Accept “lat<N>_lon<M>” anywhere in the (decoded) URL / file-name
_COORD_RE = re.compile(
    r"lat(?P<lat>-?\d+(?:\.\d+)?)_lon(?P<lon>-?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


def _latlon(url: str) -> Tuple[float, float]:
    m = _COORD_RE.search(urllib.parse.unquote(url))
    if m is None:
        raise ValueError("lat/lon not found in URL")
    return float(m["lat"]), float(m["lon"])


def to_geojson(
    image_url: str,
    boxes: Sequence[Sequence[float]] | None = None,
) -> Dict[str, Any]:
    """
    Convert *image_url* (+ optional *boxes*) → minimal GeoJSON.

    Parameters
    ----------
    image_url
        Must embed “…lat<lat>_lon<lon>…”.  Raises ValueError otherwise.
    boxes
        Iterable of ``(x1, y1, x2, y2, conf)`` in *pixel* space.

    Returns
    -------
    dict
        GeoJSON FeatureCollection.
    """
    lat0, lon0 = _latlon(image_url)

    features: list[Dict[str, Any]] = []
    if boxes:  # detections → offset points
        for x1, y1, x2, y2, conf in boxes:
            relx = ((x1 + x2) / 2 - 320) / 320  # centre-normalised
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
    else:  # no detections → just the centre
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
