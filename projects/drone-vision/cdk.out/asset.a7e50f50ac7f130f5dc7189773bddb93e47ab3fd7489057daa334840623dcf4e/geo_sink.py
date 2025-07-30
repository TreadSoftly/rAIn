"""
GeoJSON helper used inside the Lambda container.

Enhancements
────────────
• Adds UUID `id` to every feature.
• Adds top-level `timestamp`.
• Graceful fallback: if URL lacks lat/lon, outputs pixel-space features.
"""
from __future__ import annotations

import re
import urllib.parse
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

# Regex for “…lat<lat>_lon<lon>…”, signed ints/floats
_COORD_RE = re.compile(r"lat(-?\d+(?:\.\d+)?)_lon(-?\d+(?:\.\d+)?)", re.I)

def _centre_px_feature(w: float, h: float) -> Dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [w / 2.0, h / 2.0]},
        "properties": {"id": uuid.uuid4().hex},
    }


def to_geojson(
    img_url: str,
    boxes: Sequence[Sequence[float]] | None = None,
    img_size: Tuple[int, int] | None = None,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    img_url
        May embed “…lat…_lon…”. If not, pixel-space coords are produced.
    boxes
        Iterable ``[(x1,y1,x2,y2,conf), …]`` in *pixels*.
    img_size
        Optional (w,h) when pixel fallback is used and original dimensions
        are known.  Not required in Lambda – left for future extensions.
    """
    m = _COORD_RE.search(urllib.parse.unquote(img_url))
    features: List[Dict[str, Any]] = []

    if m:  # lat/lon mapped features
        lat0, lon0 = map(float, m.groups())
        if boxes:
            for x1, y1, x2, y2, conf in boxes:
                relx, rely = ((x1 + x2) / 2 - 320) / 320, ((y1 + y2) / 2 - 320) / 320
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
        else:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon0, lat0]},
                    "properties": {"id": uuid.uuid4().hex},
                }
            )
    else:  # pixel-space fallback
        w, h = (img_size or (640, 640))
        if boxes:
            for x1, y1, x2, y2, conf in boxes:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [cx, cy]},
                        "properties": {
                            "conf": float(conf),
                            "id": uuid.uuid4().hex,
                        },
                    }
                )
        else:
            features.append(_centre_px_feature(w, h))

    return {
        "type": "FeatureCollection",
        "features": features,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
