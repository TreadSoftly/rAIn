"""
panoptes.geo_sink - pixel-space ↔ geo-space GeoJSON helper
"""
from __future__ import annotations

import datetime as _dt
import re
import uuid
from importlib import import_module
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple

from PIL import Image

__all__ = ["to_geojson"]

# ───────────────────────── helpers ──────────────────────────────────────────
_LATLON_TAG = re.compile(r"#lat-?\d+(?:\.\d+)?_lon-?\d+(?:\.\d+)?", re.I)


def _image_size(src: str | Path) -> Tuple[int, int]:
    """Return (width, height) - fall back to (640,960) if unreadable."""
    try:
        with Image.open(src) as im:
            return im.width, im.height
    except Exception:  # pragma: no cover
        return 640, 960


def _point_feature(cx: float, cy: float, conf: float | None = None) -> Dict[str, Any]:
    prop: Dict[str, Any] = {"id": uuid.uuid4().hex}
    if conf is not None:
        # clamp / normalise anomalous confidences that escaped 0-1 range
        c = float(conf)
        if c > 1.0:
            c = c / 100.0 if c <= 100 else c / 255.0
        prop["conf"] = round(c, 4)
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [cx, cy]},
        "properties": prop,
    }


def _centre_feature(w: int, h: int) -> Dict[str, Any]:
    return _point_feature(w / 2.0, h / 2.0, None)


# ───────────────────────── public API ───────────────────────────────────────
def to_geojson(
    src: str | Path, boxes: Sequence[Sequence[float]] | None = None
) -> Dict[str, Any]:
    """
    Build a **GeoJSON FeatureCollection** for *src* + *boxes*.

    • If *src* (str) contains `#lat…_lon…` → delegate to legacy lambda.geo_sink
      to obtain **true** latitude / longitude features.

    • Otherwise → pixel-space Point features centred on each bounding box.
    """
    src_str = str(src)

    # — geo-aware branch ────────────────────────────────────────────────────
    if _LATLON_TAG.search(src_str):
        lam = import_module("lambda.geo_sink")
        geo = lam.to_geojson(src_str, boxes)
        # ensure timestamp & IDs are present (legacy helper may lack them)
        geo.setdefault(
            "timestamp", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        )
        for f in geo.get("features", []):
            f.setdefault("properties", {})["id"] = f["properties"].get(
                "id", uuid.uuid4().hex
            )
        return geo

    # — pixel-space branch ──────────────────────────────────────────────────
    feats: list[Dict[str, Any]] = []

    if boxes:
        for box in boxes:
            x1, y1, x2, y2, *rest = box
            conf = rest[0] if rest else None
            feats.append(_point_feature((x1 + x2) / 2.0, (y1 + y2) / 2.0, conf))
    else:
        w, h = _image_size(src_str)
        feats.append(_centre_feature(w, h))

    return {
        "type": "FeatureCollection",
        "source": src_str,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "features": feats,
    }
