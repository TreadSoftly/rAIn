# \rAIn\projects\argos\lambda\geo_sink.py
"""
Helper that converts an *image URL* containing “…lat…_lon…” plus optional
YOLO-style boxes into a GeoJSON FeatureCollection.

* If *boxes* is omitted / empty → a single point at the photo centre.
* Otherwise → one point per box, naïvely displaced from the centre.

This file purposefully stays *very* light-weight - no heavy deps.

Progress: wraps main conversion with a tiny `simple_status` (no-op if absent).
"""
from __future__ import annotations

import re
import urllib.parse
import uuid
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Dict, Sequence, Tuple

# optional progress (safe fallback)
try:
    from panoptes.progress.progress_ux import simple_status  # type: ignore
except Exception:  # pragma: no cover
    simple_status = None  # type: ignore

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


def _norm_conf(c: float | int | str | None) -> float | None:
    if c is None:
        return None
    try:
        v = float(c)
    except Exception:
        return None
    if v > 1.0:
        v = v / 100.0 if v <= 100 else v / 255.0
    return round(max(0.0, min(1.0, v)), 4)


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
    ctx = simple_status("compose geojson") if simple_status is not None else None
    if ctx is None:
        class _Null:
            def __enter__(self) -> None:
                return None

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> bool:
                return False
        ctx = _Null()

    with ctx:
        lat0, lon0 = _latlon(image_url)

        features: list[Dict[str, Any]] = []
        if boxes:  # detections → offset points
            for x1, y1, x2, y2, *rest in boxes:
                conf = _norm_conf(rest[0] if rest else None)
                relx = ((x1 + x2) / 2 - 320) / 320  # centre-normalised
                rely = ((y1 + y2) / 2 - 320) / 320
                feat: Dict[str, Any] = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon0 + relx * 0.005, lat0 - rely * 0.005],
                    },
                    "properties": {
                        "id": uuid.uuid4().hex,
                    },
                }
                if conf is not None:
                    feat["properties"]["conf"] = conf
                features.append(feat)
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
            "source": image_url,
            "features": features,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
