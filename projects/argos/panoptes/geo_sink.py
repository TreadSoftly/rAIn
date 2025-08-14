# projects/argos/panoptes/geo_sink.py
"""
panoptes.geo_sink — pixel-space ↔ geo-space GeoJSON helper.

Progress policy (single-line UX)
────────────────────────────────
* QUIET by default: no nested spinners so the CLI’s top-level spinner
  stays pinned on one line.
* Opt-in nested spinner for debugging with PANOPTES_NESTED_PROGRESS=1.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import re
import urllib.parse
import uuid
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Self,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

from PIL import Image

__all__ = ["to_geojson"]

_LOG = logging.getLogger("panoptes.geo_sink")
if not _LOG.handlers:
    import sys as _sys
    h = logging.StreamHandler(_sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
# QUIET so we don’t interfere with the one-line progress UX
_LOG.setLevel(logging.WARNING)

# ───────────────────────── optional progress (opt-in) ────────────────────────
_ENABLE_NESTED = os.getenv("PANOPTES_NESTED_PROGRESS", "").strip().lower() in {
    "1", "true", "yes", "on"
}

class SpinnerLike(Protocol):
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None: ...
    def update(self, **kwargs: Any) -> Self: ...

_SpinnerFactory = Callable[..., SpinnerLike]

class _NullSpin:
    def __enter__(self) -> Self:
        return self
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return False
    def update(self, **_: Any) -> Self:
        return self

def _get_spinner() -> Optional[_SpinnerFactory]:
    if not _ENABLE_NESTED:
        return None
    try:
        from panoptes.progress import (  # type: ignore
            percent_spinner as _percent_spinner,
        )
        return cast(_SpinnerFactory, _percent_spinner)
    except Exception:
        return None

@contextmanager
def _maybe_spinner(prefix: str, *, total: int, current: Optional[str] = None) -> Iterator[SpinnerLike]:
    ctor = _get_spinner()
    if ctor is None or total <= 1:
        null: SpinnerLike = cast(SpinnerLike, _NullSpin())
        yield null
        return
    import sys as _sys
    sp: SpinnerLike = ctor(prefix=prefix, stream=_sys.stderr)  # stderr → pinned, not stdout
    with sp:
        sp.update(total=total)
        if current is not None:
            sp.update(current=current)
        yield sp

# ───────────────────────── helpers ──────────────────────────────────────────
_LATLON_TAG = re.compile(r"#lat-?\d+(?:\.\d+)?_lon-?\d+(?:\.\d+)?", re.I)
_COORD_RE   = re.compile(r"lat(?P<lat>-?\d+(?:\.\d+)?)_lon(?P<lon>-?\d+(?:\.\d+)?)", re.I)

def _image_size(src: str | Path) -> Tuple[int, int]:
    """Return (width, height) — fall back to (640,960) if unreadable."""
    try:
        with Image.open(src) as im:
            return im.width, im.height
    except Exception:  # pragma: no cover
        return 640, 960

def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

def _point_feature(cx: float, cy: float, conf: float | None = None) -> Dict[str, Any]:
    prop: Dict[str, Any] = {"id": uuid.uuid4().hex}
    if conf is not None:
        c = float(conf)
        if c > 1.0:
            c = c / 100.0 if c <= 100 else c / 255.0
        prop["conf"] = round(max(0.0, min(1.0, c)), 4)
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [cx, cy]},
        "properties": prop,
    }

def _centre_feature(w: int, h: int) -> Dict[str, Any]:
    return _point_feature(w / 2.0, h / 2.0, None)

T = TypeVar("T")

def _iter_with_progress(items: Sequence[T], *, prefix: str, labeler: Optional[Callable[[T], str]] = None) -> Iterable[tuple[int, T]]:
    def _to_str(x: T) -> str: return str(x)
    lbl: Callable[[T], str] = labeler or _to_str
    with _maybe_spinner(prefix, total=len(items)) as sp:
        for i, it in enumerate(items, 1):
            try:
                sp.update(count=i - 1, current=lbl(it))
            except Exception:
                pass
            yield i, it
            try:
                sp.update(count=i)
            except Exception:
                pass

# ───────────────────────── public API ───────────────────────────────────────
def to_geojson(
    src: str | Path, boxes: Sequence[Sequence[float]] | None = None
) -> Dict[str, Any]:
    """
    Build a **GeoJSON FeatureCollection** for *src* + *boxes*.

    • If *src* contains “#lat…_lon…” → prefer lambda.geo_sink for true
      lat/lon features (fallback to local parser if the module is absent).
    • Otherwise → pixel-space Point features centred on each bounding box.
    """
    src_str = str(src)

    # — geo-aware branch ────────────────────────────────────────────────────
    if _LATLON_TAG.search(src_str):
        try:
            lam = import_module("lambda.geo_sink")
            geo = lam.to_geojson(src_str, boxes)
            geo.setdefault("timestamp", _now_iso())
            geo.setdefault("source", src_str)
            for f in geo.get("features", []):
                f.setdefault("properties", {})["id"] = f["properties"].get("id", uuid.uuid4().hex)
            return geo
        except Exception:
            # Fallback: parse lat/lon directly from the URL fragment and synthesize a result
            m = _COORD_RE.search(urllib.parse.unquote(src_str))
            if m:
                lat0 = float(m["lat"])
                lon0 = float(m["lon"])
                feats: list[Dict[str, Any]] = []
                if boxes:
                    for _, (x1, y1, x2, y2, *rest) in _iter_with_progress(
                        list(boxes), prefix="ARGOS GEOJSON", labeler=lambda _b: "latlon"
                    ):
                        conf = rest[0] if rest else None
                        if conf is not None:
                            try:
                                c = float(conf)
                                if c > 1.0:
                                    c = c / 100.0 if c <= 100 else c / 255.0
                                conf = round(max(0.0, min(1.0, c)), 4)
                            except Exception:
                                conf = None
                        # interpret 640x640 frame centre as (0,0) and scale to tiny geo offsets
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        relx = (cx - 320.0) / 320.0
                        rely = (cy - 320.0) / 320.0
                        props: Dict[str, Any] = {"id": uuid.uuid4().hex}
                        if conf is not None:
                            props["conf"] = conf
                        feats.append(
                            {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Point",
                                    "coordinates": [lon0 + relx * 0.005, lat0 - rely * 0.005],
                                },
                                "properties": props,
                            }
                        )
                else:
                    feats.append(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [lon0, lat0]},
                            "properties": {"id": uuid.uuid4().hex},
                        }
                    )
                return {
                    "type": "FeatureCollection",
                    "source": src_str,
                    "timestamp": _now_iso(),
                    "features": feats,
                }

    # — pixel-space branch ─────────────────────────────────────────────────
    feats: list[Dict[str, Any]] = []
    if boxes:
        for _, box in _iter_with_progress(list(boxes), prefix="ARGOS GEOJSON", labeler=lambda _b: "pixel"):
            x1, y1, x2, y2, *rest = box
            conf = rest[0] if rest else None
            feats.append(_point_feature((x1 + x2) / 2.0, (y1 + y2) / 2.0, conf))
    else:
        w, h = _image_size(src_str)
        feats.append(_centre_feature(w, h))

    return {
        "type": "FeatureCollection",
        "source": src_str,
        "timestamp": _now_iso(),
        "features": feats,
    }
