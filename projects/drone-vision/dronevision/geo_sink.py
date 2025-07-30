"""
geo_sink.py – turn detection boxes into a GeoJSON FeatureCollection.

Why another implementation?
---------------------------
* Image tasks must *not* import heavy stacks such as **torch / opencv** that
  are still NumPy-1-only.  This module therefore depends **only** on NumPy and
  Pillow.
* The function signature matches what the ONNX helper (`lambda_like.py`)
  actually needs: it receives *pixel* boxes, not YOLO-normalised ones.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


def _boxes_to_geojson(
    src_name: str, boxes: np.ndarray[Any, Any]
) -> Dict[str, Any]:
    """
    Convert *pixel* boxes ``[x1,y1,x2,y2,conf]`` to a GeoJSON FeatureCollection.
    """
    feats: list[Dict[str, Any]] = [
        {
            "type": "Feature",
            "properties": {"conf": float(conf)},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [float(x1), float(y1)],
                        [float(x2), float(y1)],
                        [float(x2), float(y2)],
                        [float(x1), float(y2)],
                        [float(x1), float(y1)],
                    ]
                ],
            },
        }
        for x1, y1, x2, y2, conf in boxes
    ]
    return {"type": "FeatureCollection", "source": src_name, "features": feats}


def to_geojson(
    src: str | Path | Image.Image,
    boxes: np.ndarray[Any, Any],
) -> Dict[str, Any]:
    """
    Public helper used by the CLI.

    Parameters
    ----------
    src
        Image path / URL / PIL image – used only for the *source* field.
    boxes
        ndarray[N, 5] – (x1, y1, x2, y2, conf) in **pixels**.

    Returns
    -------
    dict
        Valid GeoJSON FeatureCollection (empty if no boxes).
    """
    return _boxes_to_geojson(str(src), boxes)
