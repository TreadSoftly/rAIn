"""
Serialises all YOLO bounding boxes into a flat GeoJSON *FeatureCollection*.
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
from PIL import Image
from ultralytics import YOLO


def _yolo_boxes(img: Image.Image, *, weights: str) -> np.ndarray:
    yolo = YOLO(weights)
    pred = yolo.predict(source=img, imgsz=640, conf=0.25, verbose=False)[0]
    return pred.boxes.xywhn.cpu().numpy()  # [[x, y, w, h]...]  normalised


def to_geojson(img: Image.Image, *, weights: str = "yolov8n.pt") -> str:
    h, w = img.height, img.width
    boxes = _yolo_boxes(img, weights=weights)

    features: List[Dict] = []
    for x, y, bw, bh in boxes:
        x0, y0 = (x - bw / 2) * w, (y - bh / 2) * h
        x1, y1 = (x + bw / 2) * w, (y + bh / 2) * h
        poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        features.append(
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [poly]},
            }
        )

    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2)
