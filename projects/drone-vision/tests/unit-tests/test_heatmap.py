# projects/drone-vision/tests/unit-tests/test_heatmap.py
"""
Pure-python unit test for the Gaussian-blur compositor that powers heat-maps.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from dronevision.heatmap import heatmap_overlay  # type: ignore[import-untyped]


def test_basic_gaussian() -> None:
    size = (256, 256)
    img = Image.new("RGB", size, "black")
    boxes = np.array([[50, 60, 80, 90, 0.9]], dtype=np.float32)

    out = heatmap_overlay(img, boxes=boxes, alpha=0.5, return_mask=False)
    assert isinstance(out, Image.Image)
    assert out.size == img.size
