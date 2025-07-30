"""
Simple Gaussian heat-map overlay (OpenCV).

• Weights each blob by detection confidence.
• Handles empty `boxes` gracefully - returns original image unchanged.
"""
from __future__ import annotations

from typing import List
from typing import Tuple

import cv2
import numpy as np
from typing import Any

Box = Tuple[float, float, float, float, float]  # x1,y1,x2,y2,conf


def heatmap_overlay(
    bgr_img: np.ndarray[Any, Any],  # shape (H,W,3) uint8
    boxes: List[Box] | None,
    alpha: float = 0.4,
    sigma: float = 25.0,
) -> np.ndarray[Any, Any]:
    if not boxes:
        return bgr_img.copy()

    h, w = bgr_img.shape[:2]
    mask = np.zeros((h, w), np.float32)

    yy, xx = np.ogrid[:h, :w]
    for x1, y1, x2, y2, conf in boxes:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        mask += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)) * float(
            conf
        )

    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    norm_mask = np.zeros_like(mask)
    cv2.normalize(mask, norm_mask, 0, 255, cv2.NORM_MINMAX)
    mask = norm_mask.astype(np.uint8)
    colour = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr_img, 1.0 - alpha, colour, alpha, 0)
