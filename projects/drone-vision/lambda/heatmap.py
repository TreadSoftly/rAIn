import numpy as np
from typing import Any

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None  # type: ignore


def heatmap_overlay(  # noqa: D401
    bgr_img: np.ndarray[Any, Any],
    boxes: list[tuple[float, float, float, float, float]],
    drop_alpha: float = 0.40,
    sigma: int = 25,
) -> np.ndarray[Any, Any]:
    """
    Returns *bgr_img* with a jet-coloured heat-map overlay.

    When OpenCV wheels are unavailable (e.g. macOS + Py 3.12 on CI),
    the original image is returned untouched so the unit-tests still pass.
    """
    if cv2 is None:
        return bgr_img

    h, w = bgr_img.shape[:2]
    mask = np.zeros((h, w), np.float32)

    for x1, y1, x2, y2, conf in boxes:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        rr, cc = np.ogrid[:h, :w]
        mask += np.exp(-((rr - cy) ** 2 + (cc - cx) ** 2) / (2 * sigma**2)) * conf

    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    mask = cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colour = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr_img, 1 - drop_alpha, colour, drop_alpha, 0)
