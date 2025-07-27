import numpy as np
import cv2

def heatmap_overlay(bgr_img: np.ndarray, boxes, drop_alpha=0.4, sigma=25):
    """Return BGR image with jet‑colored heat‑map."""
    h, w = bgr_img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    for x1,y1,x2,y2,conf in boxes:
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        rr, cc = np.ogrid[:h, :w]
        mask  += np.exp(-((rr-cy)**2 + (cc-cx)**2)/(2*sigma**2)) * conf
    mask = cv2.GaussianBlur(mask, (0,0), sigma)
    mask = cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color= cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr_img, 1-drop_alpha, color, drop_alpha, 0)
