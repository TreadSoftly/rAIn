"""
Task adapters for live mode.

We keep adapters ultra-light with **non-ML fallbacks** so live mode still works
without large model downloads. Later, you can swap these with real models via
your central model registry.

Available builders:
  - build_detect(*, small=True, conf=0.25, iou=0.45) -> TaskAdapter
  - build_heatmap(*, small=True)                     -> TaskAdapter
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, cast

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

from ._types import NDArrayU8, Boxes, Names


class TaskAdapter(Protocol):
    def infer(self, frame_bgr: NDArrayU8) -> Any: ...
    def render(self, frame_bgr: NDArrayU8, result: Any) -> NDArrayU8: ...


# ---------------------------
# Detect (non-ML fallback)
# ---------------------------

class _ContourDetect(TaskAdapter):
    """
    Very fast, non-ML 'detector' using Canny + contour boxes.
    This is not meant to be accurate — it's a live demo fallback.
    """

    def __init__(self, conf: float = 0.25, iou: float = 0.45) -> None:
        self.conf = float(conf)
        self.iou = float(iou)
        self.names: Names = {}

    def infer(self, frame_bgr: NDArrayU8) -> list[tuple[int, int, int, int, float, Optional[int]]]:
        assert np is not None
        if cv2 is None:
            return []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        v = max(10, int(gray.mean()))
        edges = cv2.Canny(gray, v, v * 3)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int, float, Optional[int]]] = []
        H, W = gray.shape[:2]
        min_area = max(80, (H * W) // 300)  # avoid too many tiny boxes
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < min_area:
                continue
            conf = min(0.99, 0.5 + (w * h) / (W * H))  # toy confidence
            boxes.append((x, y, x + w, y + h, conf, None))
        return boxes

    def render(self, frame_bgr: NDArrayU8, result: Boxes) -> NDArrayU8:
        from .overlay import draw_boxes_bgr
        return draw_boxes_bgr(frame_bgr, result, names=self.names)


def build_detect(*, small: bool = True, conf: float = 0.25, iou: float = 0.45) -> TaskAdapter:
    """
    Build a live detection adapter. Today this returns the contour fallback.
    Later, you can switch to a YOLO/RT-DETR/etc adapter via your registry.
    """
    return _ContourDetect(conf=conf, iou=iou)


# ---------------------------
# Heatmap (non-ML fallback)
# ---------------------------

class _LaplacianHeatmap(TaskAdapter):
    def __init__(self) -> None:
        ...

    def infer(self, frame_bgr: NDArrayU8) -> NDArrayU8:
        assert np is not None
        if cv2 is None:
            # brightness mask as fallback
            gray = frame_bgr.mean(axis=2).astype("float32")
            return (255.0 * (gray / (gray.max() + 1e-6))).astype("uint8")
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, ddepth=cv2.CV_16S, ksize=3)
        lap = cv2.convertScaleAbs(lap)
        # Normalize
        m = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore[call-arg]
        return cast(NDArrayU8, m)

    def render(self, frame_bgr: NDArrayU8, result: NDArrayU8) -> NDArrayU8:
        from .overlay import draw_heatmap_bgr
        return draw_heatmap_bgr(frame_bgr, result)


def build_heatmap(*, small: bool = True) -> TaskAdapter:
    return _LaplacianHeatmap()


# ---------------------------
# Placeholders for future tasks
# ---------------------------

def build_pose(*_a: object, **_k: object) -> TaskAdapter:
    raise NotImplementedError("live pose is not enabled yet")

def build_cls(*_a: object, **_k: object) -> TaskAdapter:
    raise NotImplementedError("live classify is not enabled yet")

def build_obb(*_a: object, **_k: object) -> TaskAdapter:
    raise NotImplementedError("live OBB is not enabled yet")
