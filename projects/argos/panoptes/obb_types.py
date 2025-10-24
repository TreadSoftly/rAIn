from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

NDArrayF32 = NDArray[np.float32]


def _ensure_float32(points: Sequence[Sequence[float]]) -> NDArrayF32:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("polygon points must be shaped (N, 2[+])")
    return arr


@dataclass
class OBBDetection:
    """
    Normalised representation of an oriented bounding polygon.

    Attributes
    ----------
    points:
        Nx2 (or NxM with extra columns) array of polygon vertices in float32.
    confidence:
        Model confidence score in [0, 1].
    class_id:
        Optional class index.
    angle:
        Optional orientation in degrees (if provided by the model).
    """

    points: NDArrayF32
    confidence: float
    class_id: Optional[int] = None
    angle: Optional[float] = None

    @classmethod
    def from_iterable(
        cls,
        points: Sequence[Sequence[float]],
        confidence: float,
        class_id: Optional[int] = None,
        angle: Optional[float] = None,
    ) -> "OBBDetection":
        return cls(_ensure_float32(points), float(confidence), class_id, angle)

    def clone(self, *, points: Optional[NDArrayF32] = None) -> "OBBDetection":
        return OBBDetection(
            points=points.copy() if points is not None else self.points.copy(),
            confidence=self.confidence,
            class_id=self.class_id,
            angle=self.angle,
        )

    def as_int_points(self) -> List[Tuple[int, int]]:
        pts = self.points
        return [(int(round(float(x))), int(round(float(y)))) for x, y in pts[:, :2]]

    def centroid(self) -> Tuple[float, float]:
        pts = self.points
        if pts.size == 0:
            return (0.0, 0.0)
        xs = pts[:, 0]
        ys = pts[:, 1]
        return float(xs.mean()), float(ys.mean())

    def axis_aligned(self) -> Tuple[int, int, int, int]:
        pts = self.points
        if pts.size == 0:
            return (0, 0, 0, 0)
        xs = pts[:, 0]
        ys = pts[:, 1]
        x0 = int(np.floor(xs.min()))
        y0 = int(np.floor(ys.min()))
        x1 = int(np.ceil(xs.max()))
        y1 = int(np.ceil(ys.max()))
        return x0, y0, x1, y1

    def with_points(self, points: NDArrayF32) -> "OBBDetection":
        return OBBDetection(points=points, confidence=self.confidence, class_id=self.class_id, angle=self.angle)


def ensure_detections(items: Iterable["OBBDetection"]) -> List["OBBDetection"]:
    return [item for item in items]
