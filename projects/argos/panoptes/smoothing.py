"""
Reusable temporal smoothing helpers for Panoptes tasks.

Each smoother keeps a small amount of state so consecutive frames (live or
batch processing) can produce steadier overlays without visible jitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np
from numpy.typing import NDArray

from .obb_types import OBBDetection

NDArrayF32 = NDArray[np.float32]


class ProbabilitySmoother:
    """
    Exponential moving average over class probabilities.

    This keeps a dictionary of label -> score and blends new predictions with
    the running state. Scores naturally decay toward zero when a class stops
    appearing so stale labels eventually disappear from the HUD.
    """

    def __init__(self, *, alpha: float = 0.6, decay: float = 0.85, max_items: int = 5) -> None:
        self.alpha = float(max(0.0, min(1.0, alpha)))
        self.decay = float(max(0.0, min(1.0, decay)))
        self.max_items = max(1, int(max_items))
        self._scores: Dict[str, float] = {}

    def reset(self) -> None:
        self._scores.clear()

    def update(self, pairs: Iterable[Tuple[str, float]]) -> List[Tuple[str, float]]:
        fresh: Dict[str, float] = {}
        for name, score in pairs:
            name = str(name)
            try:
                val = float(score)
            except Exception:
                continue
            fresh[name] = max(0.0, val)

        if not self._scores and not fresh:
            return []

        updated: Dict[str, float] = {}
        alpha = self.alpha
        decay = self.decay

        # Blend existing scores with new values (or decay if absent).
        for name in set(self._scores) | set(fresh):
            current = self._scores.get(name, 0.0)
            incoming = fresh.get(name)
            if incoming is None:
                value = current * decay
            elif current == 0.0:
                value = incoming
            else:
                value = (1.0 - alpha) * current + alpha * incoming
            if value > 1e-3:
                updated[name] = value

        self._scores = updated

        ranked = sorted(self._scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[: self.max_items]


@dataclass
class _TrackedArray:
    data: NDArrayF32
    age: int = 0


@dataclass
class _TrackedPolygon:
    detection: OBBDetection
    age: int = 0


def _ensure_f32(data: Any, *, copy: bool = False) -> NDArrayF32:
    arr: NDArrayF32 = np.asarray(data, dtype=np.float32, order=None, copy=copy)
    return arr


def _align_lengths(a: NDArrayF32, b: NDArrayF32) -> Tuple[NDArrayF32, NDArrayF32]:
    len_a = int(a.shape[0]) if a.ndim > 0 else 0
    len_b = int(b.shape[0]) if b.ndim > 0 else 0
    if len_a == 0 or len_b == 0:
        return a[:0], b[:0]
    if len_a == len_b:
        return a, b
    k = min(len_a, len_b)
    return a[:k], b[:k]


def _mean_distance(a: NDArrayF32, b: NDArrayF32) -> float:
    if a.size == 0 or b.size == 0:
        return float("inf")
    a_aligned, b_aligned = _align_lengths(a, b)
    if a_aligned.size == 0 or b_aligned.size == 0:
        return float("inf")
    diff = a_aligned[:, :2] - b_aligned[:, :2]
    norms = np.sqrt(np.sum(diff * diff, axis=1))
    return float(norms.mean()) if norms.size else float("inf")


class KeypointSmoother:
    """
    Softly stabilize pose keypoints frame-to-frame by matching detections via
    centroid distance and blending coordinates.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        max_distance: float = 80.0,
        max_age: int = 5,
    ) -> None:
        self.alpha = float(max(0.0, min(1.0, alpha)))
        self.max_distance = float(max_distance)
        self.max_age = max(1, int(max_age))
        self._tracks: List[_TrackedArray] = []

    def reset(self) -> None:
        self._tracks.clear()

    def smooth(self, detections: Sequence[NDArrayF32]) -> List[NDArrayF32]:
        if not detections:
            for track in self._tracks:
                track.age += 1
            self._tracks = [t for t in self._tracks if t.age <= self.max_age]
            return []

        arrays = [_ensure_f32(det, copy=True) for det in detections]

        # Build match candidates (distance between centroids).
        candidates: List[Tuple[float, int, int]] = []
        for new_idx, arr in enumerate(arrays):
            for track_idx, track in enumerate(self._tracks):
                dist = _mean_distance(arr, track.data)
                if math.isfinite(dist):
                    candidates.append((dist, new_idx, track_idx))
        candidates.sort(key=lambda item: item[0])

        assigned_new: set[int] = set()
        assigned_tracks: set[int] = set()
        matches: Dict[int, int] = {}

        for dist, new_idx, track_idx in candidates:
            if dist > self.max_distance:
                continue
            if new_idx in assigned_new or track_idx in assigned_tracks:
                continue
            assigned_new.add(new_idx)
            assigned_tracks.add(track_idx)
            matches[new_idx] = track_idx

        smoothed: List[NDArrayF32] = []
        next_tracks: List[_TrackedArray] = []

        for new_idx, arr in enumerate(arrays):
            track_idx = matches.get(new_idx)
            if track_idx is not None:
                prev = self._tracks[track_idx].data
                prev_aligned, arr_aligned = _align_lengths(prev, arr)
                if prev_aligned.size == 0:
                    blended = arr_aligned.copy()
                else:
                    blended = prev_aligned.copy()
                    blended[:, :2] = (1.0 - self.alpha) * prev_aligned[:, :2] + self.alpha * arr_aligned[:, :2]
                if arr_aligned.shape[1] > 2:
                    blended[:, 2:] = (1.0 - self.alpha) * prev_aligned[:, 2:] + self.alpha * arr_aligned[:, 2:]
                smoothed.append(blended)
                next_tracks.append(_TrackedArray(blended, age=0))
            else:
                smoothed.append(arr)
                next_tracks.append(_TrackedArray(arr, age=0))

        # Age unmatched previous tracks (they may reappear shortly).
        for idx, track in enumerate(self._tracks):
            if idx in assigned_tracks:
                continue
            aged = track.age + 1
            if aged <= self.max_age:
                next_tracks.append(_TrackedArray(track.data, age=aged))

        self._tracks = next_tracks
        return smoothed


class PolygonSmoother:
    """
    Stabilize oriented or free-form polygons by blending coordinates of the
    nearest previous match (per class).
    """

    def __init__(
        self,
        *,
        alpha: float = 0.55,
        max_distance: float = 90.0,
        max_age: int = 5,
    ) -> None:
        self.alpha = float(max(0.0, min(1.0, alpha)))
        self.max_distance = float(max_distance)
        self.max_age = max(1, int(max_age))
        self._tracks: Dict[int, List[_TrackedPolygon]] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def smooth(
        self,
        items: Sequence[OBBDetection],
    ) -> List[OBBDetection]:
        if not items:
            for cls_tracks in self._tracks.values():
                for track in cls_tracks:
                    track.age += 1
            for cls_id in list(self._tracks.keys()):
                cls_tracks = [t for t in self._tracks[cls_id] if t.age <= self.max_age]
                if cls_tracks:
                    self._tracks[cls_id] = cls_tracks
                else:
                    self._tracks.pop(cls_id, None)
            return []

        outputs: List[OBBDetection] = []
        next_tracks: Dict[int, List[_TrackedPolygon]] = {}
        matched_indices: Dict[int, set[int]] = {}

        for det in items:
            cls_key = int(det.class_id) if det.class_id is not None else -1
            arr = _ensure_f32(det.points, copy=True)
            tracks = self._tracks.get(cls_key, [])
            used = matched_indices.setdefault(cls_key, set())

            best_idx: Optional[int] = None
            best_dist = float("inf")

            for idx, track in enumerate(tracks):
                if idx in used:
                    continue
                dist = _mean_distance(arr, track.detection.points)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None and best_dist <= self.max_distance:
                prev_det = tracks[best_idx].detection
                prev_align, arr_align = _align_lengths(prev_det.points, arr)
                if prev_align.size == 0:
                    blended = arr_align.copy()
                else:
                    blended = prev_align.copy()
                    blended[:, :2] = (1.0 - self.alpha) * prev_align[:, :2] + self.alpha * arr_align[:, :2]
                    if arr_align.shape[1] > 2:
                        blended[:, 2:] = (1.0 - self.alpha) * prev_align[:, 2:] + self.alpha * arr_align[:, 2:]
                angle = det.angle if det.angle is not None else prev_det.angle
                new_det = OBBDetection(blended, det.confidence, det.class_id, angle)
                used.add(best_idx)
            else:
                new_det = det.with_points(arr)

            outputs.append(new_det)
            next_tracks.setdefault(cls_key, []).append(_TrackedPolygon(new_det, age=0))

        # Age unmatched tracks
        for cls_key, tracks in self._tracks.items():
            used = matched_indices.get(cls_key, set())
            for idx, track in enumerate(tracks):
                if idx in used:
                    continue
                aged = track.age + 1
                if aged <= self.max_age:
                    next_tracks.setdefault(cls_key, []).append(_TrackedPolygon(track.detection, age=aged))

        self._tracks = next_tracks
        return outputs
