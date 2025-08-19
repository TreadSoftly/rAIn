# projects/argos/panoptes/live/tasks.py
"""
Task adapters for live mode.

This revision routes LIVE tasks (detect/heatmap/classify/pose/pse/obb) through
the central model registry. If a model cannot be loaded (weights missing or
ultralytics not installed), we gracefully fall back to non-ML adapters so the
demo still runs.

Key change (2025-08-18):
    • Heatmap adapter now preserves *instance masks* from the -seg model and
      renders them with distinct colors (plus optional labels), instead of
      collapsing to one binary mask that color-maps to red.

Available builders:
  - build_detect(*, small=True, conf=0.25, iou=0.45, override=None)
  - build_heatmap(*, small=True, override=None)
  - build_classify(*, small=True, topk=1, override=None)
  - build_pose(*, small=True, conf=0.25, override=None)
  - build_pse(*, small=True, override=None)  # ALIAS of pose (same model/overlay)
  - build_obb(*, small=True, conf=0.25, iou=0.45, override=None)
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Union, cast, Sequence, List, Tuple, Dict
from pathlib import Path
import math

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

# DType aliases that are safe for static analyzers even if numpy is Optional at runtime.
try:
    import numpy as _np_for_types
    f32: Any = _np_for_types.float32
    i64: Any = _np_for_types.int64
    u8: Any = _np_for_types.uint8
except Exception:
    f32 = cast(Any, "float32")
    i64 = cast(Any, "int64")
    u8 = cast(Any, "uint8")

from ._types import NDArrayU8, Boxes, Names

# Use the central model registry (your single source of truth)
# Provide a guarded import with safe fallbacks so this module always imports.
try:
    from panoptes.model_registry import (  # type: ignore[reportMissingTypeStubs]
        load_detector,    # type: ignore[no-redef]
        load_segmenter,   # type: ignore[no-redef]
        load_classifier,  # type: ignore[no-redef]
        load_pose,        # type: ignore[no-redef]
        load_obb,         # type: ignore[no-redef]
        pick_weight,      # type: ignore[no-redef]
    )
except Exception:

    def load_detector(*_a: object, **_k: object) -> Any:  # type: ignore[no-redef]
        raise RuntimeError("panoptes.model_registry not available")

    def load_segmenter(*_a: object, **_k: object) -> Any:  # type: ignore[no-redef]
        raise RuntimeError("panoptes.model_registry not available")

    def load_classifier(*_a: object, **_k: object) -> Any:  # type: ignore[no-redef]
        raise RuntimeError("panoptes.model_registry not available")

    def load_pose(*_a: object, **_k: object) -> Any:  # type: ignore[no-redef]
        raise RuntimeError("panoptes.model_registry not available")

    def load_obb(*_a: object, **_k: object) -> Any:  # type: ignore[no-redef]
        raise RuntimeError("panoptes.model_registry not available")

    def pick_weight(*_a: object, **_k: object) -> Any:  # type: ignore[no-redef]
        return None

# ─────────────────────────────────────────────────────────────────────
# Live overrides — leave as None so the registry fully controls weights
# ─────────────────────────────────────────────────────────────────────
LIVE_DETECT_OVERRIDE: Optional[Union[str, Path]]   = None
LIVE_HEATMAP_OVERRIDE: Optional[Union[str, Path]]  = None
LIVE_CLASSIFY_OVERRIDE: Optional[Union[str, Path]] = None
LIVE_POSE_OVERRIDE: Optional[Union[str, Path]]     = None
LIVE_PSE_OVERRIDE: Optional[Union[str, Path]]      = None
LIVE_OBB_OVERRIDE: Optional[Union[str, Path]]      = None


class TaskAdapter(Protocol):
    def infer(self, frame_bgr: NDArrayU8) -> Any: ...
    def render(self, frame_bgr: NDArrayU8, result: Any) -> NDArrayU8: ...
    label: str  # for HUD


# Minimal predictor protocol so static checkers know these attributes exist.
class _Predictor(Protocol):
    names: Any
    def predict(self, img: Any, *args: Any, **kwargs: Any) -> Any: ...


# ---------------------------
# Small utility: convert torch/array-like to NumPy without importing torch
# ---------------------------
def _to_numpy(x: Any, *, dtype: Optional[Any] = None) -> Any:
    np_ = cast(Any, np)
    assert np_ is not None
    y = x
    if hasattr(y, "detach"):
        try:
            y = y.detach()
        except Exception:
            pass
    if hasattr(y, "cpu"):
        try:
            y = y.cpu()
        except Exception:
            pass
    if hasattr(y, "numpy"):
        try:
            arr = y.numpy()
        except Exception:
            arr = np_.asarray(y)
    else:
        arr = np_.asarray(y)
    if dtype is not None:
        try:
            arr = arr.astype(dtype, copy=False)
        except Exception:
            arr = np_.asarray(arr, dtype=dtype)
    return arr


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
        self.label = "fast-contour (no-ML)"

    def infer(self, frame_bgr: NDArrayU8) -> List[Tuple[int, int, int, int, float, Optional[int]]]:
        np_ = cast(Any, np)
        assert np_ is not None
        if cv2 is None:
            return []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        v = max(10, int(gray.mean()))
        edges = cv2.Canny(gray, v, v * 3)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int, int, int, int, float, Optional[int]]] = []
        H, W = gray.shape[:2]
        min_area = max(80, (H * W) // 300)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < min_area:
                continue
            conf = min(0.99, 0.5 + (w * h) / (W * H))
            boxes.append((x, y, x + w, y + h, conf, None))
        return boxes

    def render(self, frame_bgr: NDArrayU8, result: Boxes) -> NDArrayU8:
        from .overlay import draw_boxes_bgr
        return draw_boxes_bgr(frame_bgr, result, names=self.names)


# ---------------------------
# Heatmap (non-ML fallback)
# ---------------------------

class _LaplacianHeatmap(TaskAdapter):
    def __init__(self) -> None:
        self.label = "laplacian-heatmap (no-ML)"

    def infer(self, frame_bgr: NDArrayU8) -> NDArrayU8:
        np_ = cast(Any, np)
        assert np_ is not None
        if cv2 is None:
            gray = frame_bgr.mean(axis=2).astype("float32")
            return (255.0 * (gray / (gray.max() + 1e-6))).astype("uint8")
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, ddepth=cv2.CV_16S, ksize=3)
        lap = cv2.convertScaleAbs(lap)
        m = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore[call-arg]
        return cast(NDArrayU8, m)

    def render(self, frame_bgr: NDArrayU8, result: NDArrayU8) -> NDArrayU8:
        from .overlay import draw_heatmap_bgr
        return draw_heatmap_bgr(frame_bgr, result)


# ---------------------------
# YOLO-backed LIVE adapters
# ---------------------------

class _YOLODetect(TaskAdapter):
    """YOLO-based detector adapter for live mode (boxes)."""

    def __init__(self, model: _Predictor, *, label: str, conf: float = 0.25, iou: float = 0.45) -> None:
        self.model = model
        self.conf = float(conf)
        self.iou = float(iou)
        self.label = label
        self.names: Dict[int, str] = {}
        names_attr: Any = getattr(model, "names", {})
        if isinstance(names_attr, dict):
            for k, v in cast(Dict[Any, Any], names_attr).items():
                try:
                    self.names[int(k)] = str(v)
                except Exception:
                    continue
        elif isinstance(names_attr, (list, tuple)):
            for i, n in enumerate(cast(Sequence[object], names_attr)):
                self.names[i] = str(n)

    def infer(self, frame_bgr: NDArrayU8) -> List[Tuple[int, int, int, int, float, Optional[int]]]:
        np_ = cast(Any, np)
        assert np_ is not None
        inp = frame_bgr.copy() if hasattr(frame_bgr, "copy") else frame_bgr
        res_any: Any = self.model.predict(inp, conf=self.conf, iou=self.iou, verbose=False)
        if isinstance(res_any, (list, tuple)):
            res_obj: object = cast(Sequence[object], res_any)[0]
        else:
            res_obj = cast(object, res_any)

        boxes: List[Tuple[int, int, int, int, float, Optional[int]]] = []
        b_obj: Optional[object] = getattr(res_obj, "boxes", None)
        if b_obj is None:
            return boxes

        xyxy_any: Any = getattr(b_obj, "xyxy", None)
        confs_any: Any = getattr(b_obj, "conf", None)
        clses_any: Any = getattr(b_obj, "cls", None)
        if xyxy_any is None or confs_any is None or clses_any is None:
            return boxes

        xyxy_np = _to_numpy(xyxy_any, dtype=f32)
        conf_np = _to_numpy(confs_any, dtype=f32).reshape(-1)
        try:
            cls_np = _to_numpy(clses_any, dtype=i64).reshape(-1)
        except Exception:
            cls_np = np_.zeros((xyxy_np.shape[0],), dtype=int)

        for (x1, y1, x2, y2), conf_v, cls_v in zip(xyxy_np, conf_np, cls_np):
            boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf_v), int(cls_v)))
        return boxes

    def render(self, frame_bgr: NDArrayU8, result: Boxes) -> NDArrayU8:
        from .overlay import draw_boxes_bgr
        return draw_boxes_bgr(frame_bgr, result, names=self.names)


class _YOLOHeatmap(TaskAdapter):
    """
    YOLO-based segmentation adapter for live mode.

    Returns *per-instance* masks with their conf and class so the renderer can
    color each instance distinctly (and label it), matching offline heatmap behavior.
    """

    def __init__(self, model: _Predictor, *, label: str, conf: float = 0.25) -> None:
        self.model = model
        self.conf = float(conf)
        self.label = label
        # Prepare class-name lookup once
        self.names: Dict[int, str] = {}
        names_attr: Any = getattr(model, "names", {})
        if isinstance(names_attr, dict):
            for k, v in cast(Dict[Any, Any], names_attr).items():
                try:
                    self.names[int(k)] = str(v)
                except Exception:
                    continue
        elif isinstance(names_attr, (list, tuple)):
            for i, n in enumerate(cast(Sequence[object], names_attr)):
                self.names[i] = str(n)

    @staticmethod
    def _resize_nn(mask: NDArrayU8, new_hw: Tuple[int, int]) -> NDArrayU8:
        np_ = cast(Any, np)
        assert np_ is not None
        H, W = new_hw
        h, w = int(mask.shape[0]), int(mask.shape[1])
        if (h, w) == (H, W):
            return mask
        y_idx = np_.round(np_.linspace(0, h - 1, H)).astype(int)
        x_idx = np_.round(np_.linspace(0, w - 1, W)).astype(int)
        return mask[y_idx[:, None], x_idx[None, :]]

    def infer(self, frame_bgr: NDArrayU8) -> List[Tuple[NDArrayU8, float, Optional[int]]]:
        """
        Returns: list of (mask_u8(H,W), conf: float, cls_id: Optional[int])
        """
        np_ = cast(Any, np)
        assert np_ is not None
        inp = frame_bgr.copy() if hasattr(frame_bgr, "copy") else frame_bgr
        res_any: Any = self.model.predict(inp, conf=self.conf, verbose=False)
        if isinstance(res_any, (list, tuple)):
            res_obj: object = cast(Sequence[object], res_any)[0]
        else:
            res_obj = cast(object, res_any)

        H, W = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])

        masks_any: Any = getattr(res_obj, "masks", None)
        if masks_any is None:
            return []

        data_any: Any = getattr(masks_any, "data", None)
        if data_any is None:
            return []

        m_np: Any = _to_numpy(data_any, dtype=f32)  # (N,h,w) float in [0,1] typically
        if getattr(m_np, "ndim", 0) == 2:
            m_np = m_np[0:1, ...]  # (1, h, w)

        # Optional: confidences + classes (aligned with masks)
        boxes_obj: Any = getattr(res_obj, "boxes", None)
        confs_seq: Optional[Sequence[float]] = None
        clses_seq: Optional[Sequence[int]] = None
        if boxes_obj is not None:
            try:
                confs_seq = cast(Sequence[float], _to_numpy(getattr(boxes_obj, "conf", None), dtype=f32).reshape(-1).tolist())
            except Exception:
                confs_seq = None
            try:
                clses_seq = cast(Sequence[int], _to_numpy(getattr(boxes_obj, "cls", None), dtype=i64).reshape(-1).astype(int).tolist())
            except Exception:
                clses_seq = None

        out: List[Tuple[NDArrayU8, float, Optional[int]]] = []
        num = int(getattr(m_np, "shape", (0,))[0] or 0)
        for i in range(num):
            m = m_np[i]
            m_bin: NDArrayU8 = (m >= 0.5).astype(u8)
            if tuple(m_bin.shape) != (H, W):
                m_bin = self._resize_nn(m_bin, (H, W))
            conf_v = float(confs_seq[i]) if confs_seq is not None and i < len(confs_seq) else 1.0
            cls_v: Optional[int] = int(clses_seq[i]) if clses_seq is not None and i < len(clses_seq) else None
            out.append((m_bin, conf_v, cls_v))
        return out

    def render(self, frame_bgr: NDArrayU8, result: List[Tuple[NDArrayU8, float, Optional[int]]]) -> NDArrayU8:
        # Per-instance compositing with distinct colors + optional labels
        from .overlay import draw_masks_bgr
        return draw_masks_bgr(frame_bgr, result, names=self.names, alpha=0.35)


# ---------------------------
# CLASSIFY (YOLO-cls) + fallback
# ---------------------------

class _YOLOClassify(TaskAdapter):
    def __init__(self, model: _Predictor, *, label: str, topk: int = 1) -> None:
        self.model = model
        self.label = label
        self.topk = max(1, int(topk))

        # Prepare class names mapping (int -> label) if available
        self.names: Dict[int, str] = {}
        names_attr: Any = getattr(model, "names", {})
        if isinstance(names_attr, dict):
            for k, v in cast(Dict[Any, Any], names_attr).items():
                try:
                    self.names[int(k)] = str(v)
                except Exception:
                    continue
        elif isinstance(names_attr, (list, tuple)):
            for i, n in enumerate(cast(Sequence[object], names_attr)):
                self.names[i] = str(n)

    def infer(self, frame_bgr: NDArrayU8) -> List[Tuple[str, float]]:
        np_ = cast(Any, np)
        assert np_ is not None
        inp = frame_bgr.copy() if hasattr(frame_bgr, "copy") else frame_bgr
        res_any: Any = self.model.predict(inp, verbose=False)

        # Ultralytics results often expose .probs with .topk etc.
        def _topk_from_probs(obj: Any) -> Optional[List[Tuple[str, float]]]:
            probs = getattr(obj, "probs", None)
            if probs is None:
                return None
            try:
                topk_fn = getattr(probs, "topk", None)
                data_vec = _to_numpy(getattr(probs, "data", []), dtype=f32).reshape(-1)
                if callable(topk_fn):
                    raw = topk_fn(self.topk)  # could be list/np/tensor-like
                    idxs = np_.asarray(raw).astype(int).ravel().tolist()
                    scores = [float(data_vec[i]) for i in idxs if 0 <= i < data_vec.shape[0]]
                else:
                    idxs = data_vec.argsort()[-self.topk:][::-1].tolist()
                    scores = [float(data_vec[i]) for i in idxs]
                labels = [self.names.get(int(i), str(int(i))) for i in idxs]
                return list(zip(labels, scores))
            except Exception:
                return None

        if isinstance(res_any, (list, tuple)):
            res_obj: object = cast(Sequence[object], res_any)[0]
        else:
            res_obj = cast(object, res_any)

        top = _topk_from_probs(res_obj)
        if top is not None:
            return top

        for attr in ("probs", "logits", "scores", "data"):
            arr = getattr(res_obj, attr, None)
            if arr is None:
                continue
            try:
                vec = _to_numpy(arr, dtype=f32).reshape(-1)
                idxs = vec.argsort()[-self.topk:][::-1].tolist()
                labels = [self.names.get(int(i), str(i)) for i in idxs]
                scores = [float(vec[i]) for i in idxs]
                return list(zip(labels, scores))
            except Exception:
                continue

        return [("unknown", 1.0)]

    def render(self, frame_bgr: NDArrayU8, result: List[Tuple[str, float]]) -> NDArrayU8:
        from .overlay import draw_classify_card_bgr
        return draw_classify_card_bgr(frame_bgr, result)


class _SimpleClassify(TaskAdapter):
    """No-ML fallback classification: brightness & saturation heuristic."""
    def __init__(self, topk: int = 1) -> None:
        self.label = "simple-classify (no-ML)"
        self.topk = max(1, int(topk))

    def infer(self, frame_bgr: NDArrayU8) -> List[Tuple[str, float]]:
        np_ = cast(Any, np)
        assert np_ is not None
        bgr = frame_bgr.astype("float32") / 255.0
        gray = (0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2])
        bright = float(gray.mean())
        sat = float((bgr.max(axis=2) - bgr.min(axis=2)).mean())
        candidates = [
            ("bright" if bright > 0.5 else "dark", abs(bright - 0.5) + 0.5),
            ("colorful" if sat > 0.25 else "flat", abs(sat - 0.25) + 0.5),
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[: self.topk]

    def render(self, frame_bgr: NDArrayU8, result: List[Tuple[str, float]]) -> NDArrayU8:
        from .overlay import draw_classify_card_bgr
        return draw_classify_card_bgr(frame_bgr, result)


# ---------------------------
# POSE (YOLO-pose) + fallback
# ---------------------------

class _YOLOPose(TaskAdapter):
    def __init__(self, model: _Predictor, *, label: str, conf: float = 0.25) -> None:
        self.model = model
        self.label = label
        self.conf = float(conf)

    def infer(self, frame_bgr: NDArrayU8) -> List[List[List[float]]]:
        np_ = cast(Any, np)
        assert np_ is not None
        inp = frame_bgr.copy() if hasattr(frame_bgr, "copy") else frame_bgr
        res_any: Any = self.model.predict(inp, conf=self.conf, verbose=False)
        if isinstance(res_any, (list, tuple)):
            res_obj: object = cast(Sequence[object], res_any)[0]
        else:
            res_obj = cast(object, res_any)

        kps_any: Any = getattr(res_obj, "keypoints", None)
        if kps_any is None:
            kps_any = getattr(res_obj, "kpts", None)
        if kps_any is None:
            return []

        data = getattr(kps_any, "data", None)
        if data is None:
            data = getattr(kps_any, "xy", None)

        arr = _to_numpy(data)

        people: List[List[List[float]]] = []
        if getattr(arr, "ndim", 0) == 3:
            for i in range(arr.shape[0]):
                kp = arr[i]
                if kp.shape[1] == 2:
                    sc = np_.ones((kp.shape[0], 1), dtype=f32)
                    kp = np_.concatenate([kp, sc], axis=1)
                kp_list: List[List[float]] = [
                    [float(kp[j, 0]), float(kp[j, 1]), float(kp[j, 2])]
                    for j in range(kp.shape[0])
                ]
                people.append(kp_list)
        return people

    def render(self, frame_bgr: NDArrayU8, result: List[List[List[float]]]) -> NDArrayU8:
        if np is not None:
            people_np = [np.asarray(kp, dtype="float32") for kp in result]  # type: ignore[no-redef]
        else:
            people_np = result  # type: ignore[assignment]
        from .overlay import draw_pose_bgr
        return draw_pose_bgr(frame_bgr, people_np)  # type: ignore[arg-type]


class _SimplePose(TaskAdapter):
    """No-ML fallback pose: draw a stick figure anchored to frame center."""
    def __init__(self) -> None:
        self.label = "simple-pose (no-ML)"

    def infer(self, frame_bgr: NDArrayU8) -> List[List[List[float]]]:
        np_ = cast(Any, np)
        assert np_ is not None
        H, W = frame_bgr.shape[:2]
        cx, cy = W // 2, H // 2
        kpts = np_.array([
            [cx, cy - H * 0.25, 1.0],
            [cx, cy - H * 0.18, 1.0],
            [cx - W * 0.08, cy - H * 0.16, 1.0],
            [cx - W * 0.15, cy - H * 0.05, 1.0],
            [cx - W * 0.20, cy + H * 0.02, 1.0],
            [cx + W * 0.08, cy - H * 0.16, 1.0],
            [cx + W * 0.15, cy - H * 0.05, 1.0],
            [cx + W * 0.20, cy + H * 0.02, 1.0],
            [cx - W * 0.05, cy, 1.0],
            [cx - W * 0.05, cy + H * 0.15, 1.0],
            [cx - W * 0.05, cy + H * 0.30, 1.0],
            [cx + W * 0.05, cy, 1.0],
            [cx + W * 0.05, cy + H * 0.15, 1.0],
            [cx + W * 0.05, cy + H * 0.30, 1.0],
            [cx - W * 0.02, cy - H * 0.22, 1.0],
            [cx + W * 0.02, cy - H * 0.22, 1.0],
            [cx, cy - H * 0.25, 1.0],
        ], dtype=f32)
        kp_list: List[List[float]] = [[float(x), float(y), float(s)] for (x, y, s) in kpts.tolist()]
        return [kp_list]

    def render(self, frame_bgr: NDArrayU8, result: List[List[List[float]]]) -> NDArrayU8:
        if np is not None:
            people_np = [np.asarray(kp, dtype="float32") for kp in result]  # type: ignore[no-redef]
        else:
            people_np = result  # type: ignore[assignment]
        from .overlay import draw_pose_bgr
        return draw_pose_bgr(frame_bgr, people_np)  # type: ignore[arg-type]


# ---------------------------
# PSE = **POSE ALIAS** (no segmentation here)
# ---------------------------

# NOTE: Keep the specialized PSE (segmentation) adapters out of the path;
# PSE must behave the same as POSE for your CLI and LV flows.

# ---------------------------
# OBB (YOLO-obb) + fallback
# ---------------------------

def _rotrect_to_pts(cx: float, cy: float, w: float, h: float, theta_deg: float) -> List[Tuple[int, int]]:
    rad = math.radians(theta_deg)
    c, s = math.cos(rad), math.sin(rad)
    hw, hh = w / 2.0, h / 2.0
    pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    out: List[Tuple[int, int]] = []
    for (px, py) in pts:
        rx = px * c - py * s
        ry = px * s + py * c
        out.append((int(cx + rx), int(cy + ry)))
    return out

class _YOLOOBB(TaskAdapter):
    def __init__(self, model: _Predictor, *, label: str, conf: float = 0.25, iou: float = 0.45) -> None:
        self.model = model
        self.label = label
        self.conf = float(conf)
        self.iou = float(iou)
        self.names: Dict[int, str] = {}
        names_attr: Any = getattr(model, "names", {})
        if isinstance(names_attr, dict):
            for k, v in cast(Dict[Any, Any], names_attr).items():
                try:
                    self.names[int(k)] = str(v)
                except Exception:
                    continue
        elif isinstance(names_attr, (list, tuple)):
            for i, n in enumerate(cast(Sequence[object], names_attr)):
                self.names[i] = str(n)

    def infer(self, frame_bgr: NDArrayU8) -> List[Tuple[List[Tuple[int, int]], float, Optional[int]]]:
        np_ = cast(Any, np)
        assert np_ is not None
        inp = frame_bgr.copy() if hasattr(frame_bgr, "copy") else frame_bgr
        res_any: Any = self.model.predict(inp, conf=self.conf, iou=self.iou, verbose=False)
        if isinstance(res_any, (list, tuple)):
            res_obj: object = cast(Sequence[object], res_any)[0]
        else:
            res_obj = cast(object, res_any)

        out: List[Tuple[List[Tuple[int, int]], float, Optional[int]]] = []

        obb_any: Any = getattr(res_obj, "obb", None)
        if obb_any is None:
            poly_any: Any = getattr(res_obj, "xyxyxyxy", None)
            if poly_any is None:
                return out
            polys = _to_numpy(poly_any)
            confs = getattr(res_obj, "conf", None)
            clses = getattr(res_obj, "cls", None)
            confs_seq: Optional[Sequence[float]] = None
            clses_seq: Optional[Sequence[int]] = None
            if confs is not None:
                confs_seq = cast(Sequence[float], _to_numpy(confs, dtype=f32).reshape(-1).tolist())
            if clses is not None:
                clses_seq = cast(Sequence[int], _to_numpy(clses, dtype=i64).reshape(-1).astype(int).tolist())
            for i in range(polys.shape[0]):
                poly = polys[i].reshape(-1, 2).astype(float)
                pts: List[Tuple[int, int]] = [(int(poly[j, 0]), int(poly[j, 1])) for j in range(poly.shape[0])]
                conf_v = float(confs_seq[i]) if confs_seq is not None and i < len(confs_seq) else 1.0
                cls_v: Optional[int] = int(clses_seq[i]) if clses_seq is not None and i < len(clses_seq) else None
                out.append((pts, conf_v, cls_v))
            return out

        data: Any = getattr(obb_any, "xywhr", None)
        if data is None:
            data = getattr(obb_any, "data", None)
        if data is None:
            return out

        arr = _to_numpy(data)

        b_obj: Any = getattr(res_obj, "boxes", None) or res_obj
        confs = getattr(b_obj, "conf", None)
        clses = getattr(b_obj, "cls", None)
        confs_seq2: Optional[Sequence[float]] = None
        clses_seq2: Optional[Sequence[int]] = None
        if confs is not None:
            confs_seq2 = cast(Sequence[float], _to_numpy(confs, dtype=f32).reshape(-1).tolist())
        if clses is not None:
            clses_seq2 = cast(Sequence[int], _to_numpy(clses, dtype=i64).reshape(-1).astype(int).tolist())

        for i in range(arr.shape[0]):
            cx, cy, w, h, theta = arr[i][:5].tolist()
            pts = _rotrect_to_pts(float(cx), float(cy), float(w), float(h), float(theta))
            conf_v = float(confs_seq2[i]) if confs_seq2 is not None and i < len(confs_seq2) else 1.0
            cls_v: Optional[int] = int(clses_seq2[i]) if clses_seq2 is not None and i < len(clses_seq2) else None
            out.append((pts, conf_v, cls_v))
        return out

    def render(self, frame_bgr: NDArrayU8, result: List[Tuple[List[Tuple[int, int]], float, Optional[int]]]) -> NDArrayU8:
        from .overlay import draw_obb_bgr
        return draw_obb_bgr(frame_bgr, result, names=self.names)


class _SimpleOBB(TaskAdapter):
    """No-ML fallback OBB: minAreaRect on strong contours."""
    def __init__(self, conf: float = 0.25) -> None:
        self.conf = float(conf)
        self.label = "simple-obb (no-ML)"

    def infer(self, frame_bgr: NDArrayU8) -> List[Tuple[List[Tuple[int, int]], float, Optional[int]]]:
        if cv2 is None:
            return []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        v = max(10, int(gray.mean()))
        edges = cv2.Canny(gray, v, v * 3)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = gray.shape[:2]
        min_area = max(120, (H * W) // 250)
        out: List[Tuple[List[Tuple[int, int]], float, Optional[int]]] = []
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            pts = [(int(x), int(y)) for (x, y) in box]
            conf = min(0.99, 0.6 + (cv2.contourArea(c) / (W * H)))
            out.append((pts, conf, None))
        return out

    def render(self, frame_bgr: NDArrayU8, result: List[Tuple[List[Tuple[int, int]], float, Optional[int]]]) -> NDArrayU8:
        from .overlay import draw_obb_bgr
        return draw_obb_bgr(frame_bgr, result, names=None)


# ---------------------------
# Builders (using model_registry)
# ---------------------------

def _label_from_override_or_pick(task: str, small: bool, override: Optional[Union[str, Path]]) -> str:
    if override is not None:
        return Path(override).name
    sel = pick_weight(task, small=small)
    try:
        return sel.name  # type: ignore[attr-defined]
    except Exception:
        return task.upper()

def build_detect(
    *,
    small: bool = True,
    conf: float = 0.25,
    iou: float = 0.45,
    override: Optional[Union[str, Path]] = LIVE_DETECT_OVERRIDE,
) -> TaskAdapter:
    try:
        label = _label_from_override_or_pick("detect", small, override)
        raw_model = load_detector(small=small, override=override)
        model = cast(_Predictor, raw_model)
        return _YOLODetect(model, label=label, conf=conf, iou=iou)
    except Exception:
        return _ContourDetect(conf=conf, iou=iou)

def build_heatmap(
    *,
    small: bool = True,
    override: Optional[Union[str, Path]] = LIVE_HEATMAP_OVERRIDE,
) -> TaskAdapter:
    try:
        label = _label_from_override_or_pick("heatmap", small, override)
        raw_model = load_segmenter(small=small, override=override)
        model = cast(_Predictor, raw_model)
        return _YOLOHeatmap(model, label=label)
    except Exception:
        return _LaplacianHeatmap()

def build_classify(
    *,
    small: bool = True,
    topk: int = 1,
    override: Optional[Union[str, Path]] = LIVE_CLASSIFY_OVERRIDE,
) -> TaskAdapter:
    try:
        label = _label_from_override_or_pick("classify", small, override)
        raw_model = load_classifier(small=small, override=override)
        model = cast(_Predictor, raw_model)
        return _YOLOClassify(model, label=label, topk=topk)
    except Exception:
        return _SimpleClassify(topk=topk)

def build_pose(
    *,
    small: bool = True,
    conf: float = 0.25,
    override: Optional[Union[str, Path]] = LIVE_POSE_OVERRIDE,
) -> TaskAdapter:
    try:
        label = _label_from_override_or_pick("pose", small, override)
        raw_model = load_pose(small=small, override=override)
        model = cast(_Predictor, raw_model)
        return _YOLOPose(model, label=label, conf=conf)
    except Exception:
        return _SimplePose()

def build_pse(
    *,
    small: bool = True,
    override: Optional[Union[str, Path]] = LIVE_PSE_OVERRIDE,
) -> TaskAdapter:
    """
    PSE is an alias of POSE: same model family, same overlay.
    """
    # Delegate directly to build_pose to guarantee identical behavior
    return build_pose(small=small, conf=0.25, override=override)

def build_obb(
    *,
    small: bool = True,
    conf: float = 0.25,
    iou: float = 0.45,
    override: Optional[Union[str, Path]] = LIVE_OBB_OVERRIDE,
) -> TaskAdapter:
    try:
        label = _label_from_override_or_pick("obb", small, override)
        raw_model = load_obb(small=small, override=override)
        model = cast(_Predictor, raw_model)
        return _YOLOOBB(model, label=label, conf=conf, iou=iou)
    except Exception:
        return _SimpleOBB(conf=conf)
