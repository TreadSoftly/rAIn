# projects/argos/panoptes/live/tasks.py
"""
Task adapters for live mode.

This revision routes LIVE detection/heatmap through the central model registry.
If a YOLO model cannot be loaded (weights missing or ultralytics not installed),
we gracefully fall back to the previous non-ML adapters so the demo still runs.

Available builders:
  - build_detect(*, small=True, conf=0.25, iou=0.45, override=None) -> TaskAdapter
  - build_heatmap(*, small=True, override=None)                      -> TaskAdapter
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Union, cast, Sequence
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

from ._types import NDArrayU8, Boxes, Names

# Use the central model registry (your single source of truth)
# (Ultralytics provides dynamic objects; we silence missing-stub warnings.)
from panoptes.model_registry import (  # type: ignore[reportMissingTypeStubs]
    MODEL_DIR,
    load_detector,
    load_segmenter,
    pick_weight,
)

# ─────────────────────────────────────────────────────────────────────
# Quick hard-force knobs for live experiments (edit these while testing)
#   • Set to an absolute path or MODEL_DIR / "file.pt"
#   • Leave as None to let model_registry choose via WEIGHT_PRIORITY
# ─────────────────────────────────────────────────────────────────────
LIVE_DETECT_OVERRIDE: Optional[Union[str, Path]]  = MODEL_DIR / "yolo12n.onnx"
LIVE_HEATMAP_OVERRIDE: Optional[Union[str, Path]] = MODEL_DIR / "yolo11n-seg.onnx"


class TaskAdapter(Protocol):
    def infer(self, frame_bgr: NDArrayU8) -> Any: ...
    def render(self, frame_bgr: NDArrayU8, result: Any) -> NDArrayU8: ...
    # Adapters set this so the HUD can show the active model.
    # Fallback adapters also set a descriptive label.
    label: str


# Minimal predictor protocol so static checkers know these attributes exist.
# (Ultralytics types are dynamic; we keep this intentionally broad.)
class _Predictor(Protocol):
    names: Any
    def predict(self, img: Any, *args: Any, **kwargs: Any) -> Any: ...


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


# ---------------------------
# Heatmap (non-ML fallback)
# ---------------------------

class _LaplacianHeatmap(TaskAdapter):
    def __init__(self) -> None:
        self.label = "laplacian-heatmap (no-ML)"

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
        # Prepare class names mapping (int -> label) for drawing
        self.names: dict[int, str] = {}
        names_attr: Any = getattr(model, "names", {})
        if isinstance(names_attr, dict):
            for k, v in cast(dict[Any, Any], names_attr).items():  # k, v are Any at runtime
                try:
                    self.names[int(k)] = str(v)
                except Exception:
                    continue
        elif isinstance(names_attr, (list, tuple)):
            for i, n in enumerate(cast(Sequence[object], names_attr)):
                self.names[i] = str(n)

    def infer(self, frame_bgr: NDArrayU8) -> list[tuple[int, int, int, int, float, Optional[int]]]:
        assert np is not None
        # IMPORTANT: predict() may flip channels in-place; give it a copy.
        inp = frame_bgr.copy() if hasattr(frame_bgr, "copy") else frame_bgr
        res_any: Any = self.model.predict(inp, conf=self.conf, iou=self.iou, verbose=False)
        if isinstance(res_any, (list, tuple)):
            seq: Sequence[object] = cast(Sequence[object], res_any)
            res_obj: object = seq[0]
        else:
            res_obj: object = cast(object, res_any)

        boxes: list[tuple[int, int, int, int, float, Optional[int]]] = []
        b_obj: object | None = getattr(res_obj, "boxes", None)
        if b_obj is None:
            return boxes

        xyxy_any: Any = getattr(b_obj, "xyxy", None)
        confs_any: Any = getattr(b_obj, "conf", None)
        clses_any: Any = getattr(b_obj, "cls", None)
        if xyxy_any is None or confs_any is None or clses_any is None:
            return boxes

        # Convert to numpy arrays (supports torch or plain lists)
        try:
            import torch  # type: ignore
            xyxy_np: Any = xyxy_any.detach().cpu().numpy()
            conf_np: Any = confs_any.detach().cpu().numpy().reshape(-1)
            cls_np: Any = clses_any.detach().cpu().numpy().astype(int).reshape(-1)
        except Exception:
            xyxy_np = np.asarray(xyxy_any, dtype=float)  # type: ignore[arg-type]
            conf_np = np.asarray(confs_any, dtype=float).reshape(-1)  # type: ignore[arg-type]
            try:
                cls_np = np.asarray(clses_any, dtype=int).reshape(-1)  # type: ignore[arg-type]
            except Exception:
                cls_np = np.zeros((xyxy_np.shape[0],), dtype=int)  # type: ignore[call-arg]

        for (x1, y1, x2, y2), conf_v, cls_v in zip(xyxy_np, conf_np, cls_np):
            boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf_v), int(cls_v)))
        return boxes

    def render(self, frame_bgr: NDArrayU8, result: Boxes) -> NDArrayU8:
        from .overlay import draw_boxes_bgr
        return draw_boxes_bgr(frame_bgr, result, names=self.names)


class _YOLOHeatmap(TaskAdapter):
    """YOLO-based segmentation adapter for live mode (instance masks → merged mask)."""

    def __init__(self, model: _Predictor, *, label: str, conf: float = 0.25) -> None:
        self.model = model
        self.conf = float(conf)
        self.label = label

    @staticmethod
    def _resize_nn(mask: NDArrayU8, new_hw: tuple[int, int]) -> NDArrayU8:
        """Nearest-neighbor resize for 2D masks (keeps OpenCV optional)."""
        assert np is not None
        H, W = new_hw
        h, w = int(mask.shape[0]), int(mask.shape[1])
        if (h, w) == (H, W):
            return mask
        y_idx = np.round(np.linspace(0, h - 1, H)).astype(int)
        x_idx = np.round(np.linspace(0, w - 1, W)).astype(int)
        return mask[y_idx[:, None], x_idx[None, :]]

    def infer(self, frame_bgr: NDArrayU8) -> NDArrayU8:
        assert np is not None
        # IMPORTANT: predict() may flip channels in-place; give it a copy.
        inp = frame_bgr.copy() if hasattr(frame_bgr, "copy") else frame_bgr
        res_any: Any = self.model.predict(inp, conf=self.conf, verbose=False)
        if isinstance(res_any, (list, tuple)):
            seq: Sequence[object] = cast(Sequence[object], res_any)
            res_obj: object = seq[0]
        else:
            res_obj: object = cast(object, res_any)

        H, W = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        mask_out = np.zeros((H, W), dtype=np.uint8)

        masks_any: Any = getattr(res_obj, "masks", None)
        if masks_any is None:
            return mask_out
        data_any: Any = getattr(masks_any, "data", None)
        if data_any is None:
            return mask_out

        try:
            import torch  # type: ignore
            m_np: Any = data_any.detach().cpu().numpy().astype(np.float32)  # type: ignore[assignment]
        except Exception:
            m_np = np.asarray(data_any, dtype=np.float32)  # type: ignore[arg-type]

        if m_np.ndim == 2:
            m_np = m_np[0:1, ...]  # (1, h, w)

        for m in m_np:
            # threshold first to keep semantics clear, then resize as uint8
            m_bin: NDArrayU8 = (m >= 0.5).astype(np.uint8)
            if tuple(m_bin.shape) != (H, W):
                m_bin = self._resize_nn(m_bin, (H, W))
            mask_out[m_bin > 0] = 255
        return mask_out

    def render(self, frame_bgr: NDArrayU8, result: NDArrayU8) -> NDArrayU8:
        from .overlay import draw_heatmap_bgr
        return draw_heatmap_bgr(frame_bgr, result)


# ---------------------------
# Builders (using model_registry)
# ---------------------------

def build_detect(
    *,
    small: bool = True,
    conf: float = 0.25,
    iou: float = 0.45,
    override: Optional[Union[str, Path]] = LIVE_DETECT_OVERRIDE,
) -> TaskAdapter:
    """
    Build a live detection adapter.

    - If `override` is provided (absolute path or basename under your model dir),
      we ask model_registry.load_detector(override=...).
    - Else we ask model_registry.load_detector(small=small) so the registry
      chooses the first existing file from WEIGHT_PRIORITY (or *_small list).
    - On any failure, we return the non-ML fallback.
    """
    try:
        # Compute a user-friendly label for HUD
        if override is not None:
            label = Path(override).name
        else:
            sel = pick_weight("detect", small=small)
            label = sel.name if sel is not None else "YOLO"
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
    """
    Build a live heatmap adapter (YOLO-Seg). On failure, fall back to Laplacian.
    """
    try:
        if override is not None:
            label = Path(override).name
        else:
            sel = pick_weight("heatmap", small=small)
            label = sel.name if sel is not None else "YOLO-Seg"
        raw_model = load_segmenter(small=small, override=override)
        model = cast(_Predictor, raw_model)
        return _YOLOHeatmap(model, label=label)
    except Exception:
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
