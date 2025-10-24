# projects/argos/panoptes/live/tasks.py
"""
Task adapters for live mode (detect/heatmap/classify/pose/pse/obb).

This module is designed to ALWAYS import cleanly (even when optional deps like
NumPy/OpenCV/Ultralytics are missing) and to give you a working live demo via
fast non-ML fallbacks. When your central model registry is available, the
YOLO-backed adapters will be used automatically.

Progress UX:
    • Uses the Halo-based spinners from panoptes.progress.
    • Builders open a single-line, fixed-width percent spinner during model
      loading (and gracefully no-op when nested or in non-TTY environments).
    • Additionally, a lightweight `running_task` wrapper is used around the
      actual load call (it auto-disables when nested under the percent spinner),
      and fallbacks emit a brief `simple_status` notice.
    • Env flags respected by the progress package:
        PANOPTES_SPINNER, PANOPTES_PROGRESS_WIDTH, PANOPTES_PROGRESS_TAIL,
        PANOPTES_PROGRESS_FINAL_NEWLINE.

Available builders:
  - build_detect(*, small=True, conf=0.25, iou=0.45, override=None)
  - build_heatmap(*, small=True, override=None)
  - build_classify(*, small=True, topk=1, override=None)
  - build_pose(*, small=True, conf=0.25, override=None)
  - build_pse(*, small=True, override=None)  # ALIAS of pose (same model/overlay)
  - build_obb(*, small=True, conf=0.25, iou=0.45, override=None)

Key change (2025-08-18):
    • Heatmap adapter now preserves *instance masks* from the -seg model and
      renders them with distinct colors (plus optional labels), instead of
      collapsing to one binary mask that color-maps to red.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Union, cast, Sequence, List, Tuple, Dict, Callable, Iterator, TYPE_CHECKING, NamedTuple
from pathlib import Path
from contextlib import contextmanager
import logging
import os

_TRUTHY = {"1", "true", "yes", "on"}


def _tensorrt_disabled() -> bool:
    return os.environ.get("ORT_DISABLE_TENSORRT", "").strip().lower() in _TRUTHY


def _filter_providers(providers: Sequence[str]) -> List[str]:
    if _tensorrt_disabled():
        return [p for p in providers if "tensorrt" not in p.lower()]
    return list(providers)
import math
import time
import warnings
from collections import Counter

# ─────────────────────────────────────────────────────────────────────
# Progress helpers (percent spinner preferred, safe fallbacks if missing)
# ─────────────────────────────────────────────────────────────────────
try:
    from panoptes.progress import percent_spinner as _progress_percent_spinner  # type: ignore[import]
except Exception:  # pragma: no cover
    class _NullSpinner:
        def __enter__(self) -> "_NullSpinner": return self
        def __exit__(self, *_: object) -> None: return None
        def update(self, **_: Any) -> "_NullSpinner": return self
    def _progress_percent_spinner(*_a: object, **_k: object) -> _NullSpinner:
        return _NullSpinner()

# We also import these convenience spinners and *use* them to avoid linter noise.
try:
    from panoptes.progress import simple_status as _progress_simple_status  # type: ignore[import]
    from panoptes.progress import running_task as _progress_running_task    # type: ignore[import]
except Exception:  # pragma: no cover
    def _progress_simple_status(*_a: object, **_k: object):
        class _N:
            def __enter__(self): return self
            def __exit__(self, *_: object) -> bool: return False
        return _N()
    def _progress_running_task(*_a: object, **_k: object):
        return _progress_simple_status()


# ─────────────────────────────────────────────────────────────────────
# Optional heavy deps (import-safe)
# ─────────────────────────────────────────────────────────────────────
try:
    import numpy as np
except Exception:
    np = None  # type: ignore

if TYPE_CHECKING:
    from panoptes.obb_types import OBBDetection as _OBBDetectionType  # type: ignore[import]
    from panoptes.smoothing import PolygonSmoother as _PolygonSmootherType  # type: ignore[import]
else:
    _OBBDetectionType = Any  # type: ignore[misc,assignment]
    _PolygonSmootherType = Any  # type: ignore[misc,assignment]

OBBDetectionType = _OBBDetectionType
PolygonSmootherType = _PolygonSmootherType

try:
    from panoptes.obb_types import OBBDetection as _RuntimeOBBDetection  # type: ignore[import]
except Exception:  # pragma: no cover
    _RuntimeOBBDetection = None  # type: ignore[misc,assignment]

try:
    from panoptes.smoothing import PolygonSmoother as _RuntimePolygonSmoother  # type: ignore[import]
except Exception:  # pragma: no cover
    _RuntimePolygonSmoother = None  # type: ignore[misc,assignment]

try:
    import onnxruntime as _ort  # type: ignore[import]
except Exception:
    _ort = None  # type: ignore

try:
    import onnx  # type: ignore[import]
    from onnx import TensorProto, helper  # type: ignore[import]
except Exception:
    onnx = None  # type: ignore
    helper = None  # type: ignore
    TensorProto = None  # type: ignore

if TYPE_CHECKING:
    from typing import Sequence as _SequenceAny

    class _OrtInferenceSession(Protocol):
        def run(
            self,
            output_names: Optional[_SequenceAny[str]],
            input_feed: Dict[str, Any],
            run_options: Any | None = None,
        ) -> _SequenceAny[Any]:
            ...

    ORTInferenceSessionType = _OrtInferenceSession
else:
    ORTInferenceSessionType = Any

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

# DType aliases that are safe for static analyzers even if numpy is Optional.
try:
    import numpy as _np_for_types
    f32: Any = _np_for_types.float32
    i64: Any = _np_for_types.int64
    u8: Any = _np_for_types.uint8
except Exception:  # pragma: no cover
    f32 = cast(Any, "float32")
    i64 = cast(Any, "int64")
    u8 = cast(Any, "uint8")

from ._types import NDArrayU8, Boxes, Names
from .preprocess import attach_preprocessor, use_preprocessor

try:
    from panoptes.runtime.resilient_yolo import (  # type: ignore[import]
        ResilientYOLO as _ResilientYOLO,
        configure_onnxruntime as _configure_onnxruntime,
    )
except Exception:  # pragma: no cover
    _ResilientYOLO = None

    def _configure_onnxruntime(
        *,
        threads: Optional[int] = None,
        execution: Optional[str] = None,
        enable_cuda_graph: bool = True,
        arena_strategy: Optional[str] = "kNextPowerOfTwo",
        dml_device_id: Optional[int] = None,
    ) -> None:
        return None

if _ResilientYOLO is not None:
    ResilientYOLORuntime = _ResilientYOLO  # type: ignore[assignment]
else:  # pragma: no cover - runtime fallback when resilient wrapper missing
    class ResilientYOLORuntime:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("panoptes.runtime.resilient_yolo not available")

        def prepare(self) -> None:
            raise RuntimeError("panoptes.runtime.resilient_yolo not available")

        def predict(self, frame: NDArrayU8, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("panoptes.runtime.resilient_yolo not available")

        def descriptor(self) -> str:
            raise RuntimeError("panoptes.runtime.resilient_yolo not available")

        def active_model(self) -> Any:
            raise RuntimeError("panoptes.runtime.resilient_yolo not available")

        def active_weight_path(self) -> Optional[Path]:
            raise RuntimeError("panoptes.runtime.resilient_yolo not available")

        def set_on_switch(self, *_: object, **__: object) -> None:
            raise RuntimeError("panoptes.runtime.resilient_yolo not available")

        def refresh_candidates(self, *_: object, **__: object) -> None:
            raise RuntimeError("panoptes.runtime.resilient_yolo not available")

        def current_metadata(self) -> Dict[str, object]:
            raise RuntimeError("panoptes.runtime.resilient_yolo not available")

        @property
        def backend(self) -> Optional[str]:
            raise RuntimeError("panoptes.runtime.resilient_yolo not available")


def configure_onnxruntime(
    *,
    threads: Optional[int] = None,
    execution: Optional[str] = None,
    enable_cuda_graph: bool = True,
    arena_strategy: Optional[str] = "kNextPowerOfTwo",
    dml_device_id: Optional[int] = None,
) -> None:
    """
    Invoke the shared ORT configuration helper when available.
    """
    _configure_onnxruntime(
        threads=threads,
        execution=execution,
        enable_cuda_graph=enable_cuda_graph,
        arena_strategy=arena_strategy,
        dml_device_id=dml_device_id,
    )

try:
    from panoptes.runtime.backend_probe import ort_available, OrtProbeStatus  # type: ignore[import]
except Exception:  # pragma: no cover
    class OrtProbeStatus(NamedTuple):
        ok: bool
        version: Optional[str]
        providers: Optional[List[str]]
        reason: Optional[str]
        providers_ok: bool
        expected_provider: Optional[str]
        healed: bool
        summary: Optional[Dict[str, Any]]

    def ort_available() -> OrtProbeStatus:
        return OrtProbeStatus(False, None, None, "backend probe unavailable", False, None, False, None)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
LOGGER.setLevel(logging.ERROR)

@contextmanager
def _silence_ultralytics() -> Iterator[None]:
    """
    Temporarily silence Ultralytics loggers so they don't spam the console.
    """
    targets = [
        "ultralytics",
        "ultralytics.yolo.engine.model",
        "ultralytics.yolo.utils",
        "ultralytics.nn.autobackend",
    ]
    saved: list[tuple[logging.Logger, int, bool]] = []
    for name in targets:
        logger = logging.getLogger(name)
        saved.append((logger, logger.level, logger.disabled))
        logger.disabled = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=DeprecationWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            yield
    finally:
        for logger, level, disabled in saved:
            logger.setLevel(level)
            logger.disabled = disabled

def _ensure_contiguous(frame: NDArrayU8) -> NDArrayU8:
    """Return a C-contiguous view of *frame* without unnecessary copies."""
    if np is None:
        return frame
    flags = getattr(frame, "flags", None)
    if flags is not None and getattr(flags, "c_contiguous", False):
        return frame
    return np.ascontiguousarray(frame)

def _warmup_wrapper(model: "ResilientYOLOProtocol", *, task: str, **kwargs: Any) -> bool:
    """
    Run a handful of dummy inferences so the first real frame is fast.

    We memoise by backend so switching from ONNX Runtime → Torch (or vice versa)
    replays the warmup, but repeated sessions on the same backend skip it.
    """
    if np is None:
        return False

    try:
        backend_name = getattr(model, "backend", None)
    except Exception:
        backend_name = None

    try:
        active_backend = (backend_name or "unknown").lower()
    except Exception:
        active_backend = "unknown"

    warmed: set[str] = getattr(model, "_argos_warmup_backends", set())
    if active_backend in warmed:
        return False

    setattr(model, "_argos_warmup_backends", warmed)

    dummy_frames: List[NDArrayU8] = [
        np.zeros((640, 640, 3), dtype=np.uint8),
        np.full((640, 640, 3), 64, dtype=np.uint8),
    ]
    try:
        rng = np.random.default_rng(0)
        dummy_frames.append(rng.integers(0, 255, size=(640, 640, 3), dtype=np.uint8))
    except Exception:
        dummy_frames.append(np.full((640, 640, 3), 192, dtype=np.uint8))

    try:
        for frame in dummy_frames:
            model.predict(frame, verbose=False, **kwargs)
    except Exception:
        return False

    warmed.add(active_backend)
    return True


_ort_nms_session: Optional[ORTInferenceSessionType] = None


def _get_ort_nms_session() -> Optional[ORTInferenceSessionType]:
    """
    Build (or reuse) a lightweight ONNX Runtime session that wraps NonMaxSuppression.
    """
    global _ort_nms_session
    if _ort_nms_session is not None:
        return _ort_nms_session
    if _ort is None or helper is None or TensorProto is None:
        return None
    try:
        boxes_vi = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, "num_boxes", 4])
        scores_vi = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, "num_boxes"])
        max_output_vi = helper.make_tensor_value_info("max_output_boxes_per_class", TensorProto.INT64, [1])
        iou_vi = helper.make_tensor_value_info("iou_threshold", TensorProto.FLOAT, [1])
        score_vi = helper.make_tensor_value_info("score_threshold", TensorProto.FLOAT, [1])
        indices_vi = helper.make_tensor_value_info("indices", TensorProto.INT64, [None, 3])

        nms_node = helper.make_node(
            "NonMaxSuppression",
            [
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
            ["indices"],
        )
        graph = helper.make_graph(
            [nms_node],
            "argos_inline_nms",
            [boxes_vi, scores_vi, max_output_vi, iou_vi, score_vi],
            [indices_vi],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
        # Older onnxruntime builds (including the one bundled with Argos) cap the supported
        # IR version at 10, so we down-level the lightweight graph before instantiating it.
        try:
            setattr(model, "ir_version", 7)
        except Exception:
            model.ir_version = 7  # type: ignore[attr-defined]
        session_options: Any = _ort.SessionOptions()  # type: ignore[attr-defined]
        session_options.log_severity_level = 3  # type: ignore[attr-defined]
        raw_providers = cast(Sequence[str], _ort.get_available_providers())  # type: ignore[attr-defined]
        providers = _filter_providers(raw_providers)
        _ort_nms_session = _ort.InferenceSession(  # type: ignore[attr-defined]
            model.SerializeToString(),
            sess_options=session_options,
            providers=providers,
        )
        return _ort_nms_session
    except Exception:
        return None


def _candidate_backend(path: Path) -> str:
    suffix = path.suffix.lower()
    stem = path.stem.lower()
    if suffix in {".engine", ".plan"}:
        return "tensorrt"
    if suffix == ".onnx":
        if "fp16" in stem or "half" in stem:
            return "onnx-fp16"
        return "onnx"
    if suffix in {".xml", ".bin"}:
        return "openvino"
    if suffix in {".torchscript"}:
        return "torch"
    return "torch"


def _preferred_backend_order(preference: str) -> List[str]:
    pref = (preference or "auto").lower()
    ort_status = ort_available()
    providers_l = [p.lower() for p in (ort_status.providers or [])]
    has_cuda = any("cuda" in p for p in providers_l)
    has_dml = any("dml" in p or "directml" in p for p in providers_l)
    disable_tensorrt = os.environ.get("ORT_DISABLE_TENSORRT", "").strip().lower() in _TRUTHY

    order: List[str]
    if pref == "tensorrt":
        if disable_tensorrt:
            order = ["onnx", "torch"]
        else:
            order = ["tensorrt", "onnx", "torch"]
    elif pref == "ort":
        order = ["onnx"]
        if not disable_tensorrt:
            order.append("tensorrt")
        order.append("torch")
    elif pref == "torch":
        order = ["torch", "onnx"]
        if not disable_tensorrt:
            order.append("tensorrt")
    else:
        order = []
        if not disable_tensorrt:
            order.append("tensorrt")
        if ort_status.ok:
            if has_cuda or has_dml:
                order.append("onnx-gpu")
            order.append("onnx")
        order.append("torch")

    # Add defaults to guarantee coverage
    base: List[str] = ["onnx-gpu", "onnx", "torch"]
    if not disable_tensorrt:
        base.insert(0, "tensorrt")
    seen: set[str] = set()
    final: List[str] = []
    for item in order + base:
        if item not in seen:
            final.append(item)
            seen.add(item)
    return final


def _sort_candidates(
    candidates: Sequence[Path],
    *,
    backend_preference: str,
    sticky_first: Optional[Path] = None,
) -> List[Path]:
    base_candidates = [Path(c) for c in candidates]
    meta_sorted = rank_candidates_for_backend(base_candidates, backend_preference)
    meta_rank_map: Dict[str, int] = {
        str(Path(p)): idx for idx, p in enumerate(meta_sorted)
    }
    order = _preferred_backend_order(backend_preference)
    if sticky_first is not None:
        try:
            sticky_norm = Path(sticky_first).expanduser().resolve(strict=False)
        except Exception:
            sticky_norm = Path(sticky_first)
    else:
        sticky_norm = None

    def backend_rank(path: Path, idx: int) -> Tuple[int, float, int]:
        meta_idx = meta_rank_map.get(str(path), len(meta_rank_map) + idx)
        backend = _candidate_backend(path)
        base = backend
        bonus = 0.0
        if backend == "onnx-fp16":
            base = "onnx"
            bonus = -0.1
        elif backend == "onnx":
            base = "onnx"
        if backend == "onnx-fp16" and "cuda" not in backend_preference:
            bonus -= 0.05
        try:
            base_idx = order.index(base)
        except ValueError:
            base_idx = len(order)
        return (meta_idx, base_idx + bonus, idx)

    enumerated = list(enumerate(base_candidates))
    sorted_enum = sorted(enumerated, key=lambda item: backend_rank(item[1], item[0]))
    if sticky_norm is not None:
        primary = None
        remainder: List[tuple[int, Path]] = []
        for item in sorted_enum:
            _, candidate = item
            try:
                candidate_norm = candidate.expanduser().resolve(strict=False)
            except Exception:
                candidate_norm = candidate
            if primary is None and candidate_norm == sticky_norm:
                primary = item
                continue
            remainder.append(item)
        if primary is not None:
            sorted_enum = [primary] + remainder
    return [entry for _, entry in sorted_enum]


def _acquire_wrapper(
    task: str,
    candidates: Sequence[Path],
    *,
    conf: float,
    hud_callback: Optional[Callable[[str], None]],
    backend_preference: str,
    ort_threads: Optional[int],
    ort_execution: Optional[str],
) -> "ResilientYOLOProtocol":
    configure_onnxruntime(threads=ort_threads, execution=ort_execution)
    metadata_map: Dict[Path, Dict[str, object]] = {}
    for cand in candidates:
        try:
            metadata_map[cand] = artifact_metadata(cand)
        except Exception as exc:
            metadata_map[cand] = {"analysis_error": f"{type(exc).__name__}:{exc}"}
    wrapper = cast(
        ResilientYOLOProtocol,
        ResilientYOLORuntime(
            candidates,
            task=task,
            conf=conf,
            on_switch=hud_callback,
            metadata=metadata_map,
        ),
    )
    LOGGER.debug(
        "Live task %s created wrapper using candidates=%s backend_pref=%s",
        task,
        [str(p) for p in candidates],
        backend_preference,
    )
    return wrapper
# ─────────────────────────────────────────────────────────────────────
# Central model registry (single source of truth) — guarded import
# ─────────────────────────────────────────────────────────────────────
try:
    from panoptes.model_registry import (  # type: ignore[reportMissingTypeStubs]
        load_detector,     # type: ignore[no-redef]
        load_segmenter,    # type: ignore[no-redef]
        load_classifier,   # type: ignore[no-redef]
        load_pose,         # type: ignore[no-redef]
        load_obb,          # type: ignore[no-redef]
        candidate_weights, # type: ignore[no-redef]
        pick_weight,       # type: ignore[no-redef]
        rank_candidates_for_backend,  # type: ignore[no-redef]
        artifact_metadata,            # type: ignore[no-redef]
        select_postprocess_strategy,  # type: ignore[no-redef]
    )
except Exception:  # pragma: no cover

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

    def candidate_weights(*_a: object, **_k: object) -> list[Path]:  # type: ignore[no-redef]
        return []

    def pick_weight(*_a: object, **_k: object) -> Any:  # type: ignore[no-redef]
        return None

    def rank_candidates_for_backend(
        *_a: object, **_k: object
    ) -> list[Path]:  # type: ignore[no-redef]
        return []

    def artifact_metadata(*_a: object, **_k: object) -> dict[str, object]:  # type: ignore[no-redef]
        return {}

    def select_postprocess_strategy(*_a: object, **_k: object) -> dict[str, object]:  # type: ignore[no-redef]
        return {"nms": "torch"}

# ─────────────────────────────────────────────────────────────────────
# Live overrides — leave as None so the registry fully controls weights
# ─────────────────────────────────────────────────────────────────────
LIVE_DETECT_OVERRIDE: Optional[Union[str, Path]]   = None
LIVE_HEATMAP_OVERRIDE: Optional[Union[str, Path]]  = None
LIVE_CLASSIFY_OVERRIDE: Optional[Union[str, Path]] = None
LIVE_POSE_OVERRIDE: Optional[Union[str, Path]]     = None
LIVE_PSE_OVERRIDE: Optional[Union[str, Path]]      = None
LIVE_OBB_OVERRIDE: Optional[Union[str, Path]]      = None



def _names_from_model(model: Any) -> Dict[int, str]:
    names: Dict[int, str] = {}
    if model is None:
        return names
    names_attr: Any = getattr(model, "names", {})
    if isinstance(names_attr, dict):
        for k, v in cast(Dict[Any, Any], names_attr).items():
            try:
                names[int(k)] = str(v)
            except Exception:
                continue
    elif isinstance(names_attr, (list, tuple)):
        for idx, val in enumerate(cast(Sequence[object], names_attr)):
            names[idx] = str(val)
    return names
class TaskAdapter(Protocol):
    """Common protocol for live task adapters."""
    def infer(self, frame_bgr: NDArrayU8) -> Any: ...
    def render(self, frame_bgr: NDArrayU8, result: Any) -> NDArrayU8: ...
    label: str  # for HUD


# Minimal predictor protocol so static checkers know these attributes exist.
class _Predictor(Protocol):
    names: Any
    def predict(self, img: Any, *args: Any, **kwargs: Any) -> Any: ...

class ResilientYOLOProtocol(Protocol):
    def prepare(self) -> None: ...
    def predict(self, frame: NDArrayU8, *args: Any, **kwargs: Any) -> Any: ...
    def descriptor(self) -> str: ...
    def active_model(self) -> _Predictor: ...
    def active_weight_path(self) -> Optional[Path]: ...
    @property
    def backend(self) -> Optional[str]: ...
    def set_on_switch(self, callback: Optional[Callable[[str], None]]) -> None: ...
    def refresh_candidates(
        self,
        candidates: Sequence[Path],
        metadata: Optional[Dict[Path, Dict[str, object]]] = None,
    ) -> None: ...
    def current_metadata(self) -> Dict[str, object]: ...


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

    def __init__(
        self,
        model: ResilientYOLOProtocol,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        nms_mode: str = "auto",
    ) -> None:
        self.model = model
        self.conf = float(conf)
        self.iou = float(iou)
        self.names = _names_from_model(model.active_model())
        self.label = model.descriptor()
        self._strategy_cache: Dict[str, object] = {}
        self._suppressed_counts: Dict[int, int] = {}
        mode_norm = (nms_mode or "auto").strip().lower()
        self._nms_override = mode_norm if mode_norm in {"auto", "graph", "torch", "ort"} else "auto"
        self._last_nms_mode: str = "auto"

    def current_label(self) -> str:
        return self.model.descriptor()

    @staticmethod
    def _coerce_float(value: object, default: float) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return default

    @staticmethod
    def _coerce_int(value: object, default: int) -> int:
        if isinstance(value, (int, float)):
            try:
                return int(value)
            except Exception:
                return default
        return default

    def _postprocess_strategy(self) -> Dict[str, object]:
        backend = getattr(self.model, "backend", None)
        active_weight: Optional[Path]
        try:
            active_weight = self.model.active_weight_path()
        except Exception:
            active_weight = None
        metadata: Dict[str, object] = {}
        try:
            metadata = self.model.current_metadata()
        except Exception:
            metadata = {}
        strategy = select_postprocess_strategy("detect", backend or "auto", weight=active_weight)
        merged: Dict[str, object] = dict(strategy)
        if metadata:
            merged.update(metadata)
        weight_label = getattr(self.model, "weight_label", None)
        if weight_label:
            merged.setdefault("weight", weight_label)
        override = self._nms_override
        backend_label = str(backend or "auto").lower()
        if override != "auto":
            if override == "graph" and not bool(merged.get("nms_in_graph")):
                ort_session = _get_ort_nms_session()
                if ort_session is not None:
                    merged["nms"] = "ort"
                    merged["nms_override"] = True
                else:
                    LOGGER.warning(
                        "live.nms.override.unavailable task=%s weight=%s override=graph",
                        "detect",
                        merged.get("weight"),
                    )
                    merged["nms_override"] = "graph_unavailable"
            else:
                merged["nms"] = override
                merged["nms_override"] = True
        nms_mode = str(merged.get("nms") or "torch").lower()
        if nms_mode != "graph":
            backend_hint = str(merged.get("backend") or backend_label).lower()
            if backend_hint in {"onnxruntime", "onnx", "onnx-fp16", "onnx-gpu", "tensorrt"}:
                if _get_ort_nms_session() is not None:
                    nms_mode = "ort"
                    merged["nms"] = "ort"
        merged["nms"] = nms_mode
        self._last_nms_mode = nms_mode
        self._strategy_cache = merged
        return merged

    def current_nms_mode(self) -> str:
        return self._last_nms_mode

    def last_strategy(self) -> Dict[str, object]:
        return dict(self._strategy_cache)

    def nms_statistics(self) -> Dict[int, int]:
        return dict(self._suppressed_counts)

    def _boxes_from_numpy(
        self,
        xyxy_any: Any,
        confs_any: Any,
        clses_any: Any,
    ) -> List[Tuple[int, int, int, int, float, Optional[int]]]:
        np_ = cast(Any, np)
        assert np_ is not None
        xyxy_np = np_.asarray(_to_numpy(xyxy_any, dtype=f32), dtype=f32)
        conf_np = np_.asarray(_to_numpy(confs_any, dtype=f32), dtype=f32).reshape(-1)
        try:
            cls_np = np_.asarray(_to_numpy(clses_any, dtype=i64), dtype=i64).reshape(-1)
        except Exception:
            shape = getattr(xyxy_np, "shape", ())
            if shape and len(shape) > 0:
                length = int(shape[0])
            else:
                size_attr = getattr(xyxy_np, "size", None)
                if size_attr is not None:
                    length = int(size_attr)
                else:
                    try:
                        length = int(len(xyxy_np))
                    except Exception:
                        length = 0
            cls_np = np_.zeros((max(length, 0),), dtype=int)

        boxes: List[Tuple[int, int, int, int, float, Optional[int]]] = []
        for (x1, y1, x2, y2), conf_v, cls_v in zip(xyxy_np, conf_np, cls_np):
            boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf_v), int(cls_v)))
        return boxes

    def _ort_nms(
        self,
        xyxy_any: Any,
        confs_any: Any,
        clses_any: Any,
        strategy: Dict[str, object],
    ) -> List[Tuple[int, int, int, int, float, Optional[int]]]:
        if np is None:
            return self._torch_gpu_nms(xyxy_any, confs_any, clses_any, strategy)
        session = _get_ort_nms_session()
        if session is None:
            return self._torch_gpu_nms(xyxy_any, confs_any, clses_any, strategy)

        np_ = cast(Any, np)
        xyxy_np = np_.asarray(_to_numpy(xyxy_any, dtype=f32), dtype=np.float32)
        scores_np = np_.asarray(_to_numpy(confs_any, dtype=f32), dtype=np.float32).reshape(-1)
        try:
            cls_np = np_.asarray(_to_numpy(clses_any, dtype=i64), dtype=np.int64).reshape(-1)
        except Exception:
            cls_np = np_.zeros((max(int(getattr(xyxy_np, "shape", [0])[0]), 0),), dtype=np.int64)

        if xyxy_np.ndim != 2:
            xyxy_np = xyxy_np.reshape(-1, 4)

        conf_thres = self._coerce_float(strategy.get("within_graph_conf_thres"), self.conf)
        conf_thres = self._coerce_float(strategy.get("conf"), conf_thres)
        mask = scores_np >= conf_thres
        if not bool(np_.any(mask)):
            return []

        xyxy_sel = xyxy_np[mask]
        scores_sel = scores_np[mask]
        cls_sel = cls_np[mask]

        order = np_.argsort(scores_sel)[::-1]
        pre_nms_limit = self._coerce_int(strategy.get("max_candidates"), 0)
        if pre_nms_limit <= 0:
            max_det_hint = self._coerce_int(strategy.get("max_det"), 300)
            pre_nms_limit = max(max_det_hint * 4, 512)
        if order.size > pre_nms_limit > 0:
            order = order[:pre_nms_limit]

        xyxy_ordered = xyxy_sel[order]
        scores_ordered = scores_sel[order]
        cls_ordered = cls_sel[order]

        max_det = self._coerce_int(strategy.get("max_det"), 300)
        scores_input = scores_ordered.reshape(1, 1, -1).astype(np.float32, copy=False)
        boxes_input = xyxy_ordered.reshape(1, -1, 4).astype(np.float32, copy=False)
        max_output = np_.array([max_det if max_det > 0 else xyxy_ordered.shape[0]], dtype=np.int64)
        iou_thresh = np_.array([float(self.iou)], dtype=np.float32)
        score_thresh = np_.array([float(conf_thres)], dtype=np.float32)

        try:
            ort_outputs = session.run(
                None,
                {
                    "boxes": boxes_input,
                    "scores": scores_input,
                    "max_output_boxes_per_class": max_output,
                    "iou_threshold": iou_thresh,
                    "score_threshold": score_thresh,
                },
            )
        except Exception:
            return self._torch_gpu_nms(xyxy_any, confs_any, clses_any, strategy)

        if not ort_outputs:
            return []

        indices_any = ort_outputs[0]
        if isinstance(indices_any, dict):
            indices_values = list(cast(Dict[Any, Any], indices_any).values())
            indices_arr = np_.asarray(indices_values, dtype=np.int64)
        else:
            indices_arr = np_.asarray(indices_any, dtype=np.int64)

        if indices_arr.size == 0:
            return []

        if indices_arr.ndim == 1:
            if indices_arr.size % 3 != 0:
                return self._torch_gpu_nms(xyxy_any, confs_any, clses_any, strategy)
            indices_arr = indices_arr.reshape(-1, 3)

        if indices_arr.ndim != 2 or indices_arr.shape[1] < 3:
            return self._torch_gpu_nms(xyxy_any, confs_any, clses_any, strategy)

        # Ensure indices are unique while preserving selection order.
        unique_vals, first_idx = np_.unique(indices_arr[:, 2], return_index=True)
        order_by_first = np_.argsort(first_idx)
        keep_idx_ordered = unique_vals[order_by_first]
        keep_idx_ordered = np_.clip(keep_idx_ordered, 0, xyxy_ordered.shape[0] - 1)
        if max_det > 0 and keep_idx_ordered.size > max_det:
            keep_idx_ordered = keep_idx_ordered[:max_det]

        suppressed = int(xyxy_ordered.shape[0] - keep_idx_ordered.size)
        if suppressed > 0:
            all_cls_vals = [int(val) for val in cls_ordered.tolist()]
            kept_cls_vals = [int(cls_ordered[idx]) for idx in keep_idx_ordered.tolist()]
            before_counts = Counter(all_cls_vals)
            after_counts = Counter(kept_cls_vals)
            for cls_idx, total_before in before_counts.items():
                removed = total_before - after_counts.get(cls_idx, 0)
                if removed > 0:
                    self._suppressed_counts[cls_idx] = self._suppressed_counts.get(cls_idx, 0) + int(removed)
            self._suppressed_counts[-1] = self._suppressed_counts.get(-1, 0) + suppressed

        boxes: List[Tuple[int, int, int, int, float, Optional[int]]] = []
        for idx in keep_idx_ordered.tolist():
            x1, y1, x2, y2 = xyxy_ordered[idx]
            conf_v = float(scores_ordered[idx])
            cls_v = int(cls_ordered[idx])
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf_v, cls_v))
        return boxes
    def _torch_gpu_nms(
        self,
        xyxy_any: Any,
        confs_any: Any,
        clses_any: Any,
        strategy: Dict[str, object],
    ) -> List[Tuple[int, int, int, int, float, Optional[int]]]:
        try:
            import torch  # type: ignore
            from torchvision.ops import nms as tv_nms  # type: ignore
        except Exception:
            return self._boxes_from_numpy(xyxy_any, confs_any, clses_any)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xyxy_t = torch.as_tensor(xyxy_any, dtype=torch.float32, device=device)
        scores_t = torch.as_tensor(confs_any, dtype=torch.float32, device=device).reshape(-1)
        cls_t = torch.as_tensor(clses_any, dtype=torch.int64, device=device).reshape(-1)

        if xyxy_t.ndim != 2:
            xyxy_t = xyxy_t.reshape(-1, 4)

        conf_thres = self._coerce_float(strategy.get("within_graph_conf_thres"), self.conf)
        conf_thres = self._coerce_float(strategy.get("conf"), conf_thres)
        mask = scores_t >= conf_thres
        if not torch.any(mask):
            return []

        xyxy_t = xyxy_t[mask]
        scores_t = scores_t[mask]
        cls_t = cls_t[mask]

        order = torch.argsort(scores_t, descending=True)
        pre_nms_limit = self._coerce_int(strategy.get("max_candidates"), 0)
        if pre_nms_limit <= 0:
            max_det_hint = self._coerce_int(strategy.get("max_det"), 300)
            pre_nms_limit = max(max_det_hint * 4, 512)
        if order.numel() > pre_nms_limit:
            order = order[:pre_nms_limit]

        xyxy_t = xyxy_t[order]
        scores_t = scores_t[order]
        cls_t = cls_t[order]

        keep = tv_nms(xyxy_t, scores_t, float(self.iou))
        max_det = self._coerce_int(strategy.get("max_det"), 300)
        if keep.numel() > max_det:
            keep = keep[:max_det]

        suppressed = int(xyxy_t.shape[0] - keep.numel())
        if suppressed > 0:
            kept_cls = cls_t[keep]
            all_cls_vals = [int(val.item()) for val in cls_t.to(torch.int64).cpu().view(-1)]
            kept_cls_vals = [int(val.item()) for val in kept_cls.to(torch.int64).cpu().view(-1)]
            before_counts = Counter(all_cls_vals)
            after_counts = Counter(kept_cls_vals)
            for cls_idx, total_before in before_counts.items():
                removed = total_before - after_counts.get(cls_idx, 0)
                if removed > 0:
                    self._suppressed_counts[cls_idx] = self._suppressed_counts.get(cls_idx, 0) + int(removed)
            self._suppressed_counts[-1] = self._suppressed_counts.get(-1, 0) + suppressed

        xyxy_list: List[List[float]] = []
        for bbox_tensor in xyxy_t[keep].to(torch.float32).cpu():
            coords = [float(coord.item()) for coord in bbox_tensor.view(-1)]
            xyxy_list.append(coords)

        scores_list: List[float] = [
            float(score.item()) for score in scores_t[keep].to(torch.float32).cpu().view(-1)
        ]
        cls_list: List[int] = [
            int(cls_val.item()) for cls_val in cls_t[keep].to(torch.int64).cpu().view(-1)
        ]

        boxes: List[Tuple[int, int, int, int, float, Optional[int]]] = []
        for idx, conf_v in enumerate(scores_list):
            x1, y1, x2, y2 = xyxy_list[idx]
            cls_v = cls_list[idx]
            boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf_v), int(cls_v)))
        return boxes

    def infer(self, frame_bgr: NDArrayU8) -> List[Tuple[int, int, int, int, float, Optional[int]]]:
        np_ = cast(Any, np)
        assert np_ is not None
        inp = _ensure_contiguous(frame_bgr)
        with use_preprocessor(getattr(self, "_preprocessor", None)):
            res_any: Any = self.model.predict(inp, conf=self.conf, iou=self.iou, verbose=False)
        if isinstance(res_any, (list, tuple)):
            res_obj: object = cast(Sequence[object], res_any)[0]
        else:
            res_obj = cast(object, res_any)

        b_obj: Optional[object] = getattr(res_obj, "boxes", None)
        if b_obj is None:
            return []

        xyxy_any: Any = getattr(b_obj, "xyxy", None)
        confs_any: Any = getattr(b_obj, "conf", None)
        clses_any: Any = getattr(b_obj, "cls", None)
        if xyxy_any is None or confs_any is None or clses_any is None:
            return []

        self.names = _names_from_model(self.model.active_model())
        self.label = self.model.descriptor()

        strategy = self._postprocess_strategy()
        nms_mode = str(strategy.get("nms") or "torch").lower()
        if nms_mode == "graph":
            return self._boxes_from_numpy(xyxy_any, confs_any, clses_any)
        if nms_mode == "ort":
            return self._ort_nms(xyxy_any, confs_any, clses_any, strategy)
        return self._torch_gpu_nms(xyxy_any, confs_any, clses_any, strategy)

    def render(self, frame_bgr: NDArrayU8, result: Boxes) -> NDArrayU8:
        from .overlay import draw_boxes_bgr
        return draw_boxes_bgr(frame_bgr, result, names=self.names)


class _YOLOHeatmap(TaskAdapter):
    """
    YOLO-based segmentation adapter for live mode.

    Returns *per-instance* masks with their conf and class so the renderer can
    color each instance distinctly (and label it), matching offline heatmap behavior.
    """

    _MAX_INSTANCES: int = 12
    _MIN_AREA_PX: int = 96
    _MIN_AREA_FRAC: float = 5e-4

    def __init__(self, model: ResilientYOLOProtocol, *, conf: float = 0.25) -> None:
        self.model = model
        self.conf = float(conf)
        self.names = _names_from_model(model.active_model())
        self.label = model.descriptor()
        from .overlay import new_instance_color_tracker, new_mask_smoother
        self._mask_tracker = new_instance_color_tracker()
        self._mask_smoother = new_mask_smoother()
        # Cache resize index arrays keyed by (src_h, src_w, dst_h, dst_w) to avoid
        # recomputing np.linspace for every frame.
        self._resize_cache: dict[Tuple[int, int, int, int], Tuple[Any, Any]] = {}

    def current_label(self) -> str:
        return self.model.descriptor()

    def _resize_nn(self, mask: NDArrayU8, new_hw: Tuple[int, int]) -> NDArrayU8:
        np_ = cast(Any, np)
        assert np_ is not None
        H, W = new_hw
        h, w = int(mask.shape[0]), int(mask.shape[1])
        if (h, w) == (H, W):
            return mask
        cache = self._resize_cache
        key = (h, w, H, W)
        cached = cache.get(key)
        if cached is None:
            y_idx = np_.round(np_.linspace(0, h - 1, H)).astype(int)
            x_idx = np_.round(np_.linspace(0, w - 1, W)).astype(int)
            cache[key] = (y_idx, x_idx)
        else:
            y_idx, x_idx = cached
        return mask[y_idx[:, None], x_idx[None, :]]

    def _refine_mask(self, mask: NDArrayU8) -> NDArrayU8:
        if cv2 is None:
            return mask
        np_ = cast(Any, np)
        area = float(np_.count_nonzero(mask))
        if area <= 0:
            return mask
        span = int(max(3, min(15, round(math.sqrt(area) / 3.0))))
        if span % 2 == 0:
            span += 1
        mask_u8 = mask.astype(np_.uint8, copy=False)
        if mask_u8.max() <= 1:
            mask_u8 = mask_u8 * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (span, span))
        refined = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)
        if span >= 5:
            refined = cv2.GaussianBlur(refined, (span, span), 0)
        _, refined = cv2.threshold(refined, 96, 255, cv2.THRESH_BINARY)
        refined_bool = refined > 0
        if refined_bool.any():
            refined_u8 = refined_bool.astype(np_.uint8, copy=False)
            num_labels, labels = cv2.connectedComponents(refined_u8)
            if num_labels > 2:
                counts = np_.bincount(labels.reshape(-1))[1:]
                if counts.size:
                    total = counts.sum()
                    keep_threshold = max(1, int(total * 0.05))
                    keep_mask = np_.zeros_like(labels, dtype=bool)
                    for idx, count in enumerate(counts, start=1):
                        if count >= keep_threshold:
                            keep_mask |= labels == idx
                    if keep_mask.any():
                        refined_bool = keep_mask
        return refined_bool.astype(u8, copy=False)

    def infer(self, frame_bgr: NDArrayU8) -> List[Tuple[NDArrayU8, float, Optional[int]]]:
        """
        Returns: list of (mask_u8(H,W), conf: float, cls_id: Optional[int])
        """
        np_ = cast(Any, np)
        assert np_ is not None
        inp = _ensure_contiguous(frame_bgr)
        with use_preprocessor(getattr(self, "_preprocessor", None)):
            res_any: Any = self.model.predict(inp, conf=self.conf, verbose=False)
        if isinstance(res_any, (list, tuple)):
            res_obj: object = cast(Sequence[object], res_any)[0]
        else:
            res_obj = cast(object, res_any)

        H, W = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        total_px = max(1, H * W)

        masks_any: Any = getattr(res_obj, "masks", None)
        if masks_any is None:
            return []

        data_any: Any = getattr(masks_any, "data", None)
        if data_any is None:
            return []

        boxes_obj: Any = getattr(res_obj, "boxes", None)
        if boxes_obj is None:
            return []
        try:
            num_boxes = int(len(boxes_obj))
        except Exception:
            num_boxes = 0
        if num_boxes <= 0:
            return []

        m_np: Any = _to_numpy(data_any, dtype=f32)  # (N,h,w) float in [0,1] typically
        if getattr(m_np, "ndim", 0) == 2:
            m_np = m_np[0:1, ...]  # (1, h, w)

        # Optional: confidences + classes (aligned with masks)
        confs_seq: Optional[Sequence[float]] = None
        clses_seq: Optional[Sequence[int]] = None
        try:
            confs_seq = cast(
                Sequence[float],
                _to_numpy(getattr(boxes_obj, "conf", None), dtype=f32).reshape(-1),
            )
        except Exception:
            confs_seq = None
        try:
            clses_seq = cast(
                Sequence[int],
                _to_numpy(getattr(boxes_obj, "cls", None), dtype=i64).reshape(-1).astype(int),
            )
        except Exception:
            clses_seq = None

        out: List[Tuple[NDArrayU8, float, Optional[int]]] = []
        num_masks = int(getattr(m_np, "shape", (0,))[0] or 0)
        count = min(num_masks, num_boxes)
        if count <= 0:
            return []
        max_keep = self._MAX_INSTANCES
        need_resize = tuple(getattr(m_np, "shape", (0, 0, 0))[1:3]) != (H, W)
        for i in range(count):
            conf_v = float(confs_seq[i]) if confs_seq is not None and i < len(confs_seq) else 0.0
            if conf_v < self.conf:
                continue
            m = m_np[i]
            m_bin: NDArrayU8 = (m >= 0.5).astype(u8, copy=False)
            m_bin = self._refine_mask(m_bin)
            if need_resize:
                m_bin = self._resize_nn(m_bin, (H, W))
            area = float(np_.count_nonzero(m_bin))
            if area < self._MIN_AREA_PX or (area / total_px) < self._MIN_AREA_FRAC:
                continue
            cls_v: Optional[int]
            if clses_seq is not None and i < len(clses_seq):
                cls_v = int(clses_seq[i])
            else:
                cls_v = None
            out.append((m_bin, conf_v, cls_v))
            if len(out) >= max_keep:
                break

        if not out:
            return []
        out.sort(key=lambda item: item[1], reverse=True)
        return out[:max_keep]

    def render(self, frame_bgr: NDArrayU8, result: List[Tuple[NDArrayU8, float, Optional[int]]]) -> NDArrayU8:
        # Per-instance compositing with distinct colors + optional labels
        from .overlay import draw_masks_bgr
        return draw_masks_bgr(
            frame_bgr,
            result,
            names=self.names,
            alpha=0.35,
            tracker=self._mask_tracker,
            smoother=self._mask_smoother,
        )


# ---------------------------
# CLASSIFY (YOLO-cls) + fallback
# ---------------------------

class _YOLOClassify(TaskAdapter):
    def __init__(self, model: ResilientYOLOProtocol, *, topk: int = 1) -> None:
        self.model = model
        self.label = model.descriptor()
        self.topk = max(1, int(topk))
        self.names = _names_from_model(model.active_model())

    def current_label(self) -> str:
        return self.model.descriptor()

    def infer(self, frame_bgr: NDArrayU8) -> List[Tuple[str, float]]:
        np_ = cast(Any, np)
        assert np_ is not None
        inp = _ensure_contiguous(frame_bgr)
        with use_preprocessor(getattr(self, "_preprocessor", None)):
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
    def __init__(self, model: ResilientYOLOProtocol, *, conf: float = 0.25) -> None:
        self.model = model
        self.conf = float(conf)
        self.label = model.descriptor()
        self.names = _names_from_model(model.active_model())

    def current_label(self) -> str:
        return self.model.descriptor()

    def infer(self, frame_bgr: NDArrayU8) -> List[List[List[float]]]:
        np_ = cast(Any, np)
        assert np_ is not None
        inp = _ensure_contiguous(frame_bgr)
        with use_preprocessor(getattr(self, "_preprocessor", None)):
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

def build_pse(
    *,
    small: bool = True,
    override: Optional[Union[str, Path]] = LIVE_PSE_OVERRIDE,
    input_size: Optional[Tuple[int, int]] = None,
    preprocess_device: str = "cpu",
    warmup: bool = True,
    backend: str = "auto",
    ort_threads: Optional[int] = None,
    ort_execution: Optional[str] = None,
    hud_callback: Optional[Callable[[str], None]] = None,
) -> TaskAdapter:
    """
    PSE is an alias of POSE: same model family, same overlay.
    """
    return build_pose(
        small=small,
        conf=0.25,
        override=override,
        input_size=input_size,
        preprocess_device=preprocess_device,
        warmup=warmup,
        backend=backend,
        ort_threads=ort_threads,
        ort_execution=ort_execution,
        hud_callback=hud_callback,
    )


# ---------------------------
# OBB (YOLO-obb) + fallback
# ---------------------------

def _rotrect_to_pts(cx: float, cy: float, w: float, h: float, theta_deg: float) -> List[Tuple[float, float]]:
    rad = math.radians(theta_deg)
    c, s = math.cos(rad), math.sin(rad)
    hw, hh = w / 2.0, h / 2.0
    pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    out: List[Tuple[float, float]] = []
    for (px, py) in pts:
        rx = px * c - py * s
        ry = px * s + py * c
        out.append((cx + rx, cy + ry))
    return out

class _YOLOOBB(TaskAdapter):
    def __init__(self, model: ResilientYOLOProtocol, *, conf: float = 0.25, iou: float = 0.45) -> None:
        self.model = model
        self.conf = float(conf)
        self.iou = float(iou)
        self.label = model.descriptor()
        self.names = _names_from_model(model.active_model())
        self._smoother: Optional[PolygonSmootherType] = None
        if _RuntimePolygonSmoother is not None:
            self._smoother = _RuntimePolygonSmoother(alpha=0.55, max_distance=90.0, max_age=6)

    def current_label(self) -> str:
        return self.model.descriptor()

    def infer(self, frame_bgr: NDArrayU8) -> List[OBBDetectionType]:
        if _RuntimeOBBDetection is None or np is None:
            return []
        np_ = cast(Any, np)
        inp = _ensure_contiguous(frame_bgr)
        with use_preprocessor(getattr(self, "_preprocessor", None)):
            res_any: Any = self.model.predict(inp, conf=self.conf, iou=self.iou, verbose=False)
        if isinstance(res_any, (list, tuple)):
            res_obj: object = cast(Sequence[object], res_any)[0]
        else:
            res_obj = cast(object, res_any)

        out: List[OBBDetectionType] = []

        obb_any: Any = getattr(res_obj, "obb", None)
        if obb_any is None:
            poly_any: Any = getattr(res_obj, "xyxyxyxy", None)
            if poly_any is None:
                return out
            polys = _to_numpy(poly_any, dtype=f32)
            confs = getattr(res_obj, "conf", None)
            clses = getattr(res_obj, "cls", None)
            confs_seq: Optional[Sequence[float]] = None
            clses_seq: Optional[Sequence[int]] = None
            if confs is not None:
                confs_seq = cast(Sequence[float], _to_numpy(confs, dtype=f32).reshape(-1).tolist())
            if clses is not None:
                clses_seq = cast(Sequence[int], _to_numpy(clses, dtype=i64).reshape(-1).astype(int).tolist())
            for i in range(polys.shape[0]):
                poly = np_.asarray(polys[i].reshape(-1, 2), dtype=np.float32)
                conf_v = float(confs_seq[i]) if confs_seq is not None and i < len(confs_seq) else 1.0
                cls_v: Optional[int] = int(clses_seq[i]) if clses_seq is not None and i < len(clses_seq) else None
                out.append(_RuntimeOBBDetection(points=poly, confidence=conf_v, class_id=cls_v))
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
            pts = np_.asarray(
                _rotrect_to_pts(float(cx), float(cy), float(w), float(h), float(theta)),
                dtype=np.float32,
            )
            conf_v = float(confs_seq2[i]) if confs_seq2 is not None and i < len(confs_seq2) else 1.0
            cls_v: Optional[int] = int(clses_seq2[i]) if clses_seq2 is not None and i < len(clses_seq2) else None
            out.append(_RuntimeOBBDetection(points=pts, confidence=conf_v, class_id=cls_v, angle=float(theta)))
        if self._smoother is not None:
            smoothed = self._smoother.smooth(out)
            return list(smoothed)
        return out

    def render(self, frame_bgr: NDArrayU8, result: List[OBBDetectionType]) -> NDArrayU8:
        from .overlay import draw_obb_bgr
        return draw_obb_bgr(frame_bgr, result, names=self.names)


class _SimpleOBB(TaskAdapter):
    """No-ML fallback OBB: minAreaRect on strong contours."""
    def __init__(self, conf: float = 0.25) -> None:
        self.conf = float(conf)
        self.label = "simple-obb (no-ML)"

    def infer(self, frame_bgr: NDArrayU8) -> List[OBBDetectionType]:
        if cv2 is None or np is None or _RuntimeOBBDetection is None:
            return []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        v = max(10, int(gray.mean()))
        edges = cv2.Canny(gray, v, v * 3)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = gray.shape[:2]
        min_area = max(120, (H * W) // 250)
        out: List[OBBDetectionType] = []
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            pts = np.asarray(box, dtype=np.float32)
            conf = min(0.99, 0.6 + (cv2.contourArea(c) / (W * H)))
            out.append(_RuntimeOBBDetection(points=pts, confidence=conf, class_id=None))
        return out

    def render(self, frame_bgr: NDArrayU8, result: List[OBBDetectionType]) -> NDArrayU8:
        from .overlay import draw_obb_bgr
        return draw_obb_bgr(frame_bgr, result, names=None)


# ---------------------------
# Builders (using model_registry) + Percent Spinner UX
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
    input_size: Optional[Tuple[int, int]] = None,
    preprocess_device: str = "cpu",
    warmup: bool = True,
    backend: str = "auto",
    ort_threads: Optional[int] = None,
    ort_execution: Optional[str] = None,
    nms_mode: str = "auto",
    hud_callback: Optional[Callable[[str], None]] = None,
) -> TaskAdapter:
    try:
        candidates = candidate_weights("detect", small=small, override=override)
        if not candidates:
            raise RuntimeError("no-detect-weights")
        nms_mode_norm = (nms_mode or "auto").strip().lower()
        if nms_mode_norm not in {"auto", "graph", "torch"}:
            nms_mode_norm = "auto"
        override_path = Path(override).expanduser() if override is not None else None
        ordered_candidates = _sort_candidates(
            candidates,
            backend_preference=backend,
            sticky_first=override_path,
        )
        label_hint = _label_from_override_or_pick("detect", small, override)
        wrapper = _acquire_wrapper(
            "detect",
            ordered_candidates,
            conf=conf,
            hud_callback=hud_callback,
            backend_preference=backend,
            ort_threads=ort_threads,
            ort_execution=ort_execution,
        )
        with _progress_percent_spinner(prefix="LIVE") as sp:
            prepared = bool(getattr(wrapper, "_argos_prepared", False))
            job_label = "Load" if not prepared else "reuse"
            sp.update(total=1, count=0, job=job_label, model=label_hint, current="detector")
            if not prepared:
                with _progress_running_task("Load", f"detector:{label_hint}"):
                    with _silence_ultralytics():
                        wrapper.prepare()
                setattr(wrapper, "_argos_prepared", True)
            preproc = attach_preprocessor(wrapper, target_size=input_size, device=preprocess_device)
            if warmup and _warmup_wrapper(wrapper, task="detect", conf=conf, iou=iou):
                sp.update(job="warmup", model=wrapper.descriptor())
            sp.update(count=1, job="ready", model=wrapper.descriptor())
        adapter = _YOLODetect(wrapper, conf=conf, iou=iou, nms_mode=nms_mode_norm)
        if preproc is not None:
            setattr(adapter, "_preprocessor", preproc)
        return adapter
    except Exception:
        with _progress_simple_status("FALLBACK: fast-contour (no-ML)"):
            time.sleep(0.05)
        return _ContourDetect(conf=conf, iou=iou)

def build_heatmap(
    *,
    small: bool = True,
    override: Optional[Union[str, Path]] = LIVE_HEATMAP_OVERRIDE,
    input_size: Optional[Tuple[int, int]] = None,
    preprocess_device: str = "cpu",
    warmup: bool = True,
    backend: str = "auto",
    ort_threads: Optional[int] = None,
    ort_execution: Optional[str] = None,
    hud_callback: Optional[Callable[[str], None]] = None,
) -> TaskAdapter:
    try:
        candidates = candidate_weights("heatmap", small=small, override=override)
        if not candidates:
            raise RuntimeError("no-heatmap-weights")
        override_path = Path(override).expanduser() if override is not None else None
        ordered_candidates = _sort_candidates(
            candidates,
            backend_preference=backend,
            sticky_first=override_path,
        )
        label_hint = _label_from_override_or_pick("heatmap", small, override)
        wrapper = _acquire_wrapper(
            "heatmap",
            ordered_candidates,
            conf=0.25,
            hud_callback=hud_callback,
            backend_preference=backend,
            ort_threads=ort_threads,
            ort_execution=ort_execution,
        )
        with _progress_percent_spinner(prefix="LIVE") as sp:
            prepared = bool(getattr(wrapper, "_argos_prepared", False))
            job_label = "Load" if not prepared else "reuse"
            sp.update(total=1, count=0, job=job_label, model=label_hint, current="segmenter")
            if not prepared:
                with _progress_running_task("Load", f"segmenter:{label_hint}"):
                    with _silence_ultralytics():
                        wrapper.prepare()
                setattr(wrapper, "_argos_prepared", True)
            preproc = attach_preprocessor(wrapper, target_size=input_size, device=preprocess_device)
            if warmup and _warmup_wrapper(wrapper, task="heatmap", conf=0.25):
                sp.update(job="warmup", model=wrapper.descriptor())
            sp.update(count=1, job="ready", model=wrapper.descriptor())
        adapter = _YOLOHeatmap(wrapper, conf=0.25)
        if preproc is not None:
            setattr(adapter, "_preprocessor", preproc)
        return adapter
    except Exception:
        with _progress_simple_status("FALLBACK: laplacian-heatmap (no-ML)"):
            time.sleep(0.05)
        return _LaplacianHeatmap()

def build_classify(
    *,
    small: bool = True,
    topk: int = 1,
    override: Optional[Union[str, Path]] = LIVE_CLASSIFY_OVERRIDE,
    input_size: Optional[Tuple[int, int]] = None,
    preprocess_device: str = "cpu",
    warmup: bool = True,
    backend: str = "auto",
    ort_threads: Optional[int] = None,
    ort_execution: Optional[str] = None,
    hud_callback: Optional[Callable[[str], None]] = None,
) -> TaskAdapter:
    try:
        candidates = candidate_weights("classify", small=small, override=override)
        if not candidates:
            raise RuntimeError("no-classify-weights")
        override_path = Path(override).expanduser() if override is not None else None
        ordered_candidates = _sort_candidates(
            candidates,
            backend_preference=backend,
            sticky_first=override_path,
        )
        label_hint = _label_from_override_or_pick("classify", small, override)
        wrapper = _acquire_wrapper(
            "classify",
            ordered_candidates,
            conf=0.25,
            hud_callback=hud_callback,
            backend_preference=backend,
            ort_threads=ort_threads,
            ort_execution=ort_execution,
        )
        with _progress_percent_spinner(prefix="LIVE") as sp:
            prepared = bool(getattr(wrapper, "_argos_prepared", False))
            job_label = "Load" if not prepared else "reuse"
            sp.update(total=1, count=0, job=job_label, model=label_hint, current="classifier")
            if not prepared:
                with _progress_running_task("Load", f"classifier:{label_hint}"):
                    with _silence_ultralytics():
                        wrapper.prepare()
                setattr(wrapper, "_argos_prepared", True)
            preproc = attach_preprocessor(wrapper, target_size=input_size, device=preprocess_device)
            if warmup and _warmup_wrapper(wrapper, task="classify"):
                sp.update(job="warmup", model=wrapper.descriptor())
            sp.update(count=1, job="ready", model=wrapper.descriptor())
        adapter = _YOLOClassify(wrapper, topk=topk)
        if preproc is not None:
            setattr(adapter, "_preprocessor", preproc)
        return adapter
    except Exception:
        with _progress_simple_status("FALLBACK: simple-classify (no-ML)"):
            time.sleep(0.05)
        return _SimpleClassify(topk=topk)

def build_pose(
    *,
    small: bool = True,
    conf: float = 0.25,
    override: Optional[Union[str, Path]] = LIVE_POSE_OVERRIDE,
    input_size: Optional[Tuple[int, int]] = None,
    preprocess_device: str = "cpu",
    warmup: bool = True,
    backend: str = "auto",
    ort_threads: Optional[int] = None,
    ort_execution: Optional[str] = None,
    hud_callback: Optional[Callable[[str], None]] = None,
) -> TaskAdapter:
    try:
        candidates = candidate_weights("pose", small=small, override=override)
        if not candidates:
            raise RuntimeError("no-pose-weights")
        override_path = Path(override).expanduser() if override is not None else None
        ordered_candidates = _sort_candidates(
            candidates,
            backend_preference=backend,
            sticky_first=override_path,
        )
        label_hint = _label_from_override_or_pick("pose", small, override)
        wrapper = _acquire_wrapper(
            "pose",
            ordered_candidates,
            conf=conf,
            hud_callback=hud_callback,
            backend_preference=backend,
            ort_threads=ort_threads,
            ort_execution=ort_execution,
        )
        with _progress_percent_spinner(prefix="LIVE") as sp:
            prepared = bool(getattr(wrapper, "_argos_prepared", False))
            job_label = "Load" if not prepared else "reuse"
            sp.update(total=1, count=0, job=job_label, model=label_hint, current="pose")
            if not prepared:
                with _progress_running_task("Load", f"pose:{label_hint}"):
                    with _silence_ultralytics():
                        wrapper.prepare()
                setattr(wrapper, "_argos_prepared", True)
            preproc = attach_preprocessor(wrapper, target_size=input_size, device=preprocess_device)
            if warmup and _warmup_wrapper(wrapper, task="pose", conf=conf):
                sp.update(job="warmup", model=wrapper.descriptor())
            sp.update(count=1, job="ready", model=wrapper.descriptor())
        adapter = _YOLOPose(wrapper, conf=conf)
        if preproc is not None:
            setattr(adapter, "_preprocessor", preproc)
        return adapter
    except Exception:
        with _progress_simple_status("FALLBACK: simple-pose (no-ML)"):
            time.sleep(0.05)
        return _SimplePose()

def build_obb(
    *,
    small: bool = True,
    conf: float = 0.25,
    iou: float = 0.45,
    override: Optional[Union[str, Path]] = LIVE_OBB_OVERRIDE,
    input_size: Optional[Tuple[int, int]] = None,
    preprocess_device: str = "cpu",
    warmup: bool = True,
    backend: str = "auto",
    ort_threads: Optional[int] = None,
    ort_execution: Optional[str] = None,
    hud_callback: Optional[Callable[[str], None]] = None,
) -> TaskAdapter:
    try:
        candidates = candidate_weights("obb", small=small, override=override)
        if not candidates:
            raise RuntimeError("no-obb-weights")
        label_hint = _label_from_override_or_pick("obb", small, override)
        override_path = Path(override).expanduser() if override is not None else None
        ordered_candidates = _sort_candidates(
            candidates,
            backend_preference=backend,
            sticky_first=override_path,
        )
        wrapper = _acquire_wrapper(
            "obb",
            ordered_candidates,
            conf=conf,
            hud_callback=hud_callback,
            backend_preference=backend,
            ort_threads=ort_threads,
            ort_execution=ort_execution,
        )
        with _progress_percent_spinner(prefix="LIVE") as sp:
            prepared = bool(getattr(wrapper, "_argos_prepared", False))
            job_label = "Load" if not prepared else "reuse"
            sp.update(total=1, count=0, job=job_label, model=label_hint, current="obb")
            if not prepared:
                with _progress_running_task("Load", f"obb:{label_hint}"):
                    with _silence_ultralytics():
                        wrapper.prepare()
                setattr(wrapper, "_argos_prepared", True)
            preproc = attach_preprocessor(wrapper, target_size=input_size, device=preprocess_device)
            if warmup and _warmup_wrapper(wrapper, task="obb", conf=conf, iou=iou):
                sp.update(job="warmup", model=wrapper.descriptor())
            sp.update(count=1, job="ready", model=wrapper.descriptor())
        adapter = _YOLOOBB(wrapper, conf=conf, iou=iou)
        if preproc is not None:
            setattr(adapter, "_preprocessor", preproc)
        return adapter
    except Exception:
        with _progress_simple_status("FALLBACK: simple-obb (no-ML)"):
            time.sleep(0.05)
        return _SimpleOBB(conf=conf)
