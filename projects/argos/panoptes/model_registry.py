from __future__ import annotations

import contextlib
import functools
import importlib
import logging
import time
from pathlib import Path
from typing import Final, Optional, Union

from .logging_config import bind_context

# ────────────────────────────────────────────────────────────────
#  Logging (explicit, human-friendly, no stack noise)
# ────────────────────────────────────────────────────────────────
# — Logging helpers
LOGGER = logging.getLogger(__name__)

def _log(event: str, **info: object) -> None:
    if info:
        detail = ' '.join(f"{k}={info[k]}" for k in sorted(info) if info[k] is not None)
        LOGGER.info("%s %s", event, detail)
    else:
        LOGGER.info("%s", event)

def set_log_level(level: int) -> None:
    LOGGER.setLevel(level)

def set_verbose(enabled: bool = True) -> None:
    set_log_level(logging.INFO if enabled else logging.WARNING)

#  Model folder(s)
# ────────────────────────────────────────────────────────────────
_ROOT: Final[Path] = Path(__file__).resolve().parent            # …/panoptes
_MODEL_DIR_A = _ROOT / "model"                                  # packaged weights (preferred)
_MODEL_DIR_B = _ROOT.parent / "model"                           # legacy path (fallback)

# Prefer packaged dir; only fall back to legacy if it already exists
MODEL_DIR: Final[Path] = _MODEL_DIR_A if _MODEL_DIR_A.exists() or not _MODEL_DIR_B.exists() else _MODEL_DIR_B
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────
#  *** WEIGHT SELECTION TABLE ***
#  Edit this ONE place if you add/rename checkpoints
#  Only list **real** upstream files you actually keep in panoptes/model.
# ────────────────────────────────────────────────────────────────

# DETECTION (boxes)
_DETECT_LIST: list[Path] = [
    MODEL_DIR / "yolov8x.pt",
    MODEL_DIR / "yolo11x.pt",
    MODEL_DIR / "yolo11x.onnx",
    MODEL_DIR / "yolo11s.onnx",
    MODEL_DIR / "yolo12x.pt",
    MODEL_DIR / "yolov8s.pt",
    MODEL_DIR / "yolov8s.onnx",
    MODEL_DIR / "yolo11s.pt",
    MODEL_DIR / "yolov8n.pt",
    MODEL_DIR / "yolov8n.onnx",
    MODEL_DIR / "yolo11n.pt",
    MODEL_DIR / "yolo11n.onnx",
    MODEL_DIR / "yolo12n.onnx",
    MODEL_DIR / "yolo12n.pt",
]

# SEGMENTATION (heatmaps)
_HEATMAP_LIST: list[Path] = [
    MODEL_DIR / "yolov8x-seg.pt",
    MODEL_DIR / "yolo11x-seg.pt",
    MODEL_DIR / "yolo11x-seg.onnx",
    MODEL_DIR / "yolov8x-seg.onnx",
    MODEL_DIR / "yolo11s-seg.pt",
    MODEL_DIR / "yolo11s-seg.onnx",
    MODEL_DIR / "yolo11n-seg.pt",
    MODEL_DIR / "yolo11n-seg.onnx",
    MODEL_DIR / "yolov8n-seg.pt",
    MODEL_DIR / "yolov8n-seg.onnx",
]

# CLASSIFICATION
_CLASSIFY_LIST: list[Path] = [
    MODEL_DIR / "yolo11x-cls.pt",
    MODEL_DIR / "yolov8x-cls.pt",
    MODEL_DIR / "yolov8s-cls.pt",
    MODEL_DIR / "yolo11s-cls.pt",
    MODEL_DIR / "yolov8n-cls.pt",
    MODEL_DIR / "yolo11n-cls.pt",
]

# POSE (keypoints)
_POSE_LIST: list[Path] = [
    MODEL_DIR / "yolo11x-pose.pt",
    MODEL_DIR / "yolov8x-pose.pt",
    MODEL_DIR / "yolov8s-pose.pt",
    MODEL_DIR / "yolo11s-pose.pt",
    MODEL_DIR / "yolov8n-pose.pt",
    MODEL_DIR / "yolo11n-pose.pt",
]

# OBB (oriented boxes)
_OBB_LIST: list[Path] = [
    MODEL_DIR / "yolo11x-obb.pt",
    MODEL_DIR / "yolov8x-obb.pt",
    MODEL_DIR / "yolov8s-obb.pt",
    MODEL_DIR / "yolo11s-obb.pt",
    MODEL_DIR / "yolov8n-obb.pt",
    MODEL_DIR / "yolo11n-obb.pt",
]

# “small / fast” (live video / tiny devices)
_DETECT_SMALL_LIST: list[Path] = [
    MODEL_DIR / "yolov8s.onnx",
    MODEL_DIR / "yolo11s.onnx",
    MODEL_DIR / "yolo12s.onnx",
    MODEL_DIR / "yolo11n.onnx",
    MODEL_DIR / "yolo11n.pt",
]

_HEATMAP_SMALL_LIST: list[Path] = [
    MODEL_DIR / "yolov8n-seg.onnx",
    MODEL_DIR / "yolo11n-seg.onnx",
    MODEL_DIR / "yolo11n-seg.pt",
    MODEL_DIR / "yolov8n-seg.pt",
]

_CLASSIFY_SMALL_LIST: list[Path] = [
    MODEL_DIR / "yolo11x-cls.pt",
    MODEL_DIR / "yolov8x-cls.pt",
]

_POSE_SMALL_LIST: list[Path] = [
    MODEL_DIR / "yolo11s-pose.onnx",
    MODEL_DIR / "yolov8s-pose.onnx",
]

_OBB_SMALL_LIST: list[Path] = [
    MODEL_DIR / "yolo11s-obb.onnx",
    MODEL_DIR / "yolov8s-obb.onnx",
]

WEIGHT_PRIORITY: dict[str, list[Path]] = {
    "detect": _DETECT_LIST,
    "geojson": list(_DETECT_LIST),  # uses detection picks
    "heatmap": _HEATMAP_LIST,
    "classify": _CLASSIFY_LIST,
    "pose": _POSE_LIST,
    "obb": _OBB_LIST,

    # small / fast
    "detect_small": _DETECT_SMALL_LIST,
    "heatmap_small": _HEATMAP_SMALL_LIST,
    "classify_small": _CLASSIFY_SMALL_LIST,
    "pose_small": _POSE_SMALL_LIST,
    "obb_small": _OBB_SMALL_LIST,
}


# ────────────────────────────────────────────────────────────────
#  Internal helpers
# ────────────────────────────────────────────────────────────────
def _first_existing(paths: list[Path]) -> Optional[Path]:
    """Return the first path that exists on disk or *None*."""
    return next((p for p in paths if p.exists()), None)


def _resolve_yolo_class() -> Optional[type]:
    """
    Import and return ultralytics.YOLO lazily.

    This honors any monkeypatch (e.g., unit tests that install a fake
    ``ultralytics`` module **before** the loader is called) and avoids
    binding to the real library at module import time.
    """
    try:
        mod = importlib.import_module("ultralytics")
        yolo_cls = getattr(mod, "YOLO", None)
        # best-effort: silence banners/loguru noise if present
        try:
            from ultralytics.utils import LOGGER as _ULOG  # type: ignore
            remover = getattr(_ULOG, "remove", None)
            if callable(remover):
                try:
                    remover()
                except Exception:
                    pass
            else:
                for h in list(getattr(_ULOG, "handlers", [])):
                    try:
                        _ULOG.removeHandler(h)  # type: ignore[attr-defined]
                    except Exception:
                        pass
        except Exception:
            pass
        return yolo_cls
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def _load(
    weight: Optional[Path],
    *,
    task: str,  # keep liberal typing to keep static checkers happy
) -> Optional[object]:
    """
    Cached wrapper around ``YOLO(path, task=...)`` to avoid re-inits and kill
    the “Unable to automatically guess model task” warning when supported.

    PROGRESS POLICY:
    This function **never** opens or renders progress/status UI. The only
    progress surface is the parent Halo/Rich spinner from the CLI.
    """
    yolo_cls = _resolve_yolo_class()
    if yolo_cls is None or weight is None:
        if weight is None:
            _log("weights.load.skip", task=task, reason="weight-missing")
        else:
            _log("weights.load.skip", task=task, reason="ultralytics-missing")
        return None

    weight_str = str(weight)
    with bind_context(model_task=task, weight_path=weight_str):
        start = time.perf_counter()
        try:
            _log("weights.load.start", task=task, weight=weight_str)
            with contextlib.nullcontext():
                try:
                    model = yolo_cls(str(weight), task=task)  # type: ignore[call-arg]
                except TypeError:
                    model = yolo_cls(str(weight))  # type: ignore[call-arg]
        except Exception:
            _log("weights.load.error", task=task, weight=weight_str)
            raise
        else:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            runtime = getattr(model, "task", task)
            model_type = type(model).__name__
            provider = None
            try:
                predictor = getattr(model, "predictor", None)
                provider = getattr(predictor, "provider", None)
                if provider is None:
                    providers = getattr(predictor, "providers", None)
                    if providers:
                        provider = ",".join(map(str, providers))
            except Exception:
                provider = None
            device = None
            for attr in ("device", "model"):
                candidate = getattr(model, attr, None)
                device_attr = getattr(candidate, "device", None)
                if device_attr is not None:
                    try:
                        device = str(device_attr)
                        break
                    except Exception:
                        continue
                if attr == "device" and candidate is not None:
                    try:
                        device = str(candidate)
                        break
                    except Exception:
                        continue
            _log(
                "weights.load.success",
                task=task,
                weight=weight_str,
                runtime=runtime,
                model=model_type,
                provider=provider,
                device=device,
                ms=f"{elapsed_ms:.1f}",
            )
            return model


def _require(model: Optional[object], task: str) -> object:
    """Abort loudly when the chosen weight is missing."""
    if model is None:
        raise RuntimeError(f"[model_registry] no weight configured for task “{task}”")
    return model


# ────────────────────────────────────────────────────────────────
#  Public helpers
# ────────────────────────────────────────────────────────────────
def pick_weight(task: str, *, small: bool = False) -> Optional[Path]:
    """
    Return the path to the **first** existing weight for *task* or *None*.

    When *small=True* prefer the *_small* list, falling back to the normal
    list if none of those files exist.
    """
    if small:
        pref = WEIGHT_PRIORITY.get(f"{task}_small", [])
        chosen_small = _first_existing(pref)
        if chosen_small is not None:
            _log("weights.pick", task=task, small=small, weight=str(chosen_small))
            return chosen_small
    paths = WEIGHT_PRIORITY.get(task, [])
    chosen = _first_existing(paths)
    if chosen is None:
        _log("weights.pick.missing", task=task, small=small)
    else:
        _log("weights.pick", task=task, small=small, weight=str(chosen))
    return chosen


def _choose(task: str, *, small: bool, override: Optional[Union[str, Path]]) -> Optional[Path]:
    if override is not None:
        return Path(override).expanduser()
    return pick_weight(task, small=small)


def load_detector(*, small: bool = False, override: Optional[Union[str, Path]] = None) -> object:
    chosen = _choose("detect", small=small, override=override)
    choice = "override" if override is not None else ("auto-small" if small else "auto")
    _log("weights.select", task="detect", source=choice, weight=(str(chosen) if chosen else None))
    model = _load(chosen, task="detect")
    return _require(model, "detect")


def load_segmenter(*, small: bool = False, override: Optional[Union[str, Path]] = None) -> object:
    chosen = _choose("heatmap", small=small, override=override)
    choice = "override" if override is not None else ("auto-small" if small else "auto")
    _log("weights.select", task="heatmap", source=choice, weight=(str(chosen) if chosen else None))
    model = _load(chosen, task="segment")
    return _require(model, "heatmap")


def load_classifier(*, small: bool = False, override: Optional[Union[str, Path]] = None) -> object:
    chosen = _choose("classify", small=small, override=override)
    choice = "override" if override is not None else ("auto-small" if small else "auto")
    _log("weights.select", task="classify", source=choice, weight=(str(chosen) if chosen else None))
    model = _load(chosen, task="classify")
    return _require(model, "classify")


def load_pose(*, small: bool = False, override: Optional[Union[str, Path]] = None) -> object:
    chosen = _choose("pose", small=small, override=override)
    choice = "override" if override is not None else ("auto-small" if small else "auto")
    _log("weights.select", task="pose", source=choice, weight=(str(chosen) if chosen else None))
    model = _load(chosen, task="pose")
    return _require(model, "pose")


def load_obb(*, small: bool = False, override: Optional[Union[str, Path]] = None) -> object:
    chosen = _choose("obb", small=small, override=override)
    choice = "override" if override is not None else ("auto-small" if small else "auto")
    _log("weights.select", task="obb", source=choice, weight=(str(chosen) if chosen else None))
    model = _load(chosen, task="obb")
    return _require(model, "obb")


__all__ = [
    "MODEL_DIR",
    "WEIGHT_PRIORITY",
    "pick_weight",
    "load_detector",
    "load_segmenter",
    "load_classifier",
    "load_pose",
    "load_obb",
    "set_log_level",
    "set_verbose",
]
