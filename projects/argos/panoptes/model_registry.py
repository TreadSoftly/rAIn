from __future__ import annotations

import contextlib
import functools
import importlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Final, Iterable, Iterator, Mapping, Optional, Sequence, Union, cast

from .logging_config import bind_context
from .runtime.backend_probe import ort_available, torch_available
from .model.artifact_metadata import analyse_artifact, ARTIFACT_METADATA_VERSION

# ────────────────────────────────────────────────────────────────
#  Logging (explicit, human-friendly, no stack noise)
# ────────────────────────────────────────────────────────────────
# — Logging helpers
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
LOGGER.setLevel(logging.ERROR)

_ERROR_TOKENS = ("error", "fail", "failed", "exception", "warning")
_ERROR_KEYS = ("error", "reason")


def _should_log(event: str, info: Mapping[str, object]) -> bool:
    event_lower = event.lower()
    if any(token in event_lower for token in _ERROR_TOKENS):
        return True
    for key in _ERROR_KEYS:
        value = info.get(key)
        if isinstance(value, str):
            if value and value.strip().lower() not in {"ok", "success"}:
                return True
        elif value not in (None, 0, False):
            return True
    return False


def _format_detail(info: Mapping[str, object]) -> str:
    return " ".join(f"{key}={info[key]}" for key in sorted(info) if info[key] is not None)


def _log(event: str, **info: object) -> None:
    if not _should_log(event, info):
        return
    detail = _format_detail(info)
    if detail:
        LOGGER.error("%s %s", event, detail)
    else:
        LOGGER.error("%s", event)

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

_MANIFEST_PATH = MODEL_DIR / "manifest.json"
_ARTIFACT_METADATA_CACHE: dict[str, dict[str, object]] = {}
_manifest_metadata_version: Optional[int] = None


def _normalize_manifest_artifacts(source: Mapping[str, object]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for key_candidate, payload_candidate in source.items():
        if not isinstance(payload_candidate, dict):
            continue
        payload_dict: dict[str, object] = cast(dict[str, object], payload_candidate)
        entry: dict[str, object] = {}
        for inner_key_candidate, inner_value in payload_dict.items():
            entry[inner_key_candidate] = inner_value
        result[key_candidate] = entry
    return result


def _string_list_from(value: object) -> list[str]:
    items: list[str] = []
    iterable_value: Optional[Iterable[Any]] = None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        iterable_value = cast(Sequence[Any], value)
    elif isinstance(value, set):
        iterable_value = cast(set[Any], value)
    if iterable_value is not None:
        for candidate_obj in iterable_value:
            if isinstance(candidate_obj, str):
                items.append(candidate_obj)
    return items


def _bootstrap_manifest_metadata() -> None:
    global _manifest_metadata_version
    if not _MANIFEST_PATH.exists():
        return
    try:
        raw = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    version = raw.get("artifact_metadata_version")
    artifacts_raw = raw.get("artifacts")
    if isinstance(version, int):
        _manifest_metadata_version = version
    if isinstance(artifacts_raw, dict):
        normalized = _normalize_manifest_artifacts(cast(Mapping[str, object], artifacts_raw))
        _ARTIFACT_METADATA_CACHE.update(normalized)


_bootstrap_manifest_metadata()


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

def _candidate_weights(
    task: str,
    *,
    small: bool,
    override: Optional[Union[str, Path]],
) -> Iterator[Path]:
    """
    Yield candidate weight paths in preference order, de-duplicated.

    Order rules:
        - User override (if provided) first.
        - Preferred size list (small vs. regular).
        - Within each list, prefer ONNX only when the backend is available and
          not explicitly de-prioritised via environment toggles.
    """
    prefer_onnx = os.environ.get("ARGOS_PREFER_ONNX", "1").strip().lower() not in {"0", "false", "no", "off"}
    ort_ok, ort_version, ort_providers, ort_reason = ort_available()
    torch_ok = torch_available()
    ort_ready = bool(ort_ok and ort_providers and not ort_reason)
    if ort_providers:
        ort_status = ",".join(ort_providers)
    elif ort_reason:
        ort_status = ort_reason
    else:
        ort_status = "OK" if ort_ready else "unavailable"

    _log(
        "weights.select.start",
        task=task,
        prefer_small=small,
        prefer_onnx=int(prefer_onnx),
        ort=ort_status,
        ort_version=ort_version or "?",
        torch="OK" if torch_ok else "missing",
    )

    seen: set[str] = set()
    ordered: list[Path] = []

    def _normalise(path_like: Union[str, Path]) -> Path:
        path = Path(path_like).expanduser()
        try:
            return path.resolve(strict=False)
        except Exception:
            return path

    def _add(path_like: Union[str, Path], *, force: bool = False) -> bool:
        path = _normalise(path_like)
        key = str(path).lower()
        if not force and key in seen:
            return False
        seen.add(key)
        ordered.append(path)
        return True

    def _extend(paths: Iterable[Path]) -> tuple[int, int, int]:
        deferred_onnx: list[Path] = []
        added_total = 0
        added_onnx = 0
        added_non_onnx = 0

        for p in paths:
            suffix = p.suffix.lower()
            if suffix == ".onnx":
                if not ort_ready:
                    _log("weights.select.skip", task=task, weight=str(p), reason=ort_reason or "onnxruntime unavailable")
                    continue
                if prefer_onnx:
                    if _add(p):
                        added_total += 1
                        added_onnx += 1
                else:
                    deferred_onnx.append(p)
                continue

            if _add(p):
                added_total += 1
                added_non_onnx += 1

        if not prefer_onnx:
            for p in deferred_onnx:
                if _add(p):
                    added_total += 1
                    added_onnx += 1

        return added_total, added_onnx, added_non_onnx

    if override is not None:
        override_path = _normalise(override)
        _add(override_path, force=True)
        if override_path.suffix.lower() == ".onnx" and not ort_ready:
            _log(
                "weights.select.warn",
                task=task,
                weight=str(override_path),
                reason=f"ORT unavailable: {ort_reason}",
            )

    if small:
        _, _, small_non_onnx = _extend(WEIGHT_PRIORITY.get(f"{task}_small", []))
        if small_non_onnx == 0:
            _extend(WEIGHT_PRIORITY.get(task, []))
    else:
        _extend(WEIGHT_PRIORITY.get(task, []))

    for candidate in ordered:
        yield candidate


def _load_with_fallback(
    task: str,
    runtime_task: str,
    *,
    small: bool,
    override: Optional[Union[str, Path]],
) -> object:
    choice = "override" if override is not None else ("auto-small" if small else "auto")
    candidates = list(_candidate_weights(task, small=small, override=override))

    if not candidates:
        _log("weights.select.fail_all", task=task, source=choice, reasons="no-candidates")
        return _require(None, task)

    last_exc: Exception | None = None
    last_weight: Optional[Path] = None
    failure_reasons: list[str] = []

    for idx, weight in enumerate(candidates):
        source = choice if idx == 0 else "fallback"
        _log("weights.select.try", task=task, source=source, weight=str(weight), index=idx)
        try:
            model = _load(weight, task=runtime_task)
        except Exception as exc:
            last_exc = exc
            last_weight = Path(weight)
            reason = f"{type(exc).__name__}: {exc}"
            failure_reasons.append(f"{weight.name}: {reason}")
            _log("weights.select.fail", task=task, source=source, weight=str(weight), reason=reason)
            continue
        if model is not None:
            _log("weights.select.hit", task=task, source=source, weight=str(weight))
            return model
        failure_reasons.append(f"{weight.name}: unavailable")
        _log("weights.select.fail", task=task, source=source, weight=str(weight), reason="unavailable")

    joined = "; ".join(failure_reasons) or "unknown"
    _log("weights.select.fail_all", task=task, reasons=joined, source=choice)

    if last_exc is not None and last_weight is not None:
        raise RuntimeError(
            f"[model_registry] failed to load any weight for task '{task}'. "
            f"Last attempt {last_weight}: {last_exc}"
        ) from last_exc

    return _require(None, task)


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


def _artifact_key(path: Path) -> str:
    return path.name


def artifact_metadata(path: Path) -> dict[str, object]:
    key = _artifact_key(path)
    cached_entry = _ARTIFACT_METADATA_CACHE.get(key)
    if cached_entry is not None:
        return cached_entry

    try:
        analysed_map: dict[str, Any] = analyse_artifact(path)
        metadata: dict[str, object] = {
            meta_key: meta_value for meta_key, meta_value in analysed_map.items()
        }
    except Exception as exc:
        metadata = {
            "path": path.name,
            "analysis_error": f"{type(exc).__name__}:{exc}",
            "nms_in_graph": False,
            "providers": [],
        }

    _ARTIFACT_METADATA_CACHE[key] = metadata
    return metadata


def rank_candidates_for_backend(candidates: Sequence[Path], backend: str) -> list[Path]:
    backend_norm = (backend or "auto").lower()
    if backend_norm in {"auto", "torch"}:
        return list(candidates)

    def _score(idx: int, path: Path) -> tuple[int, int]:
        meta = artifact_metadata(path)
        nms_in_graph = bool(meta.get("nms_in_graph"))
        providers_obj = meta.get("providers")
        providers: list[str] = [provider.lower() for provider in _string_list_from(providers_obj)]
        has_gpu_provider = any("cuda" in p or "tensorrt" in p for p in providers)
        suffix = path.suffix.lower()
        if nms_in_graph and has_gpu_provider:
            rank = 0
        elif nms_in_graph:
            rank = 1
        elif suffix == ".onnx":
            rank = 2
        else:
            rank = 3
        return (rank, idx)

    enumerated = list(enumerate([Path(p) for p in candidates]))
    enumerated.sort(key=lambda item: _score(item[0], item[1]))
    return [entry for _, entry in enumerated]


def select_postprocess_strategy(
    task: str,
    backend: str,
    *,
    weight: Optional[Union[str, Path]] = None,
) -> dict[str, object]:
    backend_norm = (backend or "auto").lower()
    resolved_weight: Optional[Path]
    if weight is None:
        resolved_weight = pick_weight(task)
    else:
        resolved_weight = Path(weight).expanduser()

    metadata: dict[str, object] = {}
    if resolved_weight is not None:
        metadata = artifact_metadata(resolved_weight)

    nms_mode = "graph" if metadata.get("nms_in_graph") else "torch"
    if backend_norm.startswith("torch"):
        nms_mode = "torch"

    return {
        "task": task,
        "backend": backend_norm,
        "weight": str(resolved_weight) if resolved_weight else None,
        "nms": nms_mode,
        "max_det": metadata.get("max_det"),
        "within_graph_conf_thres": metadata.get("within_graph_conf_thres"),
        "providers": metadata.get("providers"),
        "metadata_version": _manifest_metadata_version or ARTIFACT_METADATA_VERSION,
    }


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


def candidate_weights(
    task: str,
    *,
    small: bool = False,
    override: Optional[Union[str, Path]] = None,
) -> list[Path]:
    """Expose the ordered candidate list (used by live runtime wrappers)."""
    return list(_candidate_weights(task, small=small, override=override))


def load_detector(*, small: bool = False, override: Optional[Union[str, Path]] = None) -> object:
    return _load_with_fallback("detect", "detect", small=small, override=override)


def load_segmenter(*, small: bool = False, override: Optional[Union[str, Path]] = None) -> object:
    return _load_with_fallback("heatmap", "segment", small=small, override=override)


def load_classifier(*, small: bool = False, override: Optional[Union[str, Path]] = None) -> object:
    return _load_with_fallback("classify", "classify", small=small, override=override)


def load_pose(*, small: bool = False, override: Optional[Union[str, Path]] = None) -> object:
    return _load_with_fallback("pose", "pose", small=small, override=override)


def load_obb(*, small: bool = False, override: Optional[Union[str, Path]] = None) -> object:
    return _load_with_fallback("obb", "obb", small=small, override=override)


__all__ = [
    "MODEL_DIR",
    "WEIGHT_PRIORITY",
    "candidate_weights",
    "pick_weight",
    "load_detector",
    "load_segmenter",
    "load_classifier",
    "load_pose",
    "load_obb",
    "set_log_level",
    "set_verbose",
    "artifact_metadata",
    "rank_candidates_for_backend",
    "select_postprocess_strategy",
]
