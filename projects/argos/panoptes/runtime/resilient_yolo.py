"""
Runtime wrapper that keeps YOLO usable even when optional backends fail.

Ultralytics lazily imports ONNX Runtime when the first inference call happens.
On machines without working ORT binaries this raises an ImportError/OSError
and would normally bubble up, leaving live video without an ML backend.
This wrapper transparently retries alternative weight formats (typically the
`.pt` Torch counterparts) so Argos continues operating.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, cast

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

LOGGER = logging.getLogger(__name__)

_ORT_CONFIG: Dict[str, Any] = {
    "threads": None,
    "execution": "sequential",
    "cuda_graph": True,
    "arena_strategy": "kNextPowerOfTwo",
    "dml_device_id": None,
}
_ORT_PATCH_LOCK = threading.Lock()
_ORT_PATCHED = False


def _detect_backend(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return "onnxruntime"
    if suffix in {".pt", ".pth"}:
        return "torch"
    return "unknown"


def _should_retry(exc: BaseException) -> bool:
    """Heuristically decide if an exception warrants trying the next weight."""
    text = f"{type(exc).__name__}: {exc}".lower()
    retry_tokens = (
        "onnxruntime",
        "dll load failed",
        "no module named 'onnxruntime'",
        "onnxruntime_pybind11_state",
        "provider cudnn is not valid",
    )
    if any(tok in text for tok in retry_tokens):
        return True
    return isinstance(exc, (ImportError, ModuleNotFoundError, OSError))


class ResilientYOLO:
    """
    Thin wrapper around ``ultralytics.YOLO`` that can swap weights on failure.

    Args:
        candidates: Ordered list of candidate weights (paths or strings).
        task: Human-readable task label, used for logging.
        conf: Default confidence threshold supplied to ``predict``.
        on_switch: Optional callback invoked when the backend changes.  The
            callback receives a single string suitable for displaying in the HUD.
    """

    def __init__(
        self,
        candidates: Sequence[Path | str],
        *,
        task: str,
        conf: float = 0.25,
        on_switch: Optional[Callable[[str], None]] = None,
    ) -> None:
        if not candidates:
            raise ValueError("ResilientYOLO requires at least one candidate weight")

        self._candidates: List[Path] = [Path(c).expanduser() for c in candidates]
        self._task = task
        self._conf = float(conf)
        self._on_switch = on_switch

        self._current_idx: int = -1
        self._model: Optional[object] = None
        self._label: Optional[str] = None
        self._backend: Optional[str] = None
        self._failures: list[str] = []
        self._has_success = False
        self._pending_notice: Optional[str] = None

    # -------------------------------
    #  Properties / inspection hooks
    # -------------------------------
    @property
    def backend(self) -> Optional[str]:
        return self._backend

    @property
    def weight_label(self) -> Optional[str]:
        return self._label

    def active_model(self) -> Optional[object]:
        try:
            self._ensure_model()
        except Exception:
            return None
        return self._model

    def descriptor(self) -> str:
        label = self._label or "unknown"
        backend = (self._backend or "unknown").upper()
        return f"{label} | {backend}"

    # -------------------------------
    #  Lifecycle management
    # -------------------------------
    def prepare(self) -> None:
        """Eagerly load the first healthy model (used during bootstrap)."""
        self._ensure_model()

    # -------------------------------
    #  Internal helpers
    # -------------------------------
    def _iter_candidates(self) -> Iterator[tuple[int, Path]]:
        for idx, path in enumerate(list(self._candidates)):
            if not path.exists():
                LOGGER.debug("weights.select.skip missing=%s task=%s", path, self._task)
                continue
            yield idx, path

    def _yolo_task(self) -> str:
        mapping = {
            "heatmap": "segment",
            "geojson": "detect",
        }
        return mapping.get(self._task, self._task)

    def _instantiate(self, weight: Path) -> object:
        from ultralytics import YOLO  # type: ignore

        backend = _detect_backend(weight)
        LOGGER.info(
            "weights.select.try task=%s weight=%s backend=%s",
            self._task,
            weight,
            backend,
        )
        task_hint = self._yolo_task()
        try:
            return YOLO(str(weight), task=task_hint)  # type: ignore[call-arg]
        except TypeError:
            return YOLO(str(weight))

    def _device_of(self, model: object) -> Optional[str]:
        for attr in ("device", "model"):
            candidate = getattr(model, attr, None)
            device_attr = getattr(candidate, "device", None)
            if device_attr is not None:
                try:
                    return str(device_attr)
                except Exception:
                    continue
            if attr == "device" and candidate is not None:
                try:
                    return str(candidate)
                except Exception:
                    continue
        return None

    def _switch_to(self, idx: int, weight: Path) -> None:
        model = self._instantiate(weight)
        previous_backend = self._backend

        self._model = model
        self._label = weight.name
        self._backend = _detect_backend(weight)
        self._current_idx = idx

        device = self._device_of(model)
        LOGGER.info(
            "weights.select.success task=%s weight=%s backend=%s device=%s",
            self._task,
            weight,
            self._backend,
            device,
        )
        event = "live.model.selected" if not self._has_success else "live.model.switched"
        self._has_success = True
        LOGGER.info(
            "%s task=%s weight=%s backend=%s device=%s",
            event,
            self._task,
            weight,
            self._backend,
            device,
        )

        notice_needed = self._pending_notice is not None or previous_backend is not None
        if notice_needed and self._on_switch is not None:
            message = f"Switched backend -> {self._backend.upper()} ({weight.name})"
            if self._pending_notice:
                message = f"{message} | {self._pending_notice}"
            self._on_switch(message)
        self._pending_notice = None

    def _record_failure(self, weight: Path, exc: BaseException) -> None:
        reason = f"{weight.name}: {type(exc).__name__}: {exc}"
        self._failures.append(reason)
        self._pending_notice = reason
        LOGGER.warning(
            "weights.select.retry task=%s weight=%s reason=%s",
            self._task,
            weight,
            exc,
        )

    def set_on_switch(self, callback: Optional[Callable[[str], None]]) -> None:
        """
        Update the HUD/status callback.

        Enables wrapper reuse across sessions by letting callers swap the toast
        sink without reinstantiating the underlying model.
        """
        self._on_switch = callback

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        while self._candidates:
            for idx, weight in self._iter_candidates():
                try:
                    self._switch_to(idx, weight)
                    return
                except Exception as exc:
                    self._record_failure(weight, exc)
                    self._candidates.pop(idx)
                    break  # restart iteration with updated list
            else:
                break

        joined = "; ".join(self._failures) or "no valid weights"
        raise RuntimeError(
            f"[ResilientYOLO] failed to initialise any weight for task '{self._task}'. "
            f"Reasons: {joined}"
        )

    def _invalidate_current(self) -> None:
        if 0 <= self._current_idx < len(self._candidates):
            failed = self._candidates.pop(self._current_idx)
            LOGGER.warning(
                "weights.select.drop task=%s weight=%s", self._task, failed
            )
            self._pending_notice = self._pending_notice or f"{failed.name}"
        else:
            self._pending_notice = self._pending_notice or "runtime failure"

        self._current_idx = -1
        self._model = None
        self._label = None
        self._backend = None
        setattr(self, "_argos_prepared", False)

    def predict(self, frame: Any, **kwargs: Any) -> Any:
        self._ensure_model()
        assert self._model is not None

        try:
            predict = getattr(self._model, "predict")
            predict_callable = cast(Callable[..., Any], predict)
            call_kwargs: Dict[str, Any] = dict(kwargs)
            call_kwargs.setdefault("conf", self._conf)
            call_kwargs.setdefault("verbose", False)
            return predict_callable(frame, **call_kwargs)
        except Exception as exc:
            if _should_retry(exc) and len(self._candidates) > 1:
                LOGGER.warning(
                    "weights.runtime.retry task=%s weight=%s backend=%s reason=%s",
                    self._task,
                    self._label,
                    self._backend,
                    exc,
                )
                self._invalidate_current()
                return self.predict(frame, **kwargs)
            raise

def _default_thread_count() -> Optional[int]:
    try:
        if psutil is not None:
            physical = psutil.cpu_count(logical=False)
            if physical:
                return int(physical)
    except Exception:
        pass
    try:
        count = os.cpu_count()
        if count:
            return int(count)
    except Exception:
        pass
    return None


def configure_onnxruntime(
    *,
    threads: Optional[int] = None,
    execution: Optional[str] = None,
    enable_cuda_graph: bool = True,
    arena_strategy: Optional[str] = "kNextPowerOfTwo",
    dml_device_id: Optional[int] = None,
) -> None:
    """
    Store desired ORT session configuration and ensure the global patch is applied.

    Parameters mirror the tuning guidance so callers can adjust thread counts or
    execution mode without directly touching onnxruntime APIs.
    """
    if threads is None:
        threads = _default_thread_count()

    execution_mode = (execution or _ORT_CONFIG.get("execution") or "sequential").lower()
    if execution_mode not in {"sequential", "parallel"}:
        execution_mode = "sequential"

    with _ORT_PATCH_LOCK:
        _ORT_CONFIG.update(
            {
                "threads": threads,
                "execution": execution_mode,
                "cuda_graph": bool(enable_cuda_graph),
                "arena_strategy": arena_strategy,
                "dml_device_id": dml_device_id,
            }
        )
    _apply_ort_patch()


def _apply_ort_patch() -> None:
    global _ORT_PATCHED
    with _ORT_PATCH_LOCK:
        if _ORT_PATCHED:
            return
        try:
            import onnxruntime as ort  # type: ignore
        except Exception:
            return
        if getattr(ort, "_argos_patched", False):
            _ORT_PATCHED = True
            return

        original_session = ort.InferenceSession

        def _canonical_providers(value: Optional[Sequence[Any]]) -> Optional[List[Any]]:
            if value is None:
                return None
            tuned: List[Any] = []
            for entry in value:
                if isinstance(entry, str):
                    name = entry
                    opts: Dict[str, Any] = {}
                elif isinstance(entry, (list, tuple)) and entry:
                    name = entry[0]
                    opts = dict(entry[1]) if len(entry) > 1 else {}
                else:
                    continue
                lower = str(name).lower()
                arena_strategy = _ORT_CONFIG.get("arena_strategy")
                if lower.startswith("cuda"):
                    if arena_strategy:
                        opts.setdefault("arena_extend_strategy", arena_strategy)
                    opts.setdefault("cudnn_conv_algo_search", "DEFAULT")
                if "directml" in lower or lower.startswith("dml"):
                    device_id = _ORT_CONFIG.get("dml_device_id")
                    if device_id is not None:
                        opts.setdefault("device_id", int(device_id))
                tuned.append((name, opts))
            return tuned or None

        def _apply_session_config(options: "ort.SessionOptions", providers: Optional[Sequence[Any]]) -> None:
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            execution_mode = _ORT_CONFIG.get("execution", "sequential")
            options.execution_mode = (
                ort.ExecutionMode.ORT_PARALLEL if execution_mode == "parallel" else ort.ExecutionMode.ORT_SEQUENTIAL
            )

            has_gpu = False
            if providers:
                for entry in providers:
                    name = entry[0] if isinstance(entry, (list, tuple)) else entry
                    name_l = str(name).lower()
                    if "cuda" in name_l or "tensorrt" in name_l or "directml" in name_l:
                        has_gpu = True
                        break

            threads = _ORT_CONFIG.get("threads")
            if threads and not has_gpu:
                try:
                    options.intra_op_num_threads = int(threads)
                except Exception:
                    pass
                options.inter_op_num_threads = 1
            options.enable_mem_pattern = True
            options.enable_cpu_mem_arena = True

            if has_gpu and _ORT_CONFIG.get("cuda_graph", True):
                try:
                    options.add_session_config_entry("session.enable_cuda_graph", "1")
                except Exception:
                    pass

        def _patched_inference_session(path_or_bytes, *args, **kwargs):
            providers = kwargs.pop("providers", None)
            provider_options = kwargs.pop("provider_options", None)
            sess_options = kwargs.pop("sess_options", None)
            if args:
                # Allow positional session options/providers
                if sess_options is None and isinstance(args[0], ort.SessionOptions):
                    sess_options = args[0]
                    args = args[1:]
                if providers is None and args:
                    providers = args[0]
                    args = args[1:]
            providers = _canonical_providers(providers)
            if sess_options is None:
                sess_options = ort.SessionOptions()

            _apply_session_config(sess_options, providers)

            if providers is not None and provider_options is None:
                # Convert tuples into separate provider_options argument if required
                provider_options = [{**opts} for _, opts in providers]
                providers = [name for name, _ in providers]

            return original_session(
                path_or_bytes,
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options,
                *args,
                **kwargs,
            )

        ort.InferenceSession = _patched_inference_session  # type: ignore[assignment]
        setattr(ort, "_argos_patched", True)
        _ORT_PATCHED = True
        LOGGER.debug("Applied Argos ONNX Runtime session patch.")

    # -------------------------------
    #  Public API
    # -------------------------------
    def predict(self, frame: Any, **kwargs: Any) -> Any:
        self._ensure_model()
        assert self._model is not None

        try:
            predict = getattr(self._model, "predict")
            predict_callable = cast(Callable[..., Any], predict)
            call_kwargs: dict[str, Any] = dict(kwargs)
            call_kwargs.setdefault("conf", self._conf)
            call_kwargs.setdefault("verbose", False)
            return predict_callable(frame, **call_kwargs)
        except Exception as exc:
            if _should_retry(exc) and len(self._candidates) > 1:
                LOGGER.warning(
                    "weights.runtime.retry task=%s weight=%s backend=%s reason=%s",
                    self._task,
                    self._label,
                    self._backend,
                    exc,
                )
                self._invalidate_current()
                return self.predict(frame, **kwargs)
            raise

