"""
Backend autoconfiguration for live inference.

Invoked during the interactive ``build`` step to detect available
accelerators, ensure required runtime packages are installed, and cache a
preferred backend/resolution profile for later use by the live pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Sequence, cast

DEFAULT_INPUT_SIZE = (640, 640)
DECISION_VERSION = 2

LOGGER = logging.getLogger(__name__)


def _coerce_dim(value: object, fallback: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return fallback
    return fallback


def _parse_input_size(raw: object) -> Tuple[int, int]:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        seq = cast(Sequence[Any], raw)
        try:
            if len(seq) >= 2:
                width = _coerce_dim(seq[0], DEFAULT_INPUT_SIZE[0])
                height = _coerce_dim(seq[1], DEFAULT_INPUT_SIZE[1])
                return (width, height)
        except Exception:
            return DEFAULT_INPUT_SIZE
    return DEFAULT_INPUT_SIZE

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from panoptes.runtime import backend_probe  # type: ignore[import]
except Exception:  # pragma: no cover
    backend_probe = None  # type: ignore[assignment]

try:
    from projects.argos import bootstrap  # type: ignore[import]
except Exception:  # pragma: no cover
    bootstrap = None  # type: ignore[assignment]

CONFIG_APP_NAME = "rAIn"
BACKEND_CACHE_NAME = "live_backend.json"


def _config_dir() -> Path:
    if os.name == "nt":
        base = Path(os.getenv("APPDATA") or os.getenv("LOCALAPPDATA") or (Path.home() / "AppData" / "Roaming"))
        return base / CONFIG_APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / CONFIG_APP_NAME
    return Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / CONFIG_APP_NAME


CACHE_PATH = _config_dir() / BACKEND_CACHE_NAME


@dataclass
class BackendDecision:
    version: int
    preferred_backend: str
    preprocess_device: str
    input_size: Tuple[int, int]
    prefer_small: bool
    ort_threads: Optional[int]
    ort_execution: Optional[str]
    nms_mode: Optional[str]
    fingerprint: Dict[str, Any]
    timestamp: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackendDecision":
        decision = data.get("decision", {})
        return cls(
            version=int(data.get("version") or 1),
            preferred_backend=str(decision.get("preferred_backend") or "auto"),
            preprocess_device=str(decision.get("preprocess_device") or "auto"),
            input_size=_parse_input_size(decision.get("input_size")),
            prefer_small=bool(decision.get("prefer_small", True)),
            ort_threads=decision.get("ort_threads"),
            ort_execution=decision.get("ort_execution"),
            nms_mode=decision.get("nms_mode"),
            fingerprint=data.get("fingerprint") or {},
            timestamp=float(data.get("timestamp") or 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "fingerprint": self.fingerprint,
            "decision": {
                "preferred_backend": self.preferred_backend,
                "preprocess_device": self.preprocess_device,
                "input_size": list(self.input_size),
                "prefer_small": self.prefer_small,
                "ort_threads": self.ort_threads,
                "ort_execution": self.ort_execution,
                "nms_mode": self.nms_mode,
            },
        }


def _load_existing_decision() -> Optional[BackendDecision]:
    try:
        if CACHE_PATH.exists():
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            decision = BackendDecision.from_dict(data)
            if decision.version != DECISION_VERSION:
                return None
            return decision
    except Exception:
        pass
    return None


def _write_decision(decision: BackendDecision) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(decision.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


def _torch_gpu_name() -> Optional[str]:
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        idx = torch.cuda.current_device()
        return torch.cuda.get_device_name(idx)
    except Exception:
        return "CUDA"


def _gpu_warmup() -> None:
    """
    Run a lightweight GPU warm-up so the first live frame avoids CUDA cold-start
    penalties (Optimize Research:1696-1714).
    """
    warmed = False
    if torch is not None:
        try:
            if torch.cuda.is_available():
                tensor = torch.zeros((1,), device="cuda", dtype=torch.float32)
                tensor += 1.0
                torch.cuda.synchronize()
                warmed = True
        except Exception as exc:  # pragma: no cover - warm-up failures are non-fatal
            LOGGER.debug("GPU warmup (torch) failed: %s", exc, exc_info=True)
    try:
        import numpy as np  # type: ignore
        import cv2  # type: ignore

        cuda_mod = getattr(cv2, "cuda", None)
        gpu_mat_ctor = getattr(cuda_mod, "GpuMat", None) if cuda_mod is not None else None
        if callable(gpu_mat_ctor):
            gpu_mat = cast(Any, gpu_mat_ctor())
            gpu_mat.upload(np.zeros((32, 32, 3), dtype=np.uint8))
            gpu_mat.download()
            warmed = True
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("GPU warmup (cv2) skipped/failed: %s", exc, exc_info=True)
    if warmed:
        LOGGER.debug("GPU warmup completed successfully")


def _gather_fingerprint(ort_status: Optional[Any], gpu_name: Optional[str]) -> Dict[str, Any]:
    providers_raw = list(getattr(ort_status, "providers", []) or []) if ort_status else []
    providers: list[str] = [str(p) for p in providers_raw]
    return {
        "platform": platform.platform(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "gpu": gpu_name or "",
        "ort_version": getattr(ort_status, "version", None) if ort_status else None,
        "providers": providers,
    }


def _ensure_onnxruntime_available() -> Any:
    if backend_probe is None:
        return None
    status = backend_probe.ort_available()
    if status.ok:
        return status
    if bootstrap is not None and hasattr(bootstrap, "ensure_onnxruntime"):
        try:
            ensure = getattr(bootstrap, "ensure_onnxruntime")
            venv_py = None
            if hasattr(bootstrap, "venv_python"):
                try:
                    venv_py = bootstrap.venv_python()
                except Exception:
                    venv_py = None

            def _silent_log(_msg: str) -> None:
                return None

            ensure(venv_py, log=_silent_log)
        except Exception:
            pass
        status = backend_probe.ort_available()
    return status


def _decide_backend() -> BackendDecision:
    status = _ensure_onnxruntime_available()
    providers_raw = list(getattr(status, "providers", []) or []) if status else []
    providers: list[str] = [str(p) for p in providers_raw]
    providers_lower: list[str] = [p.lower() for p in providers]
    has_tensorrt = any("tensorrt" in p for p in providers_lower)
    has_cuda_provider = any("cuda" in p for p in providers_lower)
    gpu_name = _torch_gpu_name()
    torch_cuda = bool(gpu_name)
    cpu_count = os.cpu_count() or 0

    preferred_backend = "auto"
    preprocess_device = "auto"
    prefer_small = True
    input_size = (640, 640)
    ort_threads: Optional[int] = None
    ort_execution: Optional[str] = None
    nms_mode: Optional[str] = None

    disable_tensorrt = os.environ.get("ORT_DISABLE_TENSORRT", "").strip().lower() in {"1", "true", "yes", "on"}
    if has_tensorrt and not disable_tensorrt:
        preferred_backend = "tensorrt"
        preprocess_device = "gpu"
        prefer_small = True
        input_size = DEFAULT_INPUT_SIZE
        nms_mode = "graph"
    elif (status and status.ok and has_cuda_provider) or torch_cuda:
        if status and status.ok and has_cuda_provider:
            preferred_backend = "ort"
            preprocess_device = "gpu"
            nms_mode = "graph"
        elif torch_cuda:
            preferred_backend = "torch"
            preprocess_device = "gpu"
            nms_mode = "torch"
        prefer_small = True
        input_size = DEFAULT_INPUT_SIZE
    else:
        preferred_backend = "ort" if status and status.ok else "torch"
        preprocess_device = "cpu"
        prefer_small = True
        input_size = DEFAULT_INPUT_SIZE
        ort_threads = max(1, cpu_count) if cpu_count else None
        ort_execution = "sequential"
        nms_mode = "auto"

    fingerprint = _gather_fingerprint(status, gpu_name)
    decision = BackendDecision(
        version=DECISION_VERSION,
        preferred_backend=preferred_backend,
        preprocess_device=preprocess_device,
        input_size=input_size,
        prefer_small=prefer_small,
        ort_threads=ort_threads,
        ort_execution=ort_execution,
        nms_mode=nms_mode,
        fingerprint=fingerprint,
        timestamp=time.time(),
    )
    if decision.preprocess_device == "gpu":
        _gpu_warmup()
    return decision


def ensure_backend_decision(force: bool = False) -> BackendDecision:
    existing = _load_existing_decision()
    if not force and existing is not None:
        if existing.version != DECISION_VERSION:
            existing = None
        else:
            status = _ensure_onnxruntime_available()
            fingerprint = _gather_fingerprint(status, _torch_gpu_name())
            if fingerprint == existing.fingerprint:
                return existing
    decision = _decide_backend()
    _write_decision(decision)
    return decision


def load_backend_decision() -> Optional[BackendDecision]:
    return _load_existing_decision()


def main(argv: Optional[list[str]] = None) -> None:
    args = argv or sys.argv[1:]
    force = "--force" in args
    ensure_backend_decision(force=force)


if __name__ == "__main__":  # pragma: no cover
    main()
