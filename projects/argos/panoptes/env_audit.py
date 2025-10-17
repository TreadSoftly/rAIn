from __future__ import annotations

import json
import os
import platform
import sys
import warnings
from pathlib import Path
from typing import Any, Dict


def _safe_import(name: str):
    try:
        module = __import__(name)
        for part in name.split(".")[1:]:
            module = getattr(module, part)
        return module
    except Exception:
        return None


def collect_env() -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "os": {
            "platform": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": os.path.realpath(sys.executable),
        },
        "cpu": {
            "count": os.cpu_count(),
            "processor": platform.processor(),
        },
    }

    cv2 = _safe_import("cv2")
    if cv2 is not None:
        info: Dict[str, Any] = {"version": getattr(cv2, "__version__", "unknown")}
        try:
            info["build_info"] = cv2.getBuildInformation()
        except Exception:
            pass
        data["opencv"] = info

    ort = _safe_import("onnxruntime")
    if ort is not None:
        ort_info: Dict[str, Any] = {"version": getattr(ort, "__version__", "unknown")}
        try:
            ort_info["providers"] = ort.get_available_providers()
        except Exception:
            pass
        data["onnxruntime"] = ort_info

    torch = _safe_import("torch")
    if torch is not None:
        torch_info: Dict[str, Any] = {"version": getattr(torch, "__version__", "unknown")}
        try:
            torch_info["cuda_available"] = bool(torch.cuda.is_available())
            torch_info["cuda_device_count"] = torch.cuda.device_count()
            torch_info["cuda_version"] = torch.version.cuda if hasattr(torch.version, "cuda") else None
        except Exception:
            pass
        data["torch"] = torch_info

    pil_features = _safe_import("PIL.features")
    pil = _safe_import("PIL")
    if pil is not None:
        pil_info: Dict[str, Any] = {"version": getattr(pil, "__version__", "unknown")}
        if pil_features is not None and hasattr(pil_features, "check"):
            def _feature_enabled(name: str) -> bool:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        return bool(pil_features.check(name))
                except Exception:
                    return False

            pil_info["has_avif"] = _feature_enabled("avif")
            pil_info["has_heif"] = _feature_enabled("heif")
        data["pillow"] = pil_info

    return data


def write_snapshot(run_dir: Path, snapshot: Dict[str, Any] | None = None) -> Path:
    snap = snapshot or collect_env()
    target = run_dir / "env.json"
    target.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    return target
