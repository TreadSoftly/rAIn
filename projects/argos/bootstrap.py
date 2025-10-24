#!/usr/bin/env python3
"""
Argos bootstrap - zero-touch, idempotent, fast.

What it does (one-time on first run):
  • creates a private venv OUTSIDE the repo (no .venv mess)
  • auto-selects CPU-only Torch when CUDA isn’t present
  • installs your project in editable mode (dev extras optional)
  • fetches model weights via Ultralytics (no git/LFS)
    - preset: all | default | nano | perception (ARGOS_WEIGHT_PRESET)
    - or interactively: pick specific files from a list
    - exports missing .onnx from matching .pt when needed
  • writes portable launchers:  argos  |  argos.ps1  |  argos.cmd
    (they call this file to --ensure, then run the CLI with the venv python)
  • relocates pytest cache outside the repo
  • prints a compact first-run quickstart

After that:
  • running ./argos (macOS/Linux) or argos.cmd (Windows) “just works”
  • no venv activation needed, ever
  • quick checks skip already-installed things (no time wasted)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Set,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

HERE: Path = Path(__file__).resolve().parent
ARGOS: Path = (
    HERE
    if (HERE / "pyproject.toml").exists() and (HERE / "panoptes").exists()
    else (HERE / "projects" / "argos")
)

# Ensure the project sources are importable even under isolated (-I) Python runs.
# Windows Sandbox + fresh installs can launch python -I, which omits the script
# directory from sys.path, so we add both the bootstrap location and the repo root.
for candidate in (HERE, ARGOS):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from panoptes.logging_config import setup_logging  # type: ignore[import]
from panoptes.runtime.onnx_spec import desired_ort_spec  # type: ignore[import]

try:
    from panoptes.model_registry import WEIGHT_PRIORITY as _BOOTSTRAP_WEIGHT_PRIORITY  # type: ignore[import]
except Exception:  # pragma: no cover - bootstrap still works even if registry import fails
    _BOOTSTRAP_WEIGHT_PRIORITY: dict[str, list[Path]] = {} # type: ignore[assignment]

# Diagnostics (always try to attach; never crash on failure)
try:
    import importlib

    importlib.import_module("panoptes.diagnostics")
except Exception:
    pass

# ──────────────────────────────────────────────────────────────
# Optional: Progress UI (safe fallbacks if deps missing / CI / non-TTY)
# ──────────────────────────────────────────────────────────────
try:
    from panoptes.progress import (  # type: ignore[import]
        ProgressEngine,  # type: ignore
        live_percent,
    )
except Exception:
    ProgressEngine = None  # type: ignore[assignment]
    live_percent = None  # type: ignore[assignment]

JSONDict = Dict[str, Any]
APP = "rAIn"

if os.name == "nt":
    os.environ.setdefault("PYTHONUTF8", "1")

_TRUTHY = {"1", "true", "yes", "on"}


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in _TRUTHY


def _ensure_default_tensorrt_env() -> None:
    if os.environ.get("ORT_DISABLE_TENSORRT"):
        return
    if _is_truthy(os.environ.get("ARGOS_ENABLE_TENSORRT", "")):
        return
    os.environ["ORT_DISABLE_TENSORRT"] = "1"


_ensure_default_tensorrt_env()

setup_logging()
_LOG = logging.getLogger(__name__)
_LOG.addHandler(logging.NullHandler())
_LOG.setLevel(logging.ERROR)

StrPath = Union[str, PathLike[str]]

def _cfg_dir() -> Path:
    if os.name == "nt":
        base = Path(os.getenv("APPDATA") or os.getenv("LOCALAPPDATA") or (Path.home() / "AppData" / "Roaming"))
        return base / APP
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP
    return Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / APP

def _data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or (Path.home() / "AppData" / "Local"))
        return base / APP
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP
    return Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / APP

CFG: Path = _cfg_dir()
DATA: Path = _data_dir()
VENVS: Path = DATA / "venvs"
SENTINEL: Path = CFG / "first_run.json"

VENV: Path = VENVS / f"py{sys.version_info.major}{sys.version_info.minor}-argos"
STATE: Path = DATA / "state"
CAPABILITIES_FILE: Path = STATE / "capabilities.json"
CUDA_STATE_DIR: Path = STATE / "cuda"


def _venv_executable_path() -> Path:
    """Return the path where the Argos venv Python should live (may not exist yet)."""
    return VENV / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")


def venv_python(*, ensure: bool = True) -> Path:
    """
    Resolve the Argos venv interpreter path, creating the venv if requested and missing.

    Args:
        ensure: When True, `_create_venv()` is invoked if the interpreter is absent.
    """
    py = _venv_executable_path()
    if ensure and not py.exists():
        _create_venv()
    return py

def _print(msg: object = "") -> None:
    s = str(msg)
    if not s.endswith("\n"):
        s += "\n"
    try:
        sys.stdout.write(s)
    except UnicodeEncodeError:
        try:
            sys.stdout.buffer.write(s.encode("utf-8", "replace"))
        except Exception:
            sys.stdout.write(s.encode("ascii", "replace").decode("ascii"))

@overload
def _run(
    cmd: Sequence[str],
    *,
    cwd: Optional[StrPath] = ...,
    env: Optional[Mapping[str, str]] = ...,
    check: bool = ...,
    capture: Literal[True],
) -> subprocess.CompletedProcess[str]: ...
@overload
def _run(
    cmd: Sequence[str],
    *,
    cwd: Optional[StrPath] = ...,
    env: Optional[Mapping[str, str]] = ...,
    check: bool = ...,
    capture: Literal[False] = ...,
) -> None: ...
def _run(
    cmd: Sequence[str],
    *,
    cwd: Optional[StrPath] = None,
    env: Optional[Mapping[str, str]] = None,
    check: bool = True,
    capture: bool = False,
):
    if capture:
        return subprocess.run(
            list(cmd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
            cwd=cwd,
            env=dict(env) if isinstance(env, MutableMapping) else env,
        )
    subprocess.run(list(cmd), check=check, cwd=cwd, env=dict(env) if isinstance(env, MutableMapping) else env)
    return None

def _ensure_dirs() -> None:
    for p in (CFG, DATA, VENVS, STATE, CUDA_STATE_DIR):
        p.mkdir(parents=True, exist_ok=True)


def _site_packages_root() -> Path:
    if os.name == "nt":
        return VENV / "Lib" / "site-packages"
    return VENV / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def _as_str_key_dict(obj: object) -> Dict[str, Any]:
    if isinstance(obj, Mapping):
        mapping_view = cast(Mapping[Any, Any], obj)
        return {str(key): value for key, value in mapping_view.items()}
    return {}


def _module_present(mod: str) -> bool:
    code = f"import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('{mod}') else 1)"
    try:
        _run([str(venv_python()), "-c", code], check=True, capture=False)
        return True
    except Exception:
        return False

def _has_distribution(name: str) -> bool:
    try:
        from importlib.metadata import PackageNotFoundError, distribution  # type: ignore
    except Exception:  # pragma: no cover
        from importlib_metadata import PackageNotFoundError, distribution  # type: ignore
    try:
        distribution(name)
        return True
    except PackageNotFoundError:
        return False


def _gather_paths_from_root(root: Path, *, include_nvidia_prefix: bool) -> list[Path]:
    if not root.exists():
        return []
    patterns = ("nvidia/**/bin", "nvidia/**/lib") if include_nvidia_prefix else ("**/bin", "**/lib")
    discovered: list[Path] = []
    for pattern in patterns:
        try:
            for candidate in root.glob(pattern):
                try:
                    if candidate.is_dir():
                        resolved = candidate.resolve()
                        if resolved not in discovered:
                            discovered.append(resolved)
                except Exception:
                    continue
        except Exception:
            continue
    return discovered


_REQUIRED_CUDA_DLL_SETS: tuple[tuple[str, ...], ...] = (
    ("cublas64_12.dll",),
    ("cublasLt64_12.dll",),
    ("cudart64_12.dll",),
    ("cudnn64_9.dll",),
    ("cudnn_ops64_9.dll", "cudnn_ops_infer64_9.dll", "cudnn_ops_train64_9.dll"),
    ("cudnn_cnn64_9.dll", "cudnn_cnn_infer64_9.dll", "cudnn_cnn_train64_9.dll"),
    ("cudnn_adv64_9.dll",),
)


def _discover_cuda_library_dirs() -> list[Path]:
    dirs: list[Path] = []
    site_root = _site_packages_root()
    dirs.extend(_gather_paths_from_root(site_root, include_nvidia_prefix=True))
    dirs.extend(_gather_paths_from_root(CUDA_STATE_DIR, include_nvidia_prefix=False))
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in dirs:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _detect_missing_cuda_dlls(directories: Sequence[Path]) -> list[str]:
    missing: list[str] = []
    normalized = [p.resolve() for p in directories if p.exists()]
    for variants in _REQUIRED_CUDA_DLL_SETS:
        found = False
        for directory in normalized:
            for candidate in variants:
                if (directory / candidate).exists():
                    found = True
                    break
            if found:
                break
        if not found:
            missing.append(variants[0])
    return missing


def _collect_torch_metadata() -> Dict[str, Any]:
    vpy = str(venv_python())
    script = textwrap.dedent(
        """\
        import json
        import sys

        payload: dict[str, object] = {}
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - diagnostics only
            payload["installed"] = False
            payload["error"] = f"{type(exc).__name__}: {exc}"
        else:
            payload["installed"] = True
            payload["version"] = getattr(torch, "__version__", None)
            cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
            payload["cuda_version"] = str(cuda_version) if cuda_version is not None else None
            try:
                payload["cuda_available"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
            except Exception as exc:  # pragma: no cover - diagnostics only
                payload["cuda_available"] = False
                payload["cuda_error"] = f"{type(exc).__name__}: {exc}"
        sys.stdout.write(json.dumps(payload))
        """
    )
    proc = subprocess.run(
        [vpy, "-c", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.stdout:
        try:
            data = json.loads(proc.stdout.strip())
            return _as_str_key_dict(data)
        except Exception:
            pass
    return {"installed": False, "error": proc.stderr.strip() or "torch metadata unavailable"}


def _collect_torchaudio_metadata() -> Dict[str, Any]:
    vpy = str(venv_python())
    script = textwrap.dedent(
        """\
        import json
        import sys

        payload: dict[str, object] = {}
        try:
            import torchaudio  # type: ignore
        except Exception as exc:  # pragma: no cover - diagnostics only
            payload["installed"] = False
            payload["error"] = f"{type(exc).__name__}: {exc}"
        else:
            payload["installed"] = True
            payload["version"] = getattr(torchaudio, "__version__", None)
        sys.stdout.write(json.dumps(payload))
        """
    )
    proc = subprocess.run(
        [vpy, "-c", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.stdout:
        try:
            data = json.loads(proc.stdout.strip())
            return _as_str_key_dict(data)
        except Exception:
            pass
    return {"installed": False, "error": proc.stderr.strip() or "torchaudio metadata unavailable"}


def _torch_meta_has_cuda(meta: object) -> bool:
    meta_dict = _as_str_key_dict(meta)
    if not meta_dict:
        return False
    cuda_version_obj = meta_dict.get("cuda_version")
    if isinstance(cuda_version_obj, str) and cuda_version_obj.strip():
        return True
    cuda_available_obj = meta_dict.get("cuda_available")
    if isinstance(cuda_available_obj, bool):
        return cuda_available_obj
    return bool(cuda_available_obj)


def _collect_ort_metadata() -> Dict[str, Any]:
    vpy = str(venv_python())
    script = textwrap.dedent(
        """\
        import json
        import sys
        from typing import Optional

        def _detect_package() -> Optional[dict[str, object]]:
            try:
                import importlib.metadata as importlib_metadata  # type: ignore[attr-defined]
            except Exception:
                try:
                    import importlib_metadata  # type: ignore[type-arg]
                except Exception:
                    return None
            for name in ("onnxruntime-gpu", "onnxruntime-directml", "onnxruntime"):
                try:
                    dist = importlib_metadata.distribution(name)
                except Exception:
                    continue
                meta_name = getattr(dist, "metadata", {}).get("Name", None)
                pretty = meta_name if isinstance(meta_name, str) and meta_name.strip() else name
                return {"name": pretty, "version": getattr(dist, "version", None)}
            return None

        payload: dict[str, object] = {}
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover - diagnostics only
            payload["available"] = False
            payload["error"] = f"{type(exc).__name__}: {exc}"
        else:
            payload["available"] = True
            payload["version"] = getattr(ort, "__version__", None)
            package_meta = _detect_package()
            if package_meta is not None:
                payload["package"] = package_meta
            try:
                providers = list(getattr(ort, "get_available_providers", lambda: [])())
            except Exception as exc:  # pragma: no cover - diagnostics only
                providers = []
                payload["providers_error"] = f"{type(exc).__name__}: {exc}"
            payload["providers"] = providers
        sys.stdout.write(json.dumps(payload))
        """
    )
    proc = subprocess.run(
        [vpy, "-c", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.stdout:
        try:
            data = json.loads(proc.stdout.strip())
            return _as_str_key_dict(data)
        except Exception:
            pass
    return {"available": False, "error": proc.stderr.strip() or "onnxruntime metadata unavailable"}


def _preferred_accelerator_from_env() -> str:
    return os.getenv("ARGOS_ACCELERATOR", "").strip().lower()


def _refresh_capabilities_cache(log: bool = False) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    env_preferred = _preferred_accelerator_from_env()
    info["timestamp"] = time.time()
    info["torch"] = _collect_torch_metadata()
    info["onnxruntime"] = _collect_ort_metadata()

    if _last_onnx_summary:
        summary_map: Dict[str, Any] = {str(key): value for key, value in _last_onnx_summary.items()}
        ort_meta_obj = _as_str_key_dict(info.get("onnxruntime") or {})
        providers_obj = summary_map.get("providers")
        if isinstance(providers_obj, Sequence) and not isinstance(providers_obj, (str, bytes)):
            providers_seq = cast(Sequence[Any], providers_obj)
            ort_meta_obj["providers"] = [str(provider) for provider in providers_seq]
        expected_provider = _coerce_optional_str(summary_map.get("expected_provider"))
        if expected_provider:
            ort_meta_obj["expected_provider"] = expected_provider
        package_name = _coerce_optional_str(summary_map.get("package"))
        if package_name:
            ort_meta_obj["package"] = package_name
        spec_str = _coerce_optional_str(summary_map.get("spec"))
        if spec_str:
            ort_meta_obj["spec"] = spec_str
        providers_ok = summary_map.get("providers_ok")
        if isinstance(providers_ok, bool):
            ort_meta_obj["providers_ok"] = providers_ok
        elif providers_ok is not None:
            ort_meta_obj["providers_ok"] = bool(providers_ok)
        validation_obj = summary_map.get("validation")
        if isinstance(validation_obj, Mapping):
            validation_map_input = cast(Mapping[Any, Any], validation_obj)
            validation_map = {str(key): value for key, value in validation_map_input.items()}
            validation_payload: Dict[str, Any] = {"ok": bool(validation_map.get("ok"))}
            validation_error = _coerce_optional_str(validation_map.get("error"))
            if validation_error:
                validation_payload["error"] = validation_error
            validation_providers = validation_map.get("providers")
            if isinstance(validation_providers, Sequence) and not isinstance(validation_providers, (str, bytes)):
                validation_payload["providers"] = [
                    str(provider) for provider in cast(Sequence[Any], validation_providers)
                ]
            ort_meta_obj["validation"] = validation_payload
        heal_flag = summary_map.get("healed")
        if isinstance(heal_flag, bool):
            ort_meta_obj["last_heal"] = {"healed": heal_flag, "timestamp": time.time()}
        elif heal_flag is not None:
            ort_meta_obj["last_heal"] = {"healed": bool(heal_flag), "timestamp": time.time()}
        info["onnxruntime"] = ort_meta_obj

    cuda_dirs: List[Path] = _discover_cuda_library_dirs()
    info["cuda"] = {
        "dll_dirs": [str(path_obj) for path_obj in cuda_dirs],
        "missing_dlls": _detect_missing_cuda_dlls(cuda_dirs),
    }
    preferred_accelerator = env_preferred
    if not preferred_accelerator:
        ort_meta_obj = _as_str_key_dict(info.get("onnxruntime") or {})
        providers_obj = ort_meta_obj.get("providers")
        if isinstance(providers_obj, Sequence) and not isinstance(providers_obj, (str, bytes)):
            providers_seq = cast(Sequence[Any], providers_obj)
            lowered = [str(provider).lower() for provider in providers_seq]
            if any("cuda" in p for p in lowered):
                preferred_accelerator = "cuda"
            elif any("directml" in p or "dml" in p for p in lowered):
                preferred_accelerator = "directml"
        if not preferred_accelerator:
            cuda_meta_obj = _as_str_key_dict(info.get("cuda") or {})
            if cuda_meta_obj.get("dll_dirs"):
                preferred_accelerator = "cuda"
    info["preferred_accelerator"] = preferred_accelerator or "cpu"
    try:
        CAPABILITIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        CAPABILITIES_FILE.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        if log:
            _print(f"ensure warning: could not write capability cache ({exc})")
    return info


def read_capabilities() -> Dict[str, Any]:
    try:
        if CAPABILITIES_FILE.exists():
            data = json.loads(CAPABILITIES_FILE.read_text(encoding="utf-8"))
            return _as_str_key_dict(data)
    except Exception:
        pass
    return {}


def _record_capabilities_step() -> None:
    _refresh_capabilities_cache(log=True)

def _nvidia_smi_candidates() -> list[str]:
    candidates: list[str] = []
    smi = shutil.which("nvidia-smi")
    if smi:
        candidates.append(smi)
    windows_root = os.environ.get("SystemRoot", "")
    program_files = os.environ.get("ProgramFiles", "")
    extra = [
        os.path.join(windows_root, "System32", "nvidia-smi.exe"),
        os.path.join(program_files, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
        "/bin/nvidia-smi",
    ]
    seen: set[str] = set()
    for path in candidates + extra:
        if not path:
            continue
        norm = os.path.normpath(path)
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        if os.path.exists(norm):
            candidates.append(norm)
    # Drop the initial executable duplicates we introduced above.
    unique: list[str] = []
    seen.clear()
    for path in candidates:
        norm = os.path.normpath(path)
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(norm)
    return unique


def _has_cuda() -> bool:
    accel_hint = os.getenv("ARGOS_ACCELERATOR", "").strip().lower()
    if accel_hint == "cuda":
        return True
    if os.getenv("CUDA_VISIBLE_DEVICES", "").strip():
        return True
    for smi in _nvidia_smi_candidates():
        try:
            _run([smi, "-L"], check=True, capture=True)
            return True
        except Exception:
            continue
    return False

def _constraints_args() -> list[str]:
    c = ARGOS / "constraints.txt"
    return ["-c", str(c)] if c.exists() else []

def _create_venv() -> None:
    py = _venv_executable_path()
    if VENV.exists() and py.exists():
        return
    _print("→ creating virtual environment (outside repo)…")
    _run([sys.executable, "-m", "venv", str(VENV)], check=True, capture=False)
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True, capture=False)

def _torch_pins_from_requirements() -> Tuple[str, str]:
    torch_spec = "torch"
    tv_spec = "torchvision"
    req = ARGOS / "requirements.txt"
    if req.exists():
        for line in req.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            low = s.lower()
            if low.startswith(("torch==", "torch>", "torch<", "torch>=", "torch<=")):
                torch_spec = s
            if low.startswith(("torchvision==", "torchvision>", "torchvision<", "torchvision>=", "torchvision<=")):
                tv_spec = s
    env_torch = os.getenv("ARGOS_TORCH_SPEC", "").strip()
    if env_torch:
        torch_spec = env_torch if env_torch.startswith("torch") else f"torch=={env_torch}"
    env_tv = os.getenv("ARGOS_TORCHVISION_SPEC", "").strip()
    if env_tv:
        tv_spec = env_tv if env_tv.startswith("torch") else f"torchvision=={env_tv}"
    return torch_spec, tv_spec


_CUDA_RUNTIME_PACKAGES: tuple[str, ...] = (
    "nvidia-cuda-runtime-cu12>=12.1.105",
    "nvidia-cublas-cu12>=12.1.3.1",
    "nvidia-cudnn-cu12>=9.1.0.70",
    "nvidia-curand-cu12>=10.3.2.106",
    "nvidia-cusolver-cu12>=11.4.5.107",
    "nvidia-cusparse-cu12>=12.1.2.106",
    "nvidia-cufft-cu12>=11.0.2.107",
    "nvidia-nvjitlink-cu12>=12.3.101",
)

TorchInstallPlan = Tuple[List[str], str, Optional[List[str]], bool]

def _ensure_numpy_floor_for_torch() -> None:
    """
    Torch/Tv CPU wheels for recent Python require NumPy >=2.1 (Py>=3.10).
    When running under Python 3.9 we stick to the 1.26.x ladder, which is the
    newest NumPy available for that interpreter while still satisfying Torch/Tv.
    """
    if sys.version_info >= (3, 10):
        spec = "numpy>=2.1,<2.3"
        note = "NumPy >=2.1,<2.3"
    else:
        spec = "numpy>=1.26,<2.0"
        note = "NumPy >=1.26,<2.0"
    _print(f"-> ensuring {note} for Torch/Tv .")
    _run([str(venv_python()), "-m", "pip", "install", "--upgrade", *_constraints_args(), spec], check=True, capture=False)


def _ensure_cuda_runtime_packages() -> None:
    accelerator = os.getenv("ARGOS_ACCELERATOR", "").strip().lower()
    if not accelerator and _has_cuda():
        accelerator = "cuda"
    if accelerator != "cuda":
        return
    _print("-> ensuring CUDA runtime wheels …")
    cmd = [
        str(venv_python()),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--only-binary=:all:",
        *_constraints_args(),
        *_CUDA_RUNTIME_PACKAGES,
    ]
    _run(cmd, check=True, capture=False)

def _install_torch_if_needed(cpu_only: bool) -> None:
    """
    Install Torch/Torchvision if missing with robust fallbacks.

    Preference order:
      1) Respect explicit specs from requirements/env (with accelerator-aware indexes).
      2) Retry once NumPy floor is ensured.
      3) Try curated accelerator-specific fallbacks (CUDA first when enabled).
      4) Fall back to known CPU wheels so the environment remains usable.
    """
    torch_spec, tv_spec = _torch_pins_from_requirements()

    accelerator = os.getenv("ARGOS_ACCELERATOR", "").strip().lower()
    prefer_cuda = accelerator == "cuda" and not cpu_only
    is_macos = sys.platform.startswith("darwin")

    vpy = str(venv_python())
    base_cmd = [vpy, "-m", "pip", "install"]

    torch_present = _module_present("torch")
    tv_present = _module_present("torchvision")
    if torch_present and tv_present:
        torch_meta = _collect_torch_metadata()
        has_cuda_build = _torch_meta_has_cuda(torch_meta)
        if prefer_cuda and not has_cuda_build:
            _print("-> existing Torch build lacks CUDA; reinstalling with CUDA support .")
            try:
                _run([vpy, "-m", "pip", "uninstall", "-y", "torch", "torchvision"], check=False, capture=False)
            except Exception:
                pass
            torch_present = False
            tv_present = False
        else:
            return
    elif torch_present or tv_present:
        # Partial install; remove stale components so we can reinstall cleanly.
        try:
            _run([vpy, "-m", "pip", "uninstall", "-y", "torch", "torchvision"], check=False, capture=False)
        except Exception:
            pass
        torch_present = tv_present = False

    env_index = os.getenv("ARGOS_TORCH_INDEX_URL", "").strip()
    env_extra_index = os.getenv("ARGOS_TORCH_EXTRA_INDEX_URL", "").strip()

    primary_idx: List[str] = []
    if env_index:
        primary_idx.extend(["--index-url", env_index])
    if env_extra_index:
        primary_idx.extend(["--extra-index-url", env_extra_index])

    if not primary_idx and prefer_cuda:
        primary_idx = ["--index-url", "https://download.pytorch.org/whl/cu121", "--extra-index-url", "https://pypi.org/simple"]
    if not primary_idx and cpu_only and (os.name == "nt" or sys.platform.startswith("linux")):
        primary_idx = ["--index-url", "https://download.pytorch.org/whl/cpu"]

    gpu_idx: List[str] = primary_idx if prefer_cuda else ["--index-url", "https://download.pytorch.org/whl/cu121", "--extra-index-url", "https://pypi.org/simple"]
    cpu_idx: List[str] = [] if is_macos else ["--index-url", "https://download.pytorch.org/whl/cpu"]

    def _effective_idx(override: Optional[List[str]]) -> List[str]:
        if override is not None:
            return override
        return primary_idx

    def _try_install(pkgs: List[str], label: str, idx_override: Optional[List[str]] = None) -> bool:
        try:
            _print(f"-> installing Torch ({label}) .")
            cmd = [*base_cmd, *_effective_idx(idx_override), *_constraints_args(), *pkgs]
            _run(cmd, check=True, capture=False)
            return True
        except Exception:
            return False

    attempted: Set[Tuple[str, str]] = set()

    def _run_sequence(sequence: Sequence[TorchInstallPlan]) -> bool:
        for pkgs, label, idx_override, require_cuda in sequence:
            key = (pkgs[0], pkgs[1] if len(pkgs) > 1 else "")
            if key in attempted:
                continue
            attempted.add(key)
            if not _try_install(pkgs, label, idx_override=idx_override):
                continue
            if require_cuda:
                meta = _collect_torch_metadata()
                if _torch_meta_has_cuda(meta):
                    return True
                _print("-> Torch installed without CUDA support; trying next CUDA candidate .")
                try:
                    _run([vpy, "-m", "pip", "uninstall", "-y", "torch", "torchvision"], check=False, capture=False)
                except Exception:
                    pass
                continue
            return True
        return False

    has_gpu_capability = prefer_cuda or (_has_cuda() and not cpu_only)
    if has_gpu_capability:
        prefer_cuda = True

    gpu_candidates: List[TorchInstallPlan] = []
    if has_gpu_capability:
        gpu_candidates.extend(
            [
                (["torch==2.5.1+cu121", "torchvision==0.20.1+cu121"], "fallback torch==2.5.1+cu121/torchvision==0.20.1+cu121", gpu_idx, True),
                (["torch==2.4.1+cu121", "torchvision==0.19.1+cu121"], "fallback torch==2.4.1+cu121/torchvision==0.19.1+cu121", gpu_idx, True),
            ]
        )

    installed = False
    if gpu_candidates and _run_sequence(gpu_candidates):
        installed = True

    if not installed:
        general_candidates: List[TorchInstallPlan] = [([torch_spec, tv_spec], "requirements", None, prefer_cuda)]
        if _run_sequence(general_candidates):
            installed = True

    if not installed:
        _ensure_numpy_floor_for_torch()
        numpy_candidates: List[TorchInstallPlan] = [([torch_spec, tv_spec], "requirements+numpy", None, prefer_cuda)]
        if _run_sequence(numpy_candidates):
            installed = True

    if not installed:
        if is_macos:
            cpu_pairs: Sequence[List[str]] = (
                ["torch==2.4.1", "torchvision==0.19.1"],
                ["torch==2.5.1", "torchvision==0.20.1"],
            )
        else:
            cpu_pairs = (
                ["torch==2.4.1+cpu", "torchvision==0.19.1+cpu"],
                ["torch==2.5.1+cpu", "torchvision==0.20.1+cpu"],
            )
        cpu_candidates: List[TorchInstallPlan] = [
            (pair, f"fallback {'/'.join(pair)}", cpu_idx, False) for pair in cpu_pairs
        ]
        if _run_sequence(cpu_candidates):
            installed = True

    if not installed:
        raise RuntimeError("Torch/Torchvision installation failed after multiple attempts.")

    torch_meta = _collect_torch_metadata()
    torch_installed = bool(torch_meta.get("installed"))
    if torch_installed:
        torch_version_obj = torch_meta.get("version")
        torch_version = str(torch_version_obj) if isinstance(torch_version_obj, str) else ""
        if torch_version:
            audio_meta = _collect_torchaudio_metadata()
            audio_version_obj = audio_meta.get("version")
            audio_version = str(audio_version_obj) if isinstance(audio_version_obj, str) else ""
            if audio_version != torch_version:
                spec = f"torchaudio=={torch_version}"
                index_override: Optional[List[str]]
                if "+cu" in torch_version:
                    index_override = gpu_idx
                elif "+cpu" in torch_version:
                    index_override = cpu_idx
                else:
                    index_override = None
                _try_install([spec], f"align torchaudio {torch_version}", idx_override=index_override)

    return

def _ensure_opencv_gui() -> None:
    _print("-> ensuring OpenCV (GUI-capable) .")
    uninstall_targets = [
        pkg for pkg in ("opencv-python-headless", "opencv-contrib-python-headless")
        if _has_distribution(pkg)
    ]
    if uninstall_targets:
        _run([str(venv_python()), "-m", "pip", "uninstall", "-y", *uninstall_targets], check=False, capture=False)
    _run([str(venv_python()), "-m", "pip", "install", "--upgrade", *_constraints_args(), "opencv-python"], check=True, capture=False)

def _pip_install_editable_if_needed(
    *,
    reinstall: bool = False,
    with_dev: bool = False,
    extras: Optional[Sequence[str]] = None,
) -> None:
    need = reinstall or (not _module_present("panoptes")) or (not _module_present("ultralytics"))

    headless_present = _has_distribution("opencv-python-headless")

    if headless_present:
        need = True
        _print("→ removing opencv-python-headless (if installed) …")
        _run([str(venv_python()), "-m", "pip", "uninstall", "-y", "opencv-python-headless"], check=False, capture=False)

    if not need:
        return

    install_extras: list[str] = []
    if extras:
        for item in extras:
            cleaned = item.strip()
            if cleaned and cleaned not in install_extras:
                install_extras.append(cleaned)
    if with_dev and "dev" not in install_extras:
        install_extras.append("dev")

    suffix = ""
    if install_extras:
        suffix = " [" + ",".join(install_extras) + "]"
    _print(f"→ installing Argos package (editable){suffix} …")

    extra_token = ""
    if install_extras:
        extra_token = "[" + ",".join(install_extras) + "]"

    _run(
        [str(venv_python()), "-m", "pip", "install", "-e", str(ARGOS) + extra_token, *_constraints_args()],
        check=True,
        capture=False,
    )


def _ensure_sympy_alignment() -> None:
    try:
        from importlib.metadata import PackageNotFoundError, version as pkg_version  # type: ignore
    except Exception:  # pragma: no cover
        from importlib_metadata import PackageNotFoundError, version as pkg_version  # type: ignore

    try:
        torch_version = cast(str, pkg_version("torch"))
    except PackageNotFoundError:
        return
    except Exception:
        torch_version = None

    if not torch_version:
        return

    required_sympy: Optional[str] = None
    if torch_version.startswith("2.5."):
        required_sympy = "1.13.1"

    if not required_sympy:
        return

    try:
        sympy_version = cast(str, pkg_version("sympy"))
    except PackageNotFoundError:
        sympy_version = None
    except Exception:
        sympy_version = None

    if sympy_version == required_sympy:
        return

    _print(f"→ aligning sympy to {required_sympy} …")
    _run(
        [str(venv_python()), "-m", "pip", "install", f"sympy=={required_sympy}", *_constraints_args()],
        check=True,
        capture=False,
    )


def _probe_onnx_runtime(py: Path) -> dict[str, object]:
    """Inspect ONNX/ONNX Runtime availability inside the Argos venv."""
    probe = r"""
import json
result = {
    "onnx": False,
    "onnx_version": None,
    "onnxruntime": False,
    "version": None,
    "providers": None,
    "error": None,
}
errors = []

try:
    import onnx  # noqa: F401
    result["onnx"] = True
    result["onnx_version"] = getattr(onnx, "__version__", None)
except Exception as exc:  # pragma: no cover - diagnostics only
    errors.append(f"onnx: {type(exc).__name__}: {exc}")

try:
    import onnxruntime as ort  # noqa: F401
    result["onnxruntime"] = True
    result["version"] = getattr(ort, "__version__", None)
    try:
        result["providers"] = list(ort.get_available_providers())
    except Exception as exc:  # pragma: no cover
        errors.append(f"providers: {type(exc).__name__}: {exc}")
except Exception as exc:
    errors.append(f"onnxruntime: {type(exc).__name__}: {exc}")

if errors:
    result["error"] = "; ".join(errors)

print(json.dumps(result))
"""
    cp = _run([str(py), "-c", probe], check=False, capture=True)
    data: dict[str, object] = {}
    try:
        payload = (cp.stdout or "").strip()
        if payload:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                data = parsed  # type: ignore[assignment]
    except Exception:
        data = {}
    return data



def _probe_onnx_success(info: dict[str, object]) -> bool:
    """Return True when the probe indicates a healthy ONNX Runtime import."""
    return bool(info.get("onnxruntime")) and not info.get("error")


def _windows_missing_runtime_dlls() -> list[str]:
    """Detect missing Windows runtime DLLs required by onnxruntime."""
    if os.name != "nt":
        return []
    win_root = Path(os.environ.get("SystemRoot") or os.environ.get("WINDIR") or "C\\Windows")
    sys32 = win_root / "System32"
    required = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "vcomp140.dll"]
    missing: list[str] = []
    for name in required:
        path = sys32 / name
        try:
            exists = path.exists()
        except Exception:
            exists = False
        if not exists:
            missing.append(name)
    return missing


def _install_windows_vcredist(
    log: Callable[[str], None],
    record: Callable[[str, dict[str, object]], None],
) -> bool:
    """Download and install the MSVC redistributable silently on Windows."""
    if os.name != "nt":
        return False

    url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    try:
        import urllib.request
    except Exception as exc:  # pragma: no cover - missing stdlib component
        record("download-vcredist", {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return False

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "vc_redist.x64.exe"
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                data = resp.read()
            dest.write_bytes(data)
            record("download-vcredist", {"status": "ok", "size": dest.stat().st_size})
        except Exception as exc:  # pragma: no cover - network issues
            record("download-vcredist", {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            return False

        log("→ installing Microsoft Visual C++ Redistributable (x64)…")
        try:
            proc = subprocess.run(
                [str(dest), "/install", "/quiet", "/norestart"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            record(
                "install-vcredist",
                {
                    "status": "ok" if proc.returncode == 0 else "error",
                    "returncode": proc.returncode,
                    "stdout_tail": (proc.stdout or "")[-400:],
                    "stderr_tail": (proc.stderr or "")[-400:],
                },
            )
            return proc.returncode == 0
        except Exception as exc:  # pragma: no cover
            record("install-vcredist", {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            return False


_last_onnx_summary: Optional[dict[str, object]] = None


def _extract_ort_package(spec: str) -> str:
    """Return the distribution name portion of an onnxruntime specifier."""
    if not spec:
        return "onnxruntime"
    trimmed = spec.strip()
    for idx, ch in enumerate(trimmed):
        if ch in "<>!=[":
            return trimmed[:idx].strip() or "onnxruntime"
    return trimmed


def _replace_ort_package(spec: str, package: str) -> str:
    """Swap the distribution name in the given specifier string."""
    base = _extract_ort_package(spec)
    if not base:
        return package
    if package == base:
        return spec
    return f"{package}{spec[len(base):]}"


def _expected_provider_for_accelerator(accelerator: str) -> Optional[str]:
    """Map an accelerator hint to the provider we expect from ONNX Runtime."""
    accel = accelerator.strip().lower()
    if accel == "cuda":
        return "CUDAExecutionProvider"
    if accel == "directml":
        return "DmlExecutionProvider"
    if accel == "tensorrt":
        return "TensorrtExecutionProvider"
    return None


def _package_for_accelerator(accelerator: str, *, default: str) -> str:
    """Return the preferred onnxruntime wheel for the requested accelerator."""
    accel = accelerator.strip().lower()
    if accel in {"cuda", "tensorrt"}:
        return "onnxruntime-gpu"
    if accel == "directml":
        return "onnxruntime-directml"
    return default


def _coerce_optional_str(value: object) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def ensure_onnxruntime(
    venv_py: Optional[Path] = None,
    *,
    log: Callable[[str], None] = _print,
) -> dict[str, object]:
    """
    Guarantee that ONNX + ONNX Runtime are importable within the Argos venv.

    Returns a summary dict describing the actions taken and the resulting state.
    """
    global _last_onnx_summary
    py = venv_py or venv_python()
    summary: dict[str, object] = {
        "installed": False,
        "healed": False,
        "ort_version": None,
        "onnx_version": None,
        "providers": None,
        "attempts": [],
        "error": None,
        "dlls_missing": [],
    }
    attempts: list[dict[str, object]] = summary["attempts"]  # type: ignore[assignment]
    try:
        raw_caps = read_capabilities()
    except Exception:
        raw_caps = {}
    capabilities_snapshot = _as_str_key_dict(raw_caps)

    env_accel_env = os.getenv("ARGOS_ACCELERATOR")
    env_accelerator_raw = _coerce_optional_str(env_accel_env)
    preferred_value: object = capabilities_snapshot.get("preferred_accelerator")
    preferred_from_caps = _coerce_optional_str(preferred_value)
    accelerator_hint = (env_accelerator_raw or preferred_from_caps or "").lower()
    if accelerator_hint and (not env_accelerator_raw or env_accelerator_raw.lower() != accelerator_hint):
        os.environ["ARGOS_ACCELERATOR"] = accelerator_hint
    summary["accelerator"] = accelerator_hint or None

    base_spec = desired_ort_spec()
    env_spec_override_raw = os.getenv("ARGOS_ONNXRUNTIME_SPEC")
    env_spec_override = _coerce_optional_str(env_spec_override_raw)
    default_package = _extract_ort_package(base_spec)
    capability_package = None
    ort_meta_value: object = capabilities_snapshot.get("onnxruntime")
    ort_meta_snapshot = _as_str_key_dict(ort_meta_value)
    package_obj = ort_meta_snapshot.get("package")
    capability_package = _coerce_optional_str(package_obj)
    if capability_package is None:
        package_meta = _as_str_key_dict(package_obj)
        capability_package = _coerce_optional_str(package_meta.get("name"))

    if env_spec_override:
        ort_spec = env_spec_override
        package_name = _extract_ort_package(env_spec_override)
    else:
        package_name = default_package
        if accelerator_hint:
            package_name = _package_for_accelerator(accelerator_hint, default=default_package)
        elif capability_package:
            package_name = capability_package
        ort_spec = _replace_ort_package(base_spec, package_name)

    summary["spec"] = ort_spec
    summary["package"] = package_name

    expected_provider = _expected_provider_for_accelerator(accelerator_hint)
    if not expected_provider and package_name == "onnxruntime-gpu":
        expected_provider = "CUDAExecutionProvider"
    elif not expected_provider and package_name == "onnxruntime-directml":
        expected_provider = "DmlExecutionProvider"
    summary["expected_provider"] = expected_provider

    def providers_match_expectation(providers: Optional[object]) -> bool:
        if not expected_provider:
            return True
        if not providers:
            return False
        if isinstance(providers, Sequence) and not isinstance(providers, (str, bytes)):
            provider_items: Sequence[Any] = cast(Sequence[Any], providers)
        else:
            provider_items = (providers,)
        lowered = [str(p).lower() for p in provider_items]
        expected_lower = expected_provider.lower()
        if expected_lower in lowered:
            return True
        if expected_lower.startswith("cuda"):
            return any("cuda" in p for p in lowered)
        if expected_lower.startswith(("dml", "directml")):
            return any("directml" in p or "dml" in p for p in lowered)
        if expected_lower.startswith("tensorrt"):
            return any("tensorrt" in p for p in lowered)
        return False

    def record(action: str, info: Optional[dict[str, object]] = None) -> None:
        entry: dict[str, object] = {"action": action}
        if info:
            entry.update(info)
        attempts.append(entry)

    def validate_provider(provider_label: Optional[str]) -> dict[str, object]:
        if not provider_label:
            return {"ok": True, "providers": None, "error": None}
        script = textwrap.dedent(
            """\
            import json
            import sys
            import tempfile
            from pathlib import Path

            provider = sys.argv[1]
            result = {"ok": False, "error": None, "providers": []}
            tmp_path = None
            try:
                import onnx  # noqa: F401
                import onnxruntime as ort  # type: ignore
                from onnx import helper, TensorProto

                node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
                graph = helper.make_graph(
                    [node],
                    "ArgosValidateORT",
                    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
                    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
                )
                model = helper.make_model(
                    graph,
                    producer_name="argos.bootstrap.ensure",
                    opset_imports=[helper.make_operatorsetid("", 12)],
                )
                try:
                    model.ir_version = min(getattr(model, "ir_version", 7), 7)
                except Exception:
                    try:
                        model.ir_version = 7
                    except Exception:
                        pass
                tmp_path = Path(tempfile.gettempdir()) / "argos_validate.onnx"
                tmp_path.write_bytes(model.SerializeToString())

                session = ort.InferenceSession(tmp_path.as_posix(), providers=[provider])
                get_providers = getattr(session, "get_providers", lambda: [])
                providers = [str(p) for p in get_providers()]
                result["providers"] = providers
                wanted = provider.lower()
                result["ok"] = any(wanted == p.lower() for p in providers)
            except Exception as exc:  # pragma: no cover - validation issues bubble up
                result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if tmp_path is not None:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            sys.stdout.write(json.dumps(result))
            """
        )
        proc = _run([str(py), "-c", script, provider_label], check=False, capture=True)
        payload: dict[str, object] = {"ok": False, "providers": None, "error": None}
        out = (proc.stdout or "").strip()
        if out:
            try:
                parsed = json.loads(out)
                if isinstance(parsed, dict):
                    payload = cast(dict[str, object], parsed)
            except Exception:
                payload = {"ok": False, "providers": None, "error": "validation: invalid response"}
        if not payload.get("error") and proc.returncode != 0:
            stderr_msg = (proc.stderr or "").strip()
            if stderr_msg:
                payload["error"] = stderr_msg
        record(
            "validate-provider",
            {
                "provider": provider_label,
                "ok": bool(payload.get("ok")),
                "error": payload.get("error"),
            },
        )
        return payload

    def run_probe(label: str) -> dict[str, object]:
        info = _probe_onnx_runtime(py)
        record(label, info)
        return info

    def install_packages(packages: Sequence[str], *, label: str, binary_only: bool = True) -> bool:
        cmd = [str(py), "-m", "pip", "install", "--upgrade"]
        if binary_only:
            cmd.append("--only-binary=:all:")
        cmd.extend(_constraints_args())
        cmd.extend(list(packages))
        try:
            _run(cmd, check=True, capture=False)
            record(label, {"status": "ok", "packages": list(packages), "binary_only": binary_only})
            return True
        except Exception as exc:
            record(
                label,
                {
                    "status": "error",
                    "packages": list(packages),
                    "binary_only": binary_only,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return False

    def uninstall_packages(packages: Sequence[str], *, label: str) -> bool:
        cmd = [str(py), "-m", "pip", "uninstall", "-y"]
        cmd.extend(list(packages))
        try:
            _run(cmd, check=True, capture=False)
            record(label, {"status": "ok", "packages": list(packages)})
            return True
        except Exception as exc:
            record(
                label,
                {
                    "status": "error",
                    "packages": list(packages),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return False

    info = run_probe("probe-initial")

    def _succeed(
        data: dict[str, object],
        healed: bool,
        validation: Optional[dict[str, object]] = None,
    ) -> dict[str, object]:
        global _last_onnx_summary
        summary["installed"] = True
        summary["healed"] = healed
        summary["ort_version"] = data.get("version")
        summary["onnx_version"] = data.get("onnx_version")
        summary["providers"] = data.get("providers")
        summary["error"] = None
        summary["providers_ok"] = providers_match_expectation(data.get("providers"))
        summary["validation"] = validation
        if validation is not None:
            summary["validation_ok"] = bool(validation.get("ok"))
        else:
            summary["validation_ok"] = None if expected_provider else True
        os.environ.pop("ARGOS_DISABLE_ONNX", None)
        log(f"-> onnxruntime ready (v{data.get('version') or '?'}, providers={data.get('providers') or []})")
        providers_raw = data.get("providers")
        if isinstance(providers_raw, Sequence) and not isinstance(providers_raw, (str, bytes)):
            providers_seq: Sequence[Any] = cast(Sequence[Any], providers_raw)
        elif providers_raw is None:
            providers_seq = ()
        else:
            providers_seq = (providers_raw,)
        providers_list = [str(p) for p in providers_seq]
        if accelerator_hint == "cuda" and summary["providers_ok"]:
            log(f"-> CUDA execution provider active (providers={providers_list})")
        elif accelerator_hint == "directml" and summary["providers_ok"]:
            log(f"-> DirectML execution provider active (providers={providers_list})")
        _last_onnx_summary = summary
        try:
            _refresh_capabilities_cache(log=True)
        except Exception:
            pass
        return summary

    if _probe_onnx_success(info) and providers_match_expectation(info.get("providers")):
        validation = validate_provider(expected_provider) if expected_provider else None
        if validation and not validation.get("ok"):
            summary["validation"] = validation
            summary["validation_ok"] = bool(validation.get("ok"))
            summary["providers_ok"] = False
            summary["error"] = validation.get("error") or (
                f"onnxruntime missing validated provider ({expected_provider})"
            )
        else:
            result = _succeed(info, healed=False, validation=validation)
            return result

    summary["providers"] = info.get("providers")
    summary["providers_ok"] = providers_match_expectation(info.get("providers"))
    if _probe_onnx_success(info) and not summary["providers_ok"] and accelerator_hint:
        expected_display = expected_provider or accelerator_hint
        summary["error"] = (
            f"onnxruntime missing expected {accelerator_hint} provider ({expected_display})"
        )
        log(
            f"-> onnxruntime present but missing expected {accelerator_hint.upper()} provider; reinstalling {ort_spec}"
        )
    else:
        summary["error"] = info.get("error")
    summary["onnx_version"] = info.get("onnx_version")
    performed_heal = False

    if os.name == "nt":
        initial_missing = _windows_missing_runtime_dlls()
        if initial_missing:
            summary["dlls_missing"] = list(initial_missing)
            record("msvc-scan-initial", {"missing": list(initial_missing)})
            log(f"-> missing MSVC runtime DLLs detected: {list(initial_missing)}")
            log("-> installing Microsoft Visual C++ Redistributable (x64).")
            if _install_windows_vcredist(log, lambda act, details: record(act, details)):
                performed_heal = True
                summary["dlls_missing"] = _windows_missing_runtime_dlls()
    removal_targets: list[str] = []
    if package_name == "onnxruntime":
        removal_targets = ["onnxruntime-gpu", "onnxruntime-directml"]
    elif package_name == "onnxruntime-gpu":
        removal_targets = ["onnxruntime", "onnxruntime-directml"]
    elif package_name == "onnxruntime-directml":
        removal_targets = ["onnxruntime", "onnxruntime-gpu"]
    if removal_targets:
        if uninstall_packages(removal_targets, label="pip-uninstall-onnxruntime-pre"):
            performed_heal = True

    install_packages(["pip", "setuptools", "wheel"], label="pip-upgrade", binary_only=False)
    if install_packages(["onnx"], label="pip-onnx"):
        performed_heal = True
    if install_packages([ort_spec], label="pip-onnxruntime"):
        performed_heal = True

    info = run_probe("probe-after-pip")
    if _probe_onnx_success(info) and providers_match_expectation(info.get("providers")):
        validation = validate_provider(expected_provider) if expected_provider else None
        if validation and not validation.get("ok"):
            summary["validation"] = validation
            summary["validation_ok"] = bool(validation.get("ok"))
            summary["providers_ok"] = False
            summary["error"] = validation.get("error") or (
                f"onnxruntime missing validated provider ({expected_provider})"
            )
        else:
            result = _succeed(info, healed=performed_heal, validation=validation)
            return result
    summary["providers"] = info.get("providers")
    summary["providers_ok"] = providers_match_expectation(info.get("providers"))

    def _error_indicates_vcredist(error: Optional[object]) -> bool:
        if not error:
            return False
        text = str(error).lower()
        if "dll load failed" in text and "onnxruntime_pybind11_state" in text:
            return True
        if "initialization routine failed" in text and "onnxruntime" in text:
            return True
        return False

    reinstall_attempted = False
    if uninstall_packages(["onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"], label="pip-uninstall-onnxruntime"):
        reinstall_attempted = True
        if install_packages([ort_spec], label="pip-onnxruntime-reinstall"):
            performed_heal = True
    if reinstall_attempted:
        info = run_probe("probe-after-reinstall")
        summary["providers"] = info.get("providers")
        summary["providers_ok"] = providers_match_expectation(info.get("providers"))
        if _probe_onnx_success(info) and summary["providers_ok"]:
            validation = validate_provider(expected_provider) if expected_provider else None
            if validation and not validation.get("ok"):
                summary["validation"] = validation
                summary["validation_ok"] = bool(validation.get("ok"))
                summary["providers_ok"] = False
                summary["error"] = validation.get("error") or (
                    f"onnxruntime missing validated provider ({expected_provider})"
                )
            else:
                result = _succeed(info, healed=True, validation=validation)
                return result

    if os.name == "nt":
        missing = _windows_missing_runtime_dlls()
        summary["dlls_missing"] = missing
        record("msvc-scan", {"missing": missing})
        needs_redist = bool(missing) or _error_indicates_vcredist(info.get("error"))
        if needs_redist:
            if missing:
                log(f"-> missing MSVC runtime DLLs detected: {missing}")
            log("-> installing Microsoft Visual C++ Redistributable (x64).")
            if _install_windows_vcredist(log, lambda act, details: record(act, details)):
                performed_heal = True
                summary["dlls_missing"] = _windows_missing_runtime_dlls()
                if install_packages([ort_spec], label="pip-onnxruntime-postvcredist"):
                    performed_heal = True
                info = run_probe("probe-after-vcredist")
                summary["providers"] = info.get("providers")
                summary["providers_ok"] = providers_match_expectation(info.get("providers"))
                if _probe_onnx_success(info) and summary["providers_ok"]:
                    validation = validate_provider(expected_provider) if expected_provider else None
                    if validation and not validation.get("ok"):
                        summary["validation"] = validation
                        summary["validation_ok"] = bool(validation.get("ok"))
                        summary["providers_ok"] = False
                        summary["error"] = validation.get("error") or (
                            f"onnxruntime missing validated provider ({expected_provider})"
                        )
                    else:
                        result = _succeed(info, healed=True, validation=validation)
                        return result

    if summary.get("providers_ok") is False and accelerator_hint:
        summary["error"] = summary.get("error") or (
            f"onnxruntime missing expected {accelerator_hint} provider ({expected_provider or accelerator_hint})"
        )
    else:
        summary["error"] = info.get("error")
    summary["providers"] = info.get("providers")
    summary["healed"] = performed_heal and summary["installed"]

    if os.environ.get("ARGOS_DISABLE_ONNX") != "1":
        os.environ["ARGOS_DISABLE_ONNX"] = "1"
    log(
        f"?? onnxruntime unavailable after automated healing: {summary['error'] or 'unknown'}; "
        "ARGOS_DISABLE_ONNX=1"
    )
    _last_onnx_summary = summary
    return summary

def get_last_onnx_summary() -> Optional[dict[str, object]]:
    """Return the most recent ONNX Runtime summary captured by bootstrap ensure."""
    return _last_onnx_summary




def _ensure_onnx_runtime_packages() -> None:
    """Guarantee ONNX + ONNX Runtime availability during bootstrap."""
    global _last_onnx_summary
    summary = ensure_onnxruntime(venv_python(), log=_print)
    _last_onnx_summary = summary
    if (not summary.get("installed")) or (summary.get("providers_ok") is False):
        reason = summary.get("error") or "unknown"
        providers = summary.get("providers")
        if isinstance(providers, Sequence) and not isinstance(providers, (str, bytes)):
            provider_display: object = [str(p) for p in cast(Sequence[Any], providers)]
        else:
            provider_display = providers
        raise RuntimeError(f"ONNX Runtime validation failed: {reason}; providers={provider_display}")


def _probe_weight_presets() -> tuple[Path, list[str], list[str], list[str], list[str]]:
    import tempfile
    probe = r"""
import json, sys
from pathlib import Path
from panoptes.model_registry import MODEL_DIR, WEIGHT_PRIORITY

def uniq(xs):
    seen=set(); out=[]
    for x in xs:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def first_or_none(lst):
    return lst[0] if lst else None

all_names=[]
for _, paths in WEIGHT_PRIORITY.items():
    for p in paths:
        all_names.append(Path(p).name)

detect_first  = first_or_none(WEIGHT_PRIORITY.get('detect', []))
heatmap_first = first_or_none(WEIGHT_PRIORITY.get('heatmap', []))
pose_first    = first_or_none(WEIGHT_PRIORITY.get('pose', []))
obb_first     = first_or_none(WEIGHT_PRIORITY.get('obb', []))

def to_name(p):
    return Path(p).name if p else None

default_names = [to_name(detect_first), to_name(heatmap_first)]
nano_names    = [Path(p).name for p in WEIGHT_PRIORITY.get('detect_small', []) + WEIGHT_PRIORITY.get('heatmap_small', [])]
perception_names = [to_name(detect_first), to_name(heatmap_first), to_name(pose_first), to_name(obb_first)]

out_path = Path(sys.argv[-1])
out_path.write_text(json.dumps({
    'model_dir': str(MODEL_DIR),
    'all': uniq(all_names),
    'default': uniq(default_names),
    'nano': uniq(nano_names),
    'perception': uniq([x for x in perception_names if x]),
}), encoding='utf-8')
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as tf:
        out_path = Path(tf.name)
    try:
        vpy_path = _venv_executable_path()
        py = str(vpy_path) if VENV.exists() and vpy_path.exists() else sys.executable
        env = os.environ.copy()
        env_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (str(ARGOS) + (os.pathsep + env_pp if env_pp else ""))

        _run([py, "-c", probe, str(out_path)], check=True, capture=False, env=env)
        txt = out_path.read_text(encoding="utf-8")
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                meta = cast(JSONDict, obj)
            else:
                meta = {}
        except Exception:
            meta = {}
    finally:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _to_str_list(v: Any) -> list[str]:
        if isinstance(v, list):
            return [str(x) for x in cast(list[object], v) if x is not None]
        return []

    model_dir_str = str(meta.get("model_dir", ""))
    return (
        Path(model_dir_str).resolve(),
        _to_str_list(meta.get("all")),
        _to_str_list(meta.get("default")),
        _to_str_list(meta.get("nano")),
        _to_str_list(meta.get("perception")),
    )

def _child_json(code: str, args: list[str]) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as tf:
        out_path = Path(tf.name)
    try:
        _run([str(venv_python()), "-c", code, *args, str(out_path)], check=True, capture=False)
        txt = out_path.read_text(encoding="utf-8")
        result: JSONDict = {}
        if txt.strip():
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    result = cast(JSONDict, obj)
            except Exception:
                result = {}
        return result
    finally:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass

def _ensure_weights_ultralytics(*, preset: Optional[str] = None, explicit_names: Optional[list[str]] = None) -> None:
    model_dir, all_names, default_names, nano_names, perception_names = _probe_weight_presets()

    env_preset = (os.getenv("ARGOS_WEIGHT_PRESET") or "").strip().lower()
    choice = (preset or env_preset or "default").lower()
    if explicit_names is not None:
        want_names = [n for n in explicit_names if n]
    else:
        if choice not in {"all", "default", "nano", "perception"}:
            choice = "default"
        want_names = {
            "all": all_names,
            "default": default_names,
            "nano": nano_names,
            "perception": perception_names,
        }[choice]

    live_small_names: list[str] = []
    for key in ("detect_small", "heatmap_small"):
        seq = cast(Sequence[Path], _BOOTSTRAP_WEIGHT_PRIORITY.get(key, []))
        for p in seq:
            path_obj = Path(p)
            if path_obj.suffix.lower() == ".onnx":
                live_small_names.append(path_obj.name)
    if live_small_names:
        combined = [*want_names, *live_small_names]
        deduped: list[str] = []
        seen: set[str] = set()
        for name in combined:
            if name and name not in seen:
                seen.add(name)
                deduped.append(name)
        want_names = deduped

    need = [n for n in want_names if n and not (model_dir / n).exists()]
    if not need:
        _print("→ model weights present.")
        return

    if not _module_present("ultralytics"):
        _run([str(venv_python()), "-m", "pip", "install", "ultralytics>=8.0"], check=True, capture=False)

    pt_missing = [n for n in need if n.lower().endswith(".pt")]
    onnx_missing = [n for n in need if n.lower().endswith(".onnx")]

    if pt_missing:
        _print(f"→ fetching {len(pt_missing)} *.pt via Ultralytics …")
        dl_pt = r"""
import json, sys, shutil, logging, os
from pathlib import Path
try:
    from ultralytics.utils import LOGGER
    LOGGER.remove()
except Exception:
    pass
logging.getLogger().handlers.clear()
logging.getLogger().setLevel(logging.ERROR)
from ultralytics import YOLO
dst = Path(sys.argv[-2]).expanduser().resolve(); dst.mkdir(parents=True, exist_ok=True)
os.chdir(dst)
ok=0
for nm in sys.argv[1:-2]:
    try:
        m = YOLO(nm)
        p = Path(getattr(m,'ckpt_path', nm)).expanduser()
        if not p.exists():
            p = Path(nm).expanduser()
        target = dst / Path(nm).name
        if p.exists() and p.resolve() != target.resolve():
            shutil.copy2(p, target)
        if target.exists():
            ok += 1
    except Exception:
        pass
out_path = Path(sys.argv[-1])
out_path.write_text(json.dumps({'ok': ok}), encoding='utf-8')
"""
        data: JSONDict = _child_json(dl_pt, [*pt_missing, str(model_dir)])
        got = int(data.get("ok", 0))
        if got < len(pt_missing):
            _print("⚠️  Some *.pt weights could not be fetched (network/rate-limit?).")

    if onnx_missing:
        _print(f"→ exporting {len(onnx_missing)} *.onnx from matching *.pt …")
        exp = r"""
import json, sys, shutil, logging, os
from pathlib import Path
try:
    from ultralytics.utils import LOGGER
    LOGGER.remove()
except Exception:
    pass
logging.getLogger().handlers.clear()
logging.getLogger().setLevel(logging.ERROR)
from ultralytics import YOLO
dst = Path(sys.argv[-2]).expanduser().resolve(); dst.mkdir(parents=True, exist_ok=True)
os.chdir(dst)
ok=0
for onnx_name in sys.argv[1:-2]:
    try:
        pt_name = Path(onnx_name).with_suffix(".pt").name
        src_pt  = dst / pt_name
        if not src_pt.exists():
            m_fetch = YOLO(pt_name)
            p = Path(getattr(m_fetch,'ckpt_path', pt_name)).expanduser()
            if p.exists() and p.resolve() != src_pt.resolve():
                shutil.copy2(p, src_pt)
        m = YOLO(str(src_pt))
        outp_path = None
        try:
            outp = Path(m.export(format='onnx', dynamic=True, simplify=True, imgsz=640, opset=12, device='cpu'))
            outp_path = outp
        except Exception:
            try:
                outp = Path(m.export(format='onnx', dynamic=True, simplify=False, imgsz=640, opset=12, device='cpu'))
                outp_path = outp
            except Exception:
                try:
                    outp = Path(m.export(format='onnx', dynamic=False, simplify=False, imgsz=640, opset=12, device='cpu'))
                    outp_path = outp
                except Exception:
                    outp_path = None
        target = dst / onnx_name
        if outp_path and outp_path.exists() and outp_path.resolve() != target.resolve():
            shutil.copy2(outp_path, target)
        elif (not target.exists()) and outp_path and outp_path.exists():
            shutil.copy2(outp_path, target)
        if target.exists():
            ok += 1
    except Exception:
        pass
try:
    shutil.rmtree(dst / 'runs', ignore_errors=True)
except Exception:
    pass
out_path = Path(sys.argv[-1])
out_path.write_text(json.dumps({'ok': ok}), encoding='utf-8')
"""
        data2: JSONDict = _child_json(exp, [*onnx_missing, str(model_dir)])
        got2 = int(data2.get("ok", 0))
        if got2 < len(onnx_missing):
            _print("⚠️  Some *.onnx could not be exported.")

    _print("→ weights ensured.")

def _move_pytest_cache_out_of_repo() -> None:
    ini = HERE / "pytest.ini"
    if not ini.exists():
        cache = (DATA / "pytest_cache").as_posix()
        ini.write_text(
            textwrap.dedent(f"""\
            [pytest]
            cache_dir = {cache}
        """),
            encoding="utf-8",
        )

def _ensure_sitecustomize() -> None:
    sp = _site_packages_root()
    sc = sp / "sitecustomize.py"
    sp.mkdir(parents=True, exist_ok=True)
    pycache_dir = (DATA / "pycache").resolve()
    content = textwrap.dedent(
        f"""\
        import json
        import os
        from pathlib import Path

        _CAPS_PATH = Path({repr(str(CAPABILITIES_FILE))})
        _CUDA_STATE_DIR = Path({repr(str(CUDA_STATE_DIR))})
        _TRUTHY = {{'1', 'true', 'yes', 'on'}}


        def _is_truthy(value: str) -> bool:
            return value.strip().lower() in _TRUTHY


        def _ensure_pycache_prefix() -> None:
            target = {repr(str(pycache_dir))}
            if target:
                os.environ.setdefault("PYTHONPYCACHEPREFIX", target)


        def _load_capabilities() -> dict[str, object]:
            try:
                if _CAPS_PATH.exists():
                    data = json.loads(_CAPS_PATH.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
            return {{}}


        def _preferred_accelerator(capabilities: dict[str, object]) -> str:
            env_hint = os.environ.get("ARGOS_ACCELERATOR", "").strip().lower()
            if env_hint:
                return env_hint
            pref = capabilities.get("preferred_accelerator")
            if isinstance(pref, str):
                return pref.strip().lower()
            return ""


        def _extend_cuda_paths(capabilities: dict[str, object]) -> None:
            candidate_dirs: list[str] = []
            cuda_section = capabilities.get("cuda")
            if isinstance(cuda_section, dict):
                raw_dirs = cuda_section.get("dll_dirs")
                if isinstance(raw_dirs, (list, tuple)):
                    for entry in raw_dirs:
                        if isinstance(entry, str) and entry:
                            candidate_dirs.append(entry)
            if not candidate_dirs:
                base = Path(__file__).resolve().parent
                search_roots = [base, _CUDA_STATE_DIR]
                for root in search_roots:
                    if not root.exists():
                        continue
                    patterns = ("nvidia/**/bin", "nvidia/**/lib") if root == base else ("**/bin", "**/lib")
                    for pattern in patterns:
                        try:
                            for item in root.glob(pattern):
                                try:
                                    if item.is_dir():
                                        candidate_dirs.append(str(item.resolve()))
                                except Exception:
                                    continue
                        except Exception:
                            continue
            if not candidate_dirs:
                return

            def _merge_env_var(key: str, additions: list[str]) -> None:
                if not key:
                    return
                current = os.environ.get(key, "")
                existing = [p for p in current.split(os.pathsep) if p] if current else []
                parts = additions + existing
                seen: set[str] = set()
                merged: list[str] = []
                for part in parts:
                    norm = os.path.normcase(part) if hasattr(os.path, "normcase") else part
                    if norm in seen:
                        continue
                    seen.add(norm)
                    merged.append(part)
                os.environ[key] = os.pathsep.join(merged)

            _merge_env_var("PATH", candidate_dirs)
            if os.name != "nt":
                _merge_env_var("LD_LIBRARY_PATH", candidate_dirs)


        def _ensure_accelerator_env(capabilities: dict[str, object]) -> None:
            if os.environ.get("ARGOS_ACCELERATOR"):
                return
            pref = capabilities.get("preferred_accelerator")
            if isinstance(pref, str) and pref:
                os.environ["ARGOS_ACCELERATOR"] = pref.strip().lower()
            elif capabilities.get("cuda"):
                # If CUDA metadata exists but no explicit preference, assume CUDA.
                os.environ.setdefault("ARGOS_ACCELERATOR", "cuda")


        def _ensure_tensorrt_default() -> None:
            if os.environ.get("ORT_DISABLE_TENSORRT"):
                return
            if _is_truthy(os.environ.get("ARGOS_ENABLE_TENSORRT", "")):
                return
            os.environ["ORT_DISABLE_TENSORRT"] = "1"


        _capabilities = _load_capabilities()
        _ensure_accelerator_env(_capabilities)
        _ensure_pycache_prefix()
        _extend_cuda_paths(_capabilities)
        _ensure_tensorrt_default()
        """
    )
    if sc.exists():
        existing = sc.read_text(encoding="utf-8")
        if existing == content:
            return
    sc.write_text(content, encoding="utf-8")

def _create_launchers() -> None:
    sh = HERE / "argos"
    sh.write_text(
        textwrap.dedent(
            """\
        #!/usr/bin/env bash
        set -euo pipefail
        HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        PY="$(command -v python3 || command -v python)"
        "$PY" "$HERE/bootstrap.py" --ensure --yes >/dev/null 2>&1 || true
        VPY="$("$PY" "$HERE/bootstrap.py" --print-venv)"
        export PYTHONPYCACHEPREFIX="${XDG_CACHE_HOME:-$HOME/.cache}/rAIn/pycache"
        exec "$VPY" -m panoptes.cli "$@"
    """
        ),
        encoding="utf-8",
    )
    os.chmod(sh, 0o755)

    ps1 = HERE / "argos.ps1"
    ps1.write_text(
        textwrap.dedent(
            r"""\
        [CmdletBinding()]
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
        $ErrorActionPreference = "Stop"
        $HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
        $py = (Get-Command python -ErrorAction SilentlyContinue).Source
        if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
        & $py "$HERE\bootstrap.py" --ensure --yes | Out-Null
        $vpy = & $py "$HERE\bootstrap.py" --print-venv
        $env:PYTHONPYCACHEPREFIX = Join-Path $env:LOCALAPPDATA "rAIn\pycache"
        & $vpy -m panoptes.cli @Args
    """
        ),
        encoding="utf-8",
    )

    cmd = HERE / "argos.cmd"
    cmd.write_text(
        textwrap.dedent(
            r"""\
        @echo off
        setlocal
        set HERE=%~dp0
        where py >NUL 2>&1
        if %ERRORLEVEL% EQU 0 (
            set "PY=py -3"
        ) else (
            set "PY=python"
        )
        %PY% "%HERE%bootstrap.py" --ensure --yes >NUL 2>&1
        for /f "usebackq delims=" %%i in (`%PY% "%HERE%bootstrap.py" --print-venv`) do set "VPY=%%i"
        set "PYTHONPYCACHEPREFIX=%LOCALAPPDATA%\rAIn\pycache"
        "%VPY%" -m panoptes.cli %*
    """
        ),
        encoding="utf-8",
    )

def _print_help(cpu_only: bool) -> None:
    py = _venv_executable_path()
    msg = f"""
★ Argos is ready.

Environment
  • Venv:   {py.parent}
  • Torch:  {'CPU-only' if cpu_only else 'auto'}
  • Models: panoptes/model/  (Ultralytics fetched; preset={{default|nano|perception}}; env ARGOS_WEIGHT_PRESET)

Quick start (no venv activation)
  Windows PowerShell:
      .\\argos.ps1 tests\\assets\\assets.jpg hm --alpha 0.5
      .\\argos.ps1 tests\\assets\\assets.jpg obb

  Windows (double-click / cmd):
      argos.cmd tests\\assets\\assets.jpg d
      argos.cmd tests\\assets\\assets.jpg pse

  Linux / macOS:
      ./argos tests/raw/assets.jpg heatmap --alpha 0.5
      ./argos tests/raw/assets.jpg pose
      ./argos tests/raw/assets.jpg classify --topk 3 --annotate

  GeoJSON from URL (with #lat…_lon…):
      ./argos "https://…/image.jpg#lat37.8199_lon-122.4783" --task geojson

Power users
  • Makefile shortcuts (inside projects/argos):
      make install     # pip install -e "projects/argos[dev]"
      make test        # run pytest
      make run         # example CLI invocation (see Makefile for options)

  • Direct module:
      {py} -m panoptes.cli tests/raw/assets.jpg heatmap

CI / CD
  • Use ARGOS_WEIGHT_PRESET=[all|default|nano|perception] and run:
      python projects/argos/bootstrap.py --ensure --yes
"""
    _print(textwrap.dedent(msg).rstrip())

def _first_run_menu() -> tuple[Optional[str], Optional[list[str]], bool]:
    model_dir, all_names, _d, _n, perception_names = _probe_weight_presets()
    _print("\nModel weights will be placed under: " + str(model_dir))
    _print("\nChoose what to fetch now:")
    _print("  [1] All listed weights")
    _print("  [2] Default bundle (detect/heatmap + live ONNX)  ← recommended")
    _print("  [3] Nano pair (fastest for laptops & CI)")
    _print("  [4] Perception set (detect + heatmap + pose + obb)")
    _print("  [5] Pick from list")
    _print("  [6] Skip for now")
    while True:
        ans = input("Selection [1-6, default 2]: ").strip()
        if not ans or ans == "2":
            return ("default", None, False)
        if ans == "1":
            return ("all", None, False)
        if ans == "3":
            return ("nano", None, False)
        if ans == "4":
            return ("perception", None, False) if perception_names else ("default", None, False)
        if ans == "6":
            return (None, None, True)
        if ans == "5":
            _print("\nAvailable files:")
            for i, n in enumerate(all_names, 1):
                _print(f"  {i:>2}. {n}")
            sel = input("Enter numbers (comma-separated): ").strip()
            if not sel:
                continue
            try:
                idxs = [int(x) for x in sel.replace(" ", "").split(",") if x]
                chosen = [all_names[i - 1] for i in idxs if 1 <= i <= len(all_names)]
                if chosen:
                    return (None, chosen, False)
            except Exception:
                pass
            _print("Invalid selection. Try again.")
        else:
            _print("Please enter 1, 2, 3, 4, 5 or 6.")

def _first_run() -> bool:
    return not SENTINEL.exists()

def _write_sentinel() -> None:
    SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    SENTINEL.write_text(json.dumps({"version": 6}, indent=2), encoding="utf-8")

def _ask_yes_no(prompt: str, default_yes: bool = True) -> bool:
    d = "Y/n" if default_yes else "y/N"
    while True:
        ans = input(f"{prompt} [{d}]: ").strip().lower()
        if not ans:
            return default_yes
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False

def _pip_check_soft() -> None:
    try:
        _run([str(venv_python()), "-m", "pip", "check"], check=True, capture=False)
    except Exception:
        _print("⚠️  'pip check' reported issues; continuing. Details follow:")
        try:
            cp = _run([str(venv_python()), "-m", "pip", "check"], check=False, capture=True)
            if cp and hasattr(cp, "stdout") and cp.stdout:
                _print(cp.stdout.strip())
            if cp and hasattr(cp, "stderr") and cp.stderr:
                _print(cp.stderr.strip())
        except Exception:
            pass

def _ensure(
    cpu_only: bool,
    *,
    preset: Optional[str] = None,
    weight_names: Optional[list[str]] = None,
    skip_weights: bool = False,
    reinstall: bool = False,
    with_dev: bool = False,
    extras: Optional[Sequence[str]] = None,
) -> None:
    extras_tuple: tuple[str, ...] = tuple(x.strip() for x in (extras or []) if x.strip())
    steps: list[tuple[str, Callable[[], None]]] = [
        ("Create venv", _create_venv),
        ("Install Torch", lambda: _install_torch_if_needed(cpu_only)),
        ("Ensure CUDA runtime (wheels)", _ensure_cuda_runtime_packages),
        ("Ensure OpenCV (GUI)", _ensure_opencv_gui),
        (
            "Install Argos (editable)",
            lambda: _pip_install_editable_if_needed(
                reinstall=reinstall,
                with_dev=with_dev,
                extras=extras_tuple,
            ),
        ),
        ("Sitecustomize (pycache & CUDA env)", _ensure_sitecustomize),
        (
            "Ensure ONNX Runtime",
            _ensure_onnx_runtime_packages,
        ),
        ("Align Torch deps", _ensure_sympy_alignment),
        ("Record capabilities", _record_capabilities_step),
        (
            "Ensure weights",
            (lambda: None)
            if (skip_weights or os.getenv("ARGOS_SKIP_WEIGHTS", "").lower() in {"1", "true", "yes"})
            else (lambda: _ensure_weights_ultralytics(preset=preset, explicit_names=weight_names)),
        ),
        ("Move pytest cache", _move_pytest_cache_out_of_repo),
        ("pip check (soft)", _pip_check_soft),
    ]

    compact_progress = os.getenv("ARGOS_PROGRESS_COMPACT", "").strip().lower() in {"1", "true", "yes", "on"}
    if compact_progress:
        total_steps = len(steps)
        last_len = 0

        def _update_progress(message: str) -> None:
            nonlocal last_len
            line = f"\r{message}"
            sys.stdout.write(line)
            padding = last_len - len(message)
            if padding > 0:
                sys.stdout.write(" " * padding)
            sys.stdout.flush()
            last_len = len(message)

        try:
            for idx, (name, fn) in enumerate(steps, start=1):
                label = f"BOOTSTRAP [{idx}/{total_steps}] {name}"
                _update_progress(f"{label} ...")
                try:
                    fn()
                except Exception:
                    _update_progress(f"{label} [FAIL]")
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    raise
                _update_progress(f"{label} [OK]")
            sys.stdout.write("\n")
            sys.stdout.flush()
        finally:
            sys.stdout.flush()
        return

    if ProgressEngine is None or live_percent is None:
        for name, fn in steps:
            _print(f"→ {name} …")
            fn()
        return

    eng = ProgressEngine()  # type: ignore[call-arg]
    with live_percent(eng, prefix="BOOTSTRAP"):  # type: ignore[misc]
        eng.set_total(len(steps))
        for name, fn in steps:
            eng.set_current(name)
            try:
                fn()
            finally:
                eng.add(1)

def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--ensure", action="store_true", help="Ensure env non-interactively (fast, idempotent)")
    p.add_argument("--print-venv", action="store_true", help="Print venv python path")
    p.add_argument("--cpu", dest="cpu_only", action="store_true", help="Force CPU-only Torch")
    p.add_argument("--weights-preset", choices=["all", "default", "nano", "perception"], help="Preset for weights (overrides env)")
    p.add_argument("--reinstall", action="store_true", help="Force reinstall of Argos editable package")
    p.add_argument("--with-dev", action="store_true", help="Install Argos with [dev] extras as well")
    p.add_argument("--print-capabilities", action="store_true", help="Print cached accelerator/runtime capabilities")
    p.add_argument(
        "--extras",
        help="Comma-separated Argos extras to install (e.g., audio,onnx-tools)",
    )
    p.add_argument("--yes", action="store_true", help="Assume Yes to prompts (non-interactive)")
    args, _ = p.parse_known_args(argv)
    _ensure_dirs()

    if args.print_venv:
        _print(str(venv_python()))
        return 0

    if getattr(args, "print_capabilities", False):
        info = _refresh_capabilities_cache(log=False)
        _print(json.dumps(info, indent=2, sort_keys=True))
        return 0

    cpu_only = bool(args.cpu_only or not _has_cuda())
    with_dev_flag = bool(
        args.with_dev or (os.getenv("ARGOS_WITH_DEV", "").strip().lower() in {"1", "true", "yes"})
    )

    extras_sources: list[str] = []
    env_extras = os.getenv("ARGOS_BOOTSTRAP_EXTRAS", "").strip()
    if env_extras:
        extras_sources.extend(env_extras.split(","))
    if getattr(args, "extras", None):
        extras_sources.extend(str(args.extras).split(","))
    extras_list: list[str] = []
    for item in extras_sources:
        cleaned = item.strip()
        if cleaned and cleaned not in extras_list:
            extras_list.append(cleaned)

    if args.ensure:
        try:
            _ensure(
                cpu_only,
                preset=args.weights_preset,
                reinstall=args.reinstall,
                with_dev=with_dev_flag,
                extras=extras_list,
            )
        except Exception as e:
            traceback.print_exc()
            _print(f"ensure failed: {e}")
            return 1
        return 0

    if _first_run():
        _print("First run detected for Argos.")
        if args.yes or _ask_yes_no("Install dependencies and set up now?", default_yes=True):
            preset = args.weights_preset
            picks: Optional[list[str]] = None
            skip = False

            if not args.yes and preset is None:
                preset, picks, skip = _first_run_menu()
                if preset:
                    os.environ["ARGOS_WEIGHT_PRESET"] = preset
            elif preset:
                os.environ["ARGOS_WEIGHT_PRESET"] = preset

            _ensure(
                cpu_only,
                preset=preset,
                weight_names=picks,
                skip_weights=skip,
                reinstall=args.reinstall,
                with_dev=with_dev_flag,
                extras=extras_list,
            )
            _create_launchers()
            _write_sentinel()
            _print_help(cpu_only)
        else:
            _print("Setup skipped. You can run later with:  python bootstrap.py --ensure")
        return 0

    _ensure(
        cpu_only,
        preset=args.weights_preset,
        reinstall=args.reinstall,
        with_dev=with_dev_flag,
        extras=extras_list,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
