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
import traceback
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

from panoptes.logging_config import setup_logging

try:
    from panoptes.model_registry import WEIGHT_PRIORITY as _BOOTSTRAP_WEIGHT_PRIORITY
except Exception:  # pragma: no cover - bootstrap still works even if registry import fails
    _BOOTSTRAP_WEIGHT_PRIORITY: dict[str, list[Path]] = {}

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

setup_logging()
_LOG = logging.getLogger(__name__)

StrPath = Union[str, PathLike[str]]

HERE: Path = Path(__file__).resolve().parent
ARGOS: Path = (
    HERE
    if (HERE / "pyproject.toml").exists() and (HERE / "panoptes").exists()
    else (HERE / "projects" / "argos")
)

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
    for p in (CFG, DATA, VENVS):
        p.mkdir(parents=True, exist_ok=True)

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
    except Exception:
        return False

def _has_cuda() -> bool:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        _run([smi, "-L"], check=True, capture=True)
        return True
    except Exception:
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
    return torch_spec, tv_spec

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
    _print(f" ensuring {note} for Torch/Tv .")
    _run([str(venv_python()), "-m", "pip", "install", "--upgrade", *_constraints_args(), spec], check=True, capture=False)

def _install_torch_if_needed(cpu_only: bool) -> None:
    """
    Install Torch/Torchvision if missing with robust fallbacks:
      1) Try requirement specs + constraints (with CPU index when cpu_only).
      2) On failure, ensure NumPy floor and retry.
      3) On failure, fall back to curated pairs (with +cpu local tags on CPU index).
    """
    if _module_present("torch") and _module_present("torchvision"):
        return

    torch_spec, tv_spec = _torch_pins_from_requirements()

    # Use CPU wheels index on Win/Linux when CPU-only; macOS uses default (MPS wheels).
    vpy = str(venv_python())
    base_cmd = [vpy, "-m", "pip", "install"]
    idx: list[str] = []
    if cpu_only and (os.name == "nt" or sys.platform.startswith("linux")):
        idx = ["--index-url", "https://download.pytorch.org/whl/cpu"]

    def _try_install(pkgs: list[str], label: str) -> bool:
        try:
            _print(f"→ installing Torch ({label}) …")
            _run([*base_cmd, *idx, *_constraints_args(), *pkgs], check=True, capture=False)
            return True
        except Exception:
            return False

    # 1) Requirements + constraints (preferred)
    if _try_install([torch_spec, tv_spec], "requirements"):
        return

    # 2) Add NumPy floor and retry
    _ensure_numpy_floor_for_torch()
    if _try_install([torch_spec, tv_spec], "requirements+numpy"):
        return

    # 3) Curated fallback pairs
    if cpu_only and (os.name == "nt" or sys.platform.startswith("linux")):
        for pair in (["torch==2.4.1+cpu", "torchvision==0.19.1+cpu"],
                     ["torch==2.5.1+cpu", "torchvision==0.20.1+cpu"]):
            if _try_install(pair, f"fallback {'/'.join(pair)}"):
                return
    else:
        for pair in (["torch==2.4.1", "torchvision==0.19.1"],
                     ["torch==2.5.1", "torchvision==0.20.1"]):
            if _try_install(pair, f"fallback {'/'.join(pair)}"):
                return

    raise RuntimeError("Torch/Torchvision installation failed after multiple attempts.")

def _ensure_opencv_gui() -> None:
    _print("\x1a ensuring OpenCV (GUI-capable) .")
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
        torch_version = pkg_version("torch")
    except PackageNotFoundError:
        return

    required_sympy: Optional[str] = None
    if torch_version.startswith("2.5."):
        required_sympy = "1.13.1"

    if not required_sympy:
        return

    try:
        sympy_version = pkg_version("sympy")
    except PackageNotFoundError:
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


ORT_VERSION_LADDER: list[str] = ["1.19.2", "1.19.1", "1.19.0", "1.18.1", "1.18.0", "1.17.3"]
_LAST_ONNX_SUMMARY: Optional[dict[str, object]] = None


def ensure_onnxruntime(
    venv_py: Optional[Path] = None,
    *,
    log: Callable[[str], None] = _print,
) -> dict[str, object]:
    """
    Guarantee that ONNX + ONNX Runtime are importable within the Argos venv.

    Returns a summary dict describing the actions taken and the resulting state.
    """
    global _LAST_ONNX_SUMMARY
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

    def record(action: str, info: Optional[dict[str, object]] = None) -> None:
        entry: dict[str, object] = {"action": action}
        if info:
            entry.update(info)
        attempts.append(entry)

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

    info = run_probe("probe-initial")

    def _succeed(data: dict[str, object], healed: bool) -> dict[str, object]:
        summary["installed"] = True
        summary["healed"] = healed
        summary["ort_version"] = data.get("version")
        summary["onnx_version"] = data.get("onnx_version")
        summary["providers"] = data.get("providers")
        summary["error"] = None
        log(f"→ onnxruntime ready (v{data.get('version') or '?'}, providers={data.get('providers') or []})")
        return summary

    if _probe_onnx_success(info):
        result = _succeed(info, healed=False)
        _LAST_ONNX_SUMMARY = result
        return result

    summary["error"] = info.get("error")
    summary["onnx_version"] = info.get("onnx_version")
    performed_heal = False

    install_packages(["pip", "setuptools", "wheel"], label="pip-upgrade", binary_only=False)
    if install_packages(["onnx"], label="pip-onnx"):
        performed_heal = True
    if install_packages(["onnxruntime"], label="pip-onnxruntime"):
        performed_heal = True

    info = run_probe("probe-after-pip")
    if _probe_onnx_success(info):
        result = _succeed(info, healed=performed_heal)
        _LAST_ONNX_SUMMARY = result
        return result

    if os.name == "nt":
        missing = _windows_missing_runtime_dlls()
        summary["dlls_missing"] = missing
        record("msvc-scan", {"missing": missing})
        if missing:
            log(f"→ missing MSVC runtime DLLs detected: {missing}")
            if _install_windows_vcredist(log, lambda act, details: record(act, details)):
                performed_heal = True
            summary["dlls_missing"] = _windows_missing_runtime_dlls()
            info = run_probe("probe-after-vcredist")
            if _probe_onnx_success(info):
                result = _succeed(info, healed=True)
                _LAST_ONNX_SUMMARY = result
                return result

    for ver in ORT_VERSION_LADDER:
        log(f"→ attempting onnxruntime=={ver}")
        if install_packages([f"onnxruntime=={ver}"], label=f"pip-onnxruntime-{ver}"):
            performed_heal = True
            info = run_probe(f"probe-onnxruntime-{ver}")
            if _probe_onnx_success(info):
                result = _succeed(info, healed=True)
                _LAST_ONNX_SUMMARY = result
                return result

    summary["error"] = info.get("error")
    summary["providers"] = info.get("providers")
    summary["healed"] = performed_heal and summary["installed"]

    if os.environ.get("ARGOS_DISABLE_ONNX") != "1":
        os.environ["ARGOS_DISABLE_ONNX"] = "1"
    log(
        f"?? onnxruntime unavailable after automated healing: {summary['error'] or 'unknown'}; "
        'ARGOS_DISABLE_ONNX=1'
    )
    _LAST_ONNX_SUMMARY = summary
    return summary


def get_last_onnx_summary() -> Optional[dict[str, object]]:
    """Return the most recent ONNX Runtime summary captured by bootstrap ensure."""
    return _LAST_ONNX_SUMMARY




def _ensure_onnx_runtime_packages() -> None:
    """Guarantee ONNX + ONNX Runtime availability during bootstrap."""
    global _LAST_ONNX_SUMMARY
    summary = ensure_onnxruntime(venv_python(), log=_print)
    _LAST_ONNX_SUMMARY = summary
    if not summary.get("installed"):
        reason = summary.get("error") or "unknown"
        providers = summary.get("providers")
        _print(f"?? onnxruntime unavailable after bootstrap: {reason}; providers={providers}")


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
        for p in _BOOTSTRAP_WEIGHT_PRIORITY.get(key, []):
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
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sp = VENV / ("Lib" if os.name == "nt" else "lib") / pyver / "site-packages"
    sc = sp / "sitecustomize.py"
    if not sc.exists():
        sp.mkdir(parents=True, exist_ok=True)
        sc.write_text(
            "import os\n"
            f"os.environ.setdefault('PYTHONPYCACHEPREFIX', r'{(DATA / 'pycache')}'.replace('\\\\','/'))\n",
            encoding="utf-8",
        )

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
        ("Ensure OpenCV (GUI)", _ensure_opencv_gui),
        (
            "Install Argos (editable)",
            lambda: _pip_install_editable_if_needed(
                reinstall=reinstall,
                with_dev=with_dev,
                extras=extras_tuple,
            ),
        ),
        (
            "Ensure ONNX Runtime",
            _ensure_onnx_runtime_packages,
        ),
        ("Align Torch deps", _ensure_sympy_alignment),
        (
            "Ensure weights",
            (lambda: None)
            if (skip_weights or os.getenv("ARGOS_SKIP_WEIGHTS", "").lower() in {"1", "true", "yes"})
            else (lambda: _ensure_weights_ultralytics(preset=preset, explicit_names=weight_names)),
        ),
        ("Move pytest cache", _move_pytest_cache_out_of_repo),
        ("Sitecustomize (pycache outside)", _ensure_sitecustomize),
        ("pip check (soft)", _pip_check_soft),
    ]

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
