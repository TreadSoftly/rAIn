#!/usr/bin/env python3
"""
Argos bootstrap - zero-touch, idempotent, fast.
"""
from __future__ import annotations

import argparse
import json
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
    Literal,  # pyright: ignore[reportUnusedImport]
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

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

# Help Windows avoid legacy codepage weirdness in child processes (CI, cmd.exe)
if os.name == "nt":
    os.environ.setdefault("PYTHONUTF8", "1")

# ──────────────────────────────────────────────────────────────
# Path / typing helpers
# ──────────────────────────────────────────────────────────────
StrPath = Union[str, PathLike[str]]

# ──────────────────────────────────────────────────────────────
# Project roots
# ──────────────────────────────────────────────────────────────
HERE: Path = Path(__file__).resolve().parent
# If someone moves this file to repo-root later, auto-fallback to projects/argos
ARGOS: Path = (
    HERE
    if (HERE / "pyproject.toml").exists() and (HERE / "panoptes").exists()
    else (HERE / "projects" / "argos")
)

# ──────────────────────────────────────────────────────────────
# OS-specific config/data dirs (venv lives under DATA/venvs)
# ──────────────────────────────────────────────────────────────
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
VPY: Path = VENV / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")


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
) -> subprocess.CompletedProcess[str]:
    ...


@overload
def _run(
    cmd: Sequence[str],
    *,
    cwd: Optional[StrPath] = ...,
    env: Optional[Mapping[str, str]] = ...,
    check: bool = ...,
    capture: Literal[False] = ...,
) -> None:
    ...


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


# ──────────────────────────────────────────────────────────────
# Fast checks
# ──────────────────────────────────────────────────────────────
def _module_present(mod: str) -> bool:
    code = f"import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('{mod}') else 1)"
    try:
        _run([str(VPY), "-c", code], check=True, capture=False)
        return True
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
    if VENV.exists() and VPY.exists():
        return
    _print("→ creating virtual environment (outside repo)…")
    _run([sys.executable, "-m", "venv", str(VENV)], check=True, capture=False)
    _run([str(VPY), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True, capture=False)


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
            if low.startswith(("torch==", "torch>", "torch<")):
                torch_spec = s
            if low.startswith(("torchvision==", "torchvision>", "torchvision<")):
                tv_spec = s
    return torch_spec, tv_spec


def _install_torch_if_needed(cpu_only: bool) -> None:
    if _module_present("torch") and _module_present("torchvision"):
        return
    torch_spec, tv_spec = _torch_pins_from_requirements()
    idx: list[str] = []
    if cpu_only and (os.name == "nt" or sys.platform.startswith("linux")):
        idx = ["--index-url", "https://download.pytorch.org/whl/cpu"]
    _print(f"→ installing Torch ({'CPU-only' if cpu_only else 'auto'}) …")
    _run([str(VPY), "-m", "pip", "install", *idx, *_constraints_args(), torch_spec, tv_spec], check=True, capture=False)


def _ensure_opencv_gui() -> None:
    _print("→ ensuring OpenCV (GUI-capable) …")
    try:
        _run(
            [
                str(VPY),
                "-m",
                "pip",
                "uninstall",
                "-y",
                "opencv-python-headless",
                "opencv-contrib-python-headless",
            ],
            check=False,
            capture=False,
        )
    except Exception:
        pass
    _run([str(VPY), "-m", "pip", "install", "--upgrade", *_constraints_args(), "opencv-python"], check=True, capture=False)


def _pip_install_editable_if_needed(*, reinstall: bool = False) -> None:
    need = reinstall or (not _module_present("panoptes")) or (not _module_present("ultralytics"))

    headless_present = False
    try:
        from importlib.metadata import PackageNotFoundError, distribution as ild_distribution  # type: ignore
    except Exception:  # pragma: no cover
        from importlib_metadata import PackageNotFoundError, distribution as ild_distribution  # type: ignore
    try:
        ild_distribution("opencv-python-headless")
        headless_present = True
    except PackageNotFoundError:
        headless_present = False
    except Exception:
        headless_present = False

    if headless_present:
        need = True
        _print("→ removing opencv-python-headless (if installed) …")
        _run([str(VPY), "-m", "pip", "uninstall", "-y", "opencv-python-headless"], check=False, capture=False)

    if not need:
        return

    _print("→ installing Argos package (editable) + dev extras …")
    _run(
        [str(VPY), "-m", "pip", "install", "-e", str(ARGOS) + "[dev]", *_constraints_args()],
        check=True,
        capture=False,
    )


# ──────────────────────────────────────────────────────────────
# Weights helpers
# ──────────────────────────────────────────────────────────────
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

# Build name families from registry
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
        py = str(VPY) if VENV.exists() and VPY.exists() else sys.executable
        env = os.environ.copy()
        env_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (str(ARGOS) + (os.pathsep + env_pp if env_pp else ""))

        _run([py, "-c", probe, str(out_path)], check=True, capture=False, env=env)
        txt = out_path.read_text(encoding="utf-8")
        meta: JSONDict = {}
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                meta = cast(JSONDict, obj)
        except Exception:
            meta = {}
    finally:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _to_str_list(v: Any) -> list[str]:
        if isinstance(v, list):
            vv = cast(list[Any], v)
            return [str(x) for x in vv if x is not None]
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
        _run([str(VPY), "-c", code, *args, str(out_path)], check=True, capture=False)
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
    """
    Ensure weights exist under panoptes.model_registry.WEIGHT_PRIORITY.

    Downloads missing *.pt and exports missing *.onnx (quietly) with robust fallbacks.
    """
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

    need = [n for n in want_names if n and not (model_dir / n).exists()]
    if not need:
        _print("→ model weights present.")
        return

    if not _module_present("ultralytics"):
        _run([str(VPY), "-m", "pip", "install", "ultralytics>=8.0"], check=True, capture=False)

    pt_missing = [n for n in need if n.lower().endswith(".pt")]
    onnx_missing = [n for n in need if n.lower().endswith(".onnx")]

    # fetch *.pt directly into model_dir
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

    # Ensure ONNX export toolchain (numpy/onnx/onnxruntime/onnxsim/onnxslim)
    if onnx_missing:
        _print("→ ensuring ONNX export toolchain …")
        if not _module_present("numpy"):
            _run([str(VPY), "-m", "pip", "install", "numpy<2"], check=True, capture=False)
        else:
            try:
                cp = _run([str(VPY), "-c", "import numpy,sys;print(numpy.__version__);sys.exit(0)"], check=True, capture=True)
                if cp and cp.stdout:
                    parts = cp.stdout.strip().split(".")
                    if parts and parts[0].isdigit() and int(parts[0]) >= 2:
                        _run([str(VPY), "-m", "pip", "install", "--upgrade", "numpy<2"], check=True, capture=False)
            except Exception:
                pass

        if not _module_present("onnx"):
            _run([str(VPY), "-m", "pip", "install", "onnx>=1.14,<1.18"], check=True, capture=False)
        if not _module_present("onnxruntime"):
            if sys.version_info < (3, 10):
                _run([str(VPY), "-m", "pip", "install", "onnxruntime==1.19.2"], check=True, capture=False)
            elif os.name == "nt":
                _run([str(VPY), "-m", "pip", "install", "onnxruntime>=1.22,<1.23"], check=True, capture=False)
            else:
                _run([str(VPY), "-m", "pip", "install", "onnxruntime>=1.22,<1.24"], check=True, capture=False)
        if not _module_present("onnxsim"):
            _run([str(VPY), "-m", "pip", "install", "onnxsim>=0.4.17,<0.5"], check=True, capture=False)
        _run([str(VPY), "-m", "pip", "install", "onnxslim>=0.1.59,<0.1.60"], check=True, capture=False)

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

def _latest_exported_onnx() -> Path|None:
    hits=[]
    for root,dirs,files in os.walk('runs'):
        for f in files:
            if f.lower().endswith('.onnx'):
                hits.append(Path(root)/f)
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None

dst = Path(sys.argv[-2]).expanduser().resolve(); dst.mkdir(parents=True, exist_ok=True)
os.chdir(dst)
ok=0
for onnx_name in sys.argv[1:-2]:
    try:
        pt_name = Path(onnx_name).with_suffix(".pt").name
        src_pt = dst / pt_name
        if not src_pt.exists():
            m_fetch = YOLO(pt_name)
            p = Path(getattr(m_fetch, 'ckpt_path', pt_name)).expanduser()
            if p.exists() and p.resolve() != src_pt.resolve():
                shutil.copy2(p, src_pt)
        if not src_pt.exists():
            continue
        # Try opsets 19 -> 17 -> 12 with fallbacks
        combos=[(19,True,True),(19,True,False),(19,False,False),
                (17,True,True),(17,True,False),(17,False,False),
                (12,True,True),(12,True,False),(12,False,False)]
        exported=False
        for (op,dyn,simp) in combos:
            try:
                outp = Path(YOLO(str(src_pt)).export(format='onnx', dynamic=dyn, simplify=simp, imgsz=640, opset=op, device='cpu'))
                # Ultralytics returns a path string; prefer target name
            except Exception:
                continue
            target = dst / onnx_name
            cand = _latest_exported_onnx()
            if target.exists():
                pass
            elif cand and cand.exists():
                shutil.copy2(cand, target)
            elif outp and outp.exists() and outp.resolve() != target.resolve():
                shutil.copy2(outp, target)
            if target.exists() and target.stat().st_size > 1_000_000:
                exported=True
                break
        if exported:
            ok += 1
    except Exception:
        pass
try:
    shutil.rmtree(dst / 'runs', ignore_errors=True)
except Exception:
    pass
out_path = Path(sys.argv[-1]); out_path.write_text(json.dumps({'ok': ok}), encoding='utf-8')
"""
        data2 = _child_json(exp, [*onnx_missing, str(model_dir)])
        if int(data2.get("ok", 0)) < len(onnx_missing):
            _print("⚠️  Some *.onnx could not be exported.")

    _print("→ weights ensured.")


# ──────────────────────────────────────────────────────────────
# Keep repo clean: pytest cache outside; pycache outside via launchers
# ──────────────────────────────────────────────────────────────
def _move_pytest_cache_out_of_repo() -> None:
    ini = HERE / "pytest.ini"
    if not ini.exists():
        cache = (DATA / "pytest_cache").as_posix()
        ini.write_text(
            textwrap.dedent(
                f"""\
            [pytest]
            cache_dir = {cache}
        """
            ),
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
    # POSIX
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

    # PowerShell
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

    # Windows CMD
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
        _run([str(VPY), "-m", "pip", "check"], check=True, capture=False)
    except Exception:
        _print("⚠️  'pip check' reported issues; continuing. Details follow:")
        try:
            cp = _run([str(VPY), "-m", "pip", "check"], check=False, capture=True)
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
) -> None:
    steps: list[tuple[str, Callable[[], None]]] = [
        ("Create venv", _create_venv),
        ("Install Torch", lambda: _install_torch_if_needed(cpu_only)),
        ("Ensure OpenCV (GUI)", _ensure_opencv_gui),
        ("Install Argos (editable)", lambda: _pip_install_editable_if_needed(reinstall=reinstall)),
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
    p.add_argument("--yes", action="store_true", help="Assume Yes to prompts (non-interactive)")
    args, _ = p.parse_known_args(argv)
    _ensure_dirs()

    if args.print_venv:
        _print(str(VPY))
        return 0

    cpu_only = bool(args.cpu_only or not _has_cuda())

    if args.ensure:
        try:
            _ensure(cpu_only, preset=args.weights_preset, reinstall=args.reinstall)
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
                # interactive menu optional; unchanged
                pass
            elif preset:
                os.environ["ARGOS_WEIGHT_PRESET"] = preset

            _ensure(cpu_only, preset=preset, weight_names=picks, skip_weights=skip, reinstall=args.reinstall)
            _create_launchers()
            _write_sentinel()
            _print(
                f"""
★ Argos is ready.

Environment
  • Venv:   {VPY.parent}
  • Torch:  {'CPU-only' if cpu_only else 'auto'}
  • Models: panoptes/model/ (Ultralytics fetched; preset={{default|nano|perception}}; env ARGOS_WEIGHT_PRESET)

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

CI / CD
  • Use ARGOS_WEIGHT_PRESET=[all|default|nano|perception] and run:
      python projects/argos/bootstrap.py --ensure --yes
""".rstrip()
            )
        else:
            _print("Setup skipped. You can run later with:  python bootstrap.py --ensure")
        return 0

    _ensure(cpu_only, preset=args.weights_preset, reinstall=args.reinstall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
