#!/usr/bin/env python3
"""
Argos bootstrap – zero‑touch, idempotent, fast.

What it does (one-time on first run):
  • creates a private venv OUTSIDE the repo (no .venv mess)
  • auto-selects CPU-only Torch when CUDA isn’t present
  • installs your project in editable mode (+dev extras)
  • fetches ONLY the detector + segmenter weights via Ultralytics (no git/LFS)
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
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Sequence, Tuple, Union, overload
from typing import Literal  # pyright: ignore[reportUnusedImport]
from os import PathLike

APP = "rAIn"

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
    sys.stdout.write(str(msg) + ("" if str(msg).endswith("\n") else "\n"))

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
    """
    Thin wrapper over subprocess.run with typed overloads:
      • capture=True  -> returns CompletedProcess[str] (text mode)
      • capture=False -> returns None
    """
    if capture:
        return subprocess.run(
            list(cmd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
            cwd=cwd,
            env=dict(env) if isinstance(env, MutableMapping) else env,  # satisfy Mapping
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

# ──────────────────────────────────────────────────────────────
# Venv + pip
# ──────────────────────────────────────────────────────────────
def _create_venv() -> None:
    _print("→ creating virtual environment (outside repo)…")
    _run([sys.executable, "-m", "venv", str(VENV)], check=True, capture=False)
    _run([str(VPY), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True, capture=False)

def _torch_pins_from_requirements() -> Tuple[str, str]:
    req = ARGOS / "requirements.txt"
    torch_spec = "torch==2.3.*"
    tv_spec    = "torchvision==0.18.*"
    if req.exists():
        for line in req.read_text().splitlines():
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
    # CPU wheels index for Win/Linux when cpu_only; macOS uses default (MPS wheels on PyPI)
    idx: list[str] = []
    if cpu_only and (os.name == "nt" or sys.platform.startswith("linux")):
        idx = ["--index-url", "https://download.pytorch.org/whl/cpu"]
    _print(f"→ installing Torch ({'CPU-only' if cpu_only else 'auto'}) …")
    _run([str(VPY), "-m", "pip", "install", *idx, torch_spec, tv_spec], check=True, capture=False)

def _pip_install_editable_if_needed() -> None:
    # Also ensure Ultralytics is there (weight fetch relies on it)
    need = (not _module_present("panoptes")) or (not _module_present("ultralytics"))
    if not need:
        return
    _print("→ installing Argos package (editable) + dev extras …")
    _run([str(VPY), "-m", "pip", "install", "-e", str(ARGOS) + "[dev]"], check=True, capture=False)

# ──────────────────────────────────────────────────────────────
# Models – Ultralytics-only download of EXACT files you list
# ──────────────────────────────────────────────────────────────
def _ensure_weights_ultralytics() -> None:
    """
    Ensures the top detector + segmenter in panoptes/model_registry exist.
    Never touches Git/LFS; downloads by official Ultralytics names only.
    """
    probe = r"""
import json
from pathlib import Path
from panoptes.model_registry import MODEL_DIR, pick_weight, WEIGHT_PRIORITY
det = pick_weight('detect')  or (WEIGHT_PRIORITY.get('detect')  or [None])[0]
seg = pick_weight('heatmap') or (WEIGHT_PRIORITY.get('heatmap') or [None])[0]
print(json.dumps({'model_dir': str(MODEL_DIR),
                'det': str(det) if det else None,
                'seg': str(seg) if seg else None}))
"""
    cp = _run([str(VPY), "-c", probe], check=True, capture=True)
    info: str = cp.stdout.strip()
    meta: dict[str, Optional[str]] = json.loads(info)
    model_dir = Path(meta["model_dir"] or ".").resolve()
    det_s = meta.get("det")
    seg_s = meta.get("seg")
    want_det: Optional[Path] = Path(det_s) if det_s is not None else None
    want_seg: Optional[Path] = Path(seg_s) if seg_s is not None else None

    missing: list[Path] = [p for p in (want_det, want_seg) if p is not None and not p.exists()]  # type: ignore[arg-type]
    if not missing:
        _print("→ model weights present.")
        return

    if not _module_present("ultralytics"):
        # last-resort safety – should already be installed by -e .[dev]
        _run([str(VPY), "-m", "pip", "install", "ultralytics>=8.0"], check=True, capture=False)

    _print("→ fetching weights via Ultralytics (selected only) …")
    dl = r"""
import sys, shutil
from pathlib import Path
from ultralytics import YOLO
names   = sys.argv[1:-1]
dst_dir = Path(sys.argv[-1]).expanduser().resolve(); dst_dir.mkdir(parents=True, exist_ok=True)
ok = 0
for nm in names:
    try:
        m = YOLO(nm)      # recognised names ("yolov12s-seg.pt", "yolo11x.pt") → download
        p = Path(getattr(m, 'ckpt_path', nm)).expanduser()
        if not p.exists(): p = Path(nm).expanduser()
        if p.exists():
            shutil.copy2(p, dst_dir / p.name)
            ok += 1
    except Exception:
        pass
print(ok)
"""
    names = [p.name for p in missing]
    got_cp = _run([str(VPY), "-c", dl, *names, str(model_dir)], check=True, capture=True)
    got_s: str = (got_cp.stdout.strip() or "0")
    try:
        got = int(got_s)
    except ValueError:
        got = 0

    if got < len(names):
        _print("⚠️  Some weights could not be fetched automatically.")
        _print(f"    Place files manually under: {model_dir}")
    else:
        _print("→ weights downloaded.")

# ──────────────────────────────────────────────────────────────
# Keep repo clean: pytest cache outside; pycache outside via launchers
# ──────────────────────────────────────────────────────────────
def _move_pytest_cache_out_of_repo() -> None:
    ini = HERE / "pytest.ini"
    if not ini.exists():
        cache = (DATA / "pytest_cache").as_posix()
        ini.write_text(textwrap.dedent(f"""\
            [pytest]
            cache_dir = {cache}
        """))

# Optional extra: sitecustomize fallback to keep pycache outside even if someone runs VPY directly
def _ensure_sitecustomize() -> None:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sp = VENV / ("Lib" if os.name == "nt" else "lib") / pyver / "site-packages"
    sc = sp / "sitecustomize.py"
    if not sc.exists():
        sp.mkdir(parents=True, exist_ok=True)
        sc.write_text(
            "import os\n"
            f"os.environ.setdefault('PYTHONPYCACHEPREFIX', r'{(DATA / 'pycache')}'.replace('\\\\','/'))\n"
        )

# ──────────────────────────────────────────────────────────────
# Portable launchers that work BEFORE the venv exists
# ──────────────────────────────────────────────────────────────
def _create_launchers() -> None:
    # POSIX
    sh = HERE / "argos"
    sh.write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail
        HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        PY="$(command -v python3 || command -v python)"
        "$PY" "$HERE/bootstrap.py" --ensure --yes >/dev/null 2>&1 || true
        VPY="$("$PY" "$HERE/bootstrap.py" --print-venv)"
        # keep *.pyc out of the repo
        export PYTHONPYCACHEPREFIX="${XDG_CACHE_HOME:-$HOME/.cache}/rAIn/pycache"
        exec "$VPY" -m panoptes.cli "$@"
    """))
    os.chmod(sh, 0o755)

    # PowerShell
    ps1 = HERE / "argos.ps1"
    ps1.write_text(textwrap.dedent(r"""\
        [CmdletBinding()] param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
        $ErrorActionPreference = "Stop"
        $HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
        $py = (Get-Command python -ErrorAction SilentlyContinue).Source
        if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
        & $py "$HERE\bootstrap.py" --ensure --yes | Out-Null
        $vpy = & $py "$HERE\bootstrap.py" --print-venv
        $env:PYTHONPYCACHEPREFIX = Join-Path $env:LOCALAPPDATA "rAIn\pycache"
        & $vpy -m panoptes.cli @Args
    """))

    # Windows CMD (double-click)
    cmd = HERE / "argos.cmd"
    cmd.write_text(textwrap.dedent(r"""\
        @echo off
        setlocal
        set HERE=%~dp0
        rem choose py -3 if present, else python
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
    """))

# ──────────────────────────────────────────────────────────────
# First-run cheatsheet
# ──────────────────────────────────────────────────────────────
def _print_help(cpu_only: bool) -> None:
    py = VPY
    msg = f"""
★ Argos is ready.

Environment
  • Venv:   {py.parent}
  • Torch:  {'CPU-only' if cpu_only else 'auto'}
  • Models: panoptes/model/  (detector + segmenter)

Quick start (no venv activation)
  Windows PowerShell:
      .\\argos.ps1 tests\\assets\\shibuya.jpg hm --alpha 0.5

  Windows (double-click / cmd):
      argos.cmd tests\\assets\\shibuya.jpg d

  Linux / macOS:
      ./argos tests/assets/shibuya.jpg heatmap --alpha 0.5

  GeoJSON from URL (with #lat…_lon…):
      ./argos "https://…/image.jpg#lat37.8199_lon-122.4783" --task geojson

Power users
  • Makefile shortcuts (inside projects/argos):
      make install     # pip install -e .[dev]
      make test        # run pytest
      make run         # example CLI invocation

  • Direct module:
      {py} -m panoptes.cli tests/assets/shibuya.jpg heatmap

Lambda / API bits (local context)
  • Code + Dockerfile: projects/argos/lambda/
  • Handler name:      app.handler
  • Image expects weights under panoptes/model/.
"""
    _print(textwrap.dedent(msg).rstrip())

# ──────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────
def _first_run() -> bool:
    return not SENTINEL.exists()

def _write_sentinel() -> None:
    SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    SENTINEL.write_text(json.dumps({"version": 3}, indent=2))

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

def _ensure(cpu_only: bool) -> None:
    if not VENV.exists():
        _create_venv()
    _install_torch_if_needed(cpu_only)
    _pip_install_editable_if_needed()
    _ensure_weights_ultralytics()
    _move_pytest_cache_out_of_repo()
    _ensure_sitecustomize()

def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--ensure", action="store_true", help="Ensure env non-interactively (fast, idempotent)")
    p.add_argument("--print-venv", action="store_true", help="Print venv python path")
    p.add_argument("--cpu", dest="cpu_only", action="store_true", help="Force CPU-only Torch")
    p.add_argument("--force-ultralytics", action="store_true", help="Always download weights via Ultralytics (default behavior)")
    p.add_argument("--yes", action="store_true", help="Assume Yes to prompts (non-interactive)")
    args, _ = p.parse_known_args(argv)

    _ensure_dirs()

    if args.print_venv:
        _print(str(VPY))
        return 0

    cpu_only = bool(args.cpu_only or not _has_cuda())

    if args.ensure:
        try:
            _ensure(cpu_only)
        except Exception as e:
            _print(f"ensure failed: {e}")
            return 1
        return 0

    if _first_run():
        _print("First run detected for Argos.")
        if args.yes or _ask_yes_no("Install dependencies and set up now?", default_yes=True):
            _ensure(cpu_only)
            _create_launchers()
            _write_sentinel()
            _print_help(cpu_only)
        else:
            _print("Setup skipped. You can run later with:  python bootstrap.py --ensure")
        return 0

    # Subsequent manual invocations: still idempotent
    _ensure(cpu_only)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
