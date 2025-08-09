#!/usr/bin/env python3
"""
Argos bootstrap – zero-touch, idempotent, fast.

What it does (one-time on first run):
  • creates a private venv OUTSIDE the repo (no .venv mess)
  • auto-selects CPU-only Torch when CUDA isn’t present
  • installs your project in editable mode (+dev extras)
  • fetches model weights via Ultralytics (no git/LFS)
    - preset: all | default | nano (ARGOS_WEIGHT_PRESET)
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
    """
    Robust print that won't crash on Windows legacy codepages when emitting
    Unicode (e.g., ★, →). Falls back to UTF-8 bytes with replacement.
    """
    s = str(msg)
    if not s.endswith("\n"):
        s += "\n"
    try:
        sys.stdout.write(s)
    except UnicodeEncodeError:
        try:
            sys.stdout.buffer.write(s.encode("utf-8", "replace"))
        except Exception:
            # Last resort: strip to ASCII replacements
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
# Weights helpers
# ──────────────────────────────────────────────────────────────
def _probe_weight_presets() -> tuple[Path, list[str], list[str], list[str]]:
    """Return (model_dir, all_names, default_names, nano_names) from registry."""
    probe = r"""
import json
from pathlib import Path
from panoptes.model_registry import MODEL_DIR, WEIGHT_PRIORITY

def uniq(xs):
    seen=set(); out=[]
    for x in xs:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

all_names=[]
for _, paths in WEIGHT_PRIORITY.items():
    for p in paths: all_names.append(Path(p).name)

def first_or_none(lst): return lst[0] if lst else None
detect_first  = first_or_none(WEIGHT_PRIORITY.get('detect', []))
heatmap_first = first_or_none(WEIGHT_PRIORITY.get('heatmap', []))

default_names = [Path(detect_first).name if detect_first else None,
                 Path(heatmap_first).name if heatmap_first else None]

nano_names = [Path(p).name for p in WEIGHT_PRIORITY.get('detect_small', []) +
                               WEIGHT_PRIORITY.get('heatmap_small', [])]

print(json.dumps({'model_dir': str(MODEL_DIR),
                  'all': uniq(all_names),
                  'default': uniq(default_names),
                  'nano': uniq(nano_names)}))
"""
    cp = _run([str(VPY), "-c", probe], check=True, capture=True)
    meta = json.loads(cp.stdout.strip())
    return Path(meta["model_dir"]).resolve(), list(meta["all"]), list(meta["default"]), list(meta["nano"])

def _ensure_weights_ultralytics(*, preset: Optional[str] = None, explicit_names: Optional[list[str]] = None) -> None:
    """
    Ensure weights exist under panoptes.model_registry.WEIGHT_PRIORITY.

    Preset (env or arg):
      - "all"     → everything listed (default)
      - "default" → first detect + first heatmap
      - "nano"    → detect_small + heatmap_small

    • Downloads *.pt via Ultralytics
    • Builds missing *.onnx by exporting from the matching *.pt
    """
    model_dir, all_names, default_names, nano_names = _probe_weight_presets()

    # choose names
    env_preset = (os.getenv("ARGOS_WEIGHT_PRESET") or "").strip().lower()
    choice = (preset or env_preset or "all").lower()
    if explicit_names is not None:
        want_names = [n for n in explicit_names if n]
    else:
        if choice not in {"all", "default", "nano"}:
            choice = "all"
        want_names = {"all": all_names, "default": default_names, "nano": nano_names}[choice]

    need = [n for n in want_names if n and not (model_dir / n).exists()]
    if not need:
        _print("→ model weights present.")
        return

    # Ultralytics for fetching/exporting
    if not _module_present("ultralytics"):
        _run([str(VPY), "-m", "pip", "install", "ultralytics>=8.0"], check=True, capture=False)

    pt_missing   = [n for n in need if n.lower().endswith(".pt")]
    onnx_missing = [n for n in need if n.lower().endswith(".onnx")]

    # fetch *.pt
    if pt_missing:
        _print(f"→ fetching {len(pt_missing)} *.pt via Ultralytics …")
        dl_pt = r"""
import sys, shutil
from pathlib import Path
from ultralytics import YOLO
dst = Path(sys.argv[-1]).expanduser().resolve(); dst.mkdir(parents=True, exist_ok=True)
ok=0
for nm in sys.argv[1:-1]:
    try:
        m=YOLO(nm)
        p=Path(getattr(m,'ckpt_path',nm)).expanduser()
        if not p.exists(): p=Path(nm).expanduser()
        if p.exists():
            shutil.copy2(p, dst / p.name)
            ok+=1
    except Exception:
        pass
print(ok)
"""
        got = int((_run([str(VPY), "-c", dl_pt, *pt_missing, str(model_dir)],
                        check=True, capture=True).stdout.strip() or "0"))
        if got < len(pt_missing):
            _print("⚠️  Some *.pt weights could not be fetched.")

    # export *.onnx from matching *.pt
    if onnx_missing:
        _print(f"→ exporting {len(onnx_missing)} *.onnx from matching *.pt …")
        exp = r"""
import sys, shutil
from pathlib import Path
from ultralytics import YOLO
dst = Path(sys.argv[-1]).expanduser().resolve(); dst.mkdir(parents=True, exist_ok=True)
ok=0
for onnx_name in sys.argv[1:-1]:
    try:
        pt_name = Path(onnx_name).with_suffix(".pt").name
        src_pt  = dst / pt_name
        if not src_pt.exists():
            m_fetch = YOLO(pt_name)
            p = Path(getattr(m_fetch,'ckpt_path',pt_name)).expanduser()
            if p.exists(): shutil.copy2(p, dst / p.name)
        m = YOLO(str(dst / pt_name))
        outp = Path(m.export(format='onnx', dynamic=True, simplify=True, imgsz=640, device='cpu'))
        shutil.copy2(outp, dst / onnx_name)
        ok += 1
    except Exception:
        pass
print(ok)
"""
        got2 = int((_run([str(VPY), "-c", exp, *onnx_missing, str(model_dir)],
                         check=True, capture=True).stdout.strip() or "0"))
        if got2 < len(onnx_missing):
            _print("⚠️  Some *.onnx could not be exported.")

    _print("→ weights ensured.")

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
        """), encoding="utf-8")

# Optional extra: sitecustomize fallback to keep pycache outside even if someone runs VPY directly
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
    """), encoding="utf-8")
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
    """), encoding="utf-8")

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
    """), encoding="utf-8")

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
  • Models: panoptes/model/  (Ultralytics fetched; preset={os.getenv('ARGOS_WEIGHT_PRESET','all')})

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

CI / CD
  • Use ARGOS_WEIGHT_PRESET=[all|default|nano] and run:
      python projects/argos/bootstrap.py --ensure --yes
"""
    _print(textwrap.dedent(msg).rstrip())

# ──────────────────────────────────────────────────────────────
# Interactive first-run menu
# ──────────────────────────────────────────────────────────────
def _first_run_menu() -> tuple[Optional[str], Optional[list[str]], bool]:
    """
    Ask the user which weights to fetch.
    Returns (preset, explicit_names, skip_weights)
    """
    model_dir, all_names = _probe_weight_presets()[:2]
    _print("\nModel weights will be placed under: " + str(model_dir))
    _print("\nChoose what to fetch now:")
    _print("  [1] All listed weights (recommended)")
    _print("  [2] Default pair (first detect + first heatmap)")
    _print("  [3] Nano pair (fastest for laptops & CI)")
    _print("  [4] Pick from list")
    _print("  [5] Skip for now")
    while True:
        ans = input("Selection [1-5, default 1]: ").strip()
        if not ans or ans == "1":
            return ("all", None, False)
        if ans == "2":
            return ("default", None, False)
        if ans == "3":
            return ("nano", None, False)
        if ans == "5":
            return (None, None, True)
        if ans == "4":
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
            _print("Please enter 1, 2, 3, 4 or 5.")

# ──────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────
def _first_run() -> bool:
    return not SENTINEL.exists()

def _write_sentinel() -> None:
    SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    SENTINEL.write_text(json.dumps({"version": 4}, indent=2), encoding="utf-8")

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

def _ensure(
    cpu_only: bool,
    *,
    preset: Optional[str] = None,
    weight_names: Optional[list[str]] = None,
    skip_weights: bool = False,
) -> None:
    if not VENV.exists():
        _create_venv()
    _install_torch_if_needed(cpu_only)
    _pip_install_editable_if_needed()
    if not skip_weights:
        _ensure_weights_ultralytics(preset=preset, explicit_names=weight_names)
    _move_pytest_cache_out_of_repo()
    _ensure_sitecustomize()

def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--ensure", action="store_true", help="Ensure env non-interactively (fast, idempotent)")
    p.add_argument("--print-venv", action="store_true", help="Print venv python path")
    p.add_argument("--cpu", dest="cpu_only", action="store_true", help="Force CPU-only Torch")
    p.add_argument("--weights-preset", choices=["all", "default", "nano"], help="Preset for weights (overrides env)")
    p.add_argument("--yes", action="store_true", help="Assume Yes to prompts (non-interactive)")
    args, _ = p.parse_known_args(argv)

    _ensure_dirs()

    if args.print_venv:
        _print(str(VPY))
        return 0

    cpu_only = bool(args.cpu_only or not _has_cuda())

    if args.ensure:
        try:
            _ensure(cpu_only, preset=args.weights_preset)
        except Exception as e:
            _print(f"ensure failed: {e}")
            return 1
        return 0

    if _first_run():
        _print("First run detected for Argos.")
        if args.yes or _ask_yes_no("Install dependencies and set up now?", default_yes=True):
            # interactive weights choice unless --yes was used
            preset = args.weights_preset
            picks: Optional[list[str]] = None
            skip = False
            if not args.yes and preset is None:
                preset, picks, skip = _first_run_menu()
                if preset:
                    os.environ["ARGOS_WEIGHT_PRESET"] = preset
            elif preset:
                os.environ["ARGOS_WEIGHT_PRESET"] = preset

            _ensure(cpu_only, preset=preset, weight_names=picks, skip_weights=skip)
            _create_launchers()
            _write_sentinel()
            _print_help(cpu_only)
        else:
            _print("Setup skipped. You can run later with:  python bootstrap.py --ensure")
        return 0

    # Subsequent manual invocations: still idempotent
    _ensure(cpu_only, preset=args.weights_preset)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
