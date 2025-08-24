# projects/argos/panoptes/model/_fetch_models.py
from __future__ import annotations

import glob
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    cast,
)

import typer

# ------------------------------------------------------------------------------------
# One-time ONNX preflight/repair flags
# ------------------------------------------------------------------------------------
# Mutable session flags (lowercase to avoid constant-redefinition warnings)
_onnx_preflight_done: bool = False
_onnx_usable: Optional[bool] = None
_DIAG_LOG_NAME = "_onnx_diagnostics.txt"
_ECHO_ONNX_DIAG = os.environ.get("ARGOS_ECHO_ONNX_DIAG", "1") not in ("0", "false", "False")

# Force binary-only installs inside this process too (safety net)
os.environ.setdefault("PIP_ONLY_BINARY", ":all:")
os.environ.setdefault("PIP_NO_BUILD_ISOLATION", "1")
os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

# ---------------------------------------------------------------------
# Minimal no-op spinner (works with "with ... as sp: sp.update(...)")
# ---------------------------------------------------------------------
class _NoopSpinner:
    def __enter__(self) -> "_NoopSpinner":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        return None

    def update(self, **_: Any) -> "_NoopSpinner":
        return self


# ---------------------------------------------------------------------
# Halo-based progress (Panoptes) fallbacks + Path-safe osc8 wrapper
# ---------------------------------------------------------------------
class _ProgressLike(Protocol):
    def update(self, **kwargs: Any) -> Any: ...
    def __enter__(self) -> "_ProgressLike": ...
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None: ...


try:
    from panoptes.progress import osc8 as _osc8_raw  # type: ignore
    from panoptes.progress import percent_spinner as _percent_spinner  # type: ignore
    from panoptes.progress import simple_status as _simple_status  # type: ignore

    def percent_spinner(*args: Any, **kwargs: Any) -> ContextManager[Any]:  # type: ignore[misc]
        return _percent_spinner(*args, **kwargs)  # type: ignore[misc]

    def simple_status(*args: Any, **kwargs: Any) -> ContextManager[Any]:  # type: ignore[misc]
        return _simple_status(*args, **kwargs)  # type: ignore[misc]

except Exception:  # pragma: no cover

    def percent_spinner(*_: Any, **__: Any) -> ContextManager[Any]:  # type: ignore[no-redef]
        return _NoopSpinner()

    def simple_status(*_: Any, **__: Any) -> ContextManager[Any]:  # type: ignore[no-redef]
        return _NoopSpinner()

    def _osc8_raw(label: str, target: str) -> str:  # type: ignore[no-redef]
        return str(target)


def osc8_link(label: str, target: str | Path) -> str:
    try:
        return _osc8_raw(label, str(target))
    except Exception:
        return str(target)


# Our byte-accurate downloader (works with .update(total=..., count=..., current=...))
try:
    from panoptes.progress.integrations.download_progress import download_url  # type: ignore
except Exception:
    download_url = None  # type: ignore[assignment]


# ---------------------------------------------------------------------
# Resolve model directory from registry (fallback to repo path)
# ---------------------------------------------------------------------
try:
    from panoptes.model_registry import MODEL_DIR as _REG_MODEL_DIR  # type: ignore
    _registry_model_dir: Optional[Path] = Path(_REG_MODEL_DIR)  # type: ignore[arg-type]
except Exception:
    _registry_model_dir = None

MODEL_DIR: Path = _registry_model_dir or (Path(__file__).resolve().parents[2] / "panoptes" / "model")


# ---------------------------------------------------------------------
# Auto-install export deps (keep quiet; avoid importing libs that lack stubs)
# ---------------------------------------------------------------------
def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def _pip_quiet(*pkgs: str, force_reinstall: bool = False) -> None:
    if not pkgs:
        return
    args = [sys.executable, "-m", "pip", "install", "--no-input", "--quiet"]
    if force_reinstall:
        args.append("--force-reinstall")
    args.extend(pkgs)
    env = os.environ.copy()
    env.setdefault("PIP_ONLY_BINARY", ":all:")
    env.setdefault("PIP_NO_BUILD_ISOLATION", "1")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    try:
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    except Exception:
        # best-effort — export will still try fallbacks
        pass


def _parse_ver_tuple(s: str) -> Tuple[int, int, int]:
    nums = [int(x) for x in re.findall(r"\d+", s)[:3]]
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def _decide_opset(torch_version_str: Optional[str]) -> int:
    """
    Heuristic to pick a safe default opset for a given torch version:
    - torch >= 2.4 → opset 19
    - torch >= 2.2 → opset 17
    - else         → opset 12
    """
    tv = _parse_ver_tuple(torch_version_str or "0.0.0")
    if tv >= (2, 4, 0):
        return 19
    if tv >= (2, 2, 0):
        return 17
    return 12


def _candidate_opsets(preferred: int) -> List[int]:
    """
    Try the preferred opset first, then back off to other commonly supported ones.
    """
    order = [preferred, 19, 17, 12]
    seen: Set[int] = set()
    out: List[int] = []
    for o in order:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


def _local_onnx_shadowing() -> List[Path]:
    """
    Detect local files that would shadow the real 'onnx' package.
    """
    bad: List[Path] = []
    try:
        cwd = Path.cwd()
        if (cwd / "onnx.py").exists():
            bad.append(cwd / "onnx.py")
        if (cwd / "onnx").is_dir():
            bad.append(cwd / "onnx")
        if sys.path:
            p0 = Path(sys.path[0])
            if (p0 / "onnx.py").exists():
                bad.append(p0 / "onnx.py")
            if (p0 / "onnx").is_dir():
                bad.append(p0 / "onnx")
    except Exception:
        pass
    return bad


def _diag_log_path() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / _DIAG_LOG_NAME


def _write_diag(header: str, lines: Iterable[str]) -> None:
    try:
        log = _diag_log_path()
        with log.open("a", encoding="utf-8", errors="ignore") as fh:
            fh.write(f"\n=== {header} ===\n")
            for ln in lines:
                fh.write(str(ln).rstrip() + "\n")
    except Exception:
        pass


def _snapshot_env() -> None:
    try:
        lines: List[str] = []
        lines.append(f"Time: {time.ctime()}")
        lines.append(f"OS: {platform.platform()}  ({os.name})")
        lines.append(f"Machine/Arch: {platform.machine()}  Python: {sys.version.split()[0]}  Bits: {platform.architecture()[0]}")
        lines.append(f"Executable: {sys.executable}")
        lines.append(f"PATH[:300]: {os.environ.get('PATH','')[:300]}")
        # VC++ DLLs presence (Windows)
        if os.name == "nt":
            win = os.environ.get("WINDIR", r"C:\Windows")
            sys32 = Path(win) / "System32"
            for nm in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
                lines.append(f"{nm}: {'present' if (sys32 / nm).exists() else 'missing'}")
        # Key packages
        pkgs = ("onnx", "onnxruntime", "numpy", "torch", "protobuf", "ultralytics")
        for p in pkgs:
            try:
                out = subprocess.check_output([sys.executable, "-m", "pip", "show", p], stderr=subprocess.STDOUT).decode("utf-8", "ignore")
                ver = ""
                for ln in out.splitlines():
                    if ln.lower().startswith("version:"):
                        ver = ln.split(":", 1)[1].strip()
                        break
                lines.append(f"{p}: {ver or 'not installed'}")
            except Exception:
                lines.append(f"{p}: not installed")
        _write_diag("ENV SNAPSHOT", lines)
    except Exception:
        pass


def _ensure_export_toolchain() -> None:
    """
    Idempotently ensure a working export toolchain:
    - numpy < 2 (avoid ABI/DLL issues)
    - protobuf < 5 (safer with onnx 1.16–1.17)
    - ultralytics >=8.3,<8.6
    - onnx >=1.16,<1.18
    - onnxruntime:
          * Python <3.10 → 1.19.2
          * Windows (Py>=3.10) → >=1.22,<1.23
          * Linux/macOS (Py>=3.10) → >=1.22,<1.24
    - onnxsim >=0.4.17,<0.5
    - onnxslim >=0.1.59,<0.1.60
    """
    # NumPy cap first
    try:
        import numpy as _np  # type: ignore
        nvt = _parse_ver_tuple(getattr(_np, "__version__", "0.0.0"))
        if nvt >= (2, 0, 0):
            _pip_quiet("numpy<2", force_reinstall=True)
    except Exception:
        _pip_quiet("numpy<2")

    # protobuf cap (ONNX compatibility)
    _pip_quiet("protobuf<5")

    # Ultralytics range
    need_ultra = False
    if not _have("ultralytics"):
        need_ultra = True
    else:
        try:
            from importlib.metadata import version as _ver  # type: ignore
        except Exception:  # pragma: no cover
            try:
                from importlib_metadata import version as _ver  # type: ignore
            except Exception:
                _ver = None  # type: ignore
        try:
            raw_v = _ver("ultralytics") if _ver else None  # type: ignore[call-arg]
            v: Optional[str] = raw_v if isinstance(raw_v, str) else None
            if v is not None:
                vt = _parse_ver_tuple(v)
                if not ((8, 3, 0) <= vt < (8, 6, 0)):
                    need_ultra = True
            else:
                need_ultra = True
        except Exception:
            need_ultra = True
    if need_ultra:
        _pip_quiet("ultralytics>=8.3,<8.6")

    # ONNX bits
    if not _have("onnx"):
        _pip_quiet("onnx>=1.16,<1.18")
    if not _have("onnxruntime"):
        if sys.version_info < (3, 10):
            _pip_quiet("onnxruntime==1.19.2")
        elif os.name == "nt":
            _pip_quiet("onnxruntime>=1.22,<1.23")
        else:
            _pip_quiet("onnxruntime>=1.22,<1.24")

    # Simplification helpers
    if not _have("onnxsim"):
        _pip_quiet("onnxsim>=0.4.17,<0.5")
    _pip_quiet("onnxslim>=0.1.59,<0.1.60")


def _try_import_onnx() -> Tuple[bool, str]:
    import traceback
    details: List[str] = []
    ok = True
    # onnx
    try:
        import onnx  # type: ignore
        details.append(f"onnx.version: {getattr(onnx, 'version', 'unknown')}")
    except Exception:
        ok = False
        details.append("onnx import failed:\n" + traceback.format_exc())
    # onnxruntime
    try:
        import onnxruntime as ort  # type: ignore
        details.append(f"onnxruntime: {getattr(ort, '__version__', 'unknown')}")
        try:
            # NOTE: cast to Any so static type checkers don't flag unknown member type
            providers = cast(Any, ort).get_available_providers()
            details.append(f"providers: {providers}")
        except Exception:
            details.append("providers query failed:\n" + traceback.format_exc())
    except Exception:
        ok = False
        details.append("onnxruntime import failed:\n" + traceback.format_exc())
    return ok, (" \n".join(details) if not ok else "ok")


def _repair_onnx_stack() -> None:
    """
    Force-repair common causes of 'onnx_cpp2py_export' DLL failures within the venv.
    """
    steps: List[str] = []
    steps.append("Reinstall numpy<2, protobuf<5, onnx>=1.16,<1.18 (binary-only); refresh onnxruntime")
    _write_diag("REPAIR START", steps)

    # Hard reinstall to ensure consistent ABI
    _pip_quiet("numpy<2", "protobuf<5", "onnx>=1.16,<1.18", force_reinstall=True)
    if sys.version_info < (3, 10):
        _pip_quiet("onnxruntime==1.19.2", force_reinstall=True)
    elif os.name == "nt":
        _pip_quiet("onnxruntime>=1.22,<1.23", force_reinstall=True)
    else:
        _pip_quiet("onnxruntime>=1.22,<1.24", force_reinstall=True)

    _write_diag("REPAIR END", ["done"])


def _preflight_and_repair_onnx_once() -> bool:
    """
    Run ONNX import diagnostics and auto-repair exactly once per session.
    Writes rich info to _onnx_diagnostics.txt in the model dir.
    """
    global _onnx_preflight_done, _onnx_usable
    if _onnx_preflight_done:
        return bool(_onnx_usable)

    _onnx_preflight_done = True
    _snapshot_env()

    # Shadowing check
    bad = _local_onnx_shadowing()
    if bad:
        _write_diag("LOCAL SHADOWING", [f"- {p}" for p in bad])

    # Initial toolchain ensure (installs pins if missing)
    _ensure_export_toolchain()

    usable, err = _try_import_onnx()
    if not usable:
        _write_diag("FIRST IMPORT FAILURE", [err])

        # Attempt repair
        _repair_onnx_stack()
        usable, err2 = _try_import_onnx()
        if not usable:
            _write_diag("SECOND IMPORT FAILURE", [err2])

    _onnx_usable = usable

    # Console summary (single line)
    if usable:
        try:
            typer.secho("ONNX environment: OK (validated)", fg="green")
        except Exception:
            pass
    else:
        try:
            typer.secho(
                f"ONNX environment not usable after auto-repair. See log: {osc8_link(_DIAG_LOG_NAME, _diag_log_path())}",
                fg="red",
            )
        except Exception:
            pass
        # Echo tail of the diagnostics to the terminal for immediate visibility
        try:
            if _ECHO_ONNX_DIAG:
                log = _diag_log_path()
                if log.exists():
                    tail = log.read_text(encoding="utf-8", errors="ignore").splitlines()[-200:]
                    print("\n--- _onnx_diagnostics.txt (tail) ---")
                    print("\n".join(tail))
                    print("--- end diagnostics ---\n")
        except Exception:
            pass

    return usable


# Ensure deps before import; keep Ultralytics quiet afterwards
_ensure_export_toolchain()

# Ultralytics (used to export ONNX and as last-resort fetcher)
has_yolo: bool
try:
    from ultralytics import YOLO as _YOLO  # type: ignore

    try:
        # silence Ultralytics logger; we print succinct logs ourselves
        from ultralytics.utils import LOGGER as _ULTRA_LOGGER  # type: ignore
        _rem = getattr(_ULTRA_LOGGER, "remove", None)
        if callable(_rem):
            _rem()
        else:
            for h in list(getattr(_ULTRA_LOGGER, "handlers", [])):
                try:
                    _ULTRA_LOGGER.removeHandler(h)  # type: ignore[attr-defined]
                except Exception:
                    pass
    except Exception:
        pass
    has_yolo = True
except Exception:
    _YOLO = None  # type: ignore[assignment]
    has_yolo = False

YOLO = _YOLO  # expose once

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

# Where the official assets live (kept for direct .pt/.onnx downloads)
ASSETS_BASE = os.getenv("ULTRA_ASSETS_BASE", "https://github.com/ultralytics/assets/releases/download/v8.3.0").rstrip("/")

# ---------------------------------------------------------------------
# Packs / naming
# ---------------------------------------------------------------------
VERSIONS: Tuple[str, ...] = ("8", "11", "12")
SIZES: Tuple[str, ...] = ("x", "l", "m", "s", "n")
EXTS: Tuple[str, ...] = (".pt", ".onnx")
TASKS: Tuple[str, ...] = ("det", "seg", "pose", "cls", "obb")

# A sensible curated default (covers tasks, sizes; includes .onnx for validation)
DEFAULT_PACK: List[str] = [
    # DETECT
    "yolov8x.pt",
    "yolo11x.pt",
    "yolo12x.pt",
    "yolo12x.onnx",
    "yolov8n.pt",
    "yolov8n.onnx",
    "yolov8s.pt",
    "yolov8s.onnx",
    "yolo11n.pt",
    "yolo11n.onnx",
    "yolo12n.pt",
    "yolo12n.onnx",

    # SEG
    "yolov8x-seg.pt",
    "yolov8n-seg.onnx",
    "yolov8s-seg.pt",
    "yolov8s-seg.onnx",
    "yolo11x-seg.pt",
    "yolo11n-seg.onnx",
    "yolo11s-seg.pt",
    "yolo11s-seg.onnx",

    # POSE
    "yolov8x-pose.pt",
    "yolov8n-pose.onnx",
    "yolov8s-pose.pt",
    "yolov8s-pose.onnx",
    "yolo11x-pose.pt",
    "yolo11n-pose.onnx",
    "yolo11s-pose.pt",
    "yolo11s-pose.onnx",

    # CLS
    "yolov8x-cls.pt",
    "yolov8n-cls.onnx",
    "yolo11x-cls.pt",
    "yolo11n-cls.onnx",

    # OBB
    "yolov8x-obb.pt",
    "yolov8n-obb.onnx",
    "yolov8s-obb.pt",
    "yolov8s-obb.onnx",
    "yolo11x-obb.pt",
    "yolo11n-obb.onnx",
    "yolo11s-obb.onnx",
]

# ---------------------------------------------------------------------
# Small helpers (naming, UX, links)
# ---------------------------------------------------------------------
def _mk(ver: str, size: str, task: str, ext: str) -> str:
    """
    Construct official Ultralytics-style names.

    Ultralytics naming:
    • YOLOv8:  yolov8{s}.pt / yolov8{s}-seg.pt / -pose / -cls / -obb
    • YOLO11:  yolo11{s}.pt / yolo11{s}-seg.pt / -pose / -cls / -obb
    • YOLO12:  yolo12{s}.pt / yolo12{s}-seg.pt / -pose / -cls / -obb
    """
    base = f"yolov{ver}{size}" if ver == "8" else f"yolo{ver}{size}"
    suf = {"det": "", "seg": "-seg", "pose": "-pose", "cls": "-cls", "obb": "-obb"}[task]
    return base + suf + ext


def _full_pack() -> List[str]:
    """All versions/sizes/tasks; .pt + .onnx (deduped)."""
    out: List[str] = []
    seen: Set[str] = set()
    for ver in VERSIONS:
        for sz in SIZES:
            for task in TASKS:
                for ext in EXTS:
                    n = _mk(ver, sz, task, ext)
                    if n not in seen:
                        seen.add(n)
                        out.append(n)
    return out


def _dedupe(names: Iterable[str]) -> List[str]:
    return sorted({n.strip(): None for n in names if n and n.strip()}.keys())


def _ok(msg: str) -> None:
    typer.secho(msg, fg="green")


def _warn(msg: str) -> None:
    typer.secho(msg, fg="yellow")


def _err(msg: str) -> None:
    typer.secho(msg, fg="red", err=True)


def _human_bytes(n: float | int) -> str:
    try:
        f = float(n)
    except Exception:
        f = 0.0
    units = ["B", "KB", "MB", "GB", "TB"]
    for u in units:
        if f < 1024.0 or u == units[-1]:
            return f"{f:.1f}{u}"
        f /= 1024.0
    return f"{n}B"


# ---------------------------------------------------------------------
# Temporary env/context helpers
# ---------------------------------------------------------------------
@contextmanager
def _cd(path: Path) -> Iterator[None]:
    """Temporarily chdir to *path* (restores on exit)."""
    prev = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev)


@contextmanager
def _silence_stdio() -> Iterator[None]:
    buf_out, buf_err = StringIO(), StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        yield


@contextmanager
def _tqdm_disabled_env() -> Iterator[None]:
    old = os.environ.get("TQDM_DISABLE", None)
    os.environ["TQDM_DISABLE"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = old


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------
def _looks_like_text_header(bs: bytes) -> bool:
    s = bs.lstrip()
    if not s:
        return False
    return s.startswith(b"<") or s.startswith(b"{") or b"AccessDenied" in s or b"Error" in s or b"<!DOCTYPE" in s


def _validate_weight(p: Path) -> bool:
    """
    Basic sanity for .pt / .onnx:
    - exists
    - size ≥ 1MB (avoid truncated blobs)
    - not an HTML/JSON error page
    """
    try:
        if not p.exists():
            return False
        if p.stat().st_size < 1_000_000:  # ~1MB
            return False
        with p.open("rb") as fh:
            head = fh.read(512)
        if _looks_like_text_header(head):
            return False
        return True
    except Exception:
        return False


def _asset_url_for(name: str) -> str:
    return f"{ASSETS_BASE}/{Path(name).name}"


def _latest_exported_onnx() -> Optional[Path]:
    hits = [Path(p) for p in glob.glob("runs/**/**/*.onnx", recursive=True)]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


# ---------------------------------------------------------------------
# Spinner adapter (byte progress → single-line UX)
# ---------------------------------------------------------------------
class _DownloadSpinnerAdapter:
    """
    Adapts byte-accurate updates into spinner.update(...) without disturbing item totals.
    """

    def __init__(self, spinner: _ProgressLike, *, items_total: int, items_done: int, label: Optional[str] = None) -> None:
        self.spinner = spinner
        self.items_total = int(max(1, items_total))
        self.items_done = int(max(0, items_done))
        self.label = label or ""
        self.total_bytes: float = 0.0

    def update(self, **kwargs: Any) -> "_DownloadSpinnerAdapter":
        try:
            total = kwargs.get("total", None)
            cur = kwargs.get("current", None)
            cnt = kwargs.get("count", None)

            if cur:
                self.label = str(cur)

            if isinstance(total, (int, float)):
                self.total_bytes = float(total)

            job_txt = "downloading"
            if isinstance(cnt, (int, float)) and self.total_bytes > 0:
                done = float(cnt)
                job_txt = f"dl { _human_bytes(done) }/{ _human_bytes(self.total_bytes) }"

            self.spinner.update(
                total=self.items_total,
                count=self.items_done,
                current=self.label,
                job=job_txt,
                model=self.label,
            )
        except Exception:
            pass
        return self


# ---------------------------------------------------------------------
# Fetch logic
# ---------------------------------------------------------------------
def _fetch_one(name: str, dst: Path, *, spinner: Optional[_ProgressLike], items_total: int, items_done: int) -> Tuple[str, str]:
    """
    Obtain *name* into *dst* and return (basename, action).

    Actions: "present" | "download" | "copied" | "exported" | "failed"
    """
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / Path(name).name

    # Announce
    if spinner is not None:
        try:
            spinner.update(total=items_total, count=items_done, current=target.name, job="check", model=target.name)
        except Exception:
            pass

    # Already present (and sane)
    if target.exists() and _validate_weight(target):
        return (target.name, "present")
    elif target.exists() and not _validate_weight(target):
        try:
            target.unlink()
        except Exception:
            pass

    base = target.name

    # 1) Direct download (.pt only)
    if base.lower().endswith(".pt") and download_url is not None:
        try:
            ctx: ContextManager[Any] = simple_status("download", enabled=(os.environ.get("PANOPTES_PROGRESS_ACTIVE") != "1"))  # type: ignore[misc]
            with ctx:
                adapter = (
                    _DownloadSpinnerAdapter(spinner, items_total=items_total, items_done=items_done, label=base)
                    if spinner
                    else None
                )
                download_url(_asset_url_for(base), str(target), adapter)  # type: ignore[arg-type]
            if _validate_weight(target):
                return (target.name, "download")
            else:
                target.unlink(missing_ok=True)
        except Exception:
            pass

    # 1a) Direct download (.onnx) — avoid local export unless strictly required
    if base.lower().endswith(".onnx") and download_url is not None:
        try:
            ctx2: ContextManager[Any] = simple_status("download", enabled=(os.environ.get("PANOPTES_PROGRESS_ACTIVE") != "1"))  # type: ignore[misc]
            with ctx2:
                adapter2 = (
                    _DownloadSpinnerAdapter(spinner, items_total=items_total, items_done=items_done, label=base)
                    if spinner
                    else None
                )
                download_url(_asset_url_for(base), str(target), adapter2)  # type: ignore[arg-type]
            if _validate_weight(target):
                return (target.name, "download")
            else:
                target.unlink(missing_ok=True)
        except Exception:
            # If the asset isn't hosted, we’ll fall back to export below.
            pass

    # 2) YOLO last-resort fetcher for .pt (silenced)
    has_yolo_local = has_yolo and (YOLO is not None)
    if base.lower().endswith(".pt") and has_yolo_local:
        try:
            if spinner is not None:
                spinner.update(current=base, job="ultralytics", model=base)
            with _cd(dst), _tqdm_disabled_env(), _silence_stdio():
                m = YOLO(base)  # type: ignore
                p = Path(getattr(m, "ckpt_path", base)).expanduser()
                if not p.exists():
                    p = Path(base).expanduser()
                if p.exists():
                    if p.resolve() == target.resolve():
                        pass
                    else:
                        shutil.copy2(p, target)
        except Exception:
            pass
        if _validate_weight(target):
            return (target.name, "download" if (dst / base).exists() else "copied")

    # 3) If ONNX requested (and direct download failed), export from the matching .pt
    if base.lower().endswith(".onnx"):
        pt_name = Path(base).with_suffix(".pt").name
        pt_path = dst / pt_name

        # Ensure the .pt exists (prefer direct download; fallback to YOLO)
        if not pt_path.exists() or not _validate_weight(pt_path):
            if download_url is not None:
                try:
                    ctx2: ContextManager[Any] = simple_status("download", enabled=(os.environ.get("PANOPTES_PROGRESS_ACTIVE") != "1"))  # type: ignore[misc]
                    with ctx2:
                        adapter2 = (
                            _DownloadSpinnerAdapter(spinner, items_total=items_total, items_done=items_done, label=pt_name)
                            if spinner
                            else None
                        )
                        download_url(_asset_url_for(pt_name), str(pt_path), adapter2)  # type: ignore[arg-type]
                except Exception:
                    pass

            if (not pt_path.exists() or not _validate_weight(pt_path)) and has_yolo_local:
                try:
                    if spinner is not None:
                        spinner.update(current=pt_name, job="ultralytics", model=pt_name)
                    with _cd(dst), _tqdm_disabled_env(), _silence_stdio():
                        m_pt = YOLO(pt_name)  # type: ignore
                        p_pt = Path(getattr(m_pt, "ckpt_path", pt_name)).expanduser()
                        if not p_pt.exists():
                            p_pt = Path(pt_name).expanduser()
                        if p_pt.exists() and p_pt.resolve() != pt_path.resolve():
                            shutil.copy2(p_pt, pt_path)
                except Exception:
                    pass

        # Export ONNX (only after a single preflight & auto-repair)
        if has_yolo_local and pt_path.exists() and _validate_weight(pt_path):
            # Run preflight+repair once (won't spam)
            usable = _preflight_and_repair_onnx_once()
            if not usable:
                # Hard fail with clear reason; avoid repeated spam
                return (target.name, "failed")

            try:
                # Determine preferred opset from torch if available
                try:
                    import torch  # type: ignore
                    preferred = _decide_opset(getattr(torch, "__version__", "0.0.0"))
                except Exception:
                    preferred = 12

                with _cd(dst), _tqdm_disabled_env(), _silence_stdio():
                    exported = False
                    for ops in _candidate_opsets(preferred):
                        for (dyn, simp) in ((True, True), (True, False), (False, False)):
                            try:
                                YOLO(str(pt_path)).export(  # type: ignore
                                    format="onnx",
                                    dynamic=dyn,
                                    simplify=simp,
                                    imgsz=640,
                                    opset=ops,
                                    device="cpu",
                                )
                                # success path (a)
                                if target.exists() and _validate_weight(target):
                                    exported = True
                                    break
                                # success path (b): look under runs/
                                cand = _latest_exported_onnx()
                                if cand and cand.exists():
                                    shutil.copy2(cand, target)
                                if target.exists() and _validate_weight(target):
                                    exported = True
                                    break
                            except Exception:
                                # Try next combination
                                continue
                        if exported:
                            break

                # Validate and tidy up
                if target.exists() and _validate_weight(target):
                    try:
                        shutil.rmtree(dst / "runs", ignore_errors=True)
                    except Exception:
                        pass
                    return (target.name, "exported")
                else:
                    try:
                        shutil.rmtree(dst / "runs", ignore_errors=True)
                    except Exception:
                        pass

            except Exception:
                pass

    # If we reach here, we failed to produce the artifact
    return (target.name, "failed")


def _fetch_all(names: List[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Fetch all *names* into MODEL_DIR with an aggregated progress view.
    """
    results: List[Tuple[str, str]] = []
    failed: List[str] = []

    # If any .onnx present, run preflight+repair once up front (single summary line)
    if any(n.lower().endswith(".onnx") for n in names):
        _preflight_and_repair_onnx_once()

    with percent_spinner(prefix="FETCH MODELS") as sp:
        total = len(names)
        sp.update(total=max(1, total), count=0)
        done = 0
        for nm in names:
            try:
                sp.update(current=nm, job="start", model=nm)
            except Exception:
                pass

            base, action = _fetch_one(nm, MODEL_DIR, spinner=sp, items_total=total, items_done=done)
            results.append((base, action))
            if action == "failed":
                failed.append(base)

            done += 1
            try:
                sp.update(count=done, job="", model=base)
            except Exception:
                pass

    return results, failed


def _write_manifest(selected: List[str], installed: List[str]) -> Path:
    """
    Write a user-agnostic manifest:
      • model_dir is stored *relative* to projects/argos to avoid absolute user paths
      • paths use forward slashes for cross-platform consistency
    """
    repo_root = Path(__file__).resolve().parents[2]  # .../projects/argos
    rel_model_dir = os.path.relpath(MODEL_DIR.resolve(), repo_root.resolve()).replace("\\", "/")

    data: Dict[str, object] = {
        "model_dir": rel_model_dir,
        "selected": _dedupe(selected),
        "installed": _dedupe(installed),
    }
    man = MODEL_DIR / "manifest.json"
    man.write_text(json.dumps(data, indent=2))
    return man


# ---------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------
def _show_row(title: str, items: Iterable[str]) -> None:
    typer.secho(f"\n{title}", bold=True)
    typer.echo("  " + "  |  ".join(f"[{i+1}] {v}" for i, v in enumerate(items)))


def _parse_multi(raw: str, items: Tuple[str, ...], *, aliases: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Parse comma/space-separated numbers or names.
    Supports 'all'/'*' and simple aliases (e.g., 'v8' -> '8', 'nano' -> 'n').
    """
    if not raw:
        return []
    raw = raw.strip().lower()
    if raw in {"all", "*"}:
        return list(items)

    out: List[str] = []
    seen: Set[str] = set()
    tokens = [t for t in (p.strip() for p in raw.replace("/", ",").replace(";", ",").split(",")) if t]
    for t in tokens:
        if t.isdigit():
            i = int(t) - 1
            if 0 <= i < len(items):
                key = items[i]
                if key not in seen:
                    seen.add(key)
                    out.append(key)
            continue
        if aliases and t in aliases:
            t = aliases[t]
        if t.startswith("v") and t[1:] in items:
            t = t[1:]
        if t.startswith("yolov") and t[5:] in items:
            t = t[5:]
        if t in items and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _build_combo(
    vers: Iterable[str],
    sizes: Iterable[str],
    tasks: Iterable[str],
    *,
    formats: Iterable[str],
) -> List[str]:
    names: List[str] = []
    for ver in vers:
        for sz in sizes:
            for task in tasks:
                for ext in formats:
                    names.append(_mk(ver, sz, task, ext))
    return _dedupe(names)


# ---------------------------------------------------------------------
# Menus
# ---------------------------------------------------------------------
def _ensure_env_hint() -> None:
    typer.echo()
    typer.secho("If this is a fresh clone, the launcher ensured the Python env.", fg="yellow")
    typer.secho("You do not need to activate a venv manually.", fg="yellow")
    typer.echo()


def _menu() -> int:
    """Loop until user selects a valid option (0-4)."""
    while True:
        typer.echo(
            textwrap.dedent(
                """
                ~ Argos ~ The Many Eyed Sentinel
                An intelligent visual detection system.

                What would you like to install?
                1) Default Argos pack
                2) Full pack (ALL versions/sizes; ALL tasks; .pt + .onnx)
                3) Size pack (choose one version/size/tasks/formats)
                4) Custom builder (multi-select; preview; extras)
                0) Exit
                """
            ).strip()
        )
        pick = typer.prompt("Pick [0-4, ? for help]", default="1 or ?").strip().lower()
        if pick in {"?", "h", "help"}:
            typer.echo("Typing numbers or names are fine in the builder.")
            typer.echo("YOLO models are named by version (v8, v11, v12), size (n/s/m/l/x), task (det/seg/pose/cls/obb), and format (.pt/.onnx).")
            typer.echo("Choosing the default pack (1) is a good starting point for most users.")
            continue
        try:
            choice = int(pick)
            if choice in (0, 1, 2, 3, 4):
                return choice
        except ValueError:
            pass
        typer.secho("Invalid choice. Try 0-4 or '?' for help.", fg="yellow")


def _ask_size_pack() -> List[str]:
    ver = typer.prompt(f"Version {VERSIONS}", default="8").strip()
    while ver not in VERSIONS:
        ver = typer.prompt(f"Version {VERSIONS}", default="8").strip()

    sz = typer.prompt(f"Size {SIZES}", default="x").strip().lower()
    while sz not in SIZES:
        sz = typer.prompt(f"Size {SIZES}", default="x").strip().lower()

    # Tasks
    task_alias = {"detect": "det", "segmentation": "seg", "classification": "cls"}
    _show_row("Tasks:", TASKS)
    raw_tasks = typer.prompt("Pick tasks (e.g. 'det,seg,pose' or 'all')", default="det,seg").strip()
    chosen_tasks = _parse_multi(raw_tasks, TASKS, aliases=task_alias)
    if not chosen_tasks:
        chosen_tasks = ["det"]

    fmt = typer.prompt("Formats: .pt / .onnx / both", default="both").strip().lower()
    if fmt not in {"pt", "onnx", "both"}:
        fmt = "both"
    exts = (".pt", ".onnx") if fmt == "both" else (f".{fmt}",)

    names = _build_combo([ver], [sz], chosen_tasks, formats=exts)
    return _dedupe(names)


def _ask_custom() -> List[str]:
    """
    Guided, multi-select custom builder.
    """
    # Versions
    _show_row("Versions:", VERSIONS)
    ver_alias: Dict[str, str] = {"v8": "8", "v11": "11", "v12": "12"}
    vers: List[str] = []
    while not vers:
        raw = typer.prompt("Pick versions (e.g. '1', '2', '3' / 'all')", default="all")
        vers = _parse_multi(raw, VERSIONS, aliases=ver_alias)
        if not vers:
            typer.secho("Pick at least one version (try typing '1' '2' '3' or 'all').", fg="yellow")

    # Sizes
    _show_row("Model sizes:", SIZES)
    size_alias: Dict[str, str] = {"nano": "n", "small": "s", "medium": "m", "large": "l", "xlarge": "x"}
    sizes: List[str] = []
    while not sizes:
        raw = typer.prompt("Pick sizes (e.g. 'n,s' or '5' or 'all')", default="x,n")
        sizes = _parse_multi(raw, SIZES, aliases=size_alias)
        if not sizes:
            typer.secho("Pick at least one size (e.g., 'n' or 'all').", fg="yellow")

    # Tasks
    _show_row("Tasks:", TASKS)
    task_alias = {"detect": "det", "segmentation": "seg", "classification": "cls"}
    tasks: List[str] = []
    while not tasks:
        raw = typer.prompt("Pick tasks (e.g. 'det,seg,pose,cls,obb' or 'all')", default="all")
        tasks = _parse_multi(raw, TASKS, aliases=task_alias)
        if not tasks:
            typer.secho("Pick at least one task (e.g., 'det').", fg="yellow")

    # Formats
    _show_row("Formats:", EXTS)
    fmt_raw = typer.prompt("Formats (.pt / .onnx / both)", default="both").strip().lower()
    if fmt_raw not in {"pt", "onnx", "both"}:
        fmt_raw = "both"
    formats: Tuple[str, ...] = (".pt", ".onnx") if fmt_raw == "both" else (f".{fmt_raw}",)

    # Build + optional extras
    names = _build_combo(vers, sizes, tasks, formats=formats)

    typer.secho("\nPreview (will be fetched):", bold=True)
    for n in names:
        typer.echo(f"  • {n}")

    extra = typer.prompt(
        "Optional: add exact extra names (comma-sep) or press Enter",
        default="",
    ).strip()
    if extra:
        names += _dedupe(n for n in (p.strip() for p in extra.split(",")) if n)

    return _dedupe(names)


# ---------------------------------------------------------------------
# Quick smoke (optional)
# ---------------------------------------------------------------------
def _quick_check() -> None:
    return


# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------
@app.command()
def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Show environment hint on first run (no manifest yet)
    if not (MODEL_DIR / "manifest.json").exists():
        _ensure_env_hint()

    choice = _menu()
    if choice == 0:
        raise typer.Exit(0)
    if choice == 1:
        selected = _dedupe(DEFAULT_PACK)
    elif choice == 2:
        selected = _full_pack()
    elif choice == 3:
        selected = _ask_size_pack()
    elif choice == 4:
        selected = _ask_custom()
    else:
        _err("Invalid choice.")
        raise typer.Exit(2)

    if not selected:
        _warn("Nothing selected, exiting.")
        raise typer.Exit(0)

    typer.echo()
    typer.secho("Selected:", bold=True)
    for nm in selected:
        typer.echo(f"  • {nm}")
    typer.echo()

    if not typer.confirm("Proceed to download/install into panoptes/model/?", default=True):
        raise typer.Exit(1)

    results, bad = _fetch_all(selected)

    # Per-file, clear log
    action_icons = {
        "present": "↺",
        "download": "↓",
        "copied": "⇢",
        "exported": "⎘",
        "failed": "✗",
    }
    for name, action in results:
        icon = action_icons.get(action, "•")
        if action == "failed":
            _warn(f"{icon} {name}  (failed)")
        elif action == "present":
            typer.echo(f"{icon} {name}  (already present)")
        elif action == "download":
            _ok(f"{icon} {name}  (downloaded)")
        elif action == "copied":
            _ok(f"{icon} {name}  (downloaded → copied into model dir)")
        elif action == "exported":
            _ok(f"{icon} {name}  (exported from matching .pt)")
        else:
            typer.echo(f"{icon} {name}  ({action})")

    installed = [n for (n, a) in results if a != "failed"]

    # Summary + clickable list
    typer.echo()
    _ok(f"Installed/ready: {len(installed)}")
    if installed:
        typer.secho("Ready (Ctl + Click To Open):", bold=True)
        for bn in _dedupe(installed):
            typer.echo(f"  • {osc8_link(bn, MODEL_DIR / bn)}")

    if bad:
        _warn(f"Skipped/failed: {len(bad)}")
        for n in _dedupe(bad):
            typer.echo(f"  - {n}")
        if any(b.endswith(".onnx") for b in bad):
            _warn(f"ONNX items failed. Diagnostics: {osc8_link(_DIAG_LOG_NAME, _diag_log_path())}")
        else:
            _warn("Those names may not be hosted yet, or export failed.")

    # Manifest for reproducibility (user-agnostic, relative paths)
    man = _write_manifest(selected, installed)
    typer.echo(f"\nManifest: {osc8_link('manifest.json', man)}")

    _quick_check()


if __name__ == "__main__":
    main()
