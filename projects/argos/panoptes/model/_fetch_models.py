# \rAIn\projects\argos\panoptes\model\_fetch_models.py
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    cast,
)

import typer

# Optional progress (safe fallback if not importable or in CI/non-TTY)
try:
    import panoptes.progress as _prog  # type: ignore[reportMissingTypeStubs]
except Exception:
    _prog = None  # type: ignore[assignment]

ProgressEngine: Optional[Type[Any]]
live_percent: Optional[Callable[..., ContextManager[Any]]]
simple_status: Optional[Callable[..., ContextManager[Any]]]

if _prog is not None:
    ProgressEngine = cast(Optional[Type[Any]], getattr(_prog, "ProgressEngine", None))
    live_percent = cast(Optional[Callable[..., ContextManager[Any]]], getattr(_prog, "live_percent", None))
    simple_status = cast(Optional[Callable[..., ContextManager[Any]]], getattr(_prog, "simple_status", None))
else:
    ProgressEngine = None
    live_percent = None
    simple_status = None

# Our byte-accurate downloader (shows progress via ProgressEngine if provided)
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

# Ultralytics (used to export ONNX and as last-resort fetcher)
has_yolo: bool
try:
    from ultralytics import YOLO as _YOLO  # type: ignore[reportMissingTypeStubs]
    try:
        # Keep Ultralytics quiet; we print our own succinct logs.
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

YOLO = _YOLO

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

# Where the official assets live (override with ULTRA_ASSETS_BASE if needed)
ASSETS_BASE = os.getenv("ULTRA_ASSETS_BASE", "https://github.com/ultralytics/assets/releases/download/v8.3.0").rstrip("/")

# ---------------------------------------------------------------------
# Packs / naming
# ---------------------------------------------------------------------
VERSIONS: Tuple[str, ...] = ("8", "11", "12")
SIZES: Tuple[str, ...] = ("x", "l", "m", "s", "n")
EXTS: Tuple[str, ...] = (".pt", ".onnx")
TASKS: Tuple[str, ...] = ("det", "seg", "pose", "cls", "obb")

# A sensible curated default (small but covers multiple tasks)
DEFAULT_PACK: List[str] = [
    # DETECT
    "yolov8x.pt",
    "yolo11x.pt",
    "yolo12x.onnx",

    # SEG
    "yolo11x-seg.pt",
    "yolo11m-seg.pt",
    "yolov8n-seg.pt",

    # POSE
    "yolo11s-pose.pt",
    "yolov8n-pose.pt",

    # CLS
    "yolo11s-cls.pt",
    "yolov8n-cls.pt",

    # OBB
    "yolo11s-obb.pt",
    "yolov8n-obb.pt",

    # LIGHT/DEV extras
    "yolo12m.onnx",
    "yolo12n.onnx",
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


def _osc8(label: str, path: Path, *, yellow: bool = True) -> str:
    """
    Return a clickable label.
    • In capable terminals: OSC-8 hyperlink with bright-yellow bold label.
    • In VS Code terminal: plain absolute path (no OSC-8) so Ctrl+Click opens.
    """
    # VS Code integrated terminal doesn’t reliably honor OSC-8 for file:// on custom labels.
    is_vscode = (os.environ.get("TERM_PROGRAM", "").lower() == "vscode") or bool(os.environ.get("VSCODE_PID"))
    if is_vscode:
        return str(path.resolve())  # plain absolute path → VS Code opens it

    try:
        uri = path.resolve().as_uri()
    except Exception:
        uri = "file:///" + str(path.resolve()).replace("\\", "/")

    ESC   = "\x1b"
    BR_YE = "\x1b[93m"   # bright yellow
    BOLD  = "\x1b[1m"
    RST   = "\x1b[0m"

    text = f"{BR_YE}{BOLD}{label}{RST}" if yellow else label
    return f"{ESC}]8;;{uri}{ESC}\\{text}{ESC}]8;;{ESC}\\"



def _latest_exported_onnx() -> Optional[Path]:
    hits = [Path(p) for p in glob.glob("runs/**/**/*.onnx", recursive=True)]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


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


class ProgressLike(Protocol):
    def set_total(self, total_units: float) -> None: ...
    def set_current(self, label: str) -> None: ...
    def add(self, units: float, *, current_item: str | None = None) -> None: ...


class _ScaledProgress:
    """
    Wrap a parent ProgressEngine so a byte download maps to +1.0 for this item.
    We ignore set_total() from the child and compute our own scale.
    """
    def __init__(self, parent: ProgressLike | None) -> None:
        self.parent = parent
        self.total_bytes: Optional[float] = None
        self.scale: float = 0.0
        self.added: float = 0.0

    def set_total(self, total_units: float) -> None:
        try:
            self.total_bytes = float(total_units)
            self.scale = (1.0 / self.total_bytes) if self.total_bytes and self.total_bytes > 0 else 0.0
        except Exception:
            self.total_bytes, self.scale = None, 0.0

    def set_current(self, label: str) -> None:
        if self.parent is not None:
            try:
                self.parent.set_current(label)
            except Exception:
                pass

    def add(self, units: float, *, current_item: str | None = None) -> None:
        if self.parent is None:
            return
        if self.scale > 0.0:
            delta = float(units) * self.scale
            if delta > 0:
                try:
                    self.parent.add(delta)
                    self.added += delta
                except Exception:
                    pass

    # Top-up to exactly +1.0 for this item
    def finalize(self) -> None:
        if self.parent is None:
            return
        try:
            if self.added < 1.0:
                self.parent.add(1.0 - self.added)
                self.added = 1.0
        except Exception:
            pass


# ---------------------------------------------------------------------
# Fetch logic
# ---------------------------------------------------------------------
def _fetch_one(name: str, dst: Path, engine: Optional[ProgressLike] = None) -> Tuple[str, str]:
    """
    Obtain *name* into *dst* and return (basename, action).

    We prefer our own downloader (byte-accurate progress) and only fall
    back to Ultralytics when needed (and silence its tqdm bars).
    Possible actions:
      • "present"   – already existed in dst
      • "download"  – fetched via direct URL (Argos progress)
      • "copied"    – YOLO fetched elsewhere; we copied into dst
      • "exported"  – exported ONNX from a matching .pt
      • "failed"    – nothing worked
    """
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / Path(name).name

    # Already present (and sane)
    if target.exists() and _validate_weight(target):
        if engine is not None:
            try:
                engine.add(1.0)
            except Exception:
                pass
        return (target.name, "present")
    elif target.exists() and not _validate_weight(target):
        try:
            target.unlink()
        except Exception:
            pass

    # 1) Direct download (.pt only) with byte progress
    base = target.name
    if base.lower().endswith(".pt") and download_url is not None:
        scaler = _ScaledProgress(engine)
        try:
            ctx: ContextManager[Any] = (
                simple_status("download", enabled=(os.environ.get("PANOPTES_PROGRESS_ACTIVE") != "1"))  # type: ignore[misc]
                if simple_status is not None else cast(ContextManager[Any], nullcontext())
            )
            with ctx:
                download_url(_asset_url_for(base), str(target), scaler)
            scaler.finalize()
            if _validate_weight(target):
                return (target.name, "download")
            else:
                target.unlink(missing_ok=True)
        except Exception:
            pass

    # 2) YOLO as a last-resort fetcher for .pt (silenced)
    if base.lower().endswith(".pt") and has_yolo and YOLO is not None:
        try:
            with _cd(dst), _tqdm_disabled_env(), _silence_stdio():
                m = YOLO(base)  # type: ignore
                p = Path(getattr(m, "ckpt_path", base)).expanduser()
                if not p.exists():
                    p = Path(base).expanduser()
                if p.exists():
                    if p.resolve() == target.resolve():
                        # downloaded straight to target
                        pass
                    else:
                        shutil.copy2(p, target)
        except Exception:
            pass
        if _validate_weight(target):
            if engine is not None:
                try:
                    engine.add(1.0)
                except Exception:
                    pass
            # If YOLO wrote into cwd directly, treat as "download"
            return (target.name, "download" if (dst / base).exists() else "copied")

    # 3) If ONNX requested, export from the matching .pt (ensure it exists)
    if base.lower().endswith(".onnx"):
        pt_name = Path(base).with_suffix(".pt").name
        pt_path = dst / pt_name

        # Ensure the .pt exists (prefer our downloader)
        if not pt_path.exists() or not _validate_weight(pt_path):
            # try direct
            if download_url is not None:
                scaler = _ScaledProgress(engine)
                try:
                    ctx: ContextManager[Any] = (
                        simple_status("download", enabled=(os.environ.get("PANOPTES_PROGRESS_ACTIVE") != "1"))  # type: ignore[misc]
                        if simple_status is not None else cast(ContextManager[Any], nullcontext())
                    )
                    with ctx:
                        download_url(_asset_url_for(pt_name), str(pt_path), scaler)
                    scaler.finalize()
                except Exception:
                    pass

            # YOLO fallback for fetching the .pt
            if (not pt_path.exists() or not _validate_weight(pt_path)) and has_yolo and YOLO is not None:
                try:
                    with _cd(dst), _tqdm_disabled_env(), _silence_stdio():
                        m_pt = YOLO(pt_name)  # type: ignore
                        p_pt = Path(getattr(m_pt, "ckpt_path", pt_name)).expanduser()
                        if not p_pt.exists():
                            p_pt = Path(pt_name).expanduser()
                        if p_pt.exists() and p_pt.resolve() != pt_path.resolve():
                            shutil.copy2(p_pt, pt_path)
                except Exception:
                    pass

        # Export ONNX
        if has_yolo and YOLO is not None and pt_path.exists() and _validate_weight(pt_path):
            try:
                with _cd(dst), _tqdm_disabled_env(), _silence_stdio():
                    try:
                        YOLO(str(pt_path)).export(format="onnx", dynamic=True, simplify=True, imgsz=640, opset=12, device="cpu")  # type: ignore
                    except Exception:
                        try:
                            YOLO(str(pt_path)).export(format="onnx", dynamic=False, simplify=True, imgsz=640, opset=12, device="cpu")  # type: ignore
                        except Exception:
                            YOLO(str(pt_path)).export(format="onnx", simplify=False, imgsz=640, opset=12, device="cpu")  # type: ignore

                # success path (a)
                if target.exists() and _validate_weight(target):
                    if engine is not None:
                        try:
                            engine.add(1.0)
                        except Exception:
                            pass
                    return (target.name, "exported")

                # success path (b): look under runs/
                cand = _latest_exported_onnx()
                if cand and cand.exists():
                    try:
                        shutil.copy2(cand, target)
                    finally:
                        try:
                            shutil.rmtree(dst / "runs", ignore_errors=True)
                        except Exception:
                            pass
                if target.exists() and _validate_weight(target):
                    if engine is not None:
                        try:
                            engine.add(1.0)
                        except Exception:
                            pass
                    return (target.name, "exported")
            except Exception:
                pass

    # If we reach here, we failed to produce the artifact
    if engine is not None:
        try:
            engine.add(1.0)
        except Exception:
            pass
    return (target.name, "failed")


def _fetch_all(names: List[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Fetch all *names* into MODEL_DIR with an aggregated progress view.

    We set the global total to len(names) and let each item contribute +1.0.
    For downloads, bytes are scaled into that +1.0; for present/export/copy we
    add +1.0 when done.
    """
    results: List[Tuple[str, str]] = []
    failed: List[str] = []

    # Fast path if progress infra is unavailable
    if (ProgressEngine is None) or (live_percent is None):
        for nm in names:
            base, action = _fetch_one(nm, MODEL_DIR, None)
            results.append((base, action))
            if action == "failed":
                failed.append(base)
        return results, failed

    # Narrow types for engine and context manager
    assert ProgressEngine is not None
    assert live_percent is not None

    eng_any = ProgressEngine()
    eng = cast(ProgressLike, eng_any)
    with live_percent(eng, prefix="WEIGHTS"):
        eng.set_total(float(len(names)))
        for nm in names:
            try:
                eng.set_current(nm)
            except Exception:
                pass
            base, action = _fetch_one(nm, MODEL_DIR, eng)
            results.append((base, action))
            if action == "failed":
                failed.append(base)
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
    """Loop until user selects a valid option (0–4)."""
    while True:
        typer.echo(
            textwrap.dedent(
                """
                What would you like to install?

                1) Default Argos pack
                2) Full pack (ALL versions/sizes; ALL tasks; .pt + .onnx)
                3) Size pack (choose 1 version/size/tasks/formats)
                4) Custom builder (multi-select; preview; extras)
                0) Exit
                """
            ).strip()
        )
        pick = typer.prompt("Pick [0–4, ? for help]", default="1").strip().lower()
        if pick in {"?", "h", "help"}:
            typer.echo("Tips: numbers or names are fine in the builder; 'all' works at every step.")
            continue
        try:
            choice = int(pick)
            if choice in (0, 1, 2, 3, 4):
                return choice
        except ValueError:
            pass
        typer.secho("Invalid choice. Try 0–4 or '?' for help.", fg="yellow")


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
# Quick smoke (progress-enabled)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Quick smoke (progress-enabled)  — ALWAYS produces fresh outputs
# ---------------------------------------------------------------------
def _quick_check() -> None:
    if not typer.confirm("Run a quick smoke check now?", default=False):
        return

    # Accept full names and short aliases
    raw_task = typer.prompt(
        "Task [detect|heatmap|geojson] [d|hm|gj]", default="heatmap"
    ).strip().lower()
    task_alias = {
        "d": "detect", "det": "detect", "detect": "detect",
        "hm": "heatmap", "heatmap": "heatmap",
        "gj": "geojson", "geojson": "geojson",
    }
    task = task_alias.get(raw_task, "heatmap")

    inp = typer.prompt(
        "Input (Files in tests/raw; enter a file name or 'all' to process the whole folder)",
        default="all",
    ).strip()

    py = sys.executable
    args: List[str] = ["-m", "panoptes.cli", ("all" if inp.lower() == "all" else inp), "--task", task]

    repo_root = Path(__file__).resolve().parents[2]
    raw_dir = repo_root / "tests" / "raw"
    results_dir = repo_root / "tests" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Always rotate/clear previous results so we detect new files every run.
    # Set PANOPTES_SMOKE_KEEP_PREV=1 to preserve the folder as-is.
    keep_prev = (os.environ.get("PANOPTES_SMOKE_KEEP_PREV", "").lower() in {"1", "true", "yes"})
    if any(results_dir.iterdir()) and not keep_prev:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = results_dir.parent / f"results.prev-{stamp}"
        try:
            backup.mkdir(parents=True, exist_ok=True)
            for p in list(results_dir.iterdir()):
                try:
                    shutil.move(str(p), str(backup / p.name))
                except Exception:
                    # fallback: try to remove in place
                    try:
                        if p.is_file():
                            p.unlink()
                        else:
                            shutil.rmtree(p, ignore_errors=True)
                    except Exception:
                        pass
        except Exception:
            # last resort: try to clear in place
            for p in list(results_dir.iterdir()):
                try:
                    if p.is_file():
                        p.unlink()
                    else:
                        shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass

    # Expected items = number of raw inputs (bounded to ≥1)
    # Match CLI’s notion of supported inputs for a better initial estimate
    try:
        from panoptes import cli as _cli  # type: ignore
        _img: Set[str] = set(cast(Iterable[str], getattr(_cli, "_IMAGE_EXTS", ())))
        _vid: Set[str] = set(cast(Iterable[str], getattr(_cli, "_VIDEO_EXTS", ())))
        exts: Set[str] = _img | _vid
    except Exception:
        exts: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif",
                        ".mp4", ".mov", ".avi", ".mkv", ".webm"}
    expected = 1
    if inp.lower() == "all":
        expected = sum(1 for p in raw_dir.glob("*") if p.suffix.lower() in exts)
    expected = max(1, expected)

    _ok(f"→ running: {py} {' '.join(args)}")

    def _list_created() -> List[Path]:
        # Flat results dir; include files only
        return sorted([p for p in results_dir.glob("*") if p.is_file()],
                    key=lambda p: p.name.lower())

    # No progress infra? Just run it (still using cwd=repo_root) and print results.
    if (ProgressEngine is None) or (live_percent is None):
        try:
            subprocess.check_call([py, *args], cwd=repo_root)
            created = _list_created()
            _ok("Smoke check finished.")
            if created:
                typer.secho("Results (Ctl + Click To Open):", bold=True)
                for p in created:
                    typer.echo(f"  • {_osc8(p.name, p)}")
            else:
                _warn("CLI ran but produced no files.")
        except Exception as e:
            _warn(f"Smoke check failed: {e!s}")
        return

    # Progress-enabled path: bound progress to expected so it never exceeds 100%.
    eng_any = ProgressEngine()  # type: ignore[call-arg]
    eng = cast(ProgressLike, eng_any)
    with live_percent(eng, prefix=f"SMOKE {task.upper()}"):  # type: ignore[misc]
        try:
            eng.set_total(float(expected))
        except Exception:
            pass
        try:
            eng.set_current("starting")
        except Exception:
            pass

        # Run CLI quiet, but from repo root so paths resolve consistently
        proc = subprocess.Popen([py, *args],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                cwd=repo_root)

        done_reported = 0
        last_poll = 0.0

        try:
            while True:
                rc = proc.poll()
                now_t = time.monotonic()
                if now_t - last_poll >= 0.25:
                    last_poll = now_t
                    created = _list_created()
                    n = len(created)

                    # If we produced more than initially expected, grow the total.
                    if n > expected:
                        try:
                            eng.set_total(float(n))
                        except Exception:
                            pass

                    delta = n - done_reported
                    if delta > 0:
                        try:
                            eng.add(float(delta))
                        except Exception:
                            pass
                        done_reported = n

                    if created:
                        try:
                            eng.set_current(created[-1].name)
                        except Exception:
                            pass

                if rc is not None:
                    break
                time.sleep(0.05)

            # Final top-up (also grow total if needed)
            created = _list_created()
            n = len(created)
            if n > expected:
                try:
                    eng.set_total(float(n))
                except Exception:
                    pass
            delta = n - done_reported
            if delta > 0:
                try:
                    eng.add(float(delta))
                except Exception:
                    pass
                done_reported = n

            if proc.returncode == 0:
                _ok("Smoke check finished.")
                if created:
                    typer.secho("Results (Ctl + Click To Open):", bold=True)
                    for p in created:
                        typer.echo(f"  • {_osc8(p.name, p)}")
                else:
                    _warn("CLI ran but produced no files.")
            else:
                _warn(f"Smoke check failed (exit code {proc.returncode}).")
        finally:
            try:
                proc.terminate()
            except Exception:
                pass


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
            typer.echo(f"  • {_osc8(bn, MODEL_DIR / bn)}")

    if bad:
        _warn(f"Skipped/failed: {len(bad)}")
        for n in _dedupe(bad):
            typer.echo(f"  – {n}")
        _warn("Those names may not be hosted yet, or export failed.")

    # Manifest for reproducibility (user-agnostic, relative paths)
    man = _write_manifest(selected, installed)
    typer.echo(f"\nManifest: {_osc8('manifest.json', man)}")

    _quick_check()


if __name__ == "__main__":
    main()

