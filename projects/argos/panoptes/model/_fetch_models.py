from __future__ import annotations

import glob
import json
import os
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
# Halo-based progress (Panoptes) with safe fallbacks
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
    from panoptes.progress import (  # type: ignore[reportMissingTypeStubs]
        percent_spinner as _percent_spinner,
        simple_status as _simple_status,
        osc8 as _osc8_raw,
    )
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

def _mk(ver: str, size: str, task: str, ext: str) -> str:
    base = f"yolov{ver}{size}" if ver == "8" else f"yolo{ver}{size}"
    suf = {"det": "", "seg": "-seg", "pose": "-pose", "cls": "-cls", "obb": "-obb"}[task]
    return base + suf + ext

def _full_pack() -> List[str]:
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

@contextmanager
def _cd(path: Path) -> Iterator[None]:
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
    try:
        if not p.exists():
            return False
        if p.stat().st_size < 1_000_000:
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

class _DownloadSpinnerAdapter:
    def __init__(self, spinner: _ProgressLike, *, items_total: int, items_done: int, label: Optional[str] = None) -> None:
        self.spinner = spinner
        self.items_total = int(max(1, items_total))
        self.items_done = int(max(0, items_done))
        self.label = label or ""
        self.total_bytes: float = 0.0
        self._last_pct: int = -1

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
                pct = int(round(100.0 * done / self.total_bytes))
                if pct != self._last_pct:
                    self._last_pct = pct
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

def _fetch_one(name: str, dst: Path, *, spinner: Optional[_ProgressLike], items_total: int, items_done: int) -> Tuple[str, str]:
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / Path(name).name

    if spinner is not None:
        try:
            spinner.update(total=items_total, count=items_done, current=target.name, job="check", model=target.name)
        except Exception:
            pass

    if target.exists() and _validate_weight(target):
        return (target.name, "present")
    elif target.exists() and not _validate_weight(target):
        try:
            target.unlink()
        except Exception:
            pass

    base = target.name
    if base.lower().endswith(".pt") and download_url is not None:
        try:
            ctx: ContextManager[Any] = simple_status("download", enabled=(os.environ.get("PANOPTES_PROGRESS_ACTIVE") != "1"))  # type: ignore[misc]
            with ctx:
                adapter = _DownloadSpinnerAdapter(spinner, items_total=items_total, items_done=items_done, label=base) if spinner else None
                download_url(_asset_url_for(base), str(target), adapter)  # type: ignore[arg-type]
            if _validate_weight(target):
                return (target.name, "download")
            else:
                target.unlink(missing_ok=True)
        except Exception:
            pass

    if base.lower().endswith(".pt") and has_yolo and YOLO is not None:
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

    if base.lower().endswith(".onnx"):
        pt_name = Path(base).with_suffix(".pt").name
        pt_path = dst / pt_name

        if not pt_path.exists() or not _validate_weight(pt_path):
            if download_url is not None:
                try:
                    ctx2: ContextManager[Any] = simple_status("download", enabled=(os.environ.get("PANOPTES_PROGRESS_ACTIVE") != "1"))  # type: ignore[misc]
                    with ctx2:
                        adapter2 = _DownloadSpinnerAdapter(spinner, items_total=items_total, items_done=items_done, label=pt_name) if spinner else None
                        download_url(_asset_url_for(pt_name), str(pt_path), adapter2)  # type: ignore[arg-type]
                except Exception:
                    pass
            if (not pt_path.exists() or not _validate_weight(pt_path)) and has_yolo and YOLO is not None:
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

        if has_yolo and YOLO is not None and pt_path.exists() and _validate_weight(pt_path):
            try:
                if spinner is not None:
                    spinner.update(current=base, job="export onnx", model=pt_name)
                with _cd(dst), _tqdm_disabled_env(), _silence_stdio():
                    try:
                        YOLO(str(pt_path)).export(format="onnx", dynamic=True, simplify=True, imgsz=640, opset=12, device="cpu")  # type: ignore
                    except Exception:
                        try:
                            YOLO(str(pt_path)).export(format="onnx", dynamic=False, simplify=True, imgsz=640, opset=12, device="cpu")  # type: ignore
                        except Exception:
                            YOLO(str(pt_path)).export(format="onnx", simplify=False, imgsz=640, opset=12, device="cpu")  # type: ignore

                if target.exists() and _validate_weight(target):
                    return (target.name, "exported")

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
                    return (target.name, "exported")
            except Exception:
                pass

    return (target.name, "failed")

def _fetch_all(names: List[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    results: List[Tuple[str, str]] = []
    failed: List[str] = []

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
    repo_root = Path(__file__).resolve().parents[2]
    rel_model_dir = os.path.relpath(MODEL_DIR.resolve(), repo_root.resolve()).replace("\\", "/")

    data: Dict[str, object] = {
        "model_dir": rel_model_dir,
        "selected": _dedupe(selected),
        "installed": _dedupe(installed),
    }
    man = MODEL_DIR / "manifest.json"
    man.write_text(json.dumps(data, indent=2))
    return man

def _show_row(title: str, items: Iterable[str]) -> None:
    typer.secho(f"\n{title}", bold=True)
    typer.echo("  " + "  |  ".join(f"[{i+1}] {v}" for i, v in enumerate(items)))

def _parse_multi(raw: str, items: Tuple[str, ...], *, aliases: Optional[Dict[str, str]] = None) -> List[str]:
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

def _ensure_env_hint() -> None:
    typer.echo()
    typer.secho("If this is a fresh clone, the launcher ensured the Python env.", fg="yellow")
    typer.secho("You do not need to activate a venv manually.", fg="yellow")
    typer.echo()

def _menu() -> int:
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
            typer.echo("You are choosing model weights for tasks Argos can perform.")
            typer.echo("YOLO models are named by version (v8, v11, v12), size (n/s/m/l/x), task (det/seg/pose/cls/obb), and format (.pt/.onnx).")
            typer.echo("Choosing the default pack (1) is a good start for most users.")
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

    _show_row("Tasks:", TASKS)
    task_alias = {"detect": "det", "segmentation": "seg", "classification": "cls"}
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
    _show_row("Versions:", VERSIONS)
    ver_alias: Dict[str, str] = {"v8": "8", "v11": "11", "v12": "12"}
    vers: List[str] = []
    while not vers:
        raw = typer.prompt("Pick versions (e.g. '1', '2', '3' / 'all')", default="all")
        vers = _parse_multi(raw, VERSIONS, aliases=ver_alias)
        if not vers:
            typer.secho("Pick at least one version (try '1', '2', '3' or 'all').", fg="yellow")

    _show_row("Model sizes:", SIZES)
    size_alias: Dict[str, str] = {"nano": "n", "small": "s", "medium": "m", "large": "l", "xlarge": "x"}
    sizes: List[str] = []
    while not sizes:
        raw = typer.prompt("Pick sizes (e.g. 'n,s' or '5' or 'all')", default="x,n")
        sizes = _parse_multi(raw, SIZES, aliases=size_alias)
        if not sizes:
            typer.secho("Pick at least one size (e.g., 'n' or 'all').", fg="yellow")

    _show_row("Tasks:", TASKS)
    task_alias = {"detect": "det", "segmentation": "seg", "classification": "cls"}
    tasks: List[str] = []
    while not tasks:
        raw = typer.prompt("Pick tasks (e.g. 'det,seg,pose,cls,obb' or 'all')", default="all")
        tasks = _parse_multi(raw, TASKS, aliases=task_alias)
        if not tasks:
            typer.secho("Pick at least one task (e.g., 'det').", fg="yellow")

    _show_row("Formats:", EXTS)
    fmt_raw = typer.prompt("Formats (.pt / .onnx / both)", default="both").strip().lower()
    if fmt_raw not in {"pt", "onnx", "both"}:
        fmt_raw = "both"
    formats: Tuple[str, ...] = (".pt", ".onnx") if fmt_raw == "both" else (f".{fmt_raw}",)

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

def _quick_check() -> None:
    if not typer.confirm("Run a quick smoke check now?", default=False):
        return

    raw_task = typer.prompt(
        "Task [detect|heatmap|geojson|classify|pose|obb]  (aliases: d|hm|gj|cls|pse|object)",
        default="heatmap",
    ).strip().lower()
    task_alias = {
        "d": "detect", "det": "detect", "detect": "detect",
        "hm": "heatmap", "heatmap": "heatmap",
        "gj": "geojson", "geojson": "geojson",
        "cls": "classify", "classify": "classify", "clf": "classify",
        "pose": "pose", "pse": "pose",
        "obb": "obb", "object": "obb",
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

    keep_prev = (os.environ.get("PANOPTES_SMOKE_KEEP_PREV", "").lower() in {"1", "true", "yes"})
    if not keep_prev:
        for p in list(results_dir.iterdir()):
            try:
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass

    try:
        from panoptes import cli as _cli  # type: ignore
        _img: Set[str] = set(cast(Iterable[str], getattr(_cli, "_IMAGE_EXTS", ())))  # type: ignore[misc]
        _vid: Set[str] = set(cast(Iterable[str], getattr(_cli, "_VIDEO_EXTS", ())))  # type: ignore[misc]
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
        return sorted([p for p in results_dir.glob("*") if p.is_file()],
                      key=lambda p: p.name.lower())

    with percent_spinner(prefix=f"ARGOS TESTING {task.upper()}") as sp:
        sp.update(total=float(expected), count=0, current="starting", job="spawn", model=task)

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
                    if n > expected:
                        try:
                            sp.update(total=float(n))
                        except Exception:
                            pass

                    delta = n - done_reported
                    if delta > 0:
                        try:
                            sp.update(count=done_reported + delta,
                                      current=(created[-1].name if created else "working"),
                                      job="write",
                                      model=task)
                        except Exception:
                            pass
                        done_reported = n

                if rc is not None:
                    break
                time.sleep(0.05)

            created = _list_created()
            n = len(created)
            if n > expected:
                try:
                    sp.update(total=float(n))
                except Exception:
                    pass
            delta = n - done_reported
            if delta > 0:
                try:
                    sp.update(count=done_reported + delta,
                              current=(created[-1].name if created else "done"),
                              job="finalize",
                              model=task)
                except Exception:
                    pass
                done_reported = n

            if proc.returncode == 0:
                _ok("Smoke check finished.")
                if created:
                    typer.secho("Results (Ctl + Click To Open):", bold=True)
                    for p in created:
                        typer.echo(f"  • {osc8_link(p.name, p)}")
                else:
                    _warn("CLI ran but produced no files.")
            else:
                _warn(f"Smoke check failed (exit code {proc.returncode}).")
        finally:
            try:
                proc.terminate()
            except Exception:
                pass

@app.command()
def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not (MODEL_DIR / "manifest.json").exists():
        _ensure_env_hint()

    choice = _menu()
    if choice == 0:
        raise typer.Exit(0)
    if choice == 1:
        selected = _dedupe(DEFAULT_PACK)
    elif choice == 2:
        selected = _dedupe(_full_pack())
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

    action_icons = {"present": "↺", "download": "↓", "copied": "⇢", "exported": "⎘", "failed": "✗"}
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
        _warn("Those names may not be hosted yet, or export failed.")

    man = _write_manifest(selected, installed)
    typer.echo(f"\nManifest: {osc8_link('manifest.json', man)}")

    _quick_check()

if __name__ == "__main__":
    main()
