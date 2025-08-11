# \rAIn\projects\argos\panoptes\tools\build_models.py
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import typer

# ---------------------------------------------------------------------
# Resolve model directory from registry (fallback to repo path)
# ---------------------------------------------------------------------
try:
    from panoptes.model_registry import MODEL_DIR as _REG_MODEL_DIR  # type: ignore
    _registry_model_dir: Optional[Path] = Path(_REG_MODEL_DIR)  # type: ignore[arg-type]
except Exception:
    _registry_model_dir = None

MODEL_DIR: Path = _registry_model_dir or (Path(__file__).resolve().parents[2] / "panoptes" / "model")

# Ultralytics (used to fetch/export weights)
has_yolo: bool
try:
    from ultralytics import YOLO as _YOLO  # type: ignore[reportMissingTypeStubs]
    try:
        # Keep Ultralytics quiet; we print our own succinct logs.
        from ultralytics.utils import LOGGER as _ULTRA_LOGGER  # type: ignore
        _rem = getattr(_ULTRA_LOGGER, "remove", None)
        if callable(_rem):
            _rem()
    except Exception:
        pass
    has_yolo = True
except Exception:
    _YOLO = None  # type: ignore[assignment]
    has_yolo = False

YOLO = _YOLO

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

# ---------------------------------------------------------------------
# Packs / naming
# ---------------------------------------------------------------------
FAMILIES: Tuple[str, ...] = ("8", "11", "12")
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


def _mk(fam: str, size: str, task: str, ext: str) -> str:
    """
    Construct official Ultralytics-style names.

    Ultralytics naming:
      • YOLOv8:  yolov8{s}.pt / yolov8{s}-seg.pt / -pose / -cls / -obb
      • YOLO11:  yolo11{s}.pt / yolo11{s}-seg.pt / -pose / -cls / -obb
      • YOLO12:  yolo12{s}.pt / yolo12{s}-seg.pt / -pose / -cls / -obb
    """
    base = f"yolov{fam}{size}" if fam == "8" else f"yolo{fam}{size}"
    suf = {
        "det": "",
        "seg": "-seg",
        "pose": "-pose",
        "cls": "-cls",
        "obb": "-obb",
    }[task]
    return base + suf + ext


def _full_pack() -> List[str]:
    """All families/sizes/tasks; .pt + .onnx (deduped)."""
    out: List[str] = []
    seen: Set[str] = set()
    for fam in FAMILIES:
        for sz in SIZES:
            for task in TASKS:
                for ext in EXTS:
                    n = _mk(fam, sz, task, ext)
                    if n not in seen:
                        seen.add(n)
                        out.append(n)
    return out


def _dedupe(names: Iterable[str]) -> List[str]:
    return sorted({n.strip(): None for n in names if n and n.strip()}.keys())


# ---------------------------------------------------------------------
# UX helpers
# ---------------------------------------------------------------------
def _ok(msg: str) -> None:
    typer.secho(msg, fg="green")


def _warn(msg: str) -> None:
    typer.secho(msg, fg="yellow")


def _err(msg: str) -> None:
    typer.secho(msg, fg="red", err=True)


def _latest_exported_onnx() -> Optional[Path]:
    hits = [Path(p) for p in glob.glob("runs/**/**/*.onnx", recursive=True)]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


@contextmanager
def _cd(path: Path):
    """Temporarily chdir to *path* (restores on exit)."""
    prev = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev)


# ---------------------------------------------------------------------
# Fetch logic
# ---------------------------------------------------------------------
def _fetch_one(name: str, dst: Path) -> Tuple[str, str]:
    """
    Obtain *name* into *dst* and return (basename, action).

    Possible actions:
      • "present"   – already existed in dst
      • "download"  – YOLO fetched directly into dst
      • "copied"    – YOLO fetched elsewhere; we copied into dst
      • "exported"  – we exported ONNX from a matching .pt
      • "failed"    – nothing worked
    """
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / Path(name).name

    # Already present
    if target.exists():
        return (target.name, "present")

    if not has_yolo or YOLO is None:
        return (target.name, "failed")

    # 1) Try to fetch directly by name from the model zoo (with CWD=dst)
    try:
        with _cd(dst):
            m = YOLO(name)  # type: ignore
            p = Path(getattr(m, "ckpt_path", name)).expanduser()
            if not p.exists():
                p = Path(name).expanduser()
            if p.exists():
                if p.resolve() == target.resolve():
                    return (target.name, "download")
                shutil.copy2(p, target)
                return (target.name, "copied")
    except Exception:
        # swallow and try ONNX export path below
        pass

    # 2) If ONNX, export from the corresponding .pt (inside dst)
    if name.endswith(".onnx"):
        try:
            pt_name = name[:-5] + ".pt"
            with _cd(dst):
                # Ensure the .pt is present in dst (YOLO will fetch into CWD=dst if needed)
                m_pt = YOLO(pt_name)  # type: ignore
                p_pt = Path(getattr(m_pt, "ckpt_path", pt_name)).expanduser()
                if p_pt.exists() and p_pt.resolve() != (dst / Path(pt_name)).resolve():
                    shutil.copy2(p_pt, (dst / Path(pt_name)))
                # Export ONNX (Ultralytics writes under runs/)
                try:
                    m_pt.export(format="onnx", dynamic=True, simplify=True, imgsz=640, opset=12, device="cpu")
                except Exception:
                    # fallback if simplification/opset settings cause issues
                    m_pt.export(format="onnx", dynamic=True, simplify=False, imgsz=640, opset=12, device="cpu")
                cand = _latest_exported_onnx()
                if cand and cand.exists():
                    shutil.copy2(cand, target)
                    try:
                        shutil.rmtree(dst / "runs", ignore_errors=True)
                    except Exception:
                        pass
                    if target.exists():
                        return (target.name, "exported")
        except Exception:
            return (target.name, "failed")

    return (target.name, "failed")


def _fetch_all(names: List[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    results: List[Tuple[str, str]] = []
    failed: List[str] = []
    for nm in names:
        base, action = _fetch_one(nm, MODEL_DIR)
        results.append((base, action))
        if action == "failed":
            failed.append(base)
    return results, failed


def _write_manifest(selected: List[str], installed: List[str]) -> None:
    """
    Write a user-agnostic manifest:
      • model_dir is stored *relative* to projects/argos to avoid absolute user paths
      • paths use forward slashes for cross-platform consistency
    """
    # repo_root here means .../projects/argos
    repo_root = Path(__file__).resolve().parents[2]
    rel_model_dir = os.path.relpath(MODEL_DIR.resolve(), repo_root.resolve()).replace("\\", "/")

    data: Dict[str, object] = {
        "model_dir": rel_model_dir,
        "selected": _dedupe(selected),
        "installed": _dedupe(installed),
    }
    (MODEL_DIR / "manifest.json").write_text(json.dumps(data, indent=2))


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
        # alias lookup
        if aliases and t in aliases:
            t = aliases[t]
        # allow 'v8'/'yolov8' → '8'
        if t.startswith("v") and t[1:] in items:
            t = t[1:]
        if t.startswith("yolov") and t[5:] in items:
            t = t[5:]
        if t in items and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _build_combo(
    fams: Iterable[str],
    sizes: Iterable[str],
    tasks: Iterable[str],
    *,
    formats: Iterable[str],
) -> List[str]:
    names: List[str] = []
    for fam in fams:
        for sz in sizes:
            for task in tasks:
                for ext in formats:
                    names.append(_mk(fam, sz, task, ext))
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
                2) Full pack (ALL families/sizes; ALL tasks; .pt + .onnx)
                3) Size pack (choose 1 family/size/tasks/formats)
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
    fam = typer.prompt(f"Family {FAMILIES}", default="8").strip()
    while fam not in FAMILIES:
        fam = typer.prompt(f"Family {FAMILIES}", default="8").strip()

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

    names = _build_combo([fam], [sz], chosen_tasks, formats=exts)
    return _dedupe(names)


def _ask_custom() -> List[str]:
    """
    Guided, multi-select custom builder:
    • choose one or more families
    • choose one or more sizes
    • choose tasks (det/seg/pose/cls/obb)
    • choose formats (.pt/.onnx/both)
    • optional extras typed as raw names
    """
    # Families
    _show_row("Families:", FAMILIES)
    fam_alias: Dict[str, str] = {"v8": "8", "v11": "11", "v12": "12"}
    fams: List[str] = []
    while not fams:
        raw = typer.prompt("Pick families (e.g. '1', '2', '3' / 'all')", default="all")
        fams = _parse_multi(raw, FAMILIES, aliases=fam_alias)
        if not fams:
            typer.secho("Pick at least one family version (try typing '1' '2' '3' or 'all').", fg="yellow")

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
    names = _build_combo(fams, sizes, tasks, formats=formats)

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
# Quick smoke
# ---------------------------------------------------------------------
def _quick_check() -> None:
    if not typer.confirm("Run a quick smoke check now?", default=False):
        return
    task = typer.prompt("Task [detect|heatmap|geojson]", default="heatmap").strip().lower()
    if task not in {"detect", "heatmap", "geojson"}:
        task = "heatmap"
    inp = typer.prompt("Input (path/URL or 'all' to process tests/raw)", default="all").strip()

    py = sys.executable
    args: List[str]
    if inp.lower() == "all":
        args = ["-m", "panoptes.cli", "all", "--task", task]
    else:
        args = ["-m", "panoptes.cli", inp, "--task", task]

    _ok(f"→ running: {py} {' '.join(args)}")
    try:
        subprocess.check_call([py, *args])
        _ok("Smoke check finished. See projects/argos/tests/results/")
    except Exception as e:
        _warn(f"Smoke check failed: {e!s}")


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

    # Summary
    typer.echo()
    _ok(f"Installed/ready: {len(installed)}")
    if bad:
        _warn(f"Skipped/failed: {len(bad)}")
        for n in _dedupe(bad):
            typer.echo(f"  – {n}")
        _warn("Those names may not be hosted by Ultralytics yet, or export failed.")

    # Manifest for reproducibility (user-agnostic, relative paths)
    _write_manifest(selected, installed)

    _quick_check()


if __name__ == "__main__":
    main()
