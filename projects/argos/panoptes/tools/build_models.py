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
    _registry_model_dir: Optional[Path] = Path(_REG_MODEL_DIR)
except Exception:
    _registry_model_dir = None

MODEL_DIR: Path = _registry_model_dir or (Path(__file__).resolve().parents[2] / "panoptes" / "model")

# Ultralytics (used to fetch/export weights)
try:
    from ultralytics import YOLO as _YOLO  # type: ignore[reportMissingTypeStubs,import-not-found]
    has_yolo: bool = True
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

# Your curated default pack
DEFAULT_PACK: List[str] = [
    # DETECT
    "yolov8x.pt",      # main
    "yolo11x.pt",      # backup
    "yolov12x.onnx",   # dev/light

    # HEATMAP (seg)
    "yolo11x-seg.pt",  # main
    "yolo11m-seg.pt",  # backup
    "yolov8n-seg.pt",  # dev/light

    # LIGHTWEIGHT family (additional fallbacks)
    "yolov12x.onnx",
    "yolov12m.onnx",
    "yolov12n.onnx",
]


def _mk(fam: str, size: str, seg: bool, ext: str) -> str:
    """Construct official Ultralytics-style names."""
    base = f"yolo11{size}" if fam == "11" else f"yolov{fam}{size}"
    if seg:
        base += "-seg"
    return base + ext


def _full_pack() -> List[str]:
    """All families/sizes; include seg and non-seg; .pt + .onnx."""
    out: List[str] = []
    seen: Set[str] = set()
    for fam in FAMILIES:
        for sz in SIZES:
            for ext in EXTS:
                for seg in (False, True):
                    n = _mk(fam, sz, seg, ext)
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
def _fetch_one(name: str, dst: Path) -> Tuple[str, bool]:
    """
    Try to obtain *name* into *dst*.

    Strategy (performed INSIDE dst to avoid polluting repo-root):
    1) YOLO(name) – works for official names (e.g., yolov8x.pt, yolo11x.pt)
    2) If name endswith .onnx and (1) failed:
        - YOLO(<same>.pt); export(..., format='onnx')
    """
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / Path(name).name

    # Already present
    if target.exists():
        return (target.name, True)

    # Need Ultralytics to download/export
    if not has_yolo or YOLO is None:
        return (target.name, False)

    # Direct download by name from the model zoo (run with CWD=dst)
    try:
        with _cd(dst):
            m = YOLO(name)  # type: ignore
            p = Path(getattr(m, "ckpt_path", name)).expanduser()
            if not p.exists():
                p = Path(name).expanduser()
            if p.exists():
                # If YOLO wrote elsewhere (cache), copy into dst/target
                if p.resolve() != target.resolve():
                    shutil.copy2(p, target)
                else:
                    # p is already the file at target path – nothing to copy
                    pass
                return (target.name, True)
    except Exception:
        pass

    # Export ONNX from the corresponding .pt (run entirely inside dst)
    if name.endswith(".onnx"):
        try:
            pt_name = name[:-5] + ".pt"
            with _cd(dst):
                # Ensure the .pt is here (YOLO will fetch into CWD=dst if needed)
                m_pt = YOLO(pt_name)  # type: ignore
                # Export ONNX (Ultralytics writes under runs/)
                m_pt.export(format="onnx", dynamic=True, simplify=True, imgsz=640, opset=12, device="cpu")
                cand = _latest_exported_onnx()
                if cand and cand.exists():
                    shutil.copy2(cand, target)
                    # optional tidy
                    try:
                        shutil.rmtree(dst / "runs", ignore_errors=True)
                    except Exception:
                        pass
                    return (target.name, True)
        except Exception:
            return (target.name, False)

    return (target.name, False)


def _fetch_all(names: List[str]) -> Tuple[List[str], List[str]]:
    ok: List[str] = []
    bad: List[str] = []
    for nm in names:
        final, success = _fetch_one(nm, MODEL_DIR)
        (ok if success else bad).append(final)
    return ok, bad


def _write_manifest(selected: List[str], installed: List[str]) -> None:
    data: Dict[str, object] = {
        "model_dir": str(MODEL_DIR),
        "selected": selected,
        "installed": installed,
    }
    (MODEL_DIR / "manifest.json").write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------
# Interactive helpers (for a nicer “Custom” flow)
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
    *,
    want_det: bool,
    want_seg: bool,
    formats: Iterable[str],
) -> List[str]:
    names: List[str] = []
    for fam in fams:
        for sz in sizes:
            if want_det:
                for ext in formats:
                    names.append(_mk(fam, sz, False, ext))
            if want_seg:
                for ext in formats:
                    names.append(_mk(fam, sz, True, ext))
    return _dedupe(names)


# ---------------------------------------------------------------------
# Menus
# ---------------------------------------------------------------------
def _ensure_env_hint() -> None:
    typer.echo()
    _warn("If this is a fresh clone, the launcher ensured the Python env.")
    _warn("You do not need to activate a venv manually.")
    typer.echo()


def _menu() -> int:
    """Loop until user selects a valid option (0–4)."""
    while True:
        typer.echo(
            textwrap.dedent(
                """
                What would you like to install?

                1) Default Drone-Vision pack
                2) Full pack (ALL families/sizes; .pt + .onnx; includes -seg)
                3) Size pack (choose 1 family/size/formats; optional -seg)
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
        _warn("Invalid choice. Try 0–4 or '?' for help.")


def _ask_size_pack() -> List[str]:
    fam = typer.prompt(f"Family {FAMILIES}", default="11").strip()
    while fam not in FAMILIES:
        fam = typer.prompt(f"Family {FAMILIES}", default="11").strip()

    sz = typer.prompt(f"Size {SIZES}", default="x").strip().lower()
    while sz not in SIZES:
        sz = typer.prompt(f"Size {SIZES}", default="x").strip().lower()

    want_seg = typer.confirm("Include segmentation (-seg) for heatmaps?", default=True)

    fmt = typer.prompt("Formats: .pt / .onnx / both", default="both").strip().lower()
    if fmt not in {"pt", "onnx", "both"}:
        fmt = "both"

    exts = (".pt", ".onnx") if fmt == "both" else (f".{fmt}",)
    names = [_mk(fam, sz, False, ext) for ext in exts]
    if want_seg:
        names += [_mk(fam, sz, True, ext) for ext in exts]
    return _dedupe(names)


def _ask_custom() -> List[str]:
    """
    Guided, multi-select custom builder:
    • choose one or more families
    • choose one or more sizes
    • choose detect/seg (or both)
    • choose formats (.pt/.onnx/both)
    • optional extras typed as raw names
    """
    # Families
    _show_row("Families:", FAMILIES)
    fam_alias: Dict[str, str] = {"v8": "8", "v11": "11", "v12": "12"}
    fams: List[str] = []
    while not fams:
        raw = typer.prompt("Pick families (e.g. '1, '2' '3' / 'all')", default="Familiy 11 = '2' / All Families = 'all'")
        fams = _parse_multi(raw, FAMILIES, aliases=fam_alias)
        if not fams:
            _warn("Pick at least one family version (try typing '1' '2' '3' or 'all').")

    # Sizes
    _show_row("Model sizes:", SIZES)
    size_alias: Dict[str, str] = {"nano": "n", "small": "s", "medium": "m", "large": "l", "xlarge": "x"}
    sizes: List[str] = []
    while not sizes:
        raw = typer.prompt("Pick sizes (e.g. 'n,s' or '5' or 'all')", default="n,s")
        sizes = _parse_multi(raw, SIZES, aliases=size_alias)
        if not sizes:
            _warn("Pick at least one size (e.g., 'n' or 'all').")

    # Tasks (detect/seg)
    want_det = typer.confirm("Include DETECT (no -seg)?", default=True)
    want_seg = typer.confirm("Include SEG (heatmap, '-seg')?", default=True)
    if not (want_det or want_seg):
        _warn("Nothing selected; enabling SEG by default.")
        want_seg = True

    # Formats
    _show_row("Formats:", EXTS)
    fmt_raw = typer.prompt("Formats (.pt / .onnx / both)", default="both").strip().lower()
    if fmt_raw not in {"pt", "onnx", "both"}:
        fmt_raw = "both"
    formats: Tuple[str, ...] = (".pt", ".onnx") if fmt_raw == "both" else (f".{fmt_raw}",)

    # Build + optional extras
    names = _build_combo(fams, sizes, want_det=want_det, want_seg=want_seg, formats=formats)

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
    _ensure_env_hint()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    choice = _menu()
    if choice == 0:
        raise typer.Exit(0)
    if choice == 1:
        selected = list(DEFAULT_PACK)
    elif choice == 2:
        selected = _full_pack()
    elif choice == 3:
        selected = _ask_size_pack()
    elif choice == 4:
        selected = _ask_custom()
    else:
        # Should never happen because _menu() only returns 0–4
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

    ok, bad = _fetch_all(selected)

    if ok:
        _ok(f"Installed {len(ok)} file(s) to {MODEL_DIR}:")
        for n in ok:
            typer.echo(f"  ✓ {n}")
    if bad:
        _warn(f"\nSkipped/failed ({len(bad)}):")
        for n in bad:
            typer.echo(f"  – {n}")
        _warn("Those names may not be hosted by Ultralytics, or export failed. You can add them manually.")

    _write_manifest(selected, ok)
    _quick_check()


if __name__ == "__main__":
    main()
