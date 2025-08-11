"""
panoptes.cli – unified Typer front-end (“argos”)

Tasks: detect | heatmap | geojson on images **and** videos.
Model selection is delegated to *panoptes.model_registry* and is strictly enforced.

This version hardens help/man UX and adds a rich manual, examples, and a tutorial
without changing the core task flow.
"""

from __future__ import annotations

import fnmatch
import re
import sys
import textwrap
import urllib.parse
from pathlib import Path
from typing import Any, Final, List, Literal, Optional, cast

import typer

# ────────────────────────────────────────────────────────────────────────────
#  app scaffolding
# ────────────────────────────────────────────────────────────────────────────

# Recognize as many help invocations as possible (Windows-style included)
_HELP_NAMES = ["-h", "--help", "-help", "-?", "/?"]

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": _HELP_NAMES},
)

# NOTE: heavy imports (model registry / lambda_like) are deferred to avoid
# initializing models when user only wants help/man.

# ────────────────────────────────────────────────────────────────────────────
#  token aliases (positional tokens, NOT click/typer flags)
# ────────────────────────────────────────────────────────────────────────────
_ALIAS = {
    # detect
    "d": "detect",
    "-d": "detect",
    "detect": "detect",
    "--detect": "detect",
    "-detect": "detect",
    # heatmap
    "hm": "heatmap",
    "heatmap": "heatmap",
    "--heatmap": "heatmap",
    "-hm": "heatmap",
    # geojson
    "gj": "geojson",
    "geojson": "geojson",
    "--geojson": "geojson",
    "-gj": "geojson",
    # help / man as positional tokens (so `argos help` / `argos man` works)
    "help": "help",
    "man": "man",
    "?": "help",
}
_VALID = {"detect", "heatmap", "geojson"}

_URL_RE = re.compile(r"^(https?://.+|data:image/[^;]+;base64,.+)$", re.I)

_AVAILABLE_MODELS: list[str] = ["primary"]  # cosmetic placeholder
_DEFAULT_MODEL = "primary"

# Search config
_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".avif", ".bmp", ".tif", ".tiff", ".gif", ".webp", ".heic", ".heif"
}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
_MEDIA_EXTS = _IMAGE_EXTS | _VIDEO_EXTS
_PREF_EXT_ORDER = (
    ".avif", ".jpg", ".jpeg", ".png",
    ".mp4", ".mov", ".avi", ".mkv",
    ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic", ".heif",
)

# Compute /projects/argos from this file path
_PROJ_ROOT_PATH: Final[Path] = Path(__file__).resolve().parents[1]
_RAW_DIR: Path = _PROJ_ROOT_PATH / "tests" / "raw"
_RESULTS_DIR: Path = _PROJ_ROOT_PATH / "tests" / "results"
_SEARCH_DIRS: list[Path] = [
    _RAW_DIR,
    _PROJ_ROOT_PATH / "tests" / "assets",
    Path.cwd(),
]
_NOISE = {"argos", "run", "me"}
_GLOB_CHARS = set("*?[]")

# ────────────────────────────────────────────────────────────────────────────
#  rich manual / examples / tutorial text
# ────────────────────────────────────────────────────────────────────────────

def _join_lines(*parts: Optional[str]) -> str:
    return "\n".join((p or "").rstrip() for p in parts if p is not None)

def _man_header() -> str:
    return textwrap.dedent(f"""
    ARGOS(1) — Panoptes Vision CLI
    =================================

    NAME
        argos — detect, heatmap, and geojson over images & videos

    SYNOPSIS
        argos [INPUT ...] [d|hm|gj] [OPTIONS]
        argos [OPTIONS] INPUT ...

    DESCRIPTION
        A zero-fiddling front-end over Panoptes tasks:
          • detect   — object detection boxes
          • heatmap  — segmentation heat-map overlay
          • geojson  — polygon extraction to GeoJSON (images only)

        You can put task tokens anywhere:
            argos mildrone d
            detect mildrone.avif
            d mildrone
            hm camo
            argos -d all
            argos --heatmap all .jpg

        Inputs are resolved from (in order):
            1) given path or URL
            2) {str(_RAW_DIR)}
            3) {str(_PROJ_ROOT_PATH / 'tests' / 'assets')}
            4) current working directory

        Wildcards and “all”:
            d *
            hm *.png
            gj all .jpg
            argos all d
    """).strip("\n")

def _man_tasks() -> str:
    return textwrap.dedent("""
    TASKS
        detect   Boxes over images or videos.
        heatmap  Segmentation heat-map overlay (images & videos).
        geojson  Extract polygons from images as GeoJSON (videos are skipped).
    """).strip("\n")

def _man_inputs() -> str:
    return textwrap.dedent(f"""
    INPUTS
        • Plain filenames or stems (we search known dirs)
        • URLs (http/https or data:image/* base64)
        • Globs: "*.jpg", "*"
        • ALL:   "all" (optionally followed by ".ext"/"ext"/"*.ext")

        When using globs or "all", the search happens in:
            {_RAW_DIR}
    """).strip("\n")

def _man_models() -> str:
    return textwrap.dedent("""
    MODEL SELECTION
        Weights are picked by the model registry strictly (no silent fallbacks).
        Use --small/--fast to prefer lightweight models for live video.
        Per-task override:
          --det-weights PATH   (detect/geojson)
          --seg-weights PATH   (heatmap)

        If a required weight is missing, the command exits with code 1.
    """).strip("\n")

def _man_tuning() -> str:
    return textwrap.dedent("""
    TUNING
        --conf FLOAT         [detect/heatmap] confidence threshold 0–1 (default 0.40)
        --alpha FLOAT        heat-map blend 0–1 (default 0.40)
        --cmap NAME          OpenCV/Matplotlib colour-map (default COLORMAP_JET)
        -k / --k FLOAT       σ area / kernel_scale (smaller → blurrier)
        --small / --fast     use nano models for live video
    """).strip("\n")

def _man_outputs() -> str:
    return textwrap.dedent(f"""
    OUTPUTS
        Results are written under:
            {_RESULTS_DIR}
        (File names mirror the input stem with task-specific suffixes.)
    """).strip("\n")

def _man_windows() -> str:
    return textwrap.dedent("""
    WINDOWS / POWERSHELL NOTES
        • `-h`, `--help`, `-help`, `-?`, and `/?` all show help.
        • `argos man` or `argos --man` opens the full manual.
        • Typing `--help` or `-h` *by itself* in PowerShell (without `argos`) is a shell error.
    """).strip("\n")

def _man_shortcuts() -> str:
    return textwrap.dedent("""
    SHORTCUT ENTRYPOINTS (optional, via project.scripts)
        d         -> prepends "d"        e.g., `d all`
        hm        -> prepends "hm"       e.g., `hm all`
        gj        -> prepends "gj"       e.g., `gj fuji`
        detect    -> prepends "d"        e.g., `detect midtown`
        heatmap   -> prepends "hm"       e.g., `heatmap all .jpg`
        geojson   -> prepends "gj"
        all       -> prepends "all"      e.g., `all hm` (PowerShell-friendly)
    """).strip("\n")

def _man_examples() -> str:
    return textwrap.dedent("""
    EXAMPLES
        Quick help & manual:
            argos -h
            argos man

        Detect (single file by stem search):
            argos d midtown
            argos detect fuji.jpg

        Heat-map overlay:
            argos hm camo
            argos --heatmap gerbera-drones.png

        GeoJSON (images only):
            argos gj fuji
            argos geojson assets.jpg

        Batch from test set:
            argos d all
            argos hm *.png
            argos gj all .jpg

        Video:
            argos d bunny.mp4
            argos hm bunny.mp4 --small

        Explain without running (dry-run):
            argos --dry-run d all .jpg

        Force weights:
            argos d fuji --det-weights projects/argos/panoptes/model/yolov8x.pt
            argos hm camo --seg-weights projects/argos/panoptes/model/yolov8x-seg.pt
    """).strip("\n")

def _full_manual() -> str:
    return _join_lines(
        _man_header(), "",
        _man_tasks(), "",
        _man_inputs(), "",
        _man_models(), "",
        _man_tuning(), "",
        _man_outputs(), "",
        _man_windows(), "",
        _man_shortcuts(), "",
        _man_examples(), ""
    )

def _examples_page() -> str:
    samples: list[str] = []
    if _RAW_DIR.exists():
        for f in sorted(_RAW_DIR.iterdir(), key=lambda p: p.name.lower()):
            if f.is_file() and f.suffix.lower() in _MEDIA_EXTS:
                samples.append(f"  - {f.name}")
    dynamic = "\n".join(samples) if samples else "  (no sample files found)"
    return _join_lines(
        "EXAMPLES — QUICK START",
        "======================",
        "",
        "Try these first:",
        "  argos hm camo",
        "  argos d mildrone",
        "  argos gj fuji",
        "  argos d bunny.mp4",
        "",
        "Globs / ALL:",
        "  argos d all",
        "  argos hm *.png",
        "  argos gj all .jpg",
        "",
        "Samples available in tests/raw:",
        dynamic,
        "",
        "Results are written to:",
        f"  {_RESULTS_DIR}",
        ""
    )

def _tutorial_page() -> str:
    return _join_lines(
        "TUTORIAL — 5-Minute Tour",
        "=========================",
        "",
        "1) Detect something",
        "   argos d midtown",
        "",
        "2) Heat-map a camo target",
        "   argos hm camo --alpha 0.45 --cmap COLORMAP_JET",
        "",
        "3) Extract polygons",
        "   argos gj fuji",
        "",
        "4) Batch a whole folder",
        "   argos hm all .jpg --small",
        "",
        "5) Video in nano mode",
        "   argos hm bunny.mp4 --small",
        "",
        "Pro tip: not sure what would run?",
        "   argos --dry-run hm all .png",
        ""
    )

# Topic registry for `argos man <topic>`
_MAN_TOPICS = {
    "tasks": _man_tasks,
    "inputs": _man_inputs,
    "models": _man_models,
    "tuning": _man_tuning,
    "outputs": _man_outputs,
    "windows": _man_windows,
    "shortcuts": _man_shortcuts,
    "examples": _man_examples,
    "tutorial": _tutorial_page,
}

# ────────────────────────────────────────────────────────────────────────────
#  helper utils (resolution)
# ────────────────────────────────────────────────────────────────────────────
def _is_url(text: str) -> bool:
    return urllib.parse.urlparse(text).scheme in {"http", "https"} or bool(_URL_RE.match(text))

def _extract_task(tokens: List[str]) -> tuple[Optional[str], List[str]]:
    """Return (explicit_task_or_None, remaining_tokens)."""
    hits = [(i, _ALIAS[t.lower()]) for i, t in enumerate(tokens) if t.lower() in _ALIAS]
    if not hits:
        return None, tokens

    # If the user typed help/man as a positional token, surface it immediately
    uniq = {alias for _, alias in hits}
    if "help" in uniq or "man" in uniq:
        idx = next(i for i, a in hits if a in {"help", "man"})
        rest = tokens[:idx] + tokens[idx + 1 :]
        return _ALIAS[tokens[idx].lower()], rest

    # Disallow ambiguous multi-task tokens
    uniq = {alias for _, alias in hits if alias in _VALID}
    if len(uniq) > 1:
        typer.secho(
            f"[bold red]  more than one task alias supplied "
            f"({', '.join(sorted(uniq))})",
            err=True,
        )
        raise typer.Exit(2)

    # remove the first occurrence of the single task token
    if uniq:
        idx, task = next((i, _ALIAS[t.lower()]) for i, t in hits if _ALIAS[t.lower()] in _VALID)
        rest = tokens[:idx] + tokens[idx + 1 :]
        return task, rest

    return None, tokens

def _pref_key(p: Path) -> int:
    try:
        return _PREF_EXT_ORDER.index(p.suffix.lower())
    except ValueError:
        return 999

def _resolve_input_token(tok: str) -> Optional[str]:
    """
    Resolve *tok* to a usable path (str) or URL.
    Returns ''/None to indicate the token should be dropped (noise).
    """
    tlow = tok.lower().strip()
    if not tlow:
        return None
    if tlow in _NOISE:
        return ""  # drop silently

    if _is_url(tok):
        return tok

    p = Path(tok)
    if p.exists():
        return str(p)

    name = p.name
    stem = p.stem if p.suffix else name

    candidates: list[Path] = []
    if p.suffix:
        for d in _SEARCH_DIRS:
            candidates.append(d / name)
    else:
        for d in _SEARCH_DIRS:
            for ext in _PREF_EXT_ORDER:
                candidates.append(d / f"{stem}{ext}")

    for c in candidates:
        if c.exists():
            return str(c)

    matches: list[Path] = []
    for d in _SEARCH_DIRS:
        if not d.exists():
            continue
        for f in d.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in _MEDIA_EXTS:
                continue
            if f.stem.lower() == stem.lower():
                matches.append(f)

    if matches:
        best = sorted(matches, key=_pref_key)[0]
        return str(best)

    return tok

def _looks_like_glob(s: str) -> bool:
    return any(ch in s for ch in _GLOB_CHARS)

def _normalize_ext_token(t: str) -> Optional[str]:
    m = re.fullmatch(r"\*?\.*([A-Za-z0-9]+)", t.strip())
    if not m:
        return None
    return "." + m.group(1).lower()

def _expand_in_raw(*, pattern: Optional[str], exts: Optional[set[str]], task: str) -> list[str]:
    """Enumerate files in tests/raw matching pattern or extension filter."""
    base = _RAW_DIR
    out: list[str] = []
    if not base.exists():
        return out

    for f in base.iterdir():
        if not f.is_file():
            continue
        suf = f.suffix.lower()
        if suf not in _MEDIA_EXTS:
            continue
        if task == "geojson" and suf in _VIDEO_EXTS:
            continue
        if exts is not None and suf not in exts:
            continue
        if pattern is not None and not fnmatch.fnmatch(f.name.lower(), pattern.lower()):
            continue
        out.append(str(f))
    return sorted(out, key=lambda s: s.lower())

def _expand_tokens(positional: list[str], task_final: str) -> list[str]:
    """
    Expand tokens into concrete input paths.
    Supports:
      - stems/paths/URLs
      - globs: "*.jpg", "*"
      - "all" [optional ".ext"/"ext"/"*.ext"]
    Search base for globs/'all' is tests/raw.
    """
    i = 0
    found: list[str] = []
    seen: set[str] = set()

    while i < len(positional):
        tok = positional[i]
        tlow = tok.lower()

        if tlow in _NOISE:
            i += 1
            continue

        if tlow in {"all", "*"}:
            exts: Optional[set[str]] = None
            if i + 1 < len(positional):
                nxt = positional[i + 1]
                ext = _normalize_ext_token(nxt)
                if ext:
                    exts = {ext}
                    i += 1
            for p in _expand_in_raw(pattern=None, exts=exts, task=task_final):
                if p not in seen:
                    seen.add(p)
                    found.append(p)
            i += 1
            continue

        if _looks_like_glob(tlow):
            ext = None
            m = re.fullmatch(r"\*\.\s*([A-Za-z0-9]+)", tlow.replace(" ", ""))
            if m:
                ext = "." + m.group(1).lower()
            matches = (
                _expand_in_raw(pattern=None, exts={ext}, task=task_final)
                if ext else
                _expand_in_raw(pattern=tlow, exts=None, task=task_final)
            )
            for p in matches:
                if p not in seen:
                    seen.add(p)
                    found.append(p)
            i += 1
            continue

        resolved = _resolve_input_token(tok)
        if resolved == "":
            pass
        elif resolved is not None:
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
        i += 1

    return found

# ────────────────────────────────────────────────────────────────────────────
#  light wrappers for lazy imports (prevent model init during help/man)
# ────────────────────────────────────────────────────────────────────────────
def _pick_weight(task: Literal["detect", "heatmap"], *, small: bool):
    from panoptes import model_registry as _mr  # type: ignore[reportMissingTypeStubs]
    return _mr.pick_weight(task, small=small)

def _reinit_models(
    *,
    detect_small: Optional[bool] = None,
    segment_small: Optional[bool] = None,
    det_override: Optional[Path] = None,
    seg_override: Optional[Path] = None,
) -> None:
    from . import lambda_like as _ll  # type: ignore[reportMissingTypeStubs]
    _ll.reinit_models(
        detect_small=detect_small,
        segment_small=segment_small,
        det_override=det_override,
        seg_override=seg_override,
    )

def _run_single(
    src: str,
    *,
    model: str,
    task: Literal["detect", "heatmap", "geojson"],
    **kwargs: Any,
) -> Any:
    from . import lambda_like as _ll  # type: ignore[reportMissingTypeStubs]
    return _ll.run_single(src, model=model, task=task, **kwargs)

# ────────────────────────────────────────────────────────────────────────────
#  command
# ────────────────────────────────────────────────────────────────────────────
@app.command()
def target(  # noqa: C901
    inputs: List[str] = typer.Argument(..., metavar="INPUT [d|hm|gj|FLAGS|help|man]"),
    *,
    # flexible: explicit task name
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Task: detect|heatmap|geojson"),
    # convenience boolean flags for task selection
    detect_flag: bool = typer.Option(False, "--detect", "-d", help="Shortcut for --task detect"),
    heatmap_flag: bool = typer.Option(False, "--heatmap", help="Shortcut for --task heatmap"),
    geojson_flag: bool = typer.Option(False, "--geojson", help="Shortcut for --task geojson"),
    # meta/help UX
    man_flag: bool = typer.Option(False, "--man", help="Open the full manual and exit"),
    examples_flag: bool = typer.Option(False, "--examples", help="Show examples and exit"),
    tutorial_flag: bool = typer.Option(False, "--tutorial", help="Show a short tutorial and exit"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Explain what would run, then exit"),
    # model (cosmetic placeholder for compatibility)
    model: str = typer.Option(_DEFAULT_MODEL, "--model", "-m"),
    # heat-map tuning
    alpha: float = typer.Option(0.40, help="Heat-map blend 0-1"),
    cmap: str = typer.Option("COLORMAP_JET", help="OpenCV / Matplotlib colour-map"),
    kernel_scale: float = typer.Option(5.0, "--k", "-k", help="σ  area / kernel_scale (smaller  blurrier)"),
    conf: float = typer.Option(0.40, help="[detect / heat-map] confidence threshold 0-1"),
    small: bool = typer.Option(False, "--small", "--fast", help="use nano models for live video"),
    # per-task override weights
    det_override: Optional[Path] = typer.Option(
        None, "--det-weights",
        help="Force a detector weight for detect/geojson (path to .pt/.onnx).",
    ),
    seg_override: Optional[Path] = typer.Option(
        None, "--seg-weights",
        help="Force a segmentation weight for heatmap (path to .pt/.onnx).",
    ),
) -> None:
    """Batch-process images / videos with zero manual weight fiddling."""

    # Honor meta flags first (no heavy imports)
    if man_flag:
        typer.echo_via_pager(_full_manual())
        raise typer.Exit(0)
    if examples_flag:
        typer.echo_via_pager(_examples_page())
        raise typer.Exit(0)
    if tutorial_flag:
        typer.echo_via_pager(_tutorial_page())
        raise typer.Exit(0)

    if not inputs:
        typer.secho("[bold red]  no inputs given", err=True)
        raise typer.Exit(2)

    token_task, positional = _extract_task(inputs)

    # If the user passed `help` or `man` as a positional token, maybe with a topic
    if token_task in {"help", "man"}:
        topic = (positional[0].lower() if positional else "").strip()
        if topic in _MAN_TOPICS:
            typer.echo_via_pager(_MAN_TOPICS[topic]())
        else:
            typer.echo_via_pager(_full_manual())
        raise typer.Exit(0)

    # Resolve a single final task with clear precedence:
    #   1) explicit --task / -t
    #   2) boolean flags (--detect/--heatmap/--geojson)
    #   3) token_task found among positionals
    #   4) default -> detect
    flag_tasks = [name for ok, name in [
        (detect_flag, "detect"),
        (heatmap_flag, "heatmap"),
        (geojson_flag, "geojson"),
    ] if ok]
    if len(flag_tasks) > 1:
        typer.secho(f"[bold red]  conflicting flags set: {', '.join(flag_tasks)}", err=True)
        raise typer.Exit(2)

    if task is not None:
        task = task.lower()
        if task not in _VALID:
            typer.secho(f"[bold red]  invalid --task {task}", err=True)
            raise typer.Exit(2)
        if flag_tasks and flag_tasks[0] != task:
            typer.secho(
                f"[bold red]  conflicting task: flag={flag_tasks[0]!r}  --task={task!r}",
                err=True,
            )
            raise typer.Exit(2)
        if token_task is not None and token_task != task:
            typer.secho(
                f"[bold red]  conflicting task: token={token_task!r}  --task={task!r}",
                err=True,
            )
            raise typer.Exit(2)
        task_final = task
    elif flag_tasks:
        if token_task is not None and token_task != flag_tasks[0]:
            typer.secho(
                f"[bold red]  conflicting task: token={token_task!r}  flag={flag_tasks[0]!r}",
                err=True,
            )
            raise typer.Exit(2)
        task_final = flag_tasks[0]
    else:
        task_final = token_task or "detect"

    # model flag is cosmetic; kept for compatibility
    if model.lower() not in _AVAILABLE_MODELS:
        typer.secho(f"[bold yellow] unknown --model ignored: {model}", err=True)
    model = model.lower()

    hm_kwargs: dict[str, Any] = dict(alpha=alpha, cmap=cmap, kernel_scale=kernel_scale, conf=conf)

    # normalize/resolve (supports ALL/globs)
    norm_inputs = _expand_tokens(positional, task_final)

    if not norm_inputs:
        typer.secho("[bold red]  no usable inputs found", err=True)
        raise typer.Exit(2)

    # Dry run: explain what would happen and exit
    if dry_run:
        typer.secho(
            f"[panoptes] dry-run: task={task_final} small={small} inputs={len(norm_inputs)}",
            err=True,
        )
        to_show = norm_inputs if len(norm_inputs) <= 20 or small else norm_inputs[:20] + ["..."]
        for item in to_show:
            if item.endswith(tuple(_VIDEO_EXTS)):
                typer.echo(f"  video:   {item}")
            elif _is_url(item):
                typer.echo(f"  url:     {item}")
            else:
                typer.echo(f"  image:   {item}")
        raise typer.Exit(0)

    # ── Heavy stuff starts here (lazy import ensures help/man stays snappy) ──

    # Announce overall run
    typer.secho(
        f"[panoptes] cli: task={task_final} small={small} inputs={len(norm_inputs)}",
        err=True,
    )

    # Initialize appropriate model(s) with optional override (applies to still-image flows)
    if task_final == "detect":
        _reinit_models(detect_small=small, det_override=det_override)
    elif task_final == "heatmap":
        _reinit_models(segment_small=small, seg_override=seg_override)
    elif task_final == "geojson":
        _reinit_models(detect_small=small, det_override=det_override)

    # loop over inputs
    for item in norm_inputs:
        low = item.lower()

        # videos
        if low.endswith(tuple(_VIDEO_EXTS)):
            if task_final == "geojson":
                typer.secho(f"[bold yellow] skipping video for geojson: {item}", err=True)
                continue

            # choose weight (override beats registry pick)
            ov = det_override if task_final == "detect" else (
                seg_override if task_final == "heatmap" else None
            )
            weight = Path(ov) if ov is not None else _require_weight(
                cast(Literal["detect", "heatmap"], task_final), small=small
            )
            typer.secho(
                f"[panoptes] video: {item} → task={task_final} small={small} weight={weight}",
                err=True,
            )

            if task_final == "heatmap":
                from .predict_heatmap_mp4 import main as _heat_vid
                _heat_vid(item, weights=weight, **hm_kwargs)
            else:  # detect
                from .predict_mp4 import main as _detect_vid
                _detect_vid(item, conf=conf, weights=weight)
            continue

        # still images & URLs
        if low.endswith(tuple(_IMAGE_EXTS)) or _is_url(item):
            if task_final == "detect":
                w = Path(det_override) if det_override is not None else _pick_weight("detect", small=small)
                typer.secho(f"[panoptes] image/url: {item} → task=detect weight={w}", err=True)
            elif task_final == "heatmap":
                w = Path(seg_override) if seg_override is not None else _pick_weight("heatmap", small=small)
                typer.secho(f"[panoptes] image/url: {item} → task=heatmap weight={w}", err=True)
            else:  # geojson
                typer.secho(f"[panoptes] image/url: {item} → task=geojson (no model required)", err=True)

            _run_single(
                item,
                model=model,
                task=cast(Literal["detect", "heatmap", "geojson"], task_final),
                **hm_kwargs,
            )
            continue

        typer.secho(f"[bold red] unsupported input: {item}", err=True)
        raise typer.Exit(2)

def _require_weight(task: Literal["detect", "heatmap"], *, small: bool) -> Path:
    """
    Strictly pick the weight for *task*.

    Raises
    ------
    typer.Exit
        If the weight is not configured or the file does not exist.
    """
    weight = _pick_weight(task, small=small)
    if weight is None:
        typer.secho(
            f"[bold red]  no weight configured for task {task} "
            "(edit panoptes.model_registry.WEIGHT_PRIORITY)",
            err=True,
        )
        raise typer.Exit(1)
    return weight

# ────────────────────────────────────────────────────────────────────────────
#  entry-point glue (unchanged)
# ────────────────────────────────────────────────────────────────────────────
def _prepend_argv(token: str) -> None:
    sys.argv = sys.argv[:1] + [token] + sys.argv[1:]

def main() -> None:  # pragma: no cover
    try:
        app()
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[bold red]Unhandled error: {exc}", err=True)
        sys.exit(1)

# Convenience entrypoints (optional):
def main_detect() -> None:  # pragma: no cover
    _prepend_argv("d")
    main()

def main_heatmap() -> None:  # pragma: no cover
    _prepend_argv("hm")
    main()

def main_geojson() -> None:  # pragma: no cover
    _prepend_argv("gj")
    main()

def main_all() -> None:  # pragma: no cover
    _prepend_argv("all")
    main()

if __name__ == "__main__":  # pragma: no cover
    main()
