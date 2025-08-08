"""
panoptes.cli  – unified Typer front-end (“target”)

• Tasks: detect | heatmap | geojson on images **and** videos.
• Model selection is delegated to *panoptes.model_registry* and is
  strictly enforced – no silent fall-backs.

Enhanced (search + flexible order + ALL/globs):
───────────────────────────────────────────────
• Inputs and task tokens in ANY order, e.g.
      argos mildrone d
      detect mildrone.avif
      d mildrone
      hm camo
• Implicit search if pathless:
      projects/argos/tests/raw/
      projects/argos/tests/assets/
      current working directory
• Wildcards / “all”:
      d *
      hm *.png
      gj all .jpg
      argos all d
  (all, ALL, All, *, *.ext, all .ext, all ext supported;
   expansion is done against tests/raw by default.)
"""

from __future__ import annotations

import fnmatch
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any, List, Literal, Optional, cast

import typer

from panoptes.model_registry import pick_weight
from .lambda_like import run_single
from . import ROOT as _PROJ_ROOT  # /projects/argos

#  scaffolding ─
app = typer.Typer(add_completion=False, rich_markup_mode="rich")

_ALIAS = {
    "d": "detect",
    "-d": "detect",
    "detect": "detect",
    "hm": "heatmap",
    "heatmap": "heatmap",
    "gj": "geojson",
    "-gj": "geojson",
    "geojson": "geojson",
}
_VALID = {"detect", "heatmap", "geojson"}

_URL_RE = re.compile(r"^(https?://.+|data:image/[^;]+;base64,.+)$", re.I)

_AVAILABLE_MODELS: list[str] = ["primary"]  # flag kept for CLI parity
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
_RAW_DIR = _PROJ_ROOT / "tests" / "raw"
_SEARCH_DIRS = [
    _RAW_DIR,
    _PROJ_ROOT / "tests" / "assets",
    Path.cwd(),
]
_NOISE = {"argos", "run", "me"}  # ignore these if they appear as positional tokens
_GLOB_CHARS = set("*?[]")

#  helper utils 
def _is_url(text: str) -> bool:
    return urllib.parse.urlparse(text).scheme in {"http", "https"} or bool(_URL_RE.match(text))

def _extract_task(tokens: List[str]) -> tuple[str, List[str]]:
    """Return *(task, remaining_tokens)*  defaults to *detect* if none found."""
    hits = [(i, _ALIAS[t.lower()]) for i, t in enumerate(tokens) if t.lower() in _ALIAS]
    if not hits:
        return "detect", tokens
    if len(hits) > 1:
        typer.secho(
            f"[bold red]  more than one task alias supplied "
            f"({', '.join(a for _, a in hits)})",
            err=True,
        )
        raise typer.Exit(2)
    idx, task = hits[0]
    rest = tokens[:idx] + tokens[idx + 1:]
    return task, rest

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
    # direct path?
    if p.exists():
        return str(p)

    # attempt resolution from the known search locations
    name = p.name
    stem = p.stem if p.suffix else name

    candidates: list[Path] = []

    # If user already provided an extension  look for exact basename in search dirs
    if p.suffix:
        for d in _SEARCH_DIRS:
            candidates.append(d / name)
    else:
        # No extension: try preferred extension order for the given stem
        for d in _SEARCH_DIRS:
            for ext in _PREF_EXT_ORDER:
                candidates.append(d / f"{stem}{ext}")

    for c in candidates:
        if c.exists():
            return str(c)

    # As a fallback, do a case-insensitive stem match within search dirs
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

    # give up  let the loop below raise a clean error for unsupported input
    return tok

#  wildcard / "all" expansion (searches tests/raw by default) 
def _looks_like_glob(s: str) -> bool:
    return any(ch in s for ch in _GLOB_CHARS)

def _normalize_ext_token(t: str) -> Optional[str]:
    # Accept ".jpg", "jpg", "*.jpg"
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
    # stable, human-ish ordering
    return sorted(out, key=lambda s: s.lower())

def _expand_tokens(positional: list[str], task_final: str) -> list[str]:
    """
    Expand tokens into concrete input paths.
    Handles:
       plain filenames/stems (resolved across RAW/assets/CWD)
       URLs
       globs: "*.jpg", "*"
       'all', 'ALL', 'All' (optionally followed by ".ext"/"ext"/"*.ext")
    When using globs/'all', we search tests/raw by design.
    """
    i = 0
    found: list[str] = []
    seen: set[str] = set()

    while i < len(positional):
        tok = positional[i]
        tlow = tok.lower()

        # skip noise words
        if tlow in _NOISE:
            i += 1
            continue

        # Case: all / *  [+ optional extension token]
        if tlow in {"all", "*"}:
            exts: Optional[set[str]] = None
            # Lookahead: extension spec in the next token?
            if i + 1 < len(positional):
                nxt = positional[i + 1]
                ext = _normalize_ext_token(nxt)
                if ext:
                    exts = {ext}
                    i += 1  # consume the ext token

            for p in _expand_in_raw(pattern=None, exts=exts, task=task_final):
                if p not in seen:
                    seen.add(p)
                    found.append(p)

            i += 1
            continue

        # Case: explicit glob pattern like "*.jpg"
        if _looks_like_glob(tlow):
            # If it's purely an extension glob, funnel via exts
            ext = None
            m = re.fullmatch(r"\*\.\s*([A-Za-z0-9]+)", tlow.replace(" ", ""))
            if m:
                ext = "." + m.group(1).lower()

            if ext:
                matches = _expand_in_raw(pattern=None, exts={ext}, task=task_final)
            else:
                matches = _expand_in_raw(pattern=tlow, exts=None, task=task_final)

            for p in matches:
                if p not in seen:
                    seen.add(p)
                    found.append(p)
            i += 1
            continue

        # Fallback: resolve normal single item (stem/path/url)
        resolved = _resolve_input_token(tok)
        if resolved == "":
            pass
        elif resolved is not None:
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
        i += 1

    return found

#  command 
@app.command()
def target(  # noqa: C901  CLI parsing verbosity by design
    inputs: List[str] = typer.Argument(..., metavar="INPUT [d|hm|gj]"),
    *,
    task: Optional[str] = typer.Option(None, "--task", "-t"),
    model: str = typer.Option(_DEFAULT_MODEL, "--model", "-m"),
    # heat-map tuning
    alpha: float = typer.Option(0.40, help="Heat-map blend 0-1"),
    cmap: str = typer.Option("COLORMAP_JET", help="OpenCV / Matplotlib colour-map"),
    kernel_scale: float = typer.Option(
        5.0, "--k", "-k", help="σ  area / kernel_scale (smaller  blurrier)"
    ),
    conf: float = typer.Option(0.40, help="[detect / heat-map] confidence threshold 0-1"),
    small: bool = typer.Option(False, "--small", "--fast", help="use nano models for live video"),
) -> None:
    """Batch-process images / videos with zero manual weight fiddling."""
    if not inputs:
        typer.secho("[bold red]  no inputs given", err=True)
        raise typer.Exit(2)

    token_task, positional = _extract_task(inputs)

    #  task resolution ─
    if task:
        task = task.lower()
        if task not in _VALID:
            typer.secho(f"[bold red]  invalid --task {task}", err=True)
            raise typer.Exit(2)
        if token_task != "detect" and task != token_task:
            typer.secho(
                f"[bold red]  conflicting task: flag={task!r}  token={token_task!r}",
                err=True,
            )
            raise typer.Exit(2)
        task_final = task
    else:
        task_final = token_task

    # model flag is now cosmetic  kept for compatibility
    if model.lower() not in _AVAILABLE_MODELS:
        typer.secho(f"[bold yellow] unknown --model ignored: {model}", err=True)
    model = model.lower()

    hm_kwargs: dict[str, Any] = dict(alpha=alpha, cmap=cmap, kernel_scale=kernel_scale, conf=conf)

    #  normalize/resolve (now supports ALL/globs) 
    norm_inputs = _expand_tokens(positional, task_final)

    if not norm_inputs:
        typer.secho("[bold red]  no usable inputs found", err=True)
        raise typer.Exit(2)

    #  loop over inputs ─
    for item in norm_inputs:
        low = item.lower()

        #  videos 
        if low.endswith(tuple(_VIDEO_EXTS)):
            if task_final == "geojson":
                typer.secho(f"[bold yellow] skipping video for geojson: {item}", err=True)
                continue

            weight = _require_weight(cast(Literal["detect", "heatmap"], task_final), small=small)

            if task_final == "heatmap":
                from .predict_heatmap_mp4 import main as _heat_vid
                _heat_vid(item, weights=weight, **hm_kwargs)
            else:  # detect
                from .predict_mp4 import main as _detect_vid
                _detect_vid(item, conf=conf, weights=weight)
            continue

        # ─ still images & URLs ─
        if low.endswith(tuple(_IMAGE_EXTS)) or _is_url(item):
            run_single(
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
    weight = pick_weight(task, small=small)
    if weight is None:
        typer.secho(
            f"[bold red]  no weight configured for task {task} "
            "(edit panoptes.model_registry.WEIGHT_PRIORITY)",
            err=True,
        )
        raise typer.Exit(1)
    return weight

#  entry-point glue ─
def main() -> None:  # pragma: no cover
    try:
        app()
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[bold red]Unhandled error: {exc}", err=True)
        sys.exit(1)

if __name__ == "__main__":  # pragma: no cover
    main()