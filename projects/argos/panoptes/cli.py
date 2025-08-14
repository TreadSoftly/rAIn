# C:\Users\MrDra\OneDrive\Desktop\rAIn\projects\argos\panoptes\cli.py
"""
panoptes.cli – unified Typer front-end (“argos”)

Tasks: detect | heatmap | geojson on images **and** videos.
Model selection is delegated to *panoptes.model_registry* and is strictly enforced.

Progress & UX (this build)
──────────────────────────────────────────────────────────────────────────────
• Single, pinned progress line for the whole run:
    ARGOS DETECT — [DONE i/N • xx%] current_item
  Falls back to an ANSI pinned-line writer when the progress package is absent.

• Accurate counts for ALL/globs; % done is shown live.

• At the end, prints a **clickable list** of just the basenames for outputs
  (OSC-8 hyperlinks that open the real file path in supported terminals/editors).

• While the line is active, stdout/stderr from workers is silenced unless --verbose.

• Spinner writes to sys.__stderr__ so it remains visible even when stdio is redirected.

Notes
─────
• We detect per-item outputs by diffing the results directory before/after each
  item, so we can still list results even if the task functions don’t return paths
  yet. When the task layer starts returning paths, we’ll prefer those.
"""

from __future__ import annotations

import fnmatch
import os
import re
import sys
import textwrap
import time
import urllib.parse
from contextlib import contextmanager
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Final
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Optional
from typing import Protocol
from typing import Self
from typing import cast

import typer

# ────────────────────────────────────────────────────────────────────────────
#  app scaffolding
# ────────────────────────────────────────────────────────────────────────────

_IS_WINDOWS = sys.platform.startswith("win")
_HELP_NAMES = ["-h", "--help", "-help", "-?"] + (["/?"] if _IS_WINDOWS else [])

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": _HELP_NAMES},
)

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
    # help / man
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
            argos --dry-run hm all .png

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

    uniq = {alias for _, alias in hits}
    if "help" in uniq or "man" in uniq:
        idx = next(i for i, a in hits if a in {"help", "man"})
        rest = tokens[:idx] + tokens[idx + 1 :]
        return _ALIAS[tokens[idx].lower()], rest

    uniq = {alias for _, alias in hits if alias in _VALID}
    if len(uniq) > 1:
        typer.secho(
            f"[bold red]  more than one task alias supplied "
            f"({', '.join(sorted(uniq))})",
            err=True,
        )
        raise typer.Exit(2)

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
#  OSC-8 hyperlink helper (clickable file names)
# ────────────────────────────────────────────────────────────────────────────
def _as_file_uri(p: Path) -> str:
    # Path.as_uri() handles Windows + encoding, but requires absolute path.
    return p.resolve().as_uri()

def _osc8(label: str, target_uri: str, *, yellow: bool = True) -> str:
    ESC   = "\x1b"
    BR_YE = "\x1b[93m"   # bright yellow
    BOLD  = "\x1b[1m"
    RST   = "\x1b[0m"
    colored = f"{BR_YE}{BOLD}{label}{RST}" if yellow else label
    return f"{ESC}]8;;{target_uri}{ESC}\\{colored}{ESC}]8;;{ESC}\\"

def _clickable_basename(p: Path) -> str:
    """
    In VS Code terminal, return a plain absolute path so Ctrl+Click opens.
    Else, return a bright-yellow OSC-8 link with a short label.
    """
    is_vscode = (os.environ.get("TERM_PROGRAM", "").lower() == "vscode") or bool(os.environ.get("VSCODE_PID"))
    if is_vscode:
        return str(p.resolve())  # VS Code link detector picks this up
    return _osc8(p.name, _as_file_uri(p), yellow=True)


# ────────────────────────────────────────────────────────────────────────────
#  optional progress wrapper
# ────────────────────────────────────────────────────────────────────────────
class SpinnerLike(Protocol):
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None: ...
    def update(self, **kwargs: Any) -> Self: ...

_SpinnerFactory = Callable[..., SpinnerLike]

try:
    # neutral Halo/Colorama spinner (template lives in progress_ux.py)
    from panoptes.progress import percent_spinner as _percent_spinner  # type: ignore[reportMissingTypeStubs]
    _spinner_factory: Optional[_SpinnerFactory] = cast(_SpinnerFactory, _percent_spinner)
except Exception:
    _spinner_factory = None  # type: ignore[assignment]

class _ConsoleSpinner:
    """
    Minimal pinned-line progress for when the progress package is unavailable.
    Updates a single line with:  "{prefix} — [DONE i/N • xx%] current"
    Writes to sys.__stderr__ to bypass redirected stdio.
    """
    def __init__(self, *, prefix: str, stream: Any | None = None, final_newline: bool = True) -> None:
        self.prefix = prefix
        self._stream = stream or getattr(sys, "__stderr__", sys.stdout)
        self._total = 0
        self._count = 0
        self._current: Optional[str] = None
        self._start = time.time()
        self._active = False
        self._final_newline = bool(final_newline)

    def __enter__(self) -> "_ConsoleSpinner":
        self._active = True
        self._render()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        # finalize line; optional newline so next prints are clean
        self._active = False
        self._render(final=True)
        try:
            if self._final_newline:
                self._stream.write("\n")
                self._stream.flush()
        except Exception:
            pass
        return False

    def update(self, **kwargs: Any) -> "_ConsoleSpinner":
        if "total" in kwargs:
            try:
                self._total = max(0, int(float(kwargs["total"])))
            except Exception:
                pass
        if "count" in kwargs:
            try:
                self._count = max(0, int(float(kwargs["count"])))
            except Exception:
                pass
        cur = kwargs.get("current", None)
        if isinstance(cur, str) or cur is None:
            self._current = cur
        self._render()
        return self

    def _render(self, *, final: bool = False) -> None:
        if not self._active and not final:
            return
        tot = max(0, self._total)
        done = max(0, min(self._count, tot if tot else self._count))
        pct = int(round((100.0 * done / (tot or 1)), 0)) if tot else (100 if final else 0)
        cur = f" {self._current}" if self._current else ""
        line = f"{self.prefix} — [DONE {done}/{tot or '?'} • {pct:>3}%]{cur}"
        try:
            import shutil as _sh
            width = _sh.get_terminal_size((100, 20)).columns
            self._stream.write("\r" + " " * (width - 1) + "\r")
            self._stream.write(line)
            self._stream.flush()
        except Exception:
            pass

# import after definition to avoid early import costs

@contextmanager
def _silence_stdio(enabled: bool) -> Iterator[None]:
    """
    Redirect stdout/stderr to os.devnull while active when *enabled* is True.
    This prevents child libs and workers from printing and breaking the spinner.
    """
    if not enabled:
        yield
        return
    devnull = open(os.devnull, "w", buffering=1, encoding="utf-8", errors="ignore")
    try:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield
    finally:
        try:
            devnull.close()
        except Exception:
            pass

@contextmanager
def _maybe_spinner(prefix: str, *, final_newline: bool = True) -> Iterator[SpinnerLike]:
    """
    Single pinned spinner (or a console fallback) used by the whole CLI run.

    NOTE: We do NOT set any env overrides here. The rich spinner (if present)
    marks PANOPTES_PROGRESS_ACTIVE=1 when it starts, which automatically
    silences nested spinners. This prevents flicker/scroll.
    """
    stream = getattr(sys, "__stderr__", sys.stdout)  # bypass redirected sys.stderr
    if _spinner_factory is None:
        sp: SpinnerLike = cast(SpinnerLike, _ConsoleSpinner(prefix=prefix, stream=stream, final_newline=final_newline))
    else:
        # progress.percent_spinner supports final_newline
        sp: SpinnerLike = _spinner_factory(prefix=prefix, stream=stream, final_newline=final_newline)  # type: ignore[call-arg]
    with sp:
        yield sp

# ────────────────────────────────────────────────────────────────────────────
#  lightweight wrappers for lazy imports
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
    # silence any prints during model init unless verbose
    with _silence_stdio(True):
        from . import lambda_like as _ll  # type: ignore[reportMissingTypeStubs]
        _ll.reinit_models(
            detect_small=detect_small,
            segment_small=segment_small,
            det_override=det_override,
            seg_override=seg_override,
        )

# ────────────────────────────────────────────────────────────────────────────
#  Progress proxy: map child current → JOB so ITEM stays pinned
# ────────────────────────────────────────────────────────────────────────────
class _JobAwareProxy:
    """
    Intercepts update(current=...) from child layers and treats it as JOB,
    so our ITEM stays the filename/URL on the main line.
    """
    def __init__(self, spinner: SpinnerLike) -> None:
        self._sp = spinner

    def update(self, **kw: Any) -> "_JobAwareProxy":
        cur = kw.pop("current", None)
        if cur is not None:
            txt = str(cur).strip()
            if txt:
                kw.setdefault("job", txt)
        self._sp.update(**kw)
        return self

def _run_single(
    src: str,
    *,
    model: str,
    task: Literal["detect", "heatmap", "geojson"],
    progress: Optional[SpinnerLike],
    quiet: bool,
    **kwargs: Any,
) -> Any:
    from . import lambda_like as _ll  # type: ignore[reportMissingTypeStubs]
    prox = _JobAwareProxy(progress) if progress is not None else None
    # ensure child layer never spawns a spinner; it will opportunistically
    # update the parent spinner’s fields instead (JOB via proxy).
    return _ll.run_single(
        src,
        model=model,
        task=task,
        progress=prox,
        quiet=quiet,
        **kwargs,
    )

# ────────────────────────────────────────────────────────────────────────────
#  results tracking (for clickable list)
# ────────────────────────────────────────────────────────────────────────────
def _snapshot_results() -> set[Path]:
    out: set[Path] = set()
    base = _RESULTS_DIR
    if not base.exists():
        return out
    for p in base.rglob("*"):
        if p.is_file():
            out.add(p.resolve())
    return out

# ────────────────────────────────────────────────────────────────────────────
#  command
# ────────────────────────────────────────────────────────────────────────────
@app.command()
def target(  # noqa: C901
    inputs: List[str] = typer.Argument(..., metavar="INPUT [d|hm|gj|FLAGS|help|man]"),
    *,
    # flexible: explicit task name
    task: Optional[str] = typer.Option(None, "--task", "-t", help="- task detect /or/ -t d | -t heatmap /or/ -task d | -t geojson /or/ -t -gj"),
    # convenience boolean flags for task selection
    detect_flag: bool = typer.Option(False, "--detect", "-d", help="d is shortcut for --task detect | d YourFile /or/ YourFile d"),
    heatmap_flag: bool = typer.Option(False, "--heatmap", help="hm is shortcut for --task heatmap | hm YourFile /or/ YourFile hm"),
    geojson_flag: bool = typer.Option(False, "--geojson", help="gj is shortcut for --task geojson | gj YourFile /or/ YourFile gj"),
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
    kernel_scale: float = typer.Option(5.0, "--k", "-k", help="Area / kernel_scale (smaller → blurrier)"),
    conf: float = typer.Option(0.40, help="[detect / heat-map] confidence threshold 0-1"),
    small: bool = typer.Option(False, "--small", "--fast", help="Use nano models for live video"),
    # per-task override weights
    det_override: Optional[Path] = typer.Option(
        None, "--det-weights",
        help="Force a detector weight for detect/geojson (path to .pt/.onnx).",
    ),
    seg_override: Optional[Path] = typer.Option(
        None, "--seg-weights",
        help="Force a segmentation weight for heatmap (path to .pt/.onnx).",
    ),
    # output control
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Chatty logs (disables single-line-only mode)"),
    quiet: bool = typer.Option(True, "--quiet", "-q", help="Single-line progress only (default)"),
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

    if verbose:
        quiet = False

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

    # Resolve a single final task with clear precedence
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
        if not quiet:
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

    # ── Heavy stuff starts here ──

    if not quiet:
        typer.secho(
            f"[panoptes] cli: task={task_final} small={small} inputs={len(norm_inputs)}",
            err=True,
        )

    # Short status spinner during model init (covers downloads) — do NOT add a newline
    with _maybe_spinner(prefix="ARGOS INIT", final_newline=False) as sp_init:
        sp_init.update(total=1, count=0, current="init models (may download)")
        if task_final == "detect":
            _reinit_models(detect_small=small, det_override=det_override)
        elif task_final == "heatmap":
            _reinit_models(segment_small=small, seg_override=seg_override)
        elif task_final == "geojson":
            _reinit_models(detect_small=small, det_override=det_override)
        sp_init.update(count=1, current="ready")

    # Progress wrapper (counts processed inputs) — SINGLE pinned spinner
    done = 0
    produced: list[Path] = []         # all outputs from this run
    per_input_outs: dict[str, list[Path]] = {}  # optional grouping

    prefix = task_final.upper()
    with _maybe_spinner(prefix=f"ARGOS {prefix}", final_newline=True) as sp:
        sp.update(total=len(norm_inputs), count=0, current=None)

        # loop over inputs
        for item in norm_inputs:
            low = item.lower()
            label = ("URL" if _is_url(item) else Path(item).name)
            sp.update(current=label)

            # snapshot results so we can diff per item
            before = _snapshot_results()

            # videos
            if low.endswith(tuple(_VIDEO_EXTS)):
                if task_final == "geojson":
                    sp.update(current=f"skip video: {label}")
                    done += 1
                    sp.update(count=done)
                    per_input_outs[label] = []
                    continue

                # choose weight (override beats registry pick)
                ov = det_override if task_final == "detect" else (
                    seg_override if task_final == "heatmap" else None
                )
                weight = Path(ov) if ov is not None else _pick_weight(
                    cast(Literal["detect", "heatmap"], task_final), small=small
                )

                if not quiet:
                    typer.secho(
                        f"[panoptes] video: {item} → task={task_final} small={small} weight={weight}",
                        err=True,
                    )

                # Show ITEM + JOB + MODEL on the progress line up-front
                sp.update(
                    current=label,
                    job=("heatmap" if task_final == "heatmap" else "detect"),
                    model=(Path(weight).name if weight else "")
                )

                # Silence worker output while spinner is live
                with _silence_stdio(quiet):
                    if task_final == "heatmap":
                        from .predict_heatmap_mp4 import main as _heat_vid
                        _heat_vid(item, weights=weight, **hm_kwargs)
                    else:  # detect
                        from .predict_mp4 import main as _detect_vid
                        _detect_vid(item, conf=conf, weights=weight)

                # diff results
                after = _snapshot_results()
                new_files = sorted((after - before), key=lambda p: p.name.lower())
                produced.extend(new_files)
                per_input_outs[label] = new_files

                done += 1
                sp.update(count=done)
                continue

            # still images & URLs
            if low.endswith(tuple(_IMAGE_EXTS)) or _is_url(item):
                # Choose (or show) model and put all three fields on the line up-front
                if task_final == "detect":
                    w = Path(det_override) if det_override is not None else _pick_weight("detect", small=small)
                    model_name = Path(w).name if w else ""
                    sp.update(current=label, job="detect", model=model_name)
                    if not quiet:
                        typer.secho(f"[panoptes] image/url: {item} → task=detect weight={w}", err=True)
                elif task_final == "heatmap":
                    w = Path(seg_override) if seg_override is not None else _pick_weight("heatmap", small=small)
                    model_name = Path(w).name if w else ""
                    sp.update(current=label, job="heatmap", model=model_name)
                    if not quiet:
                        typer.secho(f"[panoptes] image/url: {item} → task=heatmap weight={w}", err=True)
                else:
                    w = None
                    sp.update(current=label, job="geojson", model="")
                    if not quiet:
                        typer.secho(f"[panoptes] image/url: {item} → task=geojson (no model required)", err=True)

                with _silence_stdio(quiet):
                    result = _run_single(
                        item,
                        model=model,
                        task=cast(Literal["detect", "heatmap", "geojson"], task_final),
                        progress=sp,   # child ‘current’ updates become JOB via the proxy
                        quiet=quiet,
                        **hm_kwargs,
                    )

                # Prefer returned paths if provided; fall back to results diff.
                # Normalize to List[Path]
                outs: list[Path] = []
                try:
                    if isinstance(result, (str, Path)):
                        outs = [Path(result)]
                    elif isinstance(result, (list, tuple)):
                        iterable = cast(Iterable[object], result)
                        outs = [Path(x) for x in iterable if isinstance(x, (str, Path))]
                except Exception:
                    outs = []

                if not outs:
                    after = _snapshot_results()
                    outs = sorted((after - before), key=lambda p: p.name.lower())

                produced.extend(outs)
                per_input_outs[label] = outs

                done += 1
                sp.update(count=done)
                continue

            typer.secho(f"[bold red] unsupported input: {item}", err=True)
            raise typer.Exit(2)

    # ── Final summary with clickable file names ──────────────────────────────
    if produced:
        typer.echo("")  # spacer
        typer.echo("Results (Ctl + Click To Open):")
        for p in produced:
            try:
                typer.echo(f"  - {_clickable_basename(p)}")
            except Exception:
                typer.echo(f"  - {p.name}  ({str(p)})")
    else:
        typer.echo("")  # spacer
        typer.echo("No new result files were detected.")

# ────────────────────────────────────────────────────────────────────────────
#  entry-point glue
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
