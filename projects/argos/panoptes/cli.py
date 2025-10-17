# projects/argos/panoptes/cli.py
from __future__ import annotations

try:
    # Python 3.11+ - already present
    from typing import Self as _ArgosTypingSelf  # type: ignore
except Exception:
    try:
        from typing_extensions import Self as _ArgosTypingSelf  # type: ignore
    except Exception:
        class _ArgosTypingSelf:  # minimal placeholder
            pass
    import typing as _argos_typing_mod
    setattr(_argos_typing_mod, "Self", _ArgosTypingSelf)

import fnmatch
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import urllib.parse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Final, Iterable, Iterator, List, Literal, Optional, Protocol, cast

try:
    from panoptes.runtime.venv_bootstrap import maybe_reexec_into_managed_venv

    maybe_reexec_into_managed_venv()
except Exception:
    pass

import typer

from panoptes.logging_config import bind_context, current_run_dir, setup_logging
from .ffmpeg_utils import resolve_ffmpeg
from .support_bundle import write_support_bundle

setup_logging()
LOGGER = logging.getLogger(__name__)


def _log_event(event: str, **info: object) -> None:
    if info:
        detail = ' '.join(f"{k}={info[k]}" for k in sorted(info) if info[k] is not None)
        LOGGER.info("%s %s", event, detail)
    else:
        LOGGER.info("%s", event)

# Diagnostics (always attach; best-effort — never crash if missing)
try:
    import importlib
    importlib.import_module("panoptes.diagnostics")
except Exception:
    pass

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
    # classify
    "cls": "classify",
    "classify": "classify",
    "--classify": "classify",
    "-cls": "classify",
    # pose (keypoints)
    "pose": "pose",
    "--pose": "pose",
    "-pose": "pose",
    "kp": "pose",
    # pse (alias of pose for convenience / console-script compatibility)
    "pse": "pose",
    "--pse": "pose",
    "-pse": "pose",
    # oriented bounding boxes
    "obb": "obb",
    "--obb": "obb",
    "-obb": "obb",
    # help / man
    "help": "help",
    "man": "man",
    "?": "help",
}
_VALID = {"detect", "heatmap", "geojson", "classify", "pose", "obb"}

_URL_RE = re.compile(r"^(https?://.+|data:image/[^;]+;base64,.+)$", re.I)

_AVAILABLE_MODELS: list[str] = ["primary"]  # cosmetic placeholder
_DEFAULT_MODEL = "primary"

# Search config
_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".avif", ".bmp", ".tif", ".tiff", ".gif", ".webp", ".heic", ".heif"
}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_MEDIA_EXTS = _IMAGE_EXTS | _VIDEO_EXTS
_PREF_EXT_ORDER = (
    ".avif", ".jpg", ".jpeg", ".png",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
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

# ────────────────────────────────────────────────────────────
#  media plugin registration + fallback transcode
# ────────────────────────────────────────────────────────────
def _media_debug_enabled(verbose_flag: bool) -> bool:
    # Explicit verbose always enables logging; env var can also force it
    if verbose_flag:
        return True
    v = os.environ.get("ARGOS_MEDIA_DEBUG", "").strip().lower()
    return v not in ("", "0", "false", "no", "off")

def _log_media(msg: str, *, verbose_flag: bool) -> None:
    if _media_debug_enabled(verbose_flag):
        try:
            typer.secho(f"[media] {msg}", err=True)
        except Exception:
            pass

def _register_media_plugins(*, verbose_flag: bool) -> None:
    """
    Best-effort registration of Pillow plugins for modern formats:
      - AVIF  via pillow-avif-plugin (import side-effect)
      - HEIC/HEIF via pillow-heif (register_heif_opener)
    """
    try:
        import pillow_avif as _pillow_avif  # type: ignore
        _ = _pillow_avif  # keep referenced to silence linters
        _log_media("AVIF plugin (pillow-avif-plugin) loaded", verbose_flag=verbose_flag)
    except Exception as exc:
        _log_media(f"AVIF plugin not available: {exc!s}", verbose_flag=verbose_flag)

    try:
        import pillow_heif as _pillow_heif  # type: ignore
        try:
            _pillow_heif.register_heif_opener()  # type: ignore[attr-defined]
            _log_media("HEIF/HEIC opener registered (pillow-heif)", verbose_flag=verbose_flag)
        except Exception as exc:
            _log_media(f"pillow-heif present but register_heif_opener failed: {exc!s}", verbose_flag=verbose_flag)
    except Exception as exc:
        _log_media(f"HEIF/HEIC plugin not available: {exc!s}", verbose_flag=verbose_flag)

def _pil_can_open(path: Path) -> bool:
    try:
        from PIL import Image as _PILImage  # type: ignore
        # Simplify: treat image object as Any to avoid partially unknown 'load' return type warnings.
        with _PILImage.open(path) as img:  # type: ignore[attr-defined]
            img.load()  # type: ignore[no-untyped-call]
        return True
    except Exception:
        return False

def _maybe_transcode_to_png_if_unreadable(src: str, *, verbose_flag: bool) -> str:
    """
    If *src* is .avif/.heic/.heif and Pillow can't open it, try to transcode
    to a temporary PNG using ffmpeg. Return the path to use (original or converted).
    """
    try:
        p = Path(src)
        if not p.exists() or not p.is_file():
            return src
        ext = p.suffix.lower()
        if ext not in {".avif", ".heic", ".heif"}:
            return src

        # If Pillow already opens it, keep original
        if _pil_can_open(p):
            _log_media(f"{p.name}: Pillow can decode; no transcode", verbose_flag=verbose_flag)
            return src

        ff, ff_source = resolve_ffmpeg()
        if not ff:
            _log_media(f"{p.name}: not decodable; FFmpeg unavailable → proceed with original (may fail)", verbose_flag=verbose_flag)
            return src
        _log_media(f"{p.name}: using FFmpeg ({ff_source}) for transcoding ({ff})", verbose_flag=verbose_flag)

        cache_dir = Path(tempfile.gettempdir()) / "argos_media_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"{p.stem}.png"

        cmd = [ff, "-y", "-loglevel", "error", "-i", str(p), "-frames:v", "1", str(out_path)]
        _log_media(f"Transcoding {p.name} → {out_path.name} via FFmpeg", verbose_flag=verbose_flag)
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as cpe:
            _log_media(f"FFmpeg transcode failed: {cpe!s}", verbose_flag=verbose_flag)
            return src

        if out_path.exists() and out_path.stat().st_size > 0:
            # Verify Pillow can open the result
            if _pil_can_open(out_path):
                _log_media(f"Transcode OK: using {out_path}", verbose_flag=verbose_flag)
                return str(out_path)
            _log_media(f"Transcode produced unreadable file: {out_path}", verbose_flag=verbose_flag)

        # Fallthrough: use original
        return src
    except Exception as exc:
        _log_media(f"Transcode fallback error: {exc!s}", verbose_flag=verbose_flag)
        return src

# ────────────────────────────────────────────────────────────
#  rich manual / examples / tutorial text
# ────────────────────────────────────────────────────────────

def _join_lines(*parts: Optional[str]) -> str:
    return "\n".join((p or "").rstrip() for p in parts if p is not None)

def _man_header() -> str:
    return textwrap.dedent(f"""
    ARGOS(1) — Panoptes Vision CLI
    =================================

    NAME
        argos — detect, heatmap, geojson, classify, pose, and obb over images & videos

    SYNOPSIS
        argos [INPUT ...] [d|hm|gj|cls|pose|obb] [OPTIONS]
        argos [OPTIONS] INPUT ...

    DESCRIPTION
        A zero-fiddling front-end over Panoptes tasks:
        • detect    — object detection boxes
        • heatmap   — segmentation heat-map overlay
        • geojson   — polygon extraction to GeoJSON (images only)
        • classify  — top-K image classification overlay
        • pose      — 2D keypoints + COCO-like skeleton (K=17)
        • obb       — oriented bounding boxes (quads)

        You can put task tokens anywhere:
            argos mildrone d
            detect mildrone.avif
            d mildrone
            hm camo
            cls assets
            pose human
            pse runner.jpg     (alias of pose)
            obb all
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
            cls *.jpg
            pose all
            obb all
            argos all d
    """).strip("\n")

def _man_tasks() -> str:
    return textwrap.dedent("""
    TASKS
        detect    Boxes over images or videos.
        heatmap   Segmentation heat-map overlay (images & videos).
        geojson   Extract polygons from images as GeoJSON (videos are skipped).
        classify  Top-K label card on images or per-frame in videos.
        pose      Draw keypoints and (if K==17) COCO-like skeleton.
        pse       Alias of pose (kept for entrypoint compatibility).
        obb       Draw oriented bounding boxes (quads), fallback to AABB.
    """).strip("\n")

def _man_inputs() -> str:
    return textwrap.dedent(f"""
    INPUTS
        • Plain filenames or stems (we search known dirs)
        • URLs (http/https or data:image/* base64) — URL mode currently supported
        for detect/heatmap/geojson; classify/pose/obb expect local files.
        • Globs: "*.jpg", "*"
        • ALL:   "all" (optionally followed by ".ext"/"ext"/"*.ext")

        When using globs or "all", the search happens in:
            {_RAW_DIR}
    """).strip("\n")

def _man_models() -> str:
    return textwrap.dedent("""
    MODEL SELECTION
        Weights are picked by the model registry strictly (no silent fallbacks).
        Use --small/--fast to prefer lightweight models for live video (detect/heatmap).
        Per-task override:
        --det-weights  PATH   (detect/geojson)
        --seg-weights  PATH   (heatmap)
        --cls-weights  PATH   (classify)
        --pose-weights PATH   (pose)
        --obb-weights  PATH   (obb)

        If a required weight is missing, the command exits with code 1.
    """).strip("\n")

def _man_tuning() -> str:
    return textwrap.dedent("""
    TUNING
        --conf FLOAT            confidence threshold 0-1 (default 0.40; used by detect/heatmap/pose/obb/classify*)
        --iou  FLOAT            IoU threshold (default 0.45; used by pose/obb where applicable)
        --alpha FLOAT           heat-map blend 0-1 (default 0.40)
        --cmap NAME             OpenCV/Matplotlib colour-map (default COLORMAP_JET)
        -k / --k FLOAT          σ area / kernel_scale (smaller → blurrier; heatmap)
        --topk INT              top-K labels for classify (default 1)
        --small / --fast        use nano models for live video (detect/heatmap)
    """).strip("\n")

def _man_outputs() -> str:
    return textwrap.dedent(f"""
    OUTPUTS
        Results are written under:
            {_RESULTS_DIR}
        (File names mirror the input stem with task-specific suffixes.)
        You can optionally override the output directory for image overlays with:
            --out-dir PATH
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
        gj        -> prepends "gj"       e.g., `gj assets`
        cls       -> prepends "cls"      e.g., `cls ocean.jpg`
        pose      -> prepends "pose"     e.g., `pose robodog.png`
        pse       -> prepends "pose"     e.g., `pse runner.jpg`   (alias of pose)
        obb       -> prepends "obb"      e.g., `obb subdrones.png`
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
            argos detect assets.jpg

        Heat-map overlay:
            argos hm camo
            argos --heatmap gerbera-drones.png

        GeoJSON (images only):
            argos gj assets
            argos geojson assets.jpg

        Classification:
            argos cls assets.jpg
            argos classify all .png --topk 3

        Pose (keypoints + skeleton):
            argos pose midtown.jpg
            argos pse runner.jpg
            argos pose all --conf 0.30 --iou 0.45

        Oriented Bounding Boxes:
            argos obb ship.png
            argos obb all .jpg

        Batch from test set:
            argos d all
            argos hm *.png
            argos gj all .jpg
            argos cls all .jpg
            argos pose all
            argos obb all

        Video:
            argos d bunny.mp4
            argos hm bunny.mp4 --small
            argos cls bunny.mp4
            argos pose bunny.mp4
            argos obb bunny.mp4

        Explain without running (dry-run):
            argos --dry-run hm all .png

        Force weights:
            argos d assets --det-weights projects/argos/panoptes/model/yolov8x.pt
            argos hm camo --seg-weights projects/argos/panoptes/model/yolov8x-seg.pt
            argos cls assets --cls-weights weights/classify.pt
            argos pose runner --pose-weights weights/pose.pt
            argos obb ship --obb-weights weights/obb.pt
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
        "  argos gj assets",
        "  argos cls assets.jpg",
        "  argos pose runner.jpg",
        "  argos obb ship.png",
        "  argos d bunny.mp4",
        "",
        "Globs / ALL:",
        "  argos d all",
        "  argos hm *.png",
        "  argos gj all .jpg",
        "  argos cls all .jpg",
        "  argos pose all",
        "  argos obb all .png",
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
        "   argos gj assets",
        "",
        "4) Classify an image",
        "   argos cls assets.jpg --topk 3",
        "",
        "5) Pose estimation",
        "   argos pose runner.jpg",
        "",
        "6) Oriented boxes",
        "   argos obb ship.png",
        "",
        "7) Batch a whole folder",
        "   argos hm all .jpg --small",
        "",
        "8) Video in nano mode",
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

# ────────────────────────────────────────────────────────────
#  helper utils (resolution)
# ────────────────────────────────────────────────────────────
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
            f"[bold red]  more than one task alias supplied ({', '.join(sorted(uniq))})",
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
#  progress (Halo/Rich only — hard-required)
# ────────────────────────────────────────────────────────────────────────────
class SpinnerLike(Protocol):
    def __enter__(self) -> "SpinnerLike": ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None: ...
    def update(self, **kwargs: Any) -> "SpinnerLike": ...

_SpinnerFactory = Callable[..., SpinnerLike]

# Hard-require our Halo/Rich spinner from the local progress package.
# No console/text fallback exists anymore.
try:
    from .progress import percent_spinner as _percent_spinner
    _spinner_factory: Optional[_SpinnerFactory] = cast(_SpinnerFactory, _percent_spinner)
except Exception:
    _spinner_factory = None  # type: ignore[assignment]

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
    Open the ONE Halo/Rich spinner for the whole CLI scope.
    If the progress package is unavailable, abort with a clear error.
    While active, set PANOPTES_PROGRESS_ACTIVE=1 so children never spawn their own spinners.
    """
    if _spinner_factory is None:
        typer.secho(
            "[bold red]Halo/Rich progress package is required but unavailable.\n"
            "Install the project with progress extras and retry.",
            err=True,
        )
        raise typer.Exit(2)

    stream = getattr(sys, "__stderr__", sys.stdout)  # keep progress visible even with stdout redirection
    prev_tail = os.environ.get("PANOPTES_PROGRESS_TAIL")
    if prev_tail is None or prev_tail.strip().lower() == "none":
        os.environ["PANOPTES_PROGRESS_TAIL"] = "full"
    sp: SpinnerLike = _spinner_factory(prefix=prefix, stream=stream, final_newline=final_newline)  # type: ignore[call-arg]

    # Guard against nested/local spinners
    prev_env = os.environ.get("PANOPTES_PROGRESS_ACTIVE", None)
    os.environ["PANOPTES_PROGRESS_ACTIVE"] = "1"
    try:
        with sp:
            yield sp
    finally:
        if prev_env is None:
            os.environ.pop("PANOPTES_PROGRESS_ACTIVE", None)
        else:
            os.environ["PANOPTES_PROGRESS_ACTIVE"] = prev_env
        if prev_tail is None:
            os.environ.pop("PANOPTES_PROGRESS_TAIL", None)
        else:
            os.environ["PANOPTES_PROGRESS_TAIL"] = prev_tail

# ────────────────────────────────────────────────────────────
#  lightweight wrappers for lazy imports
# ────────────────────────────────────────────────────────────
def _pick_weight(task: Literal["detect", "heatmap"], *, small: bool):
    from panoptes import model_registry as _mr  # type: ignore[reportMissingTypeStubs]
    return _mr.pick_weight(task, small=small)

def _load_classifier(override: Optional[Path] = None):
    from panoptes import model_registry as _mr  # type: ignore[reportMissingTypeStubs]
    return _mr.load_classifier(override=override)

def _load_pose(override: Optional[Path] = None):
    from panoptes import model_registry as _mr  # type: ignore[reportMissingTypeStubs]
    return _mr.load_pose(override=override)

def _load_obb(override: Optional[Path] = None):
    from panoptes import model_registry as _mr  # type: ignore[reportMissingTypeStubs]
    return _mr.load_obb(override=override)

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

# ────────────────────────────────────────────────────────────
#  Progress proxy: map child current → JOB so ITEM stays pinned
# ────────────────────────────────────────────────────────────
class _JobAwareProxy:
    """
    Intercepts update(current=...) from child layers and treats it as JOB,
    so our ITEM stays the filename/URL on the main line.

    Also normalizes common alias fields used by older or third-party workers:
      - {phase, step, stage, status, task}   → job
      - {N, total_frames, frames_total}      → total
      - {i, frame, frames_done, progress}    → count
      - {file, path}                         → item
      - {weight, weights, model_name}        → model
    """
    _JOB_KEYS   = ("job", "phase", "step", "stage", "status", "task")
    _TOT_KEYS   = ("total", "N", "total_frames", "frames_total", "progress_total")
    _CNT_KEYS   = ("count", "i", "frame", "frames_done", "progress_count")
    _ITEM_KEYS  = ("item", "file", "path")
    _MODEL_KEYS = ("model", "weight", "weights", "model_name")

    def __init__(self, spinner: SpinnerLike) -> None:
        self._sp = spinner

    @staticmethod
    def _pop_int(kw: dict[str, Any], key: str) -> int | None:
        if key not in kw:
            return None
        try:
            return int(kw.pop(key))
        except Exception:
            kw.pop(key, None)
            return None

    def update(self, **kw: Any) -> "_JobAwareProxy":
        # Map legacy 'current' to 'job'
        cur = kw.pop("current", None)
        if cur is not None:
            txt = str(cur).strip()
            if txt:
                kw.setdefault("job", txt)

        # Normalize synonyms → canonical keys
        # job
        if not any(k in kw for k in ("job",)):
            for k in self._JOB_KEYS:
                if k in kw and isinstance(kw[k], (str, int, float)):
                    kw["job"] = str(kw.pop(k))
                    break
        # total
        if "total" not in kw:
            for k in self._TOT_KEYS:
                if k in kw:
                    try:
                        kw["total"] = int(kw.pop(k))
                    except Exception:
                        kw.pop(k, None)
                    break
        # count
        if "count" not in kw:
            for k in self._CNT_KEYS:
                if k in kw:
                    try:
                        kw["count"] = int(kw.pop(k))
                    except Exception:
                        kw.pop(k, None)
                    break
        # item
        if "item" not in kw:
            for k in self._ITEM_KEYS:
                if k in kw and isinstance(kw[k], (str, Path)):
                    kw["item"] = str(kw.pop(k))
                    break
        # model
        if "model" not in kw:
            for k in self._MODEL_KEYS:
                if k in kw and isinstance(kw[k], (str, Path)):
                    val = kw.pop(k)
                    kw["model"] = Path(val).name if k != "model" else str(val)
                    break

        nested_total = self._pop_int(kw, "total")
        nested_count = self._pop_int(kw, "count")
        if nested_total is None:
            nested_total = self._pop_int(kw, "progress_total")
        if nested_count is None:
            nested_count = self._pop_int(kw, "progress_count")

        if (nested_total is not None) or (nested_count is not None):
            job_val = kw.get("job")
            if isinstance(job_val, str) and job_val:
                if (nested_total is not None) and (nested_count is not None) and ("/" not in job_val):
                    kw["job"] = f"{job_val} ({nested_count}/{nested_total})"
            elif (nested_total is not None) or (nested_count is not None):
                left = "?" if nested_count is None else str(nested_count)
                right = "?" if nested_total is None else str(nested_total)
                kw["job"] = f"{left}/{right}"

        if not kw:
            return self

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

# ────────────────────────────────────────────────────────────
#  results tracking (for clickable file names)
# ────────────────────────────────────────────────────────────
def _snapshot_results(bases: Optional[list[Path]] = None) -> set[Path]:
    """
    Snapshot all files beneath the given base directories (defaults to _RESULTS_DIR).
    """
    base_dirs = bases or [_RESULTS_DIR]
    out: set[Path] = set()
    for base in base_dirs:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                out.add(p.resolve())
    return out

# ────────────────────────────────────────────────────────────
#  command
# ────────────────────────────────────────────────────────────
@app.command()
def target(  # noqa: C901
    inputs: List[str] = typer.Argument(..., metavar="INPUT [d|hm|gj|cls|pose|obb|FLAGS|help|man]"),
    *,
    # flexible: explicit task name
    task: Optional[str] = typer.Option(None, "--task", "-t", help="- task detect | heatmap | geojson | classify | pose | obb"),
    # convenience boolean flags for task selection
    detect_flag: bool = typer.Option(False, "--detect", "-d", help="d is shortcut for --task detect | d is also a positional token"),
    heatmap_flag: bool = typer.Option(False, "--heatmap", help="hm is shortcut for --task heatmap | hm is also a positional token"),
    geojson_flag: bool = typer.Option(False, "--geojson", help="gj is shortcut for --task geojson | gj is also a positional token"),
    classify_flag: bool = typer.Option(False, "--classify", "--cls", help="cls is shortcut for --task classify | cls is also a positional token"),
    pose_flag: bool = typer.Option(False, "--pose", help="pose is shortcut for --task pose | pose is also a positional token"),
    obb_flag: bool = typer.Option(False, "--obb", help="obb is shortcut for --task obb | obb is also a positional token"),
    # meta/help UX
    man_flag: bool = typer.Option(False, "--man", help="Open the full manual and exit"),
    examples_flag: bool = typer.Option(False, "--examples", help="Show examples and exit"),
    tutorial_flag: bool = typer.Option(False, "--tutorial", help="Show a short tutorial and exit"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Explain what would run, then exit"),
    # model (cosmetic placeholder for compatibility)
    model: str = typer.Option(_DEFAULT_MODEL, "--model", "-m"),
    # heat-map + general tuning
    alpha: float = typer.Option(0.40, help="Heat-map blend 0-1"),
    cmap: str = typer.Option("COLORMAP_JET", help="OpenCV / Matplotlib colour-map"),
    kernel_scale: float = typer.Option(5.0, "--k", "-k", help="Area / kernel_scale (smaller → blurrier)"),
    conf: float = typer.Option(0.40, help="[detect/heatmap/pose/obb/classify] confidence threshold 0-1"),
    iou: float = typer.Option(0.45, help="[pose/obb] IoU threshold 0-1"),
    topk: int = typer.Option(1, "--topk", help="[classify] number of top labels to show per image/frame"),
    small: bool = typer.Option(False, "--small", "--fast", help="Use nano models for live video (detect/heatmap)"),
    # per-task override weights
    det_override: Optional[Path] = typer.Option(
        None, "--det-weights",
        help="Force a detector weight for detect/geojson (path to .pt/.onnx).",
    ),
    seg_override: Optional[Path] = typer.Option(
        None, "--seg-weights",
        help="Force a segmentation weight for heatmap (path to .pt/.onnx).",
    ),
    cls_override: Optional[Path] = typer.Option(
        None, "--cls-weights",
        help="Force a classifier weight for classify (path to .pt/.onnx).",
    ),
    pose_override: Optional[Path] = typer.Option(
        None, "--pose-weights",
        help="Force a pose weight for pose (path to .pt/.onnx).",
    ),
    obb_override: Optional[Path] = typer.Option(
        None, "--obb-weights",
        help="Force an OBB weight for obb (path to .pt/.onnx).",
    ),
    # output control
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Write image overlay outputs to this directory (defaults to tests/results)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Chatty logs (disables single-line-only mode)"),
    quiet: bool = typer.Option(True, "--quiet", "-q", help="Single-line progress only (default)"),
    support_bundle: bool = typer.Option(False, "--support-bundle", help="Write a support bundle zip for this run"),
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
        (classify_flag, "classify"),
        (pose_flag, "pose"),
        (obb_flag, "obb"),
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
                f"[bold red]  conflicting task: flag={flag_tasks[0]!r} --task={task!r}",
                err=True,
            )
            raise typer.Exit(2)
        if token_task is not None and token_task != task:
            typer.secho(
                f"[bold red]  conflicting task: token={token_task!r} --task={task!r}",
                err=True,
            )
            raise typer.Exit(2)
        task_final = task
    elif flag_tasks:
        if token_task is not None and token_task != flag_tasks[0]:
            typer.secho(
                f"[bold red]  conflicting task: token={token_task!r} flag={flag_tasks[0]!r}",
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
    cls_kwargs: dict[str, Any] = dict(conf=conf, topk=int(max(1, topk)))
    pose_kwargs: dict[str, Any] = dict(conf=conf, iou=iou)
    obb_kwargs: dict[str, Any] = dict(conf=conf, iou=iou)

    # normalize/resolve (supports ALL/globs)
    norm_inputs = _expand_tokens(positional, task_final)
    with bind_context(cli_task=task_final, mode="offline", small=small, dry_run=dry_run or None, out_dir=(str(out_dir) if out_dir else None)):
        _log_event("cli.run.start", task=task_final, inputs=len(norm_inputs), small=small, dry_run=dry_run or None, out_dir=(str(out_dir) if out_dir else None))

    if not norm_inputs:
        typer.secho("[bold red]  no usable inputs found", err=True)
        raise typer.Exit(2)

    # Register media plugins early (so downstream loaders see them)
    _register_media_plugins(verbose_flag=not quiet)

    # Dry run: explain what would happen and exit
    if dry_run:
        typer.secho(
            f"[panoptes] dry-run: task={task_final} small={small} inputs={len(norm_inputs)}",
            err=True,
        )
        to_show = norm_inputs if len(norm_inputs) <= 20 or small else norm_inputs[:20] + ["..."]
        for item in to_show:
            if item.lower().endswith(tuple(_VIDEO_EXTS)):
                typer.echo(f"  video:   {item}")
            elif _is_url(item):
                typer.echo(f"  url:     {item}")
            else:
                typer.echo(f"  image:   {item}")
        with bind_context(cli_task=task_final, mode="offline", small=small, dry_run=True):
            _log_event("cli.run.dry", task=task_final, inputs=len(norm_inputs), sample=",".join(to_show[:3]))
        raise typer.Exit(0)

    # ── Heavy stuff starts here ──

    if not quiet:
        typer.secho(
            f"[panoptes] cli: task={task_final} small={small} inputs={len(norm_inputs)}",
            err=True,
        )

    # Prepare output base directories for snapshots
    snapshot_bases: list[Path] = sorted({ _RESULTS_DIR, (out_dir or _RESULTS_DIR) }, key=lambda p: str(p))

    # Short status spinner during model init (covers downloads) — do NOT add a newline
    model_cls = model_pose = model_obb = None
    with _maybe_spinner(prefix="ARGOS INIT", final_newline=False) as sp_init:
        sp_init.update(total=1, count=0, job="init models (may download)")
        if task_final == "detect":
            _reinit_models(detect_small=small, det_override=det_override)
        elif task_final == "heatmap":
            _reinit_models(segment_small=small, seg_override=seg_override)
        elif task_final == "geojson":
            _reinit_models(detect_small=small, det_override=det_override)
        elif task_final == "classify":
            model_cls = _load_classifier(override=cls_override)
        elif task_final == "pose":
            model_pose = _load_pose(override=pose_override)
        elif task_final == "obb":
            model_obb = _load_obb(override=obb_override)
        sp_init.update(count=1, job="ready")

    # Progress wrapper (counts processed inputs) — SINGLE pinned spinner
    done = 0
    produced: list[Path] = []         # all outputs from this run
    per_input_outs: dict[str, list[Path]] = {}  # optional grouping
    printed_json = False              # if True, suppress trailing summary on stdout
    bundle_path: Optional[Path] = None

    prefix = task_final.upper()
    with _maybe_spinner(prefix=f"ARGOS {prefix}", final_newline=True) as sp:
        sp.update(total=len(norm_inputs), count=0)

        # loop over inputs
        for item in norm_inputs:
            low = item.lower()
            label = ("URL" if _is_url(item) else Path(item).name)
            media_kind = "video" if low.endswith(tuple(_VIDEO_EXTS)) else ("url" if _is_url(item) else "image")
            with bind_context(cli_task=task_final, input=item, kind=media_kind):
                _log_event("cli.item.start", input=item, kind=media_kind)
            sp.update(item=label)  # pin FILE explicitly so child 'current' can become JOB via proxy

            # snapshot results so we can diff per item (cover both default and custom out dirs)
            before = _snapshot_results(snapshot_bases)

            # videos
            if low.endswith(tuple(_VIDEO_EXTS)):
                if task_final == "geojson":
                    sp.update(item=label, job="skip video")
                    done += 1
                    sp.update(count=done)
                    per_input_outs[label] = []
                    continue

                # choose weight override for video workers (override beats registry pick)
                if task_final == "heatmap":
                    weight = Path(seg_override) if seg_override is not None else _pick_weight("heatmap", small=small)
                elif task_final == "detect":
                    weight = Path(det_override) if det_override is not None else _pick_weight("detect", small=small)
                elif task_final == "classify":
                    weight = Path(cls_override) if cls_override is not None else None  # worker will use registry if None
                elif task_final == "pose":
                    weight = Path(pose_override) if pose_override is not None else None
                elif task_final == "obb":
                    weight = Path(obb_override) if obb_override is not None else None
                else:
                    weight = None

                with bind_context(cli_task=task_final, input=item, kind="video", weight=(str(weight) if weight else None)):
                    _log_event("cli.video.config", input=item, weight=(str(weight) if weight else None), small=small, task=task_final)

                if not quiet:
                    typer.secho(
                        f"[panoptes] video: {item} → task={task_final} "
                        f"{'small='+str(small)+' ' if task_final in {'detect','heatmap'} else ''}"
                        f"weight={weight}",
                        err=True,
                    )

                # Show ITEM + JOB + MODEL on the progress line up-front
                sp.update(
                    item=label,
                    job=task_final,
                    model=(Path(weight).name if weight else "")
                )

                # Silence worker output while spinner is live; always route child updates through proxy
                with _silence_stdio(quiet):
                    if task_final == "heatmap":
                        from .predict_heatmap_mp4 import main as _heat_vid
                        try:
                            _heat_vid(item, weights=weight, progress=_JobAwareProxy(sp), **hm_kwargs)
                        except TypeError:
                            _heat_vid(item, weights=weight, **hm_kwargs)
                    elif task_final == "detect":
                        from .predict_mp4 import main as _detect_vid
                        try:
                            _detect_vid(item, conf=conf, weights=weight, progress=_JobAwareProxy(sp))
                        except TypeError:
                            _detect_vid(item, conf=conf, weights=weight)
                    elif task_final == "classify":
                        from .predict_classify_mp4 import main as _cls_vid
                        try:
                            _cls_vid(item, weights=weight, topk=int(max(1, topk)), conf=conf,
                                     progress=_JobAwareProxy(sp))
                        except TypeError:
                            _cls_vid(item, weights=weight, topk=int(max(1, topk)), conf=conf)
                    elif task_final == "pose":
                        from .predict_pose_mp4 import main as _pose_vid
                        try:
                            _pose_vid(item, weights=weight, conf=conf, iou=iou,
                                      progress=_JobAwareProxy(sp))
                        except TypeError:
                            _pose_vid(item, weights=weight, conf=conf, iou=iou)
                    elif task_final == "obb":
                        from .predict_obb_mp4 import main as _obb_vid
                        try:
                            _obb_vid(item, weights=weight, conf=conf, iou=iou,
                                     progress=_JobAwareProxy(sp))
                        except TypeError:
                            _obb_vid(item, weights=weight, conf=conf, iou=iou)

                # diff results
                after = _snapshot_results(snapshot_bases)
                new_files = sorted((after - before), key=lambda p: p.name.lower())
                produced.extend(new_files)
                per_input_outs[label] = new_files

                done += 1
                sp.update(count=done)
                continue

            # still images & URLs
            if low.endswith(tuple(_IMAGE_EXTS)) or _is_url(item):
                # URL support is limited to detect/heatmap/geojson in this build.
                if _is_url(item) and task_final in {"classify", "pose", "obb"}:
                    typer.secho(f"[bold red] URL inputs currently unsupported for task {task_final}; download the file first.", err=True)
                    raise typer.Exit(2)

                # If local file is AVIF/HEIC/HEIF and unreadable, transcode via FFmpeg to PNG in temp
                item_to_run = item
                if not _is_url(item):
                    try:
                        src_path = Path(item)
                        if src_path.suffix.lower() in {".avif", ".heic", ".heif"}:
                            item_to_run = _maybe_transcode_to_png_if_unreadable(item, verbose_flag=not quiet)
                    except Exception:
                        # Best-effort only; fall back to original on any error
                        item_to_run = item

                # Choose (or show) model and put fields on the line up-front
                if task_final == "detect":
                    w = Path(det_override) if det_override is not None else _pick_weight("detect", small=small)
                    model_name = Path(w).name if w else ""
                    sp.update(item=label, job="detect", model=model_name)
                    if not quiet:
                        typer.secho(f"[panoptes] image/url: {item} → task=detect weight={w}", err=True)
                elif task_final == "heatmap":
                    w = Path(seg_override) if seg_override is not None else _pick_weight("heatmap", small=small)
                    model_name = Path(w).name if w else ""
                    sp.update(item=label, job="heatmap", model=model_name)
                    if not quiet:
                        typer.secho(f"[panoptes] image/url: {item} → task=heatmap weight={w}", err=True)
                elif task_final == "classify":
                    sp.update(item=label, job="classify", model="")
                    if not quiet:
                        typer.secho(f"[panoptes] image: {item} → task=classify", err=True)
                elif task_final == "pose":
                    sp.update(item=label, job="pose", model="")
                    if not quiet:
                        typer.secho(f"[panoptes] image: {item} → task=pose", err=True)
                elif task_final == "obb":
                    sp.update(item=label, job="obb", model="")
                    if not quiet:
                        typer.secho(f"[panoptes] image: {item} → task=obb", err=True)
                else:
                    # geojson (image or URL) — uses the detector backbone underneath
                    w = Path(det_override) if det_override is not None else _pick_weight("detect", small=small)
                    model_name = Path(w).name if w else ""
                    sp.update(item=label, job="geojson", model=model_name)
                    if not quiet:
                        typer.secho(f"[panoptes] image/url: {item} → task=geojson weight={w}", err=True)

                # For geojson + remote URL, do NOT silence stdout — the worker prints JSON to stdout.
                suppress_stdio = quiet
                if task_final == "geojson" and _is_url(item):
                    suppress_stdio = False

                # Execute task
                with _silence_stdio(suppress_stdio):
                    result = None
                    if task_final in {"detect", "heatmap", "geojson"}:
                        result = _run_single(
                            item_to_run,
                            model=model,
                            task=cast(Literal["detect", "heatmap", "geojson"], task_final),
                            progress=sp,   # child ‘current’ updates become JOB via the proxy
                            quiet=quiet,
                            **hm_kwargs,
                        )
                    elif task_final == "classify":
                        # Strict model via registry (preloaded) with explicit out_dir
                        from . import classify as _cls_mod  # type: ignore
                        result = _cls_mod.run_image(
                            item_to_run, out_dir=(out_dir or _RESULTS_DIR),
                            model=model_cls, progress=_JobAwareProxy(sp), **cls_kwargs
                        )
                    elif task_final == "pose":
                        from . import pose as _pose_mod  # type: ignore
                        result = _pose_mod.run_image(
                            item_to_run, out_dir=(out_dir or _RESULTS_DIR),
                            model=model_pose, progress=_JobAwareProxy(sp), **pose_kwargs
                        )
                    elif task_final == "obb":
                        from . import obb as _obb_mod  # type: ignore
                        result = _obb_mod.run_image(
                            item_to_run, out_dir=(out_dir or _RESULTS_DIR),
                            model=model_obb, progress=_JobAwareProxy(sp), **obb_kwargs
                        )

                # Mark that JSON was printed for remote-URL geojson so we don't add any trailing text.
                if task_final == "geojson" and _is_url(item):
                    printed_json = True

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
                    after = _snapshot_results(snapshot_bases)
                    outs = sorted((after - before), key=lambda p: p.name.lower())

                produced.extend(outs)
                per_input_outs[label] = outs
                with bind_context(cli_task=task_final, input=item):
                    _log_event("cli.item.outputs", input=item, count=len(outs), outputs=",".join(p.name for p in outs) if outs else None)

                done += 1
                sp.update(count=done)
                continue

            typer.secho(f"[bold red] unsupported input: {item}", err=True)
            raise typer.Exit(2)

        with bind_context(cli_task=task_final, mode="offline", small=small, dry_run=dry_run or None, out_dir=(str(out_dir) if out_dir else None)):
            _log_event("cli.run.complete", task=task_final, produced=len(produced), inputs=len(norm_inputs))

    # ── Final summary with clickable file names ──────────────────────────────
    # If we already printed GeoJSON to stdout (URL case), do NOT print any summary text.
    if not printed_json:
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

    if support_bundle and bundle_path is None:
        try:
            bundle_path = write_support_bundle(extra_paths=produced)
        except Exception as exc:
            typer.secho(f"[bold red] failed to create support bundle: {exc}", err=True)
    if bundle_path:
        typer.echo(f"Support bundle written: {bundle_path}")

# ────────────────────────────────────────────────────────────
#  entry-point glue
# ────────────────────────────────────────────────────────────
def _prepend_argv(token: str) -> None:
    sys.argv = sys.argv[:1] + [token] + sys.argv[1:]

def main() -> None:  # pragma: no cover
    try:
        app()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("cli.run.error", exc_info=True)
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

def main_classify() -> None:  # pragma: no cover
    _prepend_argv("cls")
    main()

def main_pose() -> None:  # pragma: no cover
    _prepend_argv("pose")
    main()

def main_pse() -> None:  # pragma: no cover
    # Alias of pose — added to satisfy `pse.exe` console-script.
    _prepend_argv("pose")
    main()

def main_obb() -> None:  # pragma: no cover
    _prepend_argv("obb")
    main()

def main_all() -> None:  # pragma: no cover
    _prepend_argv("all")
    main()

if __name__ == "__main__":  # pragma: no cover
    main()
