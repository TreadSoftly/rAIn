"""
dronevision.cli – one unified front-end (“target”)

* Detect / heat-map / geojson for images **and** videos.
* Weight selection (including defaults) is driven entirely by the central
  ``WEIGHT_PRIORITY`` tables and the files present in *projects/drone-vision/model/*.
"""
from __future__ import annotations

import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, cast

import typer

from dronevision import MODEL_DIR, MODELS, WEIGHT_PRIORITY
from .lambda_like import run_single

# ─────────────────────────── scaffolding ────────────────────────────────
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

_AVAILABLE_MODELS = [m.lower() for m in cast(Mapping[str, Any], MODELS).keys()]
_DEFAULT_MODEL = _AVAILABLE_MODELS[0] if _AVAILABLE_MODELS else "primary"


def _is_url(text: str) -> bool:
    return urllib.parse.urlparse(text).scheme in {"http", "https"} or bool(_URL_RE.match(text))


def _extract_task(tokens: List[str]) -> tuple[str, List[str]]:
    """Return *(task, remaining_tokens)* — defaults to *detect* if none found."""
    hits = [(i, _ALIAS[t.lower()]) for i, t in enumerate(tokens) if t.lower() in _ALIAS]
    if not hits:
        return "detect", tokens
    if len(hits) > 1:
        typer.secho(
            f"[bold red]✖  more than one task alias supplied "
            f"({', '.join(a for _, a in hits)})",
            err=True,
        )
        raise typer.Exit(2)
    idx, task = hits[0]
    return task, tokens[:idx] + tokens[idx + 1:]


def _first_existing(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def _pick_weights(task: Literal["detect", "heatmap"], *, small: bool) -> Path | None:
    """
    First existing weight file according to `WEIGHT_PRIORITY`.
    Heat-maps favour segmentation → detection fallback.
    """
    key = f"{task}_small" if small else task
    cand: list[Path] = list(WEIGHT_PRIORITY.get(key, []))

    if task == "heatmap":
        # Any *-seg.* file lying around also counts
        for d in {MODEL_DIR, MODEL_DIR / "../lambda/model"}:
            cand += sorted(d.glob("*-seg.pt")) + sorted(d.glob("*-seg.onnx"))
        # If still nothing, fall back to detector list
        det_key = "detect_small" if small else "detect"
        cand += WEIGHT_PRIORITY.get(det_key, [])

    return _first_existing(cand)


# ───────────────────────────── command ───────────────────────────────────
@app.command()
def target(  # noqa: C901 – CLI parsing is inherently verbose
    inputs: List[str] = typer.Argument(..., metavar="INPUT… [d|hm|gj]"),
    *,
    task: Optional[str] = typer.Option(None, "--task", "-t"),
    model: str = typer.Option(_DEFAULT_MODEL, "--model", "-m"),
    # heat-map tuning
    alpha: float = typer.Option(0.40, help="Heat-map blend 0-1"),
    cmap: str = typer.Option("COLORMAP_JET", help="OpenCV / Matplotlib colour-map"),
    kernel_scale: float = typer.Option(
        5.0, "--k", "-k", help="σ ∝ √area / kernel_scale (smaller → blurrier)"
    ),
    conf: float = typer.Option(0.40, help="[detect / heat-map] confidence threshold 0-1"),
    small: bool = typer.Option(False, "--small", "--fast", help="use nano models for live video"),
) -> None:
    """Batch-process images and videos with zero manual weight fiddling."""
    if not inputs:
        typer.secho("[bold red]✖  no inputs given", err=True)
        raise typer.Exit(2)

    token_task, positional = _extract_task(inputs)

    # ── final task resolution ────────────────────────────────────────────
    if task:
        task = task.lower()
        if task not in _VALID:
            typer.secho(f"[bold red]✖  invalid --task {task}", err=True)
            raise typer.Exit(2)
        if token_task != "detect" and task != token_task:
            typer.secho(
                f"[bold red]✖  conflicting task: flag={task!r}  token={token_task!r}",
                err=True,
            )
            raise typer.Exit(2)
        task_final = task
    else:
        task_final = token_task

    # validate image-mode model name
    model = model.lower()
    if model not in _AVAILABLE_MODELS:
        typer.secho(
            f"[bold red]✖  unknown model {model!r} "
            f"(available: {', '.join(_AVAILABLE_MODELS)})",
            err=True,
        )
        raise typer.Exit(2)

    hm_kwargs: dict[str, Any] = dict(alpha=alpha, cmap=cmap, kernel_scale=kernel_scale, conf=conf)

    # ───────────────────── loop over user inputs ────────────────────────
    for item in positional:
        low = item.lower()

        # ── videos ───────────────────────────────────────────────────────
        if low.endswith((".mp4", ".mov", ".avi", ".mkv")):
            if task_final == "geojson":
                typer.secho(f"[bold yellow]⚠ skipping video for geojson: {item}", err=True)
                continue

            weight = _pick_weights(cast(Literal["detect", "heatmap"], task_final), small=small)

            if task_final == "heatmap":
                from .predict_heatmap_mp4 import main as _heat_vid
                _heat_vid(item, weights=weight, **hm_kwargs)
            else:  # detect
                from .predict_mp4 import main as _detect_vid
                _detect_vid(item, conf=conf, weights=weight)
            continue

        # ── images / URLs ────────────────────────────────────────────────
        if low.endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".avif",
                ".bmp",
                ".tif",
                ".tiff",
                ".gif",
                ".webp",
                ".heic",
                ".heif",
            )
        ) or _is_url(item):
            run_single(
                item,
                model=model,                      # ← no legacy literals
                task=cast(Literal["detect", "heatmap", "geojson"], task_final),
                **hm_kwargs,
            )
            continue

        typer.secho(f"[bold red]✖ unsupported input: {item}", err=True)
        raise typer.Exit(2)


# ────────────────────────── entry-point glue ─────────────────────────────
def main() -> None:  # pragma: no cover
    try:
        app()
    except Exception as exc:  # pylint: disable=broad-except
        typer.echo(f"[bold red]Unhandled error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
