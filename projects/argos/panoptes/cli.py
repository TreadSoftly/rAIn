"""
panoptes.cli  – unified Typer front‑end (“target”)

• Tasks: detect | heatmap | geojson on images **and** videos.
• Model selection is delegated to *panoptes.model_registry* and is
  now strictly enforced – no silent fall‑backs.
"""

from __future__ import annotations

import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any, List, Literal, Optional, cast

import typer

from panoptes.model_registry import pick_weight
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

_AVAILABLE_MODELS: list[str] = ["primary"]  # flag kept for CLI parity
_DEFAULT_MODEL = "primary"

# ─────────────────────────── helper utils ──────────────────────────────
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
            f"[bold red]✖  no weight configured for task “{task}” "
            "(edit panoptes.model_registry.WEIGHT_PRIORITY)",
            err=True,
        )
        raise typer.Exit(1)
    return weight

# ───────────────────────────── command ─────────────────────────────────
@app.command()
def target(  # noqa: C901 – CLI parsing verbosity by design
    inputs: List[str] = typer.Argument(..., metavar="INPUT… [d|hm|gj]"),
    *,
    task: Optional[str] = typer.Option(None, "--task", "-t"),
    model: str = typer.Option(_DEFAULT_MODEL, "--model", "-m"),
    # heat‑map tuning
    alpha: float = typer.Option(0.40, help="Heat‑map blend 0‑1"),
    cmap: str = typer.Option("COLORMAP_JET", help="OpenCV / Matplotlib colour‑map"),
    kernel_scale: float = typer.Option(
        5.0, "--k", "-k", help="σ ∝ √area / kernel_scale (smaller → blurrier)"
    ),
    conf: float = typer.Option(0.40, help="[detect / heat‑map] confidence threshold 0‑1"),
    small: bool = typer.Option(False, "--small", "--fast", help="use nano models for live video"),
) -> None:
    """Batch‑process images / videos with zero manual weight fiddling."""
    if not inputs:
        typer.secho("[bold red]✖  no inputs given", err=True)
        raise typer.Exit(2)

    token_task, positional = _extract_task(inputs)

    # ── task resolution ────────────────────────────────────────────────
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

    # model flag is now cosmetic — kept for compatibility
    if model.lower() not in _AVAILABLE_MODELS:
        typer.secho(f"[bold yellow]⚠ unknown --model ignored: {model}", err=True)
    model = model.lower()

    hm_kwargs: dict[str, Any] = dict(alpha=alpha, cmap=cmap, kernel_scale=kernel_scale, conf=conf)

    # ───────────────────── loop over inputs ────────────────────────────
    for item in positional:
        low = item.lower()

        # ── videos ─────────────────────────────────────────────────────
        if low.endswith((".mp4", ".mov", ".avi", ".mkv")):
            if task_final == "geojson":
                typer.secho(f"[bold yellow]⚠ skipping video for geojson: {item}", err=True)
                continue

            weight = _require_weight(cast(Literal["detect", "heatmap"], task_final), small=small)

            if task_final == "heatmap":
                from .predict_heatmap_mp4 import main as _heat_vid

                _heat_vid(item, weights=weight, **hm_kwargs)
            else:  # detect
                from .predict_mp4 import main as _detect_vid

                _detect_vid(item, conf=conf, weights=weight)
            continue

        # ── still images & URLs ────────────────────────────────────────
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
                model=model,
                task=cast(Literal["detect", "heatmap", "geojson"], task_final),
                **hm_kwargs,
            )
            continue

        typer.secho(f"[bold red]✖ unsupported input: {item}", err=True)
        raise typer.Exit(2)


# ───────────────────────── entry‑point glue ────────────────────────────
def main() -> None:  # pragma: no cover
    try:
        app()
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[bold red]Unhandled error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
