from __future__ import annotations

import re
import sys  # type: ignore  # noqa: F401
from pathlib import Path
from typing import Any, List

import typer
import os

from . import WEIGHT_PRIORITY, MODELS  # centralised in __init__

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

# ── helpers ──────────────────────────────────────────────────────────────
_URL_RE = re.compile(r"^(https?://.+|data:image/[^;]+;base64,+)$", re.IGNORECASE)
_VALID = {"detect", "heatmap", "geojson"}
_ALIASES = {
    "d": "detect",   "detect": "detect",
    "h": "heatmap",  "heatmap": "heatmap",
    "g": "geojson",  "geojson": "geojson",
}

def _is_url(inp: str) -> bool:
    return bool(_URL_RE.match(inp))

def _pick_weight(task: str, *, small: bool = False) -> Path:
    """Return the first existing weights file for *task* (respecting *small*)."""
    weight_env = os.getenv("DRONEVISION_TEST_WEIGHTS")
    if weight_env:
        env_path = Path(weight_env).expanduser()
        if env_path.exists():
            return env_path
    key = f"{task}_small" if small else task
    for cand in WEIGHT_PRIORITY.get(key, []):
        if cand.exists():
            return cand
    if task.startswith("heatmap"):
        alt_key = "detect_small" if small else "detect"
        for alt in WEIGHT_PRIORITY.get(alt_key, []):
            if alt.exists():
                typer.secho("[bold yellow]⚠  no segmentation model available, using detection weights for heatmap", err=True)
                return alt
    typer.secho(f"[bold red]✖  no weight file found for task '{task}'", err=True)
    raise typer.Exit(1)

# ── CLI command (single endpoint with flags) ─────────────────────────────
@app.command()
def main_cli(
    inputs: List[str] = typer.Argument(..., help="Input image(s) / video(s) / URL(s)"),
    model: str = typer.Option("primary", "--model", "-m", help="Model name (default 'primary')"),
    detect: bool = typer.Option(False, "--detect", "-d", help="Run object detection (bounding boxes)"),
    heatmap: bool = typer.Option(False, "--heatmap", "-h", help="Run segmentation heat‑map overlay"),
    geojson: bool = typer.Option(False, "--geojson", "-g", help="Run detection and emit GeoJSON"),
    conf: float = typer.Option(0.25, "--conf", help="Confidence threshold"),
    alpha: float = typer.Option(0.4, "--alpha", help="Alpha blending for overlays"),
    cmap: str = typer.Option("COLORMAP_JET", "--cmap", help="Colour‑map for heat‑maps"),
    kernel_scale: float = typer.Option(5.0, "--kernel-scale", help="Legacy Gaussian kernel scaling"),
    small: bool = typer.Option(False, "--small", help="Use fast ‘_small’ model variants when present"),
) -> None:
    """High‑level command‑line interface for Drone‑Vision tasks."""
    # Determine task
    task = (
        "detect" if detect else
        "heatmap" if heatmap else
        "geojson" if geojson else
        "detect"
    )

    # Validate model name
    model = model.lower()
    if model not in MODELS:
        typer.secho(f"[bold red]✖  unknown model '{model}'", err=True)
        raise typer.Exit(2)

    # Heat‑map‑specific kwargs (kept for backwards compatibility)
    hm_kwargs: dict[str, Any] = {
        "alpha": alpha,
        "cmap": cmap,
        "kernel_scale": kernel_scale,
        "conf": conf,
    }

    # Process each input
    from .lambda_like import run_single  # image handler (local Lambda shim)

    for item in inputs:
        low = item.lower()

        # ── videos ────────────────────────────────────────────────
        if low.endswith((".mp4", ".mov", ".avi", ".mkv")):
            weight = _pick_weight(task, small=small)
            if task == "heatmap":
                from . import predict_heatmap_mp4 as _vid
                _vid.main(item, weights=weight, **hm_kwargs)
            elif task == "detect":
                from . import predict_mp4 as _vid
                _vid.main(item, weights=weight, conf=conf)
            else:  # geojson on a video makes no sense
                typer.secho("[bold yellow]⚠  geojson ignored for video inputs", err=True)
            continue

        # ── images / URLs ─────────────────────────────────────────
        if _is_url(item) or Path(item).is_file():
            # Let the shared helper do the heavy lifting
            run_single(
                src=item,
                image_url=item,
                task=task,
                model=model,
                conf=conf,
                alpha=alpha,
                cmap=cmap,
                kernel_scale=kernel_scale,
                small=small,
            )
        else:
            typer.secho(f"[bold red]✖  unable to read input '{item}'", err=True)

# Allow “python -m dronevision” to behave like the CLI
if __name__ == "__main__":  # pragma: no cover
    app()
