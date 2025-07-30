from __future__ import annotations

import re
import sys # type: ignore
from pathlib import Path
from typing import Any, List, Literal, cast # type: ignore

import typer

from . import WEIGHT_PRIORITY, MODELS  # Removed unused MODEL_DIR

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

# ── helpers ──────────────────────────────────────────────────────────────
_URL_RE  = re.compile(r"^(https?://.+|data:image/[^;]+;base64,+)$", re.IGNORECASE)
_VALID   = {"detect", "heatmap", "geojson"}
_ALIASES = {
    "d": "detect",  "detect": "detect",
    "h": "heatmap", "heatmap": "heatmap",
    "g": "geojson", "geojson": "geojson",
}

def _is_url(inp: str) -> bool: # type: ignore
    return bool(_URL_RE.match(inp))

def _pick_weight(task: str, *, small: bool = False) -> Path:
    """Select the appropriate weights file for the given task (and size)."""
    key = f"{task}_small" if small else task
    for cand in WEIGHT_PRIORITY.get(key, []):
        if cand.exists():
            return cand
    # If no weight file is found, raise an error
    typer.secho(f"[bold red]✖  no weight file found for task '{task}'", err=True)
    raise typer.Exit(1)

# ── CLI commands (alias handling) ────────────────────────────────────────
@app.command()
def main_cli(
    inputs: List[str] = typer.Argument(..., help="Input image(s) or video(s) or URL(s)."),
    model: str = typer.Option("drone", "--model", "-m", help="Model name ('drone' or 'airplane')."),
    detect: bool = typer.Option(False, "--detect", "-d", help="Run object detection (bounding boxes)."),
    heatmap: bool = typer.Option(False, "--heatmap", "-h", help="Run segmentation heatmap overlay."),
    geojson: bool = typer.Option(False, "--geojson", "-g", help="Run detection and output GeoJSON."),
    conf: float = typer.Option(0.25, "--conf", help="Confidence threshold for detection."),
    alpha: float = typer.Option(0.4, "--alpha", help="Alpha blending for overlays."),
    cmap: str = typer.Option("COLORMAP_JET", "--cmap", help="Color map for heatmap (if applicable)."),
    kernel_scale: float = typer.Option(5.0, "--kernel-scale", help="Kernel scale (legacy, not used in new segmentation overlays)."),
    small: bool = typer.Option(False, "--small", help="Use small/fast model variants if available.")
) -> None:
    """Command-line interface for drone-vision tasks."""
    # Determine task
    if detect:
        task = "detect"
    elif heatmap:
        task = "heatmap"
    elif geojson:
        task = "geojson"
    else:
        task = "detect"

    # validate model name
    model = model.lower()
    if model not in MODELS:
        typer.secho(f"[bold red]✖  unknown model '{model}'", err=True)
        raise typer.Exit(2)

    # Heatmap-specific parameters (for backward compatibility; segmentation overlay now uses instance masks)
    hm_kwargs: dict[str, Any] = {"alpha": alpha, "cmap": cmap, "kernel_scale": kernel_scale, "conf": conf}

    # Process each input
    from .lambda_like import run_single # type: ignore  # image handler (simulate Lambda behavior locally)
    for item in inputs:
        low = item.lower()
        # ── video inputs ───────────────────────────────────────────
        if low.endswith((".mp4", ".mov", ".avi", ".mkv")):
            weight = _pick_weight(task, small=small)
            if task == "heatmap":
                from . import predict_heatmap_mp4 as _vid
                # Pass weight as an argument if supported by the function
                _vid.main(item, weights=weight, **hm_kwargs)
            elif task == "detect":
                from . import predict_mp4 as _vid
                #
