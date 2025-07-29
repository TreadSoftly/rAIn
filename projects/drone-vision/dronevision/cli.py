"""
Tiny Typer wrapper that installs as **target**.

• JPEG / PNG  → saved next to the file as *XXX_boxes.jpg* or *XXX_heat.jpg*
• MP4         → Ultralytics writes an annotated MP4 next to the video.
• HTTP/S URL  → prints base-64 JPEG (detect | heatmap) or GeoJSON to stdout.
"""
from __future__ import annotations

import urllib.parse

import typer

from .lambda_like import run_single  # lazy inferencing helper
from .predict_mp4 import main as _predict_mp4

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


def _is_url(text: str) -> bool:
    return urllib.parse.urlparse(text).scheme in {"http", "https"}


@app.command()
def target(
    inputs: list[str] = typer.Argument(..., help="Path(s) or URL(s) to image / video."),
    model: str = typer.Option("drone", help="[drone|airplane] - selects weights file"),
    task: str = typer.Option(
        "detect", help="[detect|heatmap|geojson] - processing to perform"
    ),
) -> None:
    """Batch process one or more local files / URLs."""
    for item in inputs:
        # ─── video ──────────────────────────────────────────────────────────
        if item.lower().endswith(".mp4"):
            _predict_mp4(item, model=model, task=task)
            continue

        # ─── image (local or remote) ────────────────────────────────────────
        if item.lower().endswith((".jpg", ".jpeg", ".png")) or _is_url(item):
            run_single(item, model, task)
            continue

        typer.secho(f"[bold red]✖ Unsupported input: {item}", err=True)
        raise typer.Exit(2)


def main() -> None:  # invoked by console-script hook
    app()


if __name__ == "__main__":  # python -m dronevision.cli …
    main()
