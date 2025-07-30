"""
Tiny Typer wrapper that installs as **target**.

• JPEG / PNG  → saved into tests/results as *<stem>_boxes.jpg* / *_heat.jpg* …
• MP4         → uses predict_mp4 to create an annotated MP4 in tests/results
• HTTP/S URL  → prints base-64 JPEG (detect | heatmap) or GeoJSON to stdout.
"""
from __future__ import annotations

import urllib.parse
from pathlib import Path #type: ignore[import-untyped]

import typer

from .lambda_like import run_single #type: ignore  # lazy inferencing helper

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


def _is_url(text: str) -> bool:
    return urllib.parse.urlparse(text).scheme in {"http", "https"}


@app.command()
def target(
    inputs: list[str] = typer.Argument(
        ..., help="Path(s) or URL(s) to image / video."
    ),
    model: str = typer.Option("drone", help="[drone|airplane] - selects weights file"),
    task: str = typer.Option(
        "detect", help="[detect|heatmap|geojson] - processing to perform"
    ),
) -> None:
    """Batch process one or more local files / URLs."""
    for item in inputs:
        # ─── video ──────────────────────────────────────────────────────
        if item.lower().endswith(".mp4"):
            # Import on-demand so normal image runs do NOT pull in heavy deps
            from .predict_mp4 import main as _predict_mp4

            _predict_mp4(item)  # (model/task are image-only for now)
            continue

        # ─── image (local or remote) ────────────────────────────────────
        if item.lower().endswith((".jpg", ".jpeg", ".png")) or _is_url(item):
            run_single(
                item,
                model,  # type: ignore[arg-type]
                task    # type: ignore[arg-type]
            )
            continue

        typer.secho(f"[bold red]✖ Unsupported input: {item}", err=True)
        raise typer.Exit(2)


def main() -> None:  # invoked by console-script hook
    app()


if __name__ == "__main__":  # python -m dronevision.cli …
    main()
