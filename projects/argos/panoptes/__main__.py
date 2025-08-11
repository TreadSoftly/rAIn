# \rAIn\projects\argos\panoptes\__main__.py
"""
`python -m panoptes …` forwards to the Typer CLI defined in `panoptes.cli.target`.
"""

from __future__ import annotations

import sys

import typer

from panoptes.cli import target as _target  # type: ignore

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
app.command()(_target)  # expose under same name

if __name__ == "__main__":  # pragma: no cover
    try:
        app()
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[bold red]Unhandled error: {exc}", err=True)
        sys.exit(1)
