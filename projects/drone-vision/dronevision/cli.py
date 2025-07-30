from __future__ import annotations

import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any, List, Literal, Mapping, cast

import typer

from . import lambda_like as ll
from .lambda_like import (
    MODELS,  # type: ignore[import]
    run_single,
)

# ────────────────────────── CLI scaffolding ────────────────────────────────
app = typer.Typer(add_completion=False, rich_markup_mode="rich")

_TASK_ALIAS = {
    "d": "detect", "detect": "detect", "-d": "detect", "-detect": "detect",
    "hm": "heatmap", "heatmap": "heatmap",
    "gj": "geojson", "geojson": "geojson", "-gj": "geojson", "-geojson": "geojson",
}
_VALID_TASKS = {"detect", "heatmap", "geojson"}

_URL_RE = re.compile(r"^(https?://.+|data:image/[^;]+;base64,.+)$", re.I)

_AVAILABLE_MODELS = [k.lower() for k in cast(Mapping[str, Any], MODELS).keys()]
_DEFAULT_MODEL = (
    "drone"
    if "drone" in _AVAILABLE_MODELS
    else (_AVAILABLE_MODELS[0] if _AVAILABLE_MODELS else "drone")
)


def _is_url(text: str) -> bool:
    return urllib.parse.urlparse(text).scheme in {"http", "https"} or bool(_URL_RE.match(text))


def _extract_token(tokens: List[str]) -> tuple[str, List[str]]:
    hits = [(i, _TASK_ALIAS[t.lower()]) for i, t in enumerate(tokens) if t.lower() in _TASK_ALIAS]
    if not hits:
        return "detect", tokens
    if len(hits) > 1:
        typer.secho(
            f"[bold red]✖  more than one task alias supplied "
            f"({', '.join(alias for _, alias in hits)})",
            err=True,
        )
        raise typer.Exit(2)
    idx, task = hits[0]
    return task, tokens[:idx] + tokens[idx + 1:]


# ────────────────────────────── CLI command ────────────────────────────────
@app.command()
def target(  # noqa: C901 (complexity - CLI parsing logic)
    inputs: List[str] = typer.Argument(..., metavar="INPUT… [d|hm|gj]"),
    task_flag: str | None = typer.Option(None, "--task", "-t"),
    model: str = typer.Option(_DEFAULT_MODEL, "--model", "-m"),
    # heat-map tuning
    alpha: float = typer.Option(0.4, help="Heat-map blend 0-1"),
    cmap: str = typer.Option("COLORMAP_JET", help="OpenCV / Matplotlib colour-map"),
    kernel_scale: float = typer.Option(
        5.0, "--k", "-k", help="σ ∝ √area / kernel_scale  (smaller → blurrier)"
    ),
    conf: float = typer.Option(
        0.40, "--conf", help="[detect / heat-map] confidence threshold 0-1"
    ),
    small: bool = typer.Option(False, "--small", "--fast", help="use nano models (live video)"),
) -> None:
    """Batch-process one or more image / video inputs."""
    if not inputs:
        typer.secho("[bold red]✖  no inputs given", err=True)
        raise typer.Exit(2)

    token_task, positional = _extract_token(inputs)

    # ── resolve task ───────────────────────────────────────────────────────
    if task_flag:
        task_flag = task_flag.lower()
        if task_flag not in _VALID_TASKS:
            typer.secho(f"[bold red]✖  invalid --task {task_flag}", err=True)
            raise typer.Exit(2)
        if token_task != "detect" and task_flag != token_task:
            typer.secho(
                f"[bold red]✖  conflicting task: flag={task_flag!r}  token={token_task!r}",
                err=True,
            )
            raise typer.Exit(2)
        task = task_flag
    else:
        task = token_task

    # ── validate model ────────────────────────────────────────────────────
    model = model.lower()
    if model not in _AVAILABLE_MODELS:
        typer.secho(
            f"[bold red]✖  unknown model {model!r} "
            f"(available: {', '.join(_AVAILABLE_MODELS)})",
            err=True,
        )
        raise typer.Exit(2)

    if not positional:
        typer.secho("[bold red]✖  no image / video inputs found", err=True)
        raise typer.Exit(2)

    hm_kwargs: dict[str, Any] = dict(alpha=alpha, cmap=cmap, kernel_scale=kernel_scale, conf=conf)

    # ───────────────────── iterate inputs ────────────────────────────────
    for item in positional:
        low = item.lower()

        # ──────── video inputs ────────────────────────────────────────────
        if low.endswith((".mp4", ".mov", ".avi", ".mkv")):
            if task == "geojson":
                typer.secho(f"[bold yellow]⚠  skipping video for geojson: {item}", err=True)
                continue

            # Select matching .pt weights (for video processing)
            if model in {"drone", "airplane"}:
                model_dir = getattr(ll, "MODEL_DIR", None) or Path(str(MODELS[model])).parent  # type: ignore[union-attr]
                if task == "heatmap":
                    if small:
                        weight_path = model_dir / "yolov8n-seg.pt"
                        if not weight_path.exists():
                            weight_path = model_dir / "yolo11n-seg.pt"
                            if not weight_path.exists():
                                weight_path = model_dir / "yolo11x-seg.pt"
                                if not weight_path.exists():
                                    weight_path = model_dir / "yolo11m-seg.pt"
                    else:
                        weight_path = model_dir / "yolo11x-seg.pt"
                        if not weight_path.exists():
                            weight_path = model_dir / "yolo11m-seg.pt"
                            if not weight_path.exists():
                                weight_path = model_dir / "yolov8n-seg.pt"
                                if not weight_path.exists():
                                    weight_path = model_dir / f"{model}.pt"
                else:  # detect
                    if small:
                        weight_path = model_dir / "yolov8n.pt"
                        if not weight_path.exists():
                            weight_path = model_dir / "yolo11n.pt"
                    else:
                        weight_path = model_dir / "yolov8x.pt"
                        if not weight_path.exists():
                            weight_path = model_dir / "yolo11x.pt"
                            if not weight_path.exists():
                                weight_path = model_dir / f"{model}.pt"

                if task == "heatmap":
                    from . import predict_heatmap_mp4 as _heat_mod
                    if hasattr(_heat_mod, "DEF_WEIGHTS"):
                        setattr(_heat_mod, "DEF_WEIGHTS", weight_path)
                else:  # detect
                    from . import predict_mp4 as _detect_mod
                    setattr(_detect_mod, "YOLO_WEIGHTS", weight_path)

            if task == "heatmap":
                from .predict_heatmap_mp4 import main as _heat_vid
                _heat_vid(item, **hm_kwargs)
            else:
                from .predict_mp4 import main as _detect_vid
                _detect_vid(item, conf=conf)
            continue

        # ──────── image (file / http / data-URI) ──────────────────────────
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
                model=cast(Literal["drone", "airplane"], model),
                task=cast(Literal["detect", "heatmap", "geojson"], task),
                **hm_kwargs,
            )
            continue

        typer.secho(f"[bold red]✖  unsupported input: {item}", err=True)
        raise typer.Exit(2)


# ───────────────────────────── entry-point ────────────────────────────────
def main() -> None:  # pragma: no cover
    try:
        app()
    except Exception as exc:  # pylint: disable=broad-except
        typer.echo(f"[bold red]Unhandled error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
