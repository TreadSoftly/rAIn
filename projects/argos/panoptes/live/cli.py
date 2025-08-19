# panoptes.live.cli — dedicated live/webcam entrypoint ("lv" / "live" / "livevideo")
from __future__ import annotations

import sys
from typing import List, Optional, Tuple, Union

import typer

from .pipeline import LivePipeline

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

# allow short and long spellings (only tokens you authorized)
_TASK_CHOICES = {
    "d": "detect",
    "detect": "detect",
    "hm": "heatmap",
    "heatmap": "heatmap",
    "clf": "classify",
    "classify": "classify",
    "pse": "pse",
    "pose": "pose",
    "obb": "obb",
    "object": "obb",
}

# Tokens that indicate "live intent" and should be ignored as a source
_LIVE_MARKERS = {
    "lv",
    "livevideo",
    "live-video",
    "live_video",
    "ldv",
    "lvd",
    "live",
    "video",
    "l",   # spaced variant "l d v" — treat 'l' and 'v' as noise here
    "v",
}


@app.command()
def run(
    tokens: List[str] = typer.Argument(
        None,
        metavar="[TASK] [SOURCE]",
        help="Task (d|detect|hm|heatmap|clf|classify|pse|pose|obb|object) and source (camera index, path, or 'synthetic').",
    ),
    *,
    duration: Optional[float] = typer.Option(None, "--duration", help="Seconds to run; default until quit."),
    headless: bool = typer.Option(False, "--headless", help="Disable preview window."),
    save: Optional[str] = typer.Option(None, "--save", "-o", help="Optional MP4 output path."),
    fps: Optional[int] = typer.Option(None, "--fps", help="Target FPS for writer (default 30)."),
    width: Optional[int] = typer.Option(None, "--width", help="Capture width hint."),
    height: Optional[int] = typer.Option(None, "--height", help="Capture height hint."),
    conf: float = typer.Option(0.25, "--conf", help="Detector confidence (detect/pose/obb where applicable)."),
    iou: float = typer.Option(0.45, "--iou", help="Detector IOU (detect/obb where applicable)."),
    small: bool = typer.Option(True, "--small/--no-small", help="Prefer small models for live."),
) -> None:
    """Launch the live webcam/video pipeline."""

    # Some environments + Click variadic args can mis-route option values.
    # If Typer didn't bind --save/-o, fall back to parsing sys.argv directly.
    if save is None:
        argv = sys.argv[1:]
        for flag in ("--save", "-o"):
            if flag in argv:
                idx = argv.index(flag)
                if idx + 1 < len(argv):
                    val = argv[idx + 1]
                    if not val.startswith("-"):
                        save = val
                        break

    # Flexible positional parsing:
    task_tok: Optional[str] = None
    source_tok: Optional[str] = None

    for tok in (tokens or []):
        low = tok.lower()

        # Some wrappers prepend "run"; treat it as noise.
        if low == "run":
            continue

        # Ignore any live-intent markers completely (never treat as source).
        if low in _LIVE_MARKERS:
            continue

        # First recognized task token wins, the next token becomes the source
        if low in _TASK_CHOICES and task_tok is None:
            task_tok = low
            continue

        # First non-task, non-live marker token becomes the source
        if source_tok is None:
            source_tok = tok
            continue

        # Anything beyond "<task> <source>" is unexpected
        raise typer.BadParameter(f"Got unexpected extra argument ({tok!r})")

    # Defaults
    task_tok = task_tok or "detect"
    t = _TASK_CHOICES[task_tok]

    # Normalize source: allow "synthetic", signed integers, or paths
    s: str = (source_tok or "0").strip()
    if s.lower().startswith("synthetic"):
        src: Union[int, str] = "synthetic"
    elif s.lstrip("+-").isdigit():
        src = int(s)
    else:
        src = s

    size: Optional[Tuple[int, int]] = (width, height) if (width and height) else None

    pipe = LivePipeline(
        source=src,
        task=t,
        autosave=bool(save),
        out_path=save,
        prefer_small=small,
        fps=fps,
        size=size,
        headless=headless,
        conf=conf,
        iou=iou,
        duration=duration,
    )
    out = pipe.run()
    if out:
        print(out)


def _prepend_argv(token: str) -> None:
    # Idempotently insert right before the first non-option token,
    # but only if it's not already present there.
    argv = sys.argv[1:]
    for a in argv:
        if a.startswith("-"):
            continue  # skip global options
        if a == token:
            return  # already present; do nothing
        sys.argv = sys.argv[:1] + [token] + sys.argv[1:]
        return
    # No positional args at all -> append at position 1
    sys.argv = sys.argv[:1] + [token] + sys.argv[1:]


def main() -> None:  # pragma: no cover
    try:
        # Make "python -m panoptes.live.cli hm synthetic ..." work without explicitly typing "run".
        argv = sys.argv[1:]
        if argv and not any(a in {"-h", "--help"} for a in argv):
            _prepend_argv("run")
        app()
    except KeyboardInterrupt:
        raise SystemExit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
