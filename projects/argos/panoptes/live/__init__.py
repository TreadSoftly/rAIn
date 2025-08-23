"""
ARGOS live video/webcam package.

Small, dependency-light building blocks to:
  - read frames from a camera (or synthetic source),
  - run a light "task" adapter (detect/heatmap/classify/pose/pse/obb; ML optional),
  - draw overlays / HUD,
  - send frames to one or more sinks (display / video file).

The real CLI entrypoint lives in panoptes.live.cli.
"""

from __future__ import annotations

from .camera import FrameSource, open_camera, synthetic_source
from .overlay import draw_boxes_bgr, draw_heatmap_bgr, hud  # type: ignore
from .sinks import DisplaySink, VideoSink, MultiSink
from .pipeline import LivePipeline
from . import tasks, config

# Re-export progress helpers here so apps can reach them via panoptes.live.*
# (keeps CLI and notebooks in one import space) — robust fallbacks if progress is unavailable
try:
    from panoptes.progress import (  # type: ignore
        percent_spinner as progress_percent_spinner,  # type: ignore
        simple_status as progress_simple_status,      # type: ignore
        running_task as progress_running_task,        # type: ignore
        should_enable_spinners as progress_spinners_enabled,  # type: ignore
        osc8 as progress_osc8,                        # type: ignore
    )
except Exception:  # pragma: no cover
    def progress_percent_spinner(*_a: object, **_k: object):  # type: ignore
        class _N:
            def __enter__(self): return self
            def __exit__(self, *_: object) -> bool: return False
            def update(self, **__: object): return self
        return _N()
    def progress_simple_status(*_a: object, **_k: object):
        return progress_percent_spinner()
    def progress_running_task(*_a: object, **_k: object):
        return progress_percent_spinner()
    def progress_spinners_enabled(*_a: object, **_k: object) -> bool:
        return False
    def progress_osc8(label: str, target: str) -> str:
        return label

__all__ = [
    "FrameSource",
    "open_camera",
    "synthetic_source",
    "draw_boxes_bgr",
    "draw_heatmap_bgr",
    "hud",
    "DisplaySink",
    "VideoSink",
    "MultiSink",
    "LivePipeline",
    "tasks",
    "config",
    # progress re-exports
    "progress_percent_spinner",
    "progress_simple_status",
    "progress_running_task",
    "progress_spinners_enabled",
    "progress_osc8",
]
