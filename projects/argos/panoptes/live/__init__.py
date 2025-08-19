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
]
