"""
Shim: delegate heatmap work to the shared  *dronevision*  implementation.

The Lambda side deliberately ignores the optional *masks / boxes* argument that
the handler passes in  we call the real helper with just the image so we never
crash if boundingbox shapes dont match the mask logic.
"""
from __future__ import annotations

from typing import Any, Iterable
from PIL import Image

from dronevision.heatmap import heatmap_overlay as _core_heatmap # type: ignore

def heatmap_overlay(
    img: Image.Image,
    masks: Iterable[Any] | None = None,   # signature matches the handler call
) -> Image.Image:
    """Thin wrapper that defers to the shared helper, ignoring *masks*."""
    return _core_heatmap(img)             # shared code already degrades gracefully

__all__ = ["heatmap_overlay"]
