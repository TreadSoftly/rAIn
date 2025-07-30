# projects/drone-vision/tests/unit-tests/test_avif_and_video.py
"""
AVIF round-trip + video heat-map smoke tests.

* Ensures Pillow can decode AVIF and our ``heatmap_overlay`` still returns a
  valid PIL.Image.  (Relies on `pillow-avif-plugin` being installed at runtime.)
* Generates a tiny dummy MP4 and lets the *target* CLI create a heat-mapped
  version - the file is expected as ``<stem>_heat.mp4`` inside tests/results.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dronevision.heatmap import heatmap_overlay  # type: ignore[import-untyped]


def test_avif_support(tmp_path: Path) -> None:
    img = Image.new("RGB", (64, 64), "white")
    avif = tmp_path / "pic.avif"
    img.save(avif)
    out = heatmap_overlay(Image.open(avif), boxes=None)
    assert isinstance(out, Image.Image) and out.size == (64, 64)


def test_video_heatmap(tmp_path: Path) -> None:
    vid = tmp_path / "bunny.mp4"

    # 10-frame black dummy @ 5 fps, 64x64
    fourcc = int(getattr(cv2, "VideoWriter_fourcc")(*"mp4v"))
    vw = cv2.VideoWriter(str(vid), fourcc, 5, (64, 64))
    for _ in range(10):
        vw.write(np.zeros((64, 64, 3), dtype=np.uint8))
    vw.release()

    subprocess.check_call(["target", str(vid), "--task", "heatmap"])

    expected = Path("projects/drone-vision/tests/results") / "bunny_heat.mp4"
    assert expected.exists() and expected.stat().st_size > 0
