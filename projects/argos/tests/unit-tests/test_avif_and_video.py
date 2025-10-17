# projects/argos/tests/unit-tests/test_avif_and_video.py
"""
AVIF round-trip + video heat-map smoke tests.

- Lazily import `heatmap_overlay` to avoid model init at collection time.
- Use repo-root-relative results path so the assertion works no matter
  where pytest is invoked from.
- Use `--small` for the heatmap video path to force the ONNX/nano seg.
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import Any, Callable, List, Optional, cast

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2", reason="OpenCV not installed; install opencv-python for video smoke tests")


def _repo_root() -> Path:
    # …/projects/argos
    return Path(__file__).resolve().parents[2]


def _fourcc(code: str) -> int:
    """Typed wrapper that tolerates missing stubs for VideoWriter_fourcc."""
    fn: Optional[Callable[..., Any]] = getattr(cv2, "VideoWriter_fourcc", None)
    if fn is None:
        # Some builds expose it as a static method on VideoWriter
        fn = getattr(getattr(cv2, "VideoWriter", object), "fourcc", None)
    if fn is None:
        # Last resort: 0 lets OpenCV pick a default; fine for a smoke test
        return 0
    return int(cast(Callable[..., int], fn)(*code))


def _has_avif() -> bool:
    # Works whether the plugin has already registered or not
    try:
        return (
            ".avif" in Image.registered_extensions()
            or importlib.util.find_spec("pillow_avif") is not None
        )
    except Exception:
        return False


@pytest.mark.skipif(not _has_avif(), reason="AVIF plugin missing (pip install pillow-avif-plugin)")
def test_avif_support(tmp_path: Path) -> None:
    # Create a small AVIF and ensure overlay returns a valid PIL.Image
    img = Image.new("RGB", (64, 64), "white")
    avif = tmp_path / "pic.avif"
    img.save(avif)

    # Lazy import to avoid model init at collection time
    from panoptes.heatmap import heatmap_overlay  # type: ignore[import-untyped]

    out = heatmap_overlay(Image.open(avif), boxes=None)
    assert isinstance(out, Image.Image) and out.size == (64, 64)


def test_video_heatmap(tmp_path: Path, cli_base_cmd: List[str]) -> None:
    vid = tmp_path / "bunny.mp4"

    # 10-frame black dummy @ 5 fps, 64x64
    fourcc: int = _fourcc("mp4v")
    vw = cv2.VideoWriter(str(vid), fourcc, 5.0, (64, 64))
    assert vw.isOpened(), "OpenCV VideoWriter failed to open"
    for _ in range(10):
        vw.write(np.zeros((64, 64, 3), dtype=np.uint8))
    vw.release()

    # Drive the CLI heatmap path; --small ensures the ONNX seg weight is used
    subprocess.check_call([*cli_base_cmd, str(vid), "--task", "heatmap", "--small"])

    expected = _repo_root() / "tests" / "results" / "bunny_heat.mp4"
    assert expected.exists() and expected.stat().st_size > 0
