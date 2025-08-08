# projects/argos/tests/unit-tests/test_cli_format_preservation.py
"""
Ensures the CLI preserves the source image's extension when writing heat-maps.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image


def test_png_extension_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "subdrones.png"
    Image.new("RGB", (32, 32), "white").save(src)

    subprocess.check_call(["target", str(src), "--task", "heatmap"])
    out = Path("projects/argos/tests/results") / "subdrones_heat.png"

    assert out.exists() and out.suffix == ".png"
