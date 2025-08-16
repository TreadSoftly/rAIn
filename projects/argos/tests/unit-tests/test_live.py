from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pytest


def _live_module_cmd() -> list[str]:
    """Run the module directly to mirror CI/launcher usage."""
    return [sys.executable, "-m", "panoptes.live.cli"]


def test_live_package_imports():
    import importlib
    m = importlib.import_module("panoptes.live")
    assert m is not None


def _run_ok(cmd: list[str], timeout: int = 20) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if p.returncode != 0:
        raise AssertionError(
            f"Command failed ({p.returncode}).\n"
            f"CMD: {' '.join(cmd)}\n\n"
            f"STDOUT:\n{p.stdout.decode(errors='ignore')}\n\n"
            f"STDERR:\n{p.stderr.decode(errors='ignore')}\n"
        )


def test_live_cli_detect_no_subcommand_synthetic_headless():
    # Works without explicit "run" because cli.main() injects it
    cmd = _live_module_cmd() + ["d", "synthetic", "--duration", "0.25", "--headless"]
    _run_ok(cmd, timeout=30)


def test_live_cli_heatmap_with_subcommand_synthetic_headless():
    # Explicit "run" subcommand
    cmd = _live_module_cmd() + ["run", "hm", "synthetic", "--duration", "0.25", "--headless"]
    _run_ok(cmd, timeout=30)


def test_live_cli_detect_swapped_order_synthetic():
    # Swapped order like: "lv synthetic d" — should treat 'synthetic' as source and 'd' as task
    cmd = _live_module_cmd() + ["synthetic", "d", "--duration", "0.25", "--headless"]
    _run_ok(cmd, timeout=30)


def test_live_cli_optional_save(tmp_path: Path):
    # Skip if OpenCV isn't available; also "use" the module to satisfy linters.
    cv2 = pytest.importorskip("cv2")
    assert hasattr(cv2, "__version__")

    out = tmp_path / "live-smoke.mp4"
    cmd = _live_module_cmd() + [
        "d",
        "synthetic",
        "--duration",
        "0.5",
        "--headless",
        "--save",
        str(out),
    ]
    _run_ok(cmd, timeout=45)
    assert out.exists() and out.stat().st_size > 0
