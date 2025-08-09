# projects/argos/tests/unit-tests/conftest.py
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List

import pytest
from PIL import Image

# Type-only imports so static analyzers know what these are
if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.nodes import Item


def _has_avif() -> bool:
    try:
        return (
            ".avif" in Image.registered_extensions()
            or importlib.util.find_spec("pillow_avif") is not None
        )
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def _ensure_results_dir() -> None: # type: ignore[no-untyped-def]
    """Always have projects/argos/tests/results ready (pytest autouse fixture)."""
    out = Path("projects/argos/tests/results")
    out.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def cli_base_cmd() -> List[str]:
    """
    Helper for any future tests that want a robust way to invoke the CLI:
    returns ["target"] if it's on PATH, otherwise falls back to
    [python, -m, panoptes.cli].
    """
    exe = shutil.which("target")
    if exe:
        return ["target"]
    return [sys.executable, "-m", "panoptes.cli"]


def pytest_collection_modifyitems(config: "Config", items: List["Item"]) -> None:
    """
    If the AVIF plugin isn't present, automatically skip any tests whose nodeid
    mentions 'avif' (covers functions/files with 'avif' in their names).
    """
    if _has_avif():
        return
    skip = pytest.mark.skip(reason="AVIF plugin missing (pip install pillow-avif-plugin)")
    for item in items:
        if "avif" in item.nodeid.lower():
            item.add_marker(skip)
