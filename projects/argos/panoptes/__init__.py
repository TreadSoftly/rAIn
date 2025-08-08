
from __future__ import annotations

import importlib.util as _iu
import os as _os # type: ignore
import sys as _sys
from pathlib import Path
from typing import TYPE_CHECKING, Union
from os import PathLike

# ────────────────────────────────────────────────────────────────
#  Public re‑exports
# ────────────────────────────────────────────────────────────────
from . import geo_sink as geo_sink
from .heatmap import heatmap_overlay

#  All model helpers come from one single file ↓
from .model_registry import (
    MODEL_DIR,
    WEIGHT_PRIORITY,
    load_detector,      # type: ignore[override]
    load_segmenter,     # type: ignore[override]
    pick_weight,
)

__all__ = [
    "MODEL_DIR",
    "WEIGHT_PRIORITY",
    "pick_weight",
    "load_detector",
    "load_segmenter",
    "geo_sink",
    "heatmap_overlay",
]

# ────────────────────────────────────────────────────────────────
#  Internal paths that other sub‑modules sometimes import
# ────────────────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent          # …/panoptes
ROOT  = _BASE.parent                             # …/projects/argos

# ────────────────────────────────────────────────────────────────
#  Auto‑load the test helper when the file is present
# ────────────────────────────────────────────────────────────────
_tests_path = ROOT / "tests" / "unit-tests" / "test_tasks.py"
if _tests_path.exists():
    _spec = _iu.spec_from_file_location("panoptes.test_tasks", _tests_path)
    if _spec and _spec.loader:          # pragma: no cover
        _mod = _iu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _sys.modules["panoptes.test_tasks"] = _mod

# ────────────────────────────────────────────────────────────────
#  TYPE_CHECKING stubs only – never executed at runtime
# ────────────────────────────────────────────────────────────────
if TYPE_CHECKING:
    from .model_registry import YOLO

    def load_detector(
        *, small: bool = False, override: Union[str, PathLike[str], None] = None
    ) -> "YOLO | None": ...
