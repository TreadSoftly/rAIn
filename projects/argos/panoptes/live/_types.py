from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Iterable, Tuple, Optional

"""
panoptes.live._types

Centralized, import-safe type aliases for live mode so pyright/mypy have
precise shapes without importing NumPy at runtime (where it might be absent).
"""

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    NDArrayU8 = npt.NDArray[np.uint8]
    NDArrayF32 = npt.NDArray[np.float32]

    # Type-checker-only aliases (py39-safe: no PEP 604 at runtime)
    Box   = Tuple[int, int, int, int, Optional[float], Optional[int]]
    Boxes = Iterable[Box]
    Names = Dict[int, str]

    # Surface progress Spinner protocol so optional integrations can type-check
    try:
        from panoptes.progress import Spinner as ProgressSpinner  # type: ignore[import]
    except Exception:
        class ProgressSpinner: ...  # type: ignore[no-redef, misc]
else:
    # Runtime aliases kept harmless and py39-safe
    NDArrayU8 = Any   # type: ignore[misc,assignment]
    NDArrayF32 = Any  # type: ignore[misc,assignment]

    Box   = Any       # type: ignore[assignment]
    Boxes = Iterable[Any]
    Names = Dict[int, str]

    # Runtime-friendly placeholder for progress types
    ProgressSpinner = Any  # type: ignore[assignment]
