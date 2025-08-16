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

    # Type-checker-only aliases (don’t evaluate unions at runtime on py39)
    Box   = Tuple[int, int, int, int, Optional[float], Optional[int]]
    Boxes = Iterable[Box]
    Names = Dict[int, str]
else:
    # At runtime (when NumPy may be absent), keep aliases harmless and py39-safe.
    NDArrayU8 = Any   # type: ignore[misc,assignment]
    NDArrayF32 = Any  # type: ignore[misc,assignment]

    # Keep these broad at runtime to avoid importing typing constructs.
    Box   = Any       # type: ignore[assignment]
    Boxes = Iterable[Any]
    Names = Dict[int, str]
