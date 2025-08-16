from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Iterable

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
else:
    # At runtime (when NumPy may be absent), keep annotations harmless.
    NDArrayU8 = Any   # type: ignore[misc,assignment]
    NDArrayF32 = Any  # type: ignore[misc,assignment]

# Live-mode convenience aliases
Box   = tuple[int, int, int, int, float | None, int | None]
Boxes = Iterable[Box]
Names = Dict[int, str]
