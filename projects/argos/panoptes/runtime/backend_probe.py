"""
Lightweight backend availability probes.

These utilities intentionally avoid importing heavy dependencies at module
import time.  Callers can use the helpers to decide which model weights to use
without paying the cost of initialising ONNX Runtime sessions.
"""

from __future__ import annotations

import os
from typing import Tuple


def torch_available() -> bool:
    """Return True if ``torch`` can be imported."""
    try:
        import torch  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def ort_available() -> Tuple[bool, str]:
    """
    Attempt to import ONNX Runtime and report the outcome.

    Returns:
        tuple(bool, str): ``True`` with reason ``"OK"`` if import succeeded.
        Otherwise returns ``False`` and a string describing the failure.
    """
    # Allow users to disable ONNX entirely for diagnostics/CI via env toggle.
    if os.environ.get("ARGOS_DISABLE_ONNX"):
        return False, "disabled via ARGOS_DISABLE_ONNX"

    try:
        import onnxruntime as ort  # type: ignore

        try:
            providers = ort.get_available_providers()
            return True, ",".join(providers) if providers else "OK"
        except Exception as exc:  # pragma: no cover
            # get_available_providers may fail on badly configured installs.
            return True, f"OK (providers unavailable: {exc})"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
