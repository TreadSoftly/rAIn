"""
Shared helpers for selecting the appropriate ONNX Runtime build.

Centralising the desired version specifier here keeps bootstrap, runtime
probing, and tests aligned with the packaging constraints declared in
pyproject/requirements.
"""

from __future__ import annotations

import platform
import sys


def desired_ort_spec() -> str:
    """
    Return the PEP 440 specifier string for the preferred onnxruntime build.

    The values mirror the pins declared in pyproject.toml / requirements.txt:
    - Windows + Python >= 3.10 -> >=1.22,<1.23
    - Other platforms + Python >= 3.10 -> >=1.22,<1.24
    - Legacy Python (<3.10) falls back to the last supported LTS.
    """
    if sys.version_info >= (3, 10):
        if platform.system() == "Windows":
            return "onnxruntime>=1.22,<1.23"
        return "onnxruntime>=1.22,<1.24"
    # Older Python versions (only seen on legacy hosts) stay on the final 1.19.x build.
    return "onnxruntime==1.19.2"


__all__ = ["desired_ort_spec"]
