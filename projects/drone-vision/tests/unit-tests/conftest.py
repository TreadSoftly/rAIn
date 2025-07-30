# projects/drone-vision/tests/conftest.py
"""
Shared pytest configuration.

• Silences the noisy botocore UTC-deprecation warning that now fires under
  Python 3.12+ / botocore 1.34+.
• Creates the results folder up-front so subprocess tests never fail on a race
  to `mkdir -p`.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Silence botocore warning spam
# ─────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings(
    "ignore",
    message=r"datetime\.datetime\.utcnow\(\) is deprecated",
    category=DeprecationWarning,
    module=r".*botocore",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Ensure tests/results exists (helps when parallelising)
# ─────────────────────────────────────────────────────────────────────────────
# Use the shared results folder one level up to avoid creating
# an unused tests/unit-tests/results directory.
_RESULTS = Path(__file__).resolve().parents[1] / "results"
os.makedirs(_RESULTS, exist_ok=True)
