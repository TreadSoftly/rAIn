# \rAIn\projects\argos\panoptes\tools\build_models.py
from __future__ import annotations

"""
Model installer/packer (wrapper)
--------------------------------
This tool intentionally delegates to the single, authoritative
implementation in `panoptes.model._fetch_models`. Keeping the UX
and progress logic in one place avoids drift and ensures the
build/installers show the same live progress and final clickable
results list.

Invoked by installers:
  - installers/build.ps1 → python -m panoptes.tools.build_models
"""

import sys  # type: ignore


def main() -> None:
    # Defer import so this module stays lean and import-safe
    from panoptes.model._fetch_models import main as _models_main  # type: ignore

    # Hand off to the unified implementation (Typer prompts, progress, etc.)
    _models_main()


if __name__ == "__main__":
    main()
