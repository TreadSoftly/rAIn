"""
Runtime utilities used by CLI entry points to keep Argos self-contained.

Modules in this package intentionally avoid heavy imports and side effects so
they can execute early during bootstrap across all supported platforms.
"""

from __future__ import annotations

__all__: list[str] = []
