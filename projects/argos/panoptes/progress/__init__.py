# \rAIn\projects\argos\panoptes\progress\__init__.py
"""
Panoptes Progress (HALO-ONLY)

This package exposes exactly one progress UX: the Halo/Rich spinner implemented
in `progress_ux.py`. There are **no** alternate progress bars, no console
fallback that renders a different style, and no competing look & feel.

If the environment cannot render the Halo spinner (non‑TTY, CI, nested spinner,
invalid stream handle), callers will receive a **no-op** spinner from the bridge
layer so workloads never break — but there is still no second UI.

Exports
-------
- percent_spinner(...)   → Halo/Rich single-line spinner with [File|Job|Model]
- live_percent(engine)   → bind ProgressEngine to the spinner (Halo only)
- simple_status(), running_task(), osc8()  → UX helpers (Halo-only)
- ProgressEngine / Phase / Task / ProgressState  → lightweight progress engine
- JobAwareProxy(spinner) → maps child `.update(current=...)` → `[Job: …]`

Environment knobs honoured by the Halo spinner (see progress_ux.py):
  PANOPTES_PROGRESS_WIDTH           int columns (default 100)
  PANOPTES_PROGRESS_TAIL            full | short | min (default full)
  PANOPTES_SPINNER                  Halo spinner glyph (default "dots")
  PANOPTES_SPINNER_INTERVAL         seconds (default 0.01)
  PANOPTES_PROGRESS_FINAL_NEWLINE   0/1, true/false (default 0)
  PANOPTES_PROGRESS_ACTIVE          internal: set to "1" while spinner is alive
  PANOPTES_PROGRESS_FORCE           1 → allow nested spinners anyway
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Optional, Protocol, cast

# Lightweight progress engine (no UI)
from .engine import (
    ProgressEngine,
    ProgressState,
    Phase,
    Task,
)

# The one-and-only Halo spinner lives here:
from .progress_ux import (
    percent_spinner as _halo_percent_spinner,
    simple_status,
    running_task,
    osc8,
    should_enable_spinners as _halo_should_enable,
)

__all__ = [
    # Engine
    "ProgressEngine", "ProgressState", "Phase", "Task",
    # UX & spinner
    "percent_spinner", "live_percent", "simple_status", "running_task", "osc8",
    "should_enable_spinners",
    # Utility
    "JobAwareProxy",
]


class SpinnerLike(Protocol):
    def __enter__(self) -> "SpinnerLike": ...
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> Optional[bool] | bool: ...
    def update(self, **kwargs: Any) -> "SpinnerLike": ...


def should_enable_spinners(stream: Any | None = None) -> bool:
    """Re-export: whether the Halo spinner should be enabled on this stream."""
    return _halo_should_enable(stream)


def percent_spinner(
    *,
    prefix: str = "PROGRESS",
    stream: Any | None = None,
    final_newline: bool | None = None,
    spinner_type: str | None = None,
    interval: float | None = None,
) -> SpinnerLike:
    """
    Factory for the **Halo** single-line spinner. There is no alternate
    console style — this is the only progress UI in the project.
    Parameters mirror the Halo UX and honour environment overrides.
    """
    sp = _halo_percent_spinner(
        prefix=prefix,
        enabled=True,
        spinner_type=spinner_type or "dots",
        stream=stream,
        final_newline=final_newline if final_newline is not None else False,
        interval=interval,
    )
    return cast(SpinnerLike, sp)


class JobAwareProxy:
    """
    Spinner proxy that maps child `update(current=...)` calls into the spinner's
    `[Job: …]` field while preserving the `[File: …]` (item) slot.

    Usage:
        with percent_spinner(prefix="ARGOS") as sp:
            prox = JobAwareProxy(sp)
            # Child code calling .update(current="classify") will update [Job: …]
            child_run(..., progress=prox)
    """
    def __init__(self, inner: SpinnerLike) -> None:
        self._inner = inner

    def __enter__(self) -> "JobAwareProxy":
        self._inner.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> Optional[bool] | bool:
        return self._inner.__exit__(exc_type, exc, tb)

    def update(self, **kw: Any) -> "JobAwareProxy":
        # Map "current" to "job" unless the caller explicitly sets job
        if "current" in kw and "job" not in kw:
            value = kw.pop("current")
            # Do not overwrite 'item' (the pinned [File: …]); place into [Job: …]
            kw["job"] = value
        self._inner.update(**kw)
        return self


# Import AFTER defining percent_spinner to ensure there is no circular import.
# `bridges.live_percent` pulls the spinner from progress_ux directly, so this
# top-level import does not create a cycle.
from .bridges import live_percent  # noqa: E402
