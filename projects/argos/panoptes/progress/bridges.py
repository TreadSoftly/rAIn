from __future__ import annotations

import sys
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Callable, Generator, Protocol, Self, cast

# Import via the package so we get the rich spinner OR the stdlib fallback.
from . import percent_spinner  # type: ignore
from .engine import (
    ProgressEngine,  # type: ignore
    ProgressState,
)


class _SpinnerLike(Protocol):
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None: ...
    def update(self, **kwargs: Any) -> Self: ...

def _bind_spinner(engine: ProgressEngine, spinner: _SpinnerLike) -> Callable[[], None]:
    """Subscribe engine → spinner; return unsubscribe."""
    def on_update(st: ProgressState) -> None:
        try:
            total = float(getattr(st, "total_units", 0))
            done = float(getattr(st, "done_units", 0))
            current = getattr(st, "current_item", None) or ""
            spinner.update(total=total, count=done, current=current)
        except Exception:
            # UX must not break the workload.
            pass

    try:
        return engine.on_update(on_update)  # type: ignore[attr-defined]
    except Exception:
        return lambda: None

@contextmanager
def live_percent(engine: ProgressEngine, *, prefix: str = "PROGRESS") -> Generator[_SpinnerLike, None, None]:
    """
    Context manager that opens a percent spinner and keeps it in sync with engine.
    Yields the spinner (with optional `.engine` attr attached).
    """
    sp = percent_spinner(prefix=prefix, stream=getattr(sys, "__stderr__", sys.stderr))  # prefer real stderr
    sp_typed = cast(_SpinnerLike, sp)
    try:
        setattr(sp_typed, "engine", engine)  # optional introspection
    except Exception:
        pass

    unsub = _bind_spinner(engine, sp_typed)
    try:
        with sp_typed:
            yield sp_typed
    finally:
        try:
            unsub()
        except Exception:
            pass
