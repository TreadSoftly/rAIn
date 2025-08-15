# \rAIn\projects\argos\panoptes\progress\bridges.py
from __future__ import annotations

import sys
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Callable, Generator, Protocol, cast

# Import via the package so we get the rich spinner OR the stdlib fallback.
from . import percent_spinner  # type: ignore
from .engine import (
    ProgressEngine,  # type: ignore
    ProgressState,
)


class _SpinnerLike(Protocol):
    def __enter__(self) -> "_SpinnerLike": ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None: ...
    def update(self, **kwargs: Any) -> "_SpinnerLike": ...


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


class _NullSpinner:
    """
    No-op spinner used as a safe fallback when rich spinner initialization
    fails (e.g., invalid handle on Windows under pytest capture).
    """
    def __enter__(self) -> "_NullSpinner":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return False

    def update(self, **kwargs: Any) -> "_NullSpinner":
        return self


@contextmanager
def live_percent(engine: ProgressEngine, *, prefix: str = "PROGRESS") -> Generator[_SpinnerLike, None, None]:
    """
    Context manager that opens a percent spinner and keeps it in sync with engine.
    Yields the spinner (with optional `.engine` attr attached).

    Robust against environments where the underlying stream is not a valid
    handle (e.g., Windows + pytest capture). Falls back to a no-op spinner.
    """
    # Prefer real stderr so progress remains visible even when stdio is redirected
    stream = getattr(sys, "__stderr__", sys.stderr)

    # Try to create the rich spinner
    sp = percent_spinner(prefix=prefix, stream=stream)  # type: ignore[call-arg]
    sp_typed = cast(_SpinnerLike, sp)
    try:
        setattr(sp_typed, "engine", engine)  # optional introspection
    except Exception:
        pass

    # Bind updates to the spinner
    unsub = _bind_spinner(engine, sp_typed)

    # Attempt to enter the spinner; if that fails, fall back to a Null spinner
    entered = False
    try:
        try:
            sp_typed.__enter__()  # may raise (e.g., invalid stream handle)
            entered = True
        except Exception:
            # Unbind the failed spinner and switch to a safe no-op
            try:
                unsub()
            except Exception:
                pass
            null_sp = _NullSpinner()
            unsub2 = _bind_spinner(engine, null_sp)
            try:
                with null_sp:
                    yield null_sp
            finally:
                try:
                    unsub2()
                except Exception:
                    pass
            return

        # If we got here, the rich spinner is active
        try:
            yield sp_typed
        finally:
            try:
                if entered:
                    sp_typed.__exit__(None, None, None)
            finally:
                try:
                    unsub()
                except Exception:
                    pass
    except Exception:
        # Ensure we always unbind on unexpected errors
        try:
            unsub()
        except Exception:
            pass
        raise
