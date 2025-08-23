from __future__ import annotations

import argparse
import time
from typing import Any, ContextManager, Protocol, Sequence, cast

from . import (
    ProgressEngine,
    live_percent,
    percent_spinner,
)

# Tell the type checker what the engine supports
class ProgressLike(Protocol):
    def set_total(self, total_units: float) -> None: ...
    def set_current(self, label: str) -> None: ...
    def add(self, units: float, *, current_item: str | None = None) -> None: ...


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m panoptes.progress",
        add_help=True,
    )
    p.add_argument(
        "mode",
        nargs="?",
        choices=["demo", "engine"],
        default="demo",
        help="demo: UX-only; engine: ProgressEngine + bridge",
    )
    args = p.parse_args(argv)

    items = ["alpha", "beta", "gamma", "delta", "epsilon"]

    if args.mode == "demo":
        with percent_spinner(prefix="ARGOS") as sp:
            sp.update(total=len(items))
            for i, it in enumerate(items, 1):
                sp.update(item=it)   # keep [File: …] pinned
                time.sleep(0.15)
                sp.update(count=i, job="demo")  # [Job: demo]
        return 0

    eng: ProgressLike = cast(ProgressLike, ProgressEngine())  # type: ignore[call-arg]
    ctx: ContextManager[Any] = cast(ContextManager[Any], live_percent(eng, prefix="ENGINE"))  # type: ignore[misc]
    with ctx:
        eng.set_total(float(len(items)))
        for it in items:
            eng.set_current(it)
            time.sleep(0.15)
            eng.add(1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
