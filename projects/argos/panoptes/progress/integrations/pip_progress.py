# projects/argos/panoptes/progress/integrations/pip_progress.py
"""
Lightweight pip output parser that converts notable events into "units":

  +1  on "Downloading ..."
  +1  on "Building wheel for ..."
  +N  on "Installing collected packages: a, b, c" (N packages)
  +1  on "Preparing metadata (pyproject.toml) for ..."
  +1  on "Preparing metadata (setup.py) for ..."

This is heuristic but stable across common pip versions. Callers can seed
the ProgressEngine with a total (e.g., count of packages) and then pass
on_units(Δ) here to increment as these milestones are observed.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable


_RE_DOWNLOAD = re.compile(r"^\s*Downloading\s+", re.IGNORECASE)
_RE_BUILD_WHEEL = re.compile(r"^\s*Building wheel for\s+", re.IGNORECASE)
_RE_PREPARE_META = re.compile(r"^\s*Preparing metadata\s+\(", re.IGNORECASE)
_RE_INSTALLING = re.compile(r"^\s*Installing collected packages:\s*(.+)$", re.IGNORECASE)


def _count_pkg_list(s: str) -> int:
    # Split by comma and/or space, ignore empty tokens
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return len(parts)


def parse_pip_stream(lines: Iterable[bytes], on_units: Callable[[int], None]) -> int:
    """
    Consume pip's combined stdout/stderr byte stream line-by-line, decode, and emit units.
    Returns the total units emitted.
    """
    emitted = 0
    for raw in lines:
        try:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
        except Exception:
            continue

        if _RE_DOWNLOAD.search(line):
            on_units(1)
            emitted += 1
            continue

        if _RE_BUILD_WHEEL.search(line) or _RE_PREPARE_META.search(line):
            on_units(1)
            emitted += 1
            continue

        m = _RE_INSTALLING.search(line)
        if m:
            n = _count_pkg_list(m.group(1))
            if n > 0:
                on_units(n)
                emitted += n
            continue

        # Ignore progress bars and non-signal lines.

    return emitted
