# projects/argos/panoptes/progress/integrations/download_progress.py
"""
Helpers to stream downloads and emit byte-accurate progress to the *single*
project progress engine (Halo/Rich). No UI is created here; if an engine is
provided, we send totals + increments. Writes are atomic (*.part -> final).
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional, cast

from ..engine import ProgressEngine


def _content_length(hdrs: Mapping[str, Any] | None) -> int:
    try:
        if not hdrs:
            return 0
        raw = hdrs.get("Content-Length", None)
        if raw is None:
            return 0
        if isinstance(raw, (bytes, bytearray)):
            s = raw.decode("ascii", "ignore")
        else:
            s = str(raw)
        return int(s)
    except Exception:
        return 0


def download_url(url: str, dest_path: str, engine: Optional[ProgressEngine] = None) -> None:
    """
    Download *url* to *dest_path* reporting byte progress via *engine* (optional).

    Parameters
    ----------
    url : str
        Source URL.
    dest_path : str
        Destination file path.
    engine : Optional[ProgressEngine]
        If provided, receives set_total(), set_current(), and add() calls.
        This function never creates its own UI.
    """
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    req = urllib.request.Request(url, headers={"User-Agent": "Argos/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
        headers_obj = getattr(resp, "headers", None)
        mapping_headers: Optional[Mapping[str, Any]]
        if isinstance(headers_obj, Mapping):
            # Narrow Unknown key/value types to concrete ones for the type checker
            mapping_headers = cast(Mapping[str, Any], headers_obj)
        else:
            mapping_headers = None
        total = _content_length(mapping_headers)

        if engine is not None:
            try:
                engine.set_total(max(1.0, float(total) if total > 0 else 1.0))
                engine.set_current(dest.name)
            except Exception:
                # Never let progress failures break the download
                pass

        with tmp.open("wb") as fh:
            for chunk in iter(lambda: resp.read(1024 * 1024), b""):
                fh.write(chunk)
                if engine is not None:
                    try:
                        engine.add(len(chunk))
                    except Exception:
                        pass

    # Atomic finalize
    try:
        os.replace(tmp, dest)
    except Exception:
        if dest.exists():
            dest.unlink(missing_ok=True)
        tmp.replace(dest)
