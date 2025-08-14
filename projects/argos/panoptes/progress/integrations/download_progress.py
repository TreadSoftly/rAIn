"""
Helpers to stream downloads and emit byte-accurate progress.

- Uses urllib (no external deps).
- Emits total bytes (if Content-Length known) and increments as chunks arrive.
- Writes atomically via *.part then renames into place.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, runtime_checkable


@runtime_checkable
class _HeadersLike(Protocol):
    def get(self, key: str, default: Any = ...) -> Any: ...


def _content_length(hdrs: Mapping[str, Any] | _HeadersLike | None) -> int:
    try:
        if hdrs is None:
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


class ProgressLike(Protocol):
    def set_total(self, total_units: float) -> None: ...
    def set_current(self, label: str) -> None: ...
    def add(self, units: float, *, current_item: str | None = None) -> None: ...


def download_url(url: str, dest_path: str, engine: Optional[ProgressLike] = None) -> None:
    """
    Download *url* to *dest_path* reporting byte progress via *engine* (optional).
    """
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    req = urllib.request.Request(url, headers={"User-Agent": "Argos/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
        total = _content_length(getattr(resp, "headers", None))
        if engine is not None:
            try:
                engine.set_total(max(1, float(total) if total > 0 else 1.0))
                engine.set_current(dest.name)
            except Exception:
                pass

        with tmp.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                if engine is not None:
                    try:
                        engine.add(len(chunk))
                    except Exception:
                        pass

    try:
        os.replace(tmp, dest)
    except Exception:
        if dest.exists():
            dest.unlink(missing_ok=True)
        tmp.replace(dest)
