# projects/argos/panoptes/progress/integrations/fileio_progress.py
"""
File hashing/copying with byte-accurate progress. This module never creates a
spinner or alternate progress surface: if a ProgressEngine is provided, we
emit units to it; otherwise we remain silent.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from ..engine import ProgressEngine


def hash_file(path: str, engine: Optional[ProgressEngine] = None) -> str:
    """
    Stream a file through SHA-256 and return the hex digest.
    Emits byte counts to *engine* if provided.
    """
    p = Path(path)
    total = p.stat().st_size if p.exists() else 0
    if engine is not None:
        try:
            engine.set_total(max(1.0, float(total)))
            engine.set_current(p.name)
        except Exception:
            pass

    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
            if engine is not None:
                try:
                    engine.add(len(chunk))
                except Exception:
                    pass
    return h.hexdigest()


def copy_file(src: str, dst: str, engine: Optional[ProgressEngine] = None) -> None:
    """
    Copy *src* to *dst* in 1 MiB chunks, emitting byte counts to *engine*.
    Writes are atomic (*.part -> final). Silent if *engine* is None.
    """
    s = Path(src)
    d = Path(dst)
    d.parent.mkdir(parents=True, exist_ok=True)
    tmp = d.with_suffix(d.suffix + ".part")

    total = s.stat().st_size if s.exists() else 0
    if engine is not None:
        try:
            engine.set_total(max(1.0, float(total)))
            engine.set_current(d.name)
        except Exception:
            pass

    with s.open("rb") as fin, tmp.open("wb") as fout:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            fout.write(chunk)
            if engine is not None:
                try:
                    engine.add(len(chunk))
                except Exception:
                    pass

    try:
        os.replace(tmp, d)
    except Exception:
        if d.exists():
            d.unlink(missing_ok=True)
        tmp.replace(d)
