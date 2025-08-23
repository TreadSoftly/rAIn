# projects/argos/panoptes/progress/integrations/__init__.py
"""
Integration helpers that emit *real* measurable units to the Argos ProgressEngine
(the Halo/Rich spinner). These modules **never** create their own progress UI;
they only report to a provided engine so your single spinner remains the sole
progress surface.

Included:
- pip / wheel installs (parse output, count packages/files)
- subprocess streaming (optional echo, byte counting)
- downloads (Content-Length, bytes streamed)
- hashing / file IO (bytes)
"""
from __future__ import annotations

from .subprocess_progress import run_with_progress
from .pip_progress import parse_pip_stream
from .download_progress import download_url
from .fileio_progress import hash_file, copy_file

__all__ = [
    "run_with_progress",
    "parse_pip_stream",
    "download_url",
    "hash_file",
    "copy_file",
]
