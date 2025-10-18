from __future__ import annotations

import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .logging_config import current_run_dir

DEFAULT_FILES: tuple[str, ...] = ("argos.log", "env.json")


def _iter_paths(values: Optional[Sequence[object]]) -> Iterable[Path]:
    if not values:
        return ()
    for value in values:
        if value is None:
            continue
        if isinstance(value, Path):
            yield value
            continue
        if isinstance(value, str):
            yield Path(value)


def write_support_bundle(
    *,
    run_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    include_patterns: Optional[Sequence[str]] = None,
    extra_paths: Optional[Sequence[object]] = None,
    metadata: Optional[dict[str, object]] = None,
) -> Path:
    """
    Create a compressed support bundle containing the key run artifacts.

    Parameters
    ----------
    run_dir:
        The logging run directory. Defaults to the current run directory.
    output_path:
        Where to write the bundle. Defaults to ``run_dir/support_<timestamp>.zip``.
    include_patterns:
        Additional glob patterns (relative to ``run_dir``) to include in the bundle.
    extra_paths:
        Additional absolute paths (inside or outside the run directory) to include.
    metadata:
        Extra key/value pairs to embed in ``support.json`` inside the archive.
    """
    run = run_dir or current_run_dir()
    if run is None:
        raise RuntimeError("No logging run directory available; call setup_logging() first.")
    run = run.resolve()
    if not run.exists():
        raise FileNotFoundError(run)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if output_path is None:
        output_path = run / f"support_{timestamp}.zip"
    else:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    files_to_include: list[tuple[Path, str]] = []

    # Core run files
    for name in DEFAULT_FILES:
        candidate = run / name
        if candidate.exists() and candidate.is_file():
            files_to_include.append((candidate, candidate.name))

    # Additional run patterns
    if include_patterns:
        for pattern in include_patterns:
            for match in run.glob(pattern):
                if match.is_file():
                    arcname = match.relative_to(run)
                    files_to_include.append((match, str(arcname)))

    # Extra explicit paths
    for extra in _iter_paths(extra_paths):
        extra = extra.resolve()
        if not extra.exists() or not extra.is_file():
            continue
        try:
            arcname = extra.relative_to(run)
            files_to_include.append((extra, str(arcname)))
        except ValueError:
            # Place external files under extras/
            arcname = Path("extras") / extra.name
            files_to_include.append((extra, str(arcname)))

    seen: set[str] = set()
    unique_files: list[tuple[Path, str]] = []
    for path, arcname in files_to_include:
        key = arcname.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_files.append((path, arcname))

    manifest_files: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(run),
        "files": manifest_files,
    }
    if metadata:
        manifest.update(metadata)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path, arcname in unique_files:
            try:
                bundle.write(path, arcname)
                stat = path.stat()
                manifest_files.append(
                    {
                        "arcname": arcname,
                        "size": stat.st_size,
                        "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(stat.st_mtime)),
                    }
                )
            except FileNotFoundError:
                continue
        bundle.writestr("support.json", json.dumps(manifest, indent=2, ensure_ascii=False))

    return output_path
