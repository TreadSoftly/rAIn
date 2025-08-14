# projects/argos/panoptes/progress/integrations/subprocess_progress.py
from __future__ import annotations

import subprocess
import sys
from typing import Any, Callable, Iterable, Optional


def run_with_progress(
    args: Iterable[str],
    on_bytes: Optional[Callable[[int], None]] = None,
    **kwargs: Any,
) -> int:
    """
    Run a subprocess and stream combined stdout/stderr to this process' stdout.

    Parameters
    ----------
    args
        Arg list for Popen.
    on_bytes
        Callback invoked with the number of bytes seen per chunk.
    kwargs
        Passed to Popen. We always force binary text mode (text=False).

    Returns
    -------
    int
        Process return code.
    """
    # Ensure binary mode so we can count bytes and write to stdout.buffer
    popen_kwargs = dict(kwargs)
    popen_kwargs.pop("text", None)
    popen_kwargs.pop("universal_newlines", None)
    popen_kwargs.pop("encoding", None)
    popen_kwargs.setdefault("stdout", subprocess.PIPE)
    popen_kwargs.setdefault("stderr", subprocess.STDOUT)
    popen_kwargs.setdefault("bufsize", 0)

    p: subprocess.Popen[bytes] = subprocess.Popen(  # type: ignore[type-arg]
        list(args),
        **popen_kwargs,
    )

    assert p.stdout is not None
    out = p.stdout

    try:
        for chunk in iter(lambda: out.read(8192), b""):
            if on_bytes is not None:
                try:
                    on_bytes(len(chunk))
                except Exception:
                    pass
            try:
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
            except Exception:
                # Never let logging failures kill the subprocess loop
                pass
    except KeyboardInterrupt:
        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    finally:
        try:
            rc = p.wait()
        except Exception:
            rc = -1

    return int(rc)
