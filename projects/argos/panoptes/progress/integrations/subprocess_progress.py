# projects/argos/panoptes/progress/integrations/subprocess_progress.py
"""
Subprocess runner that streams a child process' combined stdout/stderr and
(optionally) echoes it, while letting callers count bytes to drive the single
project spinner (Halo/Rich ProgressEngine).

This module does not create any UI. If you want *only* the Halo line visible,
call with echo_output=False to suppress the child output and feed byte counts
to your engine via the on_bytes callback.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Any, Callable, Iterable, Optional, IO


def run_with_progress(
    args: Iterable[str],
    on_bytes: Optional[Callable[[int], None]] = None,
    *,
    echo_output: bool = True,
    echo_stream: Optional[IO[bytes]] = None,
    **kwargs: Any,
) -> int:
    """
    Run a subprocess and stream combined stdout/stderr to this process.

    Parameters
    ----------
    args : Iterable[str]
        Arg list for Popen.
    on_bytes : Optional[Callable[[int], None]]
        Callback invoked with the number of bytes seen per chunk.
    echo_output : bool, default True
        If True, echo the child's combined output. Set False if you want *only*
        the Halo spinner visible (no child output bars).
    echo_stream : Optional[IO[bytes]]
        Where to echo. Defaults to sys.stdout.buffer if None and echo_output=True.
    **kwargs : Any
        Extra Popen kwargs. We always force binary text mode (text=False),
        stderr=STDOUT, stdout=PIPE, bufsize=0.

    Returns
    -------
    int
        Process return code.
    """
    # Ensure binary mode so we can count bytes and (optionally) write raw chunks
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
    writer: Optional[IO[bytes]] = None
    if echo_output:
        writer = echo_stream if echo_stream is not None else getattr(sys.stdout, "buffer", None)

    try:
        for chunk in iter(lambda: out.read(8192), b""):
            if on_bytes is not None:
                try:
                    on_bytes(len(chunk))
                except Exception:
                    pass
            if writer is not None:
                try:
                    writer.write(chunk)
                    writer.flush()
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
