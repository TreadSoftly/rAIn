from __future__ import annotations

import os
import sys
from typing import Any


def _truthy(val: str | None) -> bool:
    if not val:
        return False
    return val.strip().lower() in {"1", "true", "yes", "y"}


def _ensure_interactive_stdin() -> None:
    """
    If stdin isn't a TTY (some shells/launchers), try to re-attach to the real terminal
    so Typer/Click prompts work and don't silently accept defaults.
    """
    try:
        if sys.stdin and getattr(sys.stdin, "isatty", lambda: False)():
            return
        # Do not steal stdin in explicit autobuild mode
        if _truthy(os.getenv("ARGOS_AUTOBUILD")):
            return
        tty_name = "CONIN$" if os.name == "nt" else "/dev/tty"
        try:
            # Try dup2 onto existing stdin fd
            fd = os.open(tty_name, os.O_RDONLY)
            os.dup2(fd, sys.stdin.fileno())
        except Exception:
            # Fallback: replace sys.stdin object
            sys.stdin = open(tty_name, "r")  # type: ignore[assignment]
    except Exception:
        # Best effort only; if this fails, Typer will still try.
        pass


def main() -> None:
    assume_yes = _truthy(os.getenv("ARGOS_ASSUME_YES"))
    auto_build = _truthy(os.getenv("ARGOS_AUTOBUILD"))
    model_choice = (os.getenv("ARGOS_MODEL_CHOICE") or "1").strip()

    # Import the fetcher module (not just its main) so we can patch its prompt/confirm at runtime.
    import importlib

    fetch = importlib.import_module("panoptes.model._fetch_models")

    # Patch confirms if desired
    if assume_yes:
        try:
            import typer  # type: ignore

            def _always_yes(*_args: Any, **_kwargs: Any) -> bool:
                return True

            if hasattr(fetch, "typer"):
                try:
                    fetch.typer.confirm = _always_yes  # type: ignore[attr-defined]
                except Exception:
                    pass
            try:
                typer.confirm = _always_yes  # type: ignore[assignment]
            except Exception:
                pass
        except Exception:
            pass

    # Ensure we have an interactive stdin for prompts (unless explicit autobuild mode)
    _ensure_interactive_stdin()

    # If explicitly requested, pre-answer the first menu prompt with ARGOS_MODEL_CHOICE
    if auto_build:
        try:
            import typer  # type: ignore

            used = {"done": False}
            _orig_prompt = getattr(typer, "prompt", None)

            def _prompt_override(*args: Any, **kwargs: Any):
                text = ""
                if args:
                    text = str(args[0])
                elif "text" in kwargs:
                    text = str(kwargs["text"])
                if (not used["done"]) and ("Pick [0-4" in text or "Pick [0-4, ? for help]" in text):
                    used["done"] = True
                    return str(model_choice)
                if callable(_orig_prompt):
                    return _orig_prompt(*args, **kwargs)
                raise RuntimeError("prompt unavailable")

            if hasattr(fetch, "typer"):
                try:
                    fetch.typer.prompt = _prompt_override  # type: ignore[attr-defined]
                except Exception:
                    pass
            try:
                typer.prompt = _prompt_override  # type: ignore[assignment]
            except Exception:
                pass
        except Exception:
            pass

    try:
        # Defer to the real implementation (Typer prompts, progress, etc.)
        fetch.main()  # type: ignore[attr-defined]
    except Exception as e:
        # Clean exit if Click aborted because there was truly no TTY and no autobuild
        try:
            import click  # type: ignore
        except Exception:
            click = None  # type: ignore[assignment]

        if click is not None and isinstance(e, click.exceptions.Abort):  # type: ignore[attr-defined]
            sys.exit(0)
        raise


if __name__ == "__main__":
    main()
