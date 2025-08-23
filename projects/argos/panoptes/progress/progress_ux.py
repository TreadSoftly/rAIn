# progress_ux.py — terminal UX helpers (HALO/Rich spinner, status, OSC-8 links)
from __future__ import annotations

import itertools
import os
import re
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Sequence, Type, cast

try:
    from halo import Halo  # type: ignore[import]
except Exception:
    Halo = None

try:
    from colorama import Fore, Style
    from colorama import init as colorama_init
    colorama_init(autoreset=True)
except Exception:  # pragma: no cover
    CSI = "\x1b["
    class _F:
        BLACK = CSI + "30m"
        RED = CSI + "31m"
        GREEN = CSI + "32m"
        YELLOW = CSI + "33m"
        BLUE = CSI + "34m"
        MAGENTA = CSI + "35m"
        CYAN = CSI + "36m"
        WHITE = CSI + "37m"
        LIGHTBLACK_EX = CSI + "90m"
        LIGHTRED_EX = CSI + "91m"
        LIGHTGREEN_EX = CSI + "92m"
        LIGHTYELLOW_EX = CSI + "93m"
        LIGHTBLUE_EX = CSI + "94m"
        LIGHTMAGENTA_EX = CSI + "95m"
        LIGHTCYAN_EX = CSI + "96m"
        LIGHTWHITE_EX = CSI + "97m"
    class _S:
        BRIGHT = CSI + "1m"
        RESET_ALL = CSI + "0m"
    Fore, Style = _F(), _S()

DEFAULT_COLORS: List[str] = [
    Fore.RED, Fore.GREEN, Fore.BLUE, Fore.YELLOW, Fore.CYAN, Fore.MAGENTA, Fore.WHITE,
    getattr(Fore, "LIGHTBLACK_EX", Fore.WHITE),
    getattr(Fore, "LIGHTBLUE_EX", Fore.BLUE),
    getattr(Fore, "LIGHTCYAN_EX", Fore.CYAN),
    getattr(Fore, "LIGHTGREEN_EX", Fore.GREEN),
    getattr(Fore, "LIGHTMAGENTA_EX", Fore.MAGENTA),
    getattr(Fore, "LIGHTRED_EX", Fore.RED),
    getattr(Fore, "LIGHTWHITE_EX", Fore.WHITE),
    getattr(Fore, "LIGHTYELLOW_EX", Fore.YELLOW),
]

class Spinner(Protocol):
    def __enter__(self) -> "Spinner": ...
    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]) -> None: ...
    def start(self) -> "Spinner": ...
    def stop(self) -> None: ...
    def update(self, **kwargs: Any) -> "Spinner": ...
    @property
    def text(self) -> str: ...
    @text.setter
    def text(self, value: str) -> None: ...

def _should_enable_spinners(enabled: bool, stream: Any | None = None) -> bool:
    if not enabled or Halo is None:
        return False
    # Suppress nested spinners unless forced.
    if (os.environ.get("PANOPTES_PROGRESS_ACTIVE") == "1" and
        os.environ.get("PANOPTES_PROGRESS_FORCE", "").lower() not in {"1", "true", "yes"}):
        return False
    try:
        target = stream or sys.stderr
        if not getattr(target, "isatty", lambda: False)():
            return False
    except Exception:
        return False
    if os.environ.get("CI"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return True

def should_enable_spinners(stream: Any | None = None) -> bool:
    try:
        return _should_enable_spinners(True, stream)
    except Exception:
        return False

class NullSpinner:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass
    def __enter__(self) -> "Spinner":
        return self
    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]) -> None:
        self.stop()
    def start(self) -> "Spinner":
        return self
    def stop(self) -> None:
        return None
    def update(self, **_: Any) -> "Spinner":
        return self
    @property
    def text(self) -> str:
        return ""
    @text.setter
    def text(self, value: str) -> None:
        pass

if TYPE_CHECKING:
    class HaloType(Protocol):
        text: str
        def start(self, text: Optional[str] = ...) -> Any: ...
        def stop(self) -> Any: ...
        stream: Any
else:
    HaloType = Any  # at runtime, treat as Any

def _make_const_text(value: str) -> Callable[[Dict[str, Any], str], str]:
    def _fn(_s: Dict[str, Any], _c: str) -> str:
        return value
    return _fn

class DynamicSpinner:
    """
    A single-line, fixed-width spinner rendered via Halo with colored,
    bracketed middle segments: [File: …] [Job: …] [Model: …] and a percent tail.
    """
    def __init__(
        self,
        text_fn: Callable[[Dict[str, Any], str], str],
        state: Optional[Dict[str, Any]] = None,
        *,
        interval: float = 0.1,
        colors: Optional[List[str]] = None,
        spinner_type: str = "dots",
        enabled: bool = True,
        stream: Any | None = None,
        final_newline: bool = True,
    ):
        self._text_fn = text_fn
        self._state: Dict[str, Any] = dict(state or {})
        self._interval = interval
        self._colors = itertools.cycle(colors or DEFAULT_COLORS)
        self._stop = threading.Event()
        self._stream = stream or sys.stderr
        self._enabled = _should_enable_spinners(enabled, self._stream)
        self._spinner: Optional[HaloType] = None
        self._final_newline = final_newline
        if self._enabled and Halo is not None:
            self._spinner = cast(HaloType, Halo(text="", spinner=spinner_type, stream=self._stream))  # type: ignore[call-arg]
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "Spinner":
        os.environ["PANOPTES_PROGRESS_ACTIVE"] = "1"
        return self.start()

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]) -> None:
        try:
            self.stop()
        finally:
            os.environ.pop("PANOPTES_PROGRESS_ACTIVE", None)

    def start(self) -> "Spinner":
        if not self._enabled:
            return self
        self._stop.clear()
        try:
            first_text = self._text_fn(self._state, next(self._colors))
        except Exception:
            first_text = ""
        assert self._spinner is not None
        self._spinner.text = first_text
        self._spinner.start()

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    color = next(self._colors)
                    assert self._spinner is not None
                    self._spinner.text = self._text_fn(self._state, color)
                except Exception:
                    pass
                time.sleep(self._interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if not self._enabled:
            return
        self._stop.set()
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
        finally:
            try:
                assert self._spinner is not None
                self._spinner.stop()
            except Exception:
                pass
            if self._final_newline:
                try:
                    target = getattr(self._spinner, "stream", None) or self._stream or sys.stderr
                    target.write("\n")
                    target.flush()
                except Exception:
                    try:
                        sys.stderr.write("\n")
                        sys.stderr.flush()
                    except Exception:
                        pass

    def update(self, **kwargs: Any) -> "Spinner":
        # Accepts keys: total, count, item/current, job, model
        # Map legacy 'current' to [File: …] (item) — callers who want to update
        # the [Job: …] slot should pass job=... explicitly or use JobAwareProxy.
        if "current" in kwargs and "item" not in kwargs:
            kwargs["item"] = kwargs.pop("current")
        self._state.update(kwargs)
        return self

    @property
    def text(self) -> str:
        return self._text_fn(self._state, "")
    @text.setter
    def text(self, value: str) -> None:
        self._text_fn = _make_const_text(value)


def enabled_spinner(
    enabled: bool = True,
    *,
    stream: Any | None = None,
    final_newline: bool = True,
) -> Callable[..., Spinner]:
    def _ctor(
        text_fn: Callable[[Dict[str, Any], str], str],
        state: Optional[Dict[str, Any]] = None,
        *,
        interval: float = 0.1,
        colors: Optional[List[str]] = None,
        spinner_type: str = "dots",
    ) -> Spinner:
        chosen_stream = stream or sys.stderr
        if _should_enable_spinners(enabled, chosen_stream):
            return DynamicSpinner(
                text_fn,
                state,
                interval=interval,
                colors=colors,
                spinner_type=spinner_type,
                enabled=True,
                stream=chosen_stream,
                final_newline=final_newline,
            )
        return NullSpinner()
    return _ctor


def running_task(
    task: str,
    subject: str,
    *,
    enabled: bool = True,
    spinner_type: str = "dots",
    stream: Any | None = None,
    final_newline: bool = True,
) -> Spinner:
    ctor = enabled_spinner(enabled, stream=stream, final_newline=final_newline)
    def text_fn(_state: Dict[str, Any], color: str) -> str:
        return f"{color}{Style.BRIGHT}Running {task} on {subject}{Style.RESET_ALL}"
    return ctor(text_fn, spinner_type=spinner_type)


def simple_status(
    label: str,
    *,
    enabled: bool = True,
    spinner_type: str = "dots",
    stream: Any | None = None,
    final_newline: bool = True,
) -> Spinner:
    ctor = enabled_spinner(enabled, stream=stream, final_newline=final_newline)
    def text_fn(_s: Dict[str, Any], color: str) -> str:
        return f"{color}{Style.BRIGHT}{label}{Style.RESET_ALL}"
    return ctor(text_fn, spinner_type=spinner_type)


# helpers for percent spinner text
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
def _visible_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))
def _ellipsize_plain(s: str, n: int) -> str:
    s = str(s or "")
    if n <= 0:
        return ""
    if len(s) <= n:
        return s
    if n == 1:
        return "…"
    return s[: n - 1].rstrip() + "…"


def percent_spinner(
    prefix: str = "PROGRESS",
    *,
    enabled: bool = True,
    spinner_type: str = "dots",
    stream: Any | None = None,
    final_newline: bool = True,
    interval: float | None = None,
) -> Spinner:
    """
    Fixed-width, single-line progress:
    <prefix> [File:…] [Job:…] [Model:…]  [i/N] [zz.zz%]
    We keep colours by shrinking only the VALUE parts.
    The spinner is Halo-based; no other progress UI is used.
    """
    # Width (visible characters only)
    DEFAULT_LINE_COLS = 100
    try:
        env_w = os.environ.get("PANOPTES_PROGRESS_WIDTH")
        latched_cols = max(20, int(env_w)) if env_w else DEFAULT_LINE_COLS
    except Exception:
        latched_cols = DEFAULT_LINE_COLS

    tail_mode = (os.environ.get("PANOPTES_PROGRESS_TAIL", "full") or "full").lower()

    # Allow env overrides for spinner glyph, interval, and final newline
    spinner_type = os.environ.get("PANOPTES_SPINNER", spinner_type)
    if "PANOPTES_PROGRESS_FINAL_NEWLINE" in os.environ:
        final_newline = (os.environ.get("PANOPTES_PROGRESS_FINAL_NEWLINE", "0").lower() in {"1", "true", "yes"})
    if interval is None:
        try:
            interval = float(os.environ.get("PANOPTES_SPINNER_INTERVAL", "0.01"))
        except Exception:
            interval = 0.01

    ctor = enabled_spinner(enabled, stream=stream, final_newline=final_newline)

    ITEM_MIN_W = 1

    def text_fn(state: Dict[str, Any], color: str) -> str:
        from colorama import Fore as F, Style as S

        count = max(0, int(state.get("count", 0)))
        total = max(1, int(state.get("total", 1)))
        item  = str(state.get("item") or state.get("current") or "")
        job   = str(state.get("job") or "")
        model = str(state.get("model") or "")
        pct = min(100.0, max(0.0, (100.0 * count) / max(1, total)))
        w = len(str(total))

        head = f"{color}{S.BRIGHT}{prefix}{S.RESET_ALL} "

        seg_head_item = f"{F.GREEN}{S.BRIGHT}[File:{S.RESET_ALL}{F.RED}{S.BRIGHT}"
        seg_tail_item = f"{S.RESET_ALL}{F.GREEN}{S.BRIGHT}]"
        seg_head_job  = f"{F.CYAN}{S.BRIGHT}[Job:{S.RESET_ALL}{F.RED}{S.BRIGHT}"
        seg_tail_job  = f"{S.RESET_ALL}{F.CYAN}{S.BRIGHT}]"
        seg_head_mod  = f"{F.WHITE}{S.BRIGHT}[Model:{S.RESET_ALL}{F.RED}{S.BRIGHT}"
        seg_tail_mod  = f"{S.RESET_ALL}{F.WHITE}{S.BRIGHT}]"

        tail_full  = f"{F.MAGENTA}{S.BRIGHT} [{count:>{w}}/{total:{w}}] [{pct:6.2f}%]{S.RESET_ALL}"
        tail_short = f"{F.MAGENTA}{S.BRIGHT} [{count}/{total}] [{pct:5.1f}%]{S.RESET_ALL}"
        tail_min   = f"{F.MAGENTA}{S.BRIGHT} [{count}/{total}] [{int(round(pct))}%]{S.RESET_ALL}"

        preferred  = tail_full if tail_mode == "full" else (tail_short if tail_mode == "short" else tail_min)

        tail = preferred
        for t in (preferred, tail_full, tail_short, tail_min):
            if _visible_len(head) + _visible_len(t) + 3 <= latched_cols:
                tail = t
                break

        mid_budget = max(3, latched_cols - _visible_len(head) - _visible_len(tail))

        # Build visible-overhead for the bracketed coloured segments
        segments: list[tuple[str, str, str]] = []
        if item:
            segments.append((seg_head_item, item, seg_tail_item))
        if job:
            segments.append((seg_head_job, job, seg_tail_job))
        if model:
            segments.append((seg_head_mod, model, seg_tail_mod))

        if not segments:
            mid_text = ""
            mid_vis = 0
        else:
            overhead = 0
            for i, (op, _val, cl) in enumerate(segments):
                overhead += _visible_len(op) + _visible_len(cl)
                if i > 0:
                    overhead += 1  # space between segments

            avail_vals = max(0, mid_budget - overhead)
            remaining = avail_vals
            n = len(segments)
            allowed: list[int] = []
            for i, (_op, val, _cl) in enumerate(segments):
                reserve = (n - i - 1) * ITEM_MIN_W
                allow = max(ITEM_MIN_W, min(len(val), max(0, remaining - reserve)))
                allowed.append(allow)
                remaining -= allow

            parts: List[str] = []
            for i, (op, val, cl) in enumerate(segments):
                if i > 0:
                    parts.append(" ")
                parts.append(op + _ellipsize_plain(val, allowed[i]) + cl)
            mid_text = "".join(parts)
            mid_vis = _visible_len(mid_text)

        if mid_vis < mid_budget:
            mid_text = mid_text + (" " * (mid_budget - mid_vis))

        return f"{head}{mid_text}{tail}"

    # Pass selected interval to the DynamicSpinner via the ctor
    return ctor(text_fn, state={"count": 0, "total": 1, "item": None}, interval=float(interval), spinner_type=spinner_type)


def osc8(label: str, target: str) -> str:
    try:
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", target):
            uri = target
        else:
            uri = Path(target).resolve().as_uri()
    except Exception:
        uri = urllib.parse.quote(target, safe=":/#?&=%")
    return f"\x1b]8;;{uri}\x1b\\{label}\x1b]8;;\x1b\\"


class AnimatedBanner:
    def __init__(self, lines: Sequence[str], mutator: Callable[[int, List[str]], List[str]], *, interval: float = 0.05):
        self._base: List[str] = list(lines)
        self._mutator = mutator
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._step: int = 0

    def start(self) -> "AnimatedBanner":
        self._stop.clear()
        def _loop() -> None:
            while not self._stop.is_set():
                frame = self._mutator(self._step, list(self._base))
                self._render(frame)
                self._step += 1
                time.sleep(self._interval)
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def run_for(self, seconds: float) -> None:
        self.start()
        try:
            time.sleep(max(0.0, seconds))
        finally:
            self.stop()

    def run_until(self, predicate: Callable[[], bool]) -> None:
        self.start()
        try:
            while not predicate():
                time.sleep(0.05)
        finally:
            self.stop()

    def _render(self, lines: Sequence[str]) -> None:
        print("\033[H\033[J" + "\n".join(lines))
