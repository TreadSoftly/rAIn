# \rAIn\projects\argos\panoptes\progress\__init__.py
"""
Panoptes Progress — single-line, fixed-width progress with color cycling.

Behavior
• Exactly one line (no wrapping). Tail stays aligned and stable.
• Middle block flexes and shows: [ITEM …] [JOB …] [MODEL …]
• Visible width is constant for the spinner lifetime (default 75 cols).
• Uses progress_ux (Halo) when available; otherwise a stdlib clone.

Env knobs
  PANOPTES_PROGRESS_WIDTH            int (default 75)
  PANOPTES_PROGRESS_TAIL             full | short | min   (default full)
  PANOPTES_SPINNER                   line|dots|arrow|bounce (default line)
  PANOPTES_SPINNER_INTERVAL          seconds (default 0.10)
  PANOPTES_PROGRESS_FINAL_NEWLINE    0/1, true/false (default 0 → no newline)
  PANOPTES_PROGRESS_FORCE            1 to force nested spinners to print

Steady/flicker control (fallback spinner only)
  PANOPTES_SPINNER_STEADY            1 → freeze animation; repaint only on state change
  PANOPTES_PROGRESS_COLOR            e.g. GREEN / LIGHTBLUE_EX (fix color; no cycling)
"""

from __future__ import annotations

import inspect
import itertools
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Callable, ContextManager, Iterator, Optional, Protocol, TextIO, cast

# NOTE: Don't hard-import typing.Self on Py<3.11 (breaks 3.9/3.10).
# If a consumer evaluates annotations with typing.get_type_hints, ensure 'Self' resolves.
try:  # Python 3.11+
    from typing import Self  # type: ignore
except Exception:  # Python <= 3.10
    try:
        from typing_extensions import Self  # type: ignore
    except Exception:
        class Self:  # type: ignore[misc]
            """Runtime placeholder so get_type_hints doesn't explode on 'Self'."""
            pass

__all__ = [
    "ProgressEngine", "Phase", "Task", "ProgressState",
    "live_percent", "percent_spinner", "simple_status", "running_task",
    "osc8",
]

# ── 1) Engine (safe fallback) ────────────────────────────────────────────────
try:
    from .engine import (
        Phase,  # type: ignore
        ProgressEngine,  # type: ignore
        ProgressState,  # type: ignore
        Task,  # type: ignore
    )
except Exception:
    class ProgressState:  # type: ignore[no-redef]
        total_units: int = 0
        done_units: int = 0
        current_item: Optional[str] = None

    class ProgressEngine:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self._st = ProgressState()
            self._listeners: list[Callable[[ProgressState], None]] = []

        def _snapshot(self) -> ProgressState:
            s = ProgressState()
            s.total_units = int(self._st.total_units)
            s.done_units = int(self._st.done_units)
            s.current_item = self._st.current_item
            return s

        def _emit(self) -> None:
            snap = self._snapshot()
            for fn in list(self._listeners):
                try:
                    fn(snap)
                except Exception:
                    pass

        def on_update(self, fn: Callable[[ProgressState], None]) -> Callable[[], None]:
            self._listeners.append(fn)
            def _unsub() -> None:
                try:
                    self._listeners.remove(fn)
                except ValueError:
                    pass
            return _unsub

        def set_total(self, n: int | float) -> None:
            self._st.total_units = max(0, int(n))
            self._emit()

        def set_current(self, item: Optional[str]) -> None:
            self._st.current_item = item
            self._emit()

        def add(self, units: int | float = 1, current_item: Optional[str] = None) -> None:
            self._st.done_units = max(0, int(self._st.done_units) + int(units))
            if current_item is not None:
                self._st.current_item = current_item
            self._emit()

    Phase = Task = None  # type: ignore[assignment]

# ── 2) Public helpers — define fallbacks unconditionally, rebind if UX exists ─
@contextmanager
def simple_status(message: str) -> Iterator[None]:
    try:
        sys.stderr.write(f"{message}\n")
        sys.stderr.flush()
    except Exception:
        pass
    yield

def running_task(*_a: Any, **_k: Any) -> ContextManager[None]:
    return simple_status("running")

def osc8(label: str, target: str) -> str:
    ESC = "\x1b"
    return f"{ESC}]8;;{target}{ESC}\\{label}{ESC}]8;;{ESC}\\"

_have_ux = True
_ux_percent_spinner: Optional[Callable[..., Any]] = None
_ux_should_enable: Optional[Callable[[Any | None], bool]] = None

try:
    # import the module and rebind rich versions to the public names
    from . import progress_ux as _ux  # type: ignore
    _ux_percent_spinner = _ux.percent_spinner
    _ux_should_enable = _ux.should_enable_spinners
    simple_status = _ux.simple_status        # type: ignore[assignment]
    running_task = _ux.running_task          # type: ignore[assignment]
    osc8 = _ux.osc8                          # type: ignore[assignment]
except Exception:
    _have_ux = False

# ── 3) Spinner protocols ──────────────────────────────────────────────────────
class _SpinnerLike(Protocol):
    # Use protocol self-type to avoid relying on PEP 673 Self in all environments
    def __enter__(self) -> "_SpinnerLike": ...
    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: TracebackType | None) -> bool | None: ...
    def update(self, **kwargs: Any) -> "_SpinnerLike": ...

class _ProgressStateLike(Protocol):
    total_units: int
    done_units: int
    current_item: Optional[str]

class _EngineLike(Protocol):
    def on_update(self, fn: Callable[[_ProgressStateLike], None]) -> Callable[[], None]: ...

# ── 4) Stdlib fallback (same look/feel) ───────────────────────────────────────
try:
    from colorama import Fore, Style
    from colorama import init as _colorama_init
    _colorama_init(autoreset=True)
except Exception:
    CSI = "\x1b["
    class _F:
        RED = CSI + "31m"
        GREEN = CSI + "32m"
        BLUE = CSI + "34m"
        YELLOW = CSI + "33m"
        CYAN = CSI + "36m"
        MAGENTA = CSI + "35m"
        WHITE = CSI + "37m"
        LIGHTBLACK_EX = CSI + "90m"
        LIGHTBLUE_EX = CSI + "94m"
        LIGHTCYAN_EX = CSI + "96m"
        LIGHTGREEN_EX = CSI + "92m"
        LIGHTMAGENTA_EX = CSI + "95m"
        LIGHTRED_EX = CSI + "91m"
        LIGHTWHITE_EX = CSI + "97m"
        LIGHTYELLOW_EX = CSI + "93m"
    class _S:
        BRIGHT = CSI + "1m"
        RESET_ALL = CSI + "0m"
    Fore, Style = _F(), _S()

DEFAULT_COLORS: list[str] = [
    str(Fore.RED), str(Fore.GREEN), str(Fore.BLUE), str(Fore.YELLOW),
    str(Fore.CYAN), str(Fore.MAGENTA), str(Fore.WHITE),
    str(getattr(Fore, "LIGHTBLACK_EX", "")),
    str(getattr(Fore, "LIGHTBLUE_EX", "")),
    str(getattr(Fore, "LIGHTCYAN_EX", "")),
    str(getattr(Fore, "LIGHTGREEN_EX", "")),
    str(getattr(Fore, "LIGHTMAGENTA_EX", "")),
    str(getattr(Fore, "LIGHTRED_EX", "")),
    str(getattr(Fore, "LIGHTWHITE_EX", "")),
    str(getattr(Fore, "LIGHTYELLOW_EX", "")),
]

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
def _vis_len(s: str) -> int:
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

def _frames_for(name: str) -> list[str]:
    n = (name or "").lower()
    if n in {"line", "pipe", "bar"}:
        return ["|", "/", "-", "\\"]
    if n in {"arrow", "tri", "triangle"}:
        return ["▲", "▶", "▼", "◀"]
    if n in {"bounce", "bouncingbar"}:
        return ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "▊", "▋", "▌", "▍", "▎"]
    if n in {"dots", "braille"}:
        return ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    return ["|", "/", "-", "\\"]

class _FixedLineFormatter:
    """
    Build a fixed-width line.
    The middle block [ITEM …] [JOB …] [MODEL …] flexes; tail stays aligned.
    """
    def __init__(self, prefix: str, cols: int, tail_mode: str = "full") -> None:
        self.prefix = prefix
        self.cols = max(24, int(cols))
        self.tail_mode = tail_mode

    def render(self, *, count: int, total: int, item: str, job: str, model: str, color: str) -> str:
        B, Z, R, G = Style.BRIGHT, Style.RESET_ALL, Fore.RED, Fore.GREEN

        count = max(0, int(count))
        total = max(1, int(total))
        if count > total:
            total = count
        item = str(item or "")
        job = str(job or "")
        model = str(model or "")

        pct = min(100.0, max(0.0, (100.0 * count) / max(1, total)))
        w = len(str(total))

        head = f"{color}{B}{self.prefix}{Z} "

        tail_full  = f"{G}{B} [DONE:{count:>{w}}/{total:{w}}] [PROGRESS:{pct:6.2f}%]{Z}"
        tail_short = f"{G}{B} [DONE:{count}/{total}] [{pct:5.1f}%]{Z}"
        tail_min   = f"{G}{B} [{count}/{total}] [{int(round(pct))}%]{Z}"

        preferred = {"full": tail_full, "short": tail_short, "min": tail_min}.get(self.tail_mode, tail_full)

        # choose a tail that leaves at least 3 visible chars for the middle block
        tail = preferred
        for t in (preferred, tail_full, tail_short, tail_min):
            if _vis_len(head) + _vis_len(t) + 3 <= self.cols:
                tail = t
                break

        mid_budget = max(3, self.cols - _vis_len(head) - _vis_len(tail))

        # Build mid with per-value fitting (preserve colours)
        itm_open, itm_close = f"{G}{B}[ITEM:{Z}{R}{B}", f"{Z}{G}{B}]"
        job_open, job_close = f"{G}{B}[JOB:{Z}{R}{B}",  f"{Z}{G}{B}]"
        mdl_open, mdl_close = f"{G}{B}[MODEL:{Z}{R}{B}", f"{Z}{G}{B}]"

        segments: list[tuple[str, str, str]] = []
        if item:
            segments.append((itm_open, item,  itm_close))
        if job:
            segments.append((job_open, job,  job_close))
        if model:
            segments.append((mdl_open, model, mdl_close))

        if not segments:
            mid_text, mid_vis = "", 0
        else:
            overhead = 0
            for i, (op, _val, cl) in enumerate(segments):
                overhead += _vis_len(op) + _vis_len(cl)
                if i > 0:
                    overhead += 1  # space separator

            avail_vals = max(0, mid_budget - overhead)
            remaining = avail_vals
            n = len(segments)
            min_w = 1
            allowed: list[int] = []
            values = [seg[1] for seg in segments]
            for i, val in enumerate(values):
                reserve = (n - i - 1) * min_w
                allow = max(min_w, min(len(val), max(0, remaining - reserve)))
                allowed.append(allow)
                remaining -= allow

            out_parts: list[str] = []
            for i, (op, val, cl) in enumerate(segments):
                if i > 0:
                    out_parts.append(" ")
                fitted = _ellipsize_plain(val, allowed[i])
                out_parts.append(op + fitted + cl)
            mid_text = "".join(out_parts)
            mid_vis = _vis_len(mid_text)

        if mid_vis < mid_budget:
            mid_text = mid_text + (" " * (mid_budget - mid_vis))

        return f"{head}{mid_text}{tail}"

class _ConsolePercentSpinner:
    """Single-line fixed-width spinner with left glyph + color cycling."""
    def __init__(
        self, fmt: _FixedLineFormatter, *, stream: Any, interval: float, base_cols: int,
        spinner_type: str, final_newline: bool, colors: list[str] | None = None,
    ) -> None:
        self._fmt = fmt
        self._stream = stream
        self._interval = float(interval)
        self._state: dict[str, Any] = {"count": 0, "total": 1, "item": None, "job": None, "model": None}
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self._frames = _frames_for(spinner_type)
        self._frame_i = 0
        self._colors = itertools.cycle(colors or DEFAULT_COLORS)
        self._final_newline = bool(final_newline)
        self._line_cols = max(24, int(base_cols) - 2)  # reserve "<glyph> "

        # ---- Steady mode & fixed colour controls ----
        self._steady = (os.environ.get("PANOPTES_SPINNER_STEADY", "").lower() in {"1", "true", "yes"})
        fixed = os.environ.get("PANOPTES_PROGRESS_COLOR", "").strip().upper()
        if fixed:
            try:
                single = [str(getattr(Fore, fixed))]
                self._colors = itertools.cycle(single)
            except Exception:
                pass
        # Keep track of the last fully-rendered line (excluding glyph)
        self._last_line: str = ""

    def __enter__(self) -> "_ConsolePercentSpinner":
        os.environ["PANOPTES_PROGRESS_ACTIVE"] = "1"
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: TracebackType | None) -> bool | None:
        try:
            self._stop.set()
            if self._t and self._t.is_alive():
                self._t.join(timeout=1.0)
            self._render(final=True)
            if self._final_newline:
                try:
                    self._stream.write("\n")
                    self._stream.flush()
                except Exception:
                    pass
        finally:
            os.environ.pop("PANOPTES_PROGRESS_ACTIVE", None)
        return False

    def update(self, **kw: Any) -> "_ConsolePercentSpinner":
        # accept keys: total, count, item/current, job, model
        if "current" in kw and "item" not in kw:
            kw["item"] = kw.pop("current")
        self._state.update(kw)
        return self

    def _loop(self) -> None:
        try:
            env_interval = float(os.environ.get("PANOPTES_SPINNER_INTERVAL", str(self._interval)))
        except Exception:
            env_interval = self._interval
        while not self._stop.is_set():
            self._render()
            time.sleep(env_interval)
            # advance the glyph only when not in steady mode
            if not self._steady:
                self._frame_i = (self._frame_i + 1) % max(1, len(self._frames))

    def _render(self, *, final: bool = False) -> None:
        col = next(self._colors)
        line = self._fmt.render(
            count=int(self._state.get("count", 0)),
            total=int(self._state.get("total", 1)),
            item=str(self._state.get("item") or ""),
            job=str(self._state.get("job") or ""),
            model=str(self._state.get("model") or ""),
            color=col,
        )

        # If steady mode is enabled and the *line* hasn't changed, skip repaint
        if self._steady and (not final) and (line == self._last_line):
            return
        self._last_line = line

        glyph = f"{col}{self._frames[self._frame_i]}{Style.RESET_ALL}"
        out = f"{glyph} {line}"
        try:
            self._stream.write("\r\033[K")
            self._stream.write(out)
            self._stream.flush()
        except Exception:
            pass

# ── 5) Factory ───────────────────────────────────────────────────────────────
def percent_spinner(*, prefix: str = "PROGRESS", **kwargs: Any) -> _SpinnerLike:
    stream = cast(TextIO, kwargs.pop("stream", None) or getattr(sys, "__stderr__", sys.stderr))
    try:
        base_cols = max(20, int(os.environ.get("PANOPTES_PROGRESS_WIDTH", "75")))
    except Exception:
        base_cols = 75

    tail_mode = (os.environ.get("PANOPTES_PROGRESS_TAIL", "full") or "full").lower()
    spinner_type = kwargs.pop("spinner_type", os.environ.get("PANOPTES_SPINNER", "line"))
    try:
        interval = float(kwargs.pop("interval", os.environ.get("PANOPTES_SPINNER_INTERVAL", "0.10")))
    except Exception:
        interval = 0.10
    final_newline_env = (os.environ.get("PANOPTES_PROGRESS_FINAL_NEWLINE", "0") or "0").lower() in {"1", "true", "yes"}

    parent_active = (
        os.environ.get("PANOPTES_PROGRESS_ACTIVE") == "1"
        and (os.environ.get("PANOPTES_PROGRESS_FORCE", "").lower() not in {"1", "true", "yes"})
    )

    # Prefer the UX (Halo) version if available and the stream supports it.
    if _have_ux and _ux_should_enable and _ux_percent_spinner:
        try:
            if _ux_should_enable(stream):  # type: ignore[misc]
                ux_kwargs: dict[str, Any] = dict(
                    prefix=prefix, spinner_type=spinner_type, stream=stream, final_newline=final_newline_env,
                )
                try:
                    if "interval" in inspect.signature(_ux_percent_spinner).parameters:
                        ux_kwargs["interval"] = interval
                except Exception:
                    pass
                return cast(_SpinnerLike, _ux_percent_spinner(**ux_kwargs))
        except Exception:
            pass

    if parent_active:
        class _Silent:
            def __enter__(self) -> "_Silent":
                return self
            def __exit__(self, *a: Any, **k: Any) -> None:
                return None
            def update(self, **_: Any) -> "_Silent":
                return self
        return _Silent()  # type: ignore[return-value]

    fmt = _FixedLineFormatter(prefix=prefix, cols=base_cols - 2, tail_mode=tail_mode)
    return _ConsolePercentSpinner(
        fmt, stream=stream, interval=interval, base_cols=base_cols,
        spinner_type=spinner_type, final_newline=final_newline_env,
    )

# ── 6) Bridge (engine → spinner) ─────────────────────────────────────────────
try:
    from .bridges import live_percent as _bridge_live_percent  # type: ignore
    live_percent = _bridge_live_percent  # type: ignore[assignment]
except Exception:
    def _bind_spinner(engine: _EngineLike, spinner: _SpinnerLike) -> Callable[[], None]:
        def on_update(st: _ProgressStateLike) -> None:
            try:
                total = float(getattr(st, "total_units", 0))
                done = float(getattr(st, "done_units", 0))
                current = getattr(st, "current_item", None) or ""
                spinner.update(total=total, count=done, item=current)  # compat: 'item' is filename
            except Exception:
                pass
        try:
            unsub = engine.on_update(on_update)  # type: ignore[attr-defined]
        except Exception:
            def unsub() -> None:
                pass
        return unsub

    @contextmanager
    def live_percent(engine: _EngineLike, *, prefix: str = "PROGRESS"):
        sp = percent_spinner(prefix=prefix)
        unsub = _bind_spinner(engine, sp)
        try:
            with sp:
                yield sp
        finally:
            try:
                unsub()
            except Exception:
                pass
