from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


def _new_meta() -> Dict[str, object]:
    return {}


def _new_tasks() -> List["Task"]:
    return []


@dataclass
class ProgressState:
    total_units: float = 0.0
    done_units: float = 0.0
    current_item: Optional[str] = None
    started_at: float = field(default_factory=time.time)


@dataclass
class Task:
    name: str
    weight: float = 1.0
    units_hint: Optional[float] = None
    meta: Dict[str, object] = field(default_factory=_new_meta)


@dataclass
class Phase:
    name: str
    tasks: List[Task] = field(default_factory=_new_tasks)
    weight: float = 1.0


class ProgressEngine:
    """
    Minimal engine for aggregating and emitting progress updates.
    """
    def __init__(self) -> None:
        self.state = ProgressState()
        self._lock = threading.Lock()
        self._listeners: List[Callable[[ProgressState], None]] = []

    def on_update(self, f: Callable[[ProgressState], None]) -> Callable[[], None]:
        self._listeners.append(f)

        def off() -> None:
            try:
                self._listeners.remove(f)
            except ValueError:
                pass

        return off

    def set_total(self, total_units: float) -> None:
        with self._lock:
            self.state.total_units = max(0.0, float(total_units))
            self._emit()

    def add(self, units: float, *, current_item: Optional[str] = None) -> None:
        with self._lock:
            self.state.done_units += float(units)
            if current_item is not None:
                self.state.current_item = current_item
            self._emit()

    def set_current(self, label: Optional[str]) -> None:
        with self._lock:
            self.state.current_item = label
            self._emit()

    def finish(self) -> None:
        with self._lock:
            self.state.done_units = max(self.state.done_units, self.state.total_units)
            self._emit()

    def reset(self) -> None:
        with self._lock:
            self.state = ProgressState()
            self._emit()

    def _emit(self) -> None:
        for f in list(self._listeners):
            try:
                f(self.state)
            except Exception:
                pass
