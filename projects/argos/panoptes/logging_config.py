from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, MutableMapping, Optional

import contextvars

_BASE_CONTEXT: contextvars.ContextVar[MutableMapping[str, object]] = contextvars.ContextVar(
    "argos_base_context", default={}
)
_EXTRA_CONTEXT: contextvars.ContextVar[MutableMapping[str, object]] = contextvars.ContextVar(
    "argos_extra_context", default={}
)
_RUN_DIR: contextvars.ContextVar[Path | None] = contextvars.ContextVar("argos_run_dir", default=None)
_RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("argos_run_id", default=None)


class _State:
    __slots__ = ("lock", "configured")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.configured = False


_STATE = _State()


def _platform_data_dir(app: str = "rAIn") -> Path:
    if os.name == "nt":
        base = Path(
            os.getenv("LOCALAPPDATA")
            or os.getenv("APPDATA")
            or (Path.home() / "AppData" / "Local")
        )
        return base / app
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app
    return Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / app


def _detect_project_root() -> Path | None:
    env_override = os.getenv("PANOPTES_ARGOS_ROOT") or os.getenv("ARGOS_ROOT")
    if env_override:
        root = Path(env_override).expanduser()
        if root.exists():
            return root

    here = Path(__file__).resolve()
    candidates = [here.parent] + list(here.parents)
    for cand in candidates:
        if (cand / "pyproject.toml").exists() and (cand / "panoptes").exists():
            return cand

    for base in [Path.cwd().resolve(), here]:
        for parent in [base] + list(base.parents):
            probe = parent / "projects" / "argos"
            if probe.exists() and (probe / "panoptes").exists():
                return probe
    return None


def _candidate_logs_roots() -> Iterator[Path]:
    env_root = os.getenv("ARGOS_RUNS_ROOT")
    if env_root:
        yield Path(env_root).expanduser()

    project_root = _detect_project_root()
    if project_root:
        yield (project_root / "tests" / "results" / "logs")

    yield _platform_data_dir() / "tests" / "results" / "logs"
    yield Path(tempfile.gettempdir()) / "argos" / "logs"


def _pick_logs_root() -> Path:
    for candidate in _candidate_logs_roots():
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        else:
            return candidate
    raise RuntimeError("Unable to create a writable runs directory for logging output.")


def _clean_old_run_dirs(logs_root: Path) -> None:
    """
    Remove timestamped run directories under *logs_root* so we keep a single
    consolidated log location. Legacy runs populated sub-directories like
    ``20250101-010101_abcd1234`` that are no longer needed.
    """
    for child in logs_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)


def _now_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _current_context() -> Dict[str, object]:
    ctx: Dict[str, object] = {}
    base = _BASE_CONTEXT.get({})
    extra = _EXTRA_CONTEXT.get({})
    ctx.update(base)
    ctx.update(extra)
    return ctx


def current_run_dir() -> Path | None:
    return _RUN_DIR.get(None)


def current_run_id() -> str | None:
    return _RUN_ID.get(None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: Dict[str, object] = {
            "ts": _now_iso(record.created),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "file": record.filename,
            "line": record.lineno,
            "pid": record.process,
            "tid": record.thread,
        }
        ctx = _current_context()
        context_dict: Dict[str, object] = {}
        if ctx:
            context_dict = dict(ctx)
            data["context"] = context_dict
        run_id = current_run_id()
        if run_id:
            if not context_dict:
                context_dict = {}
            context_dict.update({"run_id": run_id})
            data["context"] = context_dict
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ctx = _current_context()
        ctx_frag = ""
        if ctx:
            pairs = ", ".join(f"{k}={v}" for k, v in ctx.items())
            ctx_frag = f" | {pairs}"
        run_id = current_run_id()
        if run_id:
            ctx_frag = f"{ctx_frag} | run_id={run_id}"

        base = super().format(record)
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            base = f"{base}\n{exc_text}"
        return f"{base}{ctx_frag}"


@contextmanager
def bind_context(**kwargs: object) -> Iterator[None]:
    if not kwargs:
        yield
        return
    current = dict(_EXTRA_CONTEXT.get({}))
    current.update({k: v for k, v in kwargs.items() if v is not None})
    token = _EXTRA_CONTEXT.set(current)
    try:
        yield
    finally:
        _EXTRA_CONTEXT.reset(token)


def _configure_handlers(
    level: int,
    console_level: int,
    console_fmt: str,
    use_json_console: bool,
    log_file: Path,
    file_mode: str = "w",
) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(level)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(console_level)
    if use_json_console:
        console.setFormatter(JsonFormatter())
    else:
        console.setFormatter(HumanFormatter(console_fmt))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode=file_mode)
    file_handler.setLevel(level)
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)


def setup_logging(
    *,
    level_env: str = "ARGOS_LOG_LEVEL",
    fmt_env: str = "ARGOS_LOG_FORMAT",
    file_env: str = "ARGOS_LOG_FILE",
    console_level_env: str = "ARGOS_CONSOLE_LOG_LEVEL",
) -> Path:
    with _STATE.lock:
        if _STATE.configured:
            run_dir = current_run_dir()
            if run_dir is None:
                raise RuntimeError("Logging configured but run directory unset.")
            return run_dir

        level_name = os.getenv(level_env, "WARNING").upper()
        level = getattr(logging, level_name, logging.INFO)
        console_level_name = os.getenv(console_level_env, "WARNING").upper()
        console_level = getattr(logging, console_level_name, logging.WARNING)

        fmt_choice = os.getenv(fmt_env, "human").strip().lower()
        use_json_console = fmt_choice == "json"
        console_fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

        logs_root = _pick_logs_root()
        _clean_old_run_dirs(logs_root)
        run_dir = logs_root
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = "current"

        legacy_runs = logs_root / "runs"
        if legacy_runs.exists() and legacy_runs.is_dir():
            shutil.rmtree(legacy_runs, ignore_errors=True)

        file_override = os.getenv(file_env)
        aggregate_path: Optional[Path] = None
        if file_override:
            log_path = Path(file_override).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            log_path = run_dir / "argos.log"
            aggregate_path = None

        _BASE_CONTEXT.set({"run_id": run_id, "run_dir": str(run_dir)})
        _EXTRA_CONTEXT.set({})
        _RUN_DIR.set(run_dir)
        _RUN_ID.set(run_id)

        _configure_handlers(level, console_level, console_fmt, use_json_console, log_path, file_mode="a")
        if aggregate_path is not None and aggregate_path != log_path:
            try:
                aggregate_handler = logging.FileHandler(aggregate_path, encoding="utf-8", mode="a")
                aggregate_handler.setLevel(level)
                aggregate_handler.setFormatter(JsonFormatter())
                logging.getLogger().addHandler(aggregate_handler)
            except Exception:
                logging.getLogger(__name__).debug("Failed to attach aggregate log handler", exc_info=True)
            try:
                (logs_root / "latest_run.txt").write_text(run_id, encoding="utf-8")
            except Exception:
                logging.getLogger(__name__).debug("Failed to update latest run pointer", exc_info=True)
        else:
            try:
                (logs_root / "latest_run.txt").write_text(run_id, encoding="utf-8")
            except Exception:
                logging.getLogger(__name__).debug("Failed to update latest run pointer", exc_info=True)
        _STATE.configured = True

    try:
        from . import env_audit

        snapshot = env_audit.collect_env()
        env_audit.write_snapshot(run_dir, snapshot)
    except Exception:
        logging.getLogger(__name__).debug("Failed to write environment snapshot", exc_info=True)

    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("ultralytics.yolo.engine.model").setLevel(logging.WARNING)
    logging.getLogger("ultralytics.yolo.utils").setLevel(logging.WARNING)
    logging.getLogger(__name__).debug("Logging initialised", extra={"run_dir": str(run_dir)})
    return run_dir
