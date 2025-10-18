# projects/argos/panoptes/diagnostics.py
from __future__ import annotations

import atexit
import importlib
import importlib.util as importlib_util
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from functools import wraps
from importlib import abc as importlib_abc
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Callable, Optional, Sequence, cast

# ──────────────────────────────────────────────────────────────────────────────
# Activation gate — ALWAYS ON (no env var required). Runs once per process.
# ──────────────────────────────────────────────────────────────────────────────
_prev_ready = bool(getattr(sys.modules.get(__name__), "__PANOPTES_DIAG_READY__", False))
if not _prev_ready:
    setattr(sys.modules[__name__], "__PANOPTES_DIAG_READY__", True)

# Always attach on first import; never reattach in the same process
_ACTIVE: bool = not _prev_ready

# ──────────────────────────────────────────────────────────────────────────────
# Robust path resolution for results/logs (works across machines/install modes)
# ──────────────────────────────────────────────────────────────────────────────
def _platform_data_dir() -> Path:
    """Cross-platform per-user data dir (matches bootstrap.py behavior)."""
    app = "rAIn"
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or (Path.home() / "AppData" / "Local"))
        return base / app
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app
    return Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / app


def _detect_argos_root() -> Optional[Path]:
    """
    Try hard to locate the Argos project root in a variety of launch modes:

    Priority:
      1) PANOPTES_ARGOS_ROOT / ARGOS_ROOT / PANOPTES_PROJECT_ROOT env
      2) a parent that contains pyproject.toml AND panoptes/ (repo checkout)
      3) .../projects/argos under any parent of CWD or this file
      4) parent of this module's panoptes/ directory if it has tests/
    """
    # 1) Environment overrides
    for k in ("PANOPTES_ARGOS_ROOT", "ARGOS_ROOT", "PANOPTES_PROJECT_ROOT"):
        v = os.getenv(k)
        if v:
            p = Path(v).expanduser().resolve()
            if p.exists():
                return p

    # 2) Search parents for pyproject.toml + panoptes/
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "pyproject.toml").exists() and (parent / "panoptes").exists():
            return parent

    # 3) Look for .../projects/argos under CWD parents and module parents
    def _find_projects_argos(base: Path) -> Optional[Path]:
        for parent in [base] + list(base.parents):
            cand = parent / "projects" / "argos"
            if cand.exists() and (cand / "panoptes").exists():
                return cand
        return None

    cwd = Path.cwd().resolve()
    hit = _find_projects_argos(cwd)
    if hit:
        return hit

    hit = _find_projects_argos(here)
    if hit:
        return hit

    # 4) If installed in site-packages, try parent-of-panoptes if tests/ exists
    pp = here.parent  # .../panoptes
    site_parent = pp.parent
    if (site_parent / "tests").exists():
        return site_parent

    return None


def _compute_results_dir() -> Path:
    """
    Compute a stable results directory that works everywhere.

    Priority:
      1) PANOPTES_DIAGNOSTICS_FILE → use its parent
      2) PANOPTES_DIAGNOSTICS_DIR / PANOPTES_RESULTS_DIR
      3) <argos_root>/tests/results   (if an Argos root is detected)
      4) <user_data_dir>/tests/results (fallback; always writable)
    """
    # 1) Explicit file path override
    file_override = (os.getenv("PANOPTES_DIAGNOSTICS_FILE") or "").strip()
    if file_override:
        return Path(file_override).expanduser().resolve().parent

    # 2) Explicit directory overrides
    for env_dir in ("PANOPTES_DIAGNOSTICS_DIR", "PANOPTES_RESULTS_DIR"):
        v = (os.getenv(env_dir) or "").strip()
        if v:
            return Path(v).expanduser().resolve()

    # 3) Argos repo root → tests/results
    root = _detect_argos_root()
    if root:
        return (root / "tests" / "results" / "logs").resolve()

    # 4) Per-user fallback under data dir
    return (_platform_data_dir() / "tests" / "results" / "logs").resolve()


def _compute_log_path() -> Path:
    """Determine final log file path with robust fallbacks."""
    # If PANOPTES_DIAGNOSTICS_FILE is explicitly set, respect it (create parent)
    file_override = (os.getenv("PANOPTES_DIAGNOSTICS_FILE") or "").strip()
    if file_override:
        p = Path(file_override).expanduser().resolve()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass  # fall through to other strategies

    # Otherwise, build from results dir
    res_dir = _compute_results_dir()
    try:
        res_dir.mkdir(parents=True, exist_ok=True)
        target = res_dir / "argos_diagnostics.log"
        legacy = res_dir.parent / "argos_diagnostics.log"
        if legacy.exists() and legacy != target:
            try:
                legacy.unlink()
            except Exception:
                pass
        return target
    except Exception:
        # Ultimate fallback: system temp dir
        tmp = Path(tempfile.gettempdir())
        return tmp / "argos_diagnostics.log"


# ──────────────────────────────────────────────────────────────────────────────
# Log file setup (single file; created proactively)
# ──────────────────────────────────────────────────────────────────────────────
_LOG_PATH: Path = _compute_log_path()
_LOG = logging.getLogger("panoptes.diagnostics")
_LOG.setLevel(logging.INFO)
_LOG.propagate = False

def _ensure_log_handler() -> None:
    """Attach a single file handler; avoid duplicates across repeated imports."""
    # Remove dead/duplicate handlers if any
    existing_paths: list[Optional[str]] = []  # annotate to avoid Unknown append
    for h in list(_LOG.handlers):
        try:
            if isinstance(h, logging.FileHandler):
                existing_paths.append(getattr(h, "baseFilename", None))
        except Exception:
            pass
    if str(_LOG_PATH) in (existing_paths or []):
        return

    try:
        fh = logging.FileHandler(_LOG_PATH, mode="a", encoding="utf-8", delay=False)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        _LOG.addHandler(fh)
    except Exception:
        # Last-chance fallback to temp dir if current path is unwritable
        try:
            fallback = Path(tempfile.gettempdir()) / "argos_diagnostics.log"
            fh = logging.FileHandler(fallback, mode="a", encoding="utf-8", delay=False)
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            _LOG.addHandler(fh)
        except Exception:
            # If even temp fails, swallow; we won't crash user's run
            pass

def _say(cat: str, msg: str, *, level: int = logging.INFO) -> None:
    if not _LOG.handlers:
        _ensure_log_handler()
    _LOG.log(level, f"[{cat}] {msg}")

def _err(cat: str, msg: str) -> None:
    if not _LOG.handlers:
        _ensure_log_handler()
    _LOG.error(f"[{cat}] {msg}")

def _strip_rich_markup(s: object) -> str:
    # Remove Rich/typer markup like [bold red]…[/] and ANSI escapes
    text = re.sub(r"\[(\/?)[^\[\]]+\]", "", str(s))
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    return text.strip()

def _safe_repr(obj: Any) -> str:
    try:
        return repr(obj)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return f"<{type(obj).__name__}>"

# ──────────────────────────────────────────────────────────────────────────────
# Environment & system snapshot
# ──────────────────────────────────────────────────────────────────────────────
def _log_env_snapshot() -> None:
    try:
        _say("ENV", f"Diagnostics activated → {_LOG_PATH}")
        _say("ENV", f"Python: {sys.version.replace(os.linesep, ' ')}")
        _say("ENV", f"Implementation: {platform.python_implementation()}")
        _say("ENV", f"Platform: {platform.platform()} | machine={platform.machine()} | arch={platform.architecture()[0]}")
        _say("ENV", f"Executable: {sys.executable}")
        venv_root = os.environ.get('PANOPTES_VENV_ROOT')
        in_argos_venv = False
        if venv_root:
            try:
                in_argos_venv = Path(sys.executable).resolve().is_relative_to(Path(venv_root).resolve())
            except Exception:
                in_argos_venv = False
        else:
            base_prefix = getattr(sys, 'base_prefix', sys.prefix)
            in_argos_venv = sys.prefix != base_prefix
        _say('ENV', f'In Argos venv: {in_argos_venv} | PANOPTES_VENV_ROOT={venv_root or "unset"}')
        _say('ENV', f'Working dir: {os.getcwd()}')

        # PATH entries for sanity
        try:
            for i, p in enumerate((os.getenv("PATH") or "").split(os.pathsep), 1):
                _say("ENV", f"PATH[{i:02d}] {p}")
        except Exception:
            pass

        # Every environment variable (extremely granular)
        for k in sorted(os.environ):
            _say("ENV", f"{k}={os.environ[k]}")

        # pip list (freeze)
        try:
            cp = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                text=True, capture_output=True, check=False,
            )
            if cp.stdout.strip():
                pkgs = json.loads(cp.stdout)
                for p in pkgs:
                    name = p.get("name", "?")
                    ver = p.get("version", "?")
                    _say("PIP", f"{name}=={ver}")
            else:
                # fallback
                cp2 = subprocess.run(
                    [sys.executable, "-m", "pip", "list"],
                    text=True, capture_output=True, check=False,
                )
                for line in (cp2.stdout or "").splitlines():
                    _say("PIP", line.strip())
        except Exception as e:
            _err("PIP", f"pip list failed: {e}")

        # Torch / CUDA
        try:
            import torch  # type: ignore
            _say("CUDA", f"torch.__version__={getattr(torch, '__version__', '?')}")
            has = bool(torch.cuda.is_available())
            _say("CUDA", f"torch.cuda.is_available()={has}")
            if has:
                n = int(torch.cuda.device_count())
                _say("CUDA", f"device_count={n}")
                for i in range(n):
                    try:
                        name = torch.cuda.get_device_name(i)
                    except Exception:
                        name = "?"
                    _say("CUDA", f"GPU[{i}]: {name}")
        except Exception as e:
            _say("CUDA", f"torch not available or failed to probe: {e}")

        # nvidia-smi presence
        smi = shutil.which("nvidia-smi")
        if smi:
            try:
                cp = subprocess.run([smi, "-L"], text=True, capture_output=True, check=False)
                for line in (cp.stdout or "").splitlines():
                    _say("CUDA", f"nvidia-smi -L: {line}")
            except Exception as e:
                _say("CUDA", f"nvidia-smi failed: {e}")
        else:
            _say("CUDA", "nvidia-smi not found")

        # ONNX Runtime
        try:
            import onnxruntime as ort  # type: ignore
            _say("ORT", f"onnxruntime.__version__={getattr(ort, '__version__', '?')}")
            try:
                provs = cast("list[str]", ort.get_available_providers())  # type: ignore[no-any-return]
                _say("ORT", f"providers={provs}")
            except Exception:
                pass
        except Exception as e:
            _say("ORT", f"onnxruntime not available: {e}")

        # OpenCV build info + FFMPEG
        try:
            import cv2  # type: ignore
            _say("OPENCV", f"cv2.__version__={getattr(cv2, '__version__', '?')}")
            try:
                # Make Pyright/mypy happy: cv2 has no stubs; treat as Any and coerce to str
                cv2_any = cast(Any, cv2)
                info: str = str(cv2_any.getBuildInformation())
                ffm: str = "UNKNOWN"
                gst: str = "UNKNOWN"
                for ln in info.splitlines():
                    t: str = ln.strip()
                    if t.startswith("FFMPEG:"):
                        ffm = t.split(":", 1)[1].strip()
                    if t.startswith("GStreamer:"):
                        gst = t.split(":", 1)[1].strip()
                _say("OPENCV", f"FFMPEG={ffm} | GStreamer={gst}")
            except Exception:
                pass
        except Exception as e:
            _say("OPENCV", f"cv2 not available: {e}")

        # Pillow AVIF support
        try:
            from PIL import Image  # type: ignore
            avail = (".avif" in Image.registered_extensions())
            try:
                from PIL import features as pil_features  # type: ignore
                avail = bool(pil_features.check("avif"))
            except Exception:
                pass
            _say("PIL", f"AVIF support={avail}")
        except Exception as e:
            _say("PIL", f"Pillow not available: {e}")

        # FFmpeg binary
        ff = shutil.which("ffmpeg")
        _say("FFMPEG", f"binary={'present: '+ff if ff else 'not found'}")

        # Git LFS
        git = shutil.which("git")
        if git:
            try:
                cp = subprocess.run([git, "lfs", "version"], text=True, capture_output=True, check=False)
                line = (cp.stdout or cp.stderr or "").strip().splitlines()[:1]
                _say("GITLFS", f"{line[0] if line else 'unknown'}")
            except Exception as e:
                _say("GITLFS", f"git lfs check failed: {e}")
        else:
            _say("GITLFS", "git not found")

        # Windows VC++ redistributable DLLs
        if os.name == "nt":
            sysdir = Path(os.getenv("WINDIR", r"C:\Windows")) / "System32"
            for dll in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "vcomp140.dll"):
                present = (sysdir / dll).exists()
                _say("MSVC", f"{dll}: {'Present' if present else 'Missing'}")
    except Exception as e:
        _err("ENV", f"env snapshot failed: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers to wrap/patch functions safely
# ──────────────────────────────────────────────────────────────────────────────
def _wrap(obj: Any, name: str, maker: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
    """Replace obj.name with wrapper if callable. Return True if patched."""
    try:
        fn = getattr(obj, name, None)
        if not callable(fn):
            return False
        if getattr(fn, "__panoptes_diag_wrapped__", False):
            return True
        w = maker(fn)
        setattr(w, "__panoptes_diag_wrapped__", True)
        setattr(obj, name, w)
        return True
    except Exception:
        return False

def _try_import(mod_name: str) -> Optional[Any]:
    try:
        if mod_name in sys.modules:
            return sys.modules[mod_name]
        return importlib.import_module(mod_name)
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap hooks (messages + ONNX ensure outcomes)
# ──────────────────────────────────────────────────────────────────────────────
def _patch_bootstrap() -> None:
    boot: Optional[ModuleType] = sys.modules.get("bootstrap")
    if boot is None:
        cand = _detect_argos_root()
        if cand:
            bp = cand / "bootstrap.py"
        else:
            # fallback to sibling of this file
            bp = Path(__file__).resolve().parents[1] / "bootstrap.py"
        if bp.exists():
            try:
                spec: Optional[ModuleSpec] = importlib_util.spec_from_file_location("bootstrap", bp)
                if spec and spec.loader:
                    boot_mod = importlib_util.module_from_spec(spec)
                    sys.modules["bootstrap"] = boot_mod
                    loader = spec.loader  # type: ignore[assignment]
                    assert isinstance(loader, importlib_abc.Loader)
                    loader.exec_module(boot_mod)
                    boot = boot_mod
            except Exception:
                boot = None

    if boot is None:
        return

    def _mk_print_wrapper(orig: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(orig)
        def _wrapped(msg: object = "") -> Any:
            try:
                _say("BOOTSTRAP", str(msg))
            except Exception:
                pass
            return orig(msg)
        return _wrapped

    _wrap(boot, "_print", _mk_print_wrapper)

    if hasattr(boot, "_probe_weight_presets"):
        def _mk_ensure_wrap(orig: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(orig)
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                try:
                    model_dir, all_names, default_names, nano_names, perception_names = cast(
                        tuple[Path, Sequence[str], Sequence[str], Sequence[str], Sequence[str]],
                        boot._probe_weight_presets(),  # type: ignore[attr-defined]
                    )
                    before: dict[str, bool] = {n: (model_dir / n).exists() for n in all_names}
                    _say("BOOTSTRAP", f"weights dir={model_dir}")
                    _say("BOOTSTRAP", f"names: total={len(list(all_names))} default={list(default_names)} nano={list(nano_names)} perception={list(perception_names)}")
                except Exception as e:
                    _say("BOOTSTRAP", f"probe presets failed pre-ensure: {e}")
                    before = {}
                    model_dir = ( _detect_argos_root() or Path(__file__).resolve().parents[1] ) / "panoptes" / "model"

                rv: Any = orig(*args, **kwargs)

                try:
                    if hasattr(boot, "_probe_weight_presets"):
                        model_dir2, all_names2, *_ = cast(
                            tuple[Path, Sequence[str], object, object, object],
                            boot._probe_weight_presets(),  # type: ignore[attr-defined]
                        )
                        after: dict[str, bool] = {n: (model_dir2 / n).exists() for n in all_names2}
                        created: list[str] = sorted([n for n in after if after[n] and not before.get(n, False)])
                        still_missing: list[str] = sorted([n for n in after if not after[n]])
                        if created:
                            _say("BOOTSTRAP", f"created={created}")
                        if still_missing:
                            for n in still_missing:
                                if n.lower().endswith(".onnx"):
                                    _err("BOOTSTRAP", f"ONNX missing after ensure: {n}")
                                else:
                                    _say("BOOTSTRAP", f"missing after ensure: {n}")
                except Exception as e:
                    _say("BOOTSTRAP", f"post-ensure probe failed: {e}")

                return rv
            return _wrapped
        _wrap(boot, "_ensure_weights_ultralytics", _mk_ensure_wrap)

def _patch_traceback_printing() -> None:
    _orig_print_exc = traceback.print_exc

    def _print_exc_patch(*args: Any, **kwargs: Any) -> None:
        try:
            etype, eval_, etb = sys.exc_info()
            if etype is not None:
                lines = traceback.format_exception(etype, eval_, etb)
                for ln in lines:
                    _err("ERROR", ln.rstrip("\n"))
        except Exception:
            pass
        try:
            _orig_print_exc(*args, **kwargs)  # keep original behavior
        except Exception:
            pass

    traceback.print_exc = _print_exc_patch  # type: ignore[assignment]

# ──────────────────────────────────────────────────────────────────────────────
# CLI / Typer hooks (mirror stderr/stdout, record argv)
# ──────────────────────────────────────────────────────────────────────────────
def _patch_typer_io() -> None:
    ty = _try_import("typer")
    if not ty:
        return

    if hasattr(ty, "echo"):
        _orig_echo = ty.echo

        @wraps(_orig_echo)
        def _echo_patch(message: object = "", *args: Any, **kwargs: Any) -> Any:
            try:
                msg = _strip_rich_markup(message)
                err = bool(kwargs.get("err", False))
                cat = "ERROR" if err else "CLI"
                if msg:
                    _say(cat, msg, level=(logging.ERROR if err else logging.INFO))
            except Exception:
                pass
            return _orig_echo(message, *args, **kwargs)

        ty.echo = _echo_patch  # type: ignore[assignment]

    if hasattr(ty, "secho"):
        _orig_secho = ty.secho

        @wraps(_orig_secho)
        def _secho_patch(message: object = "", *args: Any, **kwargs: Any) -> Any:
            try:
                msg = _strip_rich_markup(message)
                err = bool(kwargs.get("err", False))
                cat = "ERROR" if err else "CLI"
                if msg:
                    _say(cat, msg, level=(logging.ERROR if err else logging.INFO))
            except Exception:
                pass
            return _orig_secho(message, *args, **kwargs)

        ty.secho = _secho_patch  # type: ignore[assignment]

    try:
        _say("CLI", f"argv: {' '.join(sys.argv)}")
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Model registry hooks (weight choices + missing)
# ──────────────────────────────────────────────────────────────────────────────
def _patch_model_registry() -> None:
    mr = _try_import("panoptes.model_registry")
    if not mr:
        return

    try:
        reg_log = getattr(mr, "_LOG", None)
        if isinstance(reg_log, logging.Logger):
            reg_log.setLevel(logging.INFO)
            if _LOG.handlers:
                h0 = _LOG.handlers[0]
                _have = any((type(h) is type(h0) and getattr(h, "baseFilename", None) == getattr(h0, "baseFilename", None))  # type: ignore[attr-defined]
                            for h in reg_log.handlers)
                if not _have:
                    reg_log.addHandler(_LOG.handlers[0])
    except Exception:
        pass

    def _mk_pick_wrap(orig: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(orig)
        def _wrapped(task: str, *args: Any, **kwargs: Any) -> Any:
            w = orig(task, *args, **kwargs)
            try:
                _say("MODEL", f"pick_weight(task={task}, small={bool(kwargs.get('small', False))}) -> {w}")
            except Exception:
                pass
            return w
        return _wrapped

    _wrap(mr, "pick_weight", _mk_pick_wrap)

    def _mk_require_wrap(orig: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(orig)
        def _wrapped(model: Any, task: str) -> Any:
            if model is None:
                _err("MODEL", f"missing weight for task={task}")
            return orig(model, task)
        return _wrapped

    _wrap(mr, "_require", _mk_require_wrap)

# ──────────────────────────────────────────────────────────────────────────────
# Single-image flows (lambda_like + classify/pose/obb)
# ──────────────────────────────────────────────────────────────────────────────
def _patch_lambda_like_and_tasks() -> None:
    ll = _try_import("panoptes.lambda_like")
    if ll:
        def _mk_ll_wrap(orig: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(orig)
            def _wrapped(src: Any, *args: Any, **kwargs: Any) -> Any:
                try:
                    task = kwargs.get("task", "detect")
                    _say("RUN", f"lambda_like.run_single: task={task} src={_safe_repr(src)}")
                except Exception:
                    pass
                try:
                    out = orig(src, *args, **kwargs)
                    if out is None:
                        _say("RUN", "lambda_like.run_single: returned None (likely printed to stdout)")
                    else:
                        _say("RUN", f"lambda_like.run_single: wrote {out}")
                    return out
                except Exception as e:
                    _err("RUN", f"lambda_like.run_single error: {e}")
                    raise
            return _wrapped
        _wrap(ll, "run_single", _mk_ll_wrap)

    for mod_name, tag in [
        ("panoptes.classify", "classify"),
        ("panoptes.pose", "pose"),
        ("panoptes.obb", "obb"),
    ]:
        m = _try_import(mod_name)
        if not m:
            continue

        def _mk_runimg_wrap(orig: Callable[..., Any], _tag: str) -> Callable[..., Any]:
            @wraps(orig)
            def _wrapped(path: Any, *args: Any, **kwargs: Any) -> Any:
                try:
                    _say("RUN", f"{_tag}.run_image: {path}")
                except Exception:
                    pass
                try:
                    out = orig(path, *args, **kwargs)
                    outs: list[str] = []
                    try:
                        if out is None:
                            outs = []
                        elif isinstance(out, (str, Path)):
                            outs = [str(out)]
                        elif isinstance(out, (list, tuple)):
                            outs_temp: list[str] = []
                            for x in out:  # type: ignore[assignment]
                                if isinstance(x, (str, Path)):
                                    outs_temp.append(str(x))
                            outs = outs_temp
                    except Exception:
                        outs = []
                    if outs:
                        for o in outs:
                            _say("RUN", f"{_tag}.run_image wrote {o}")
                    return cast(Any, out)
                except Exception as e:
                    _err("RUN", f"{_tag}.run_image error: {e}")
                    raise
            return _wrapped

        def _make_runimg_wrapper(fn: Callable[..., Any], _t: str = tag) -> Callable[..., Any]:
            return _mk_runimg_wrap(fn, _t)

        _wrap(m, "run_image", _make_runimg_wrapper)

# ──────────────────────────────────────────────────────────────────────────────
# Video workers (best-effort: wrap main() to log encoder/output)
# ──────────────────────────────────────────────────────────────────────────────
def _patch_video_workers() -> None:
    def _suffix_encoder(path: str | Path) -> str:
        s = str(path).lower()
        if s.endswith(".mp4"):
            return "MP4"
        if s.endswith(".avi"):
            return "AVI"
        return Path(s).suffix.lstrip(".").upper() or "UNKNOWN"

    for mod_name, tag in [
        ("panoptes.predict_mp4", "detect"),
        ("panoptes.predict_heatmap_mp4", "heatmap"),
        ("panoptes.predict_classify_mp4", "classify"),
        ("panoptes.predict_pose_mp4", "pose"),
        ("panoptes.predict_obb_mp4", "obb"),
    ]:
        m = _try_import(mod_name)
        if not m or not hasattr(m, "main"):
            continue

        def _mk_vid_wrap(orig: Callable[..., Any], _tag: str) -> Callable[..., Any]:
            @wraps(orig)
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                try:
                    _say("VIDEO", f"{_tag}.main: start")
                except Exception:
                    pass
                try:
                    out = orig(*args, **kwargs)
                    out_s: Optional[str] = None
                    if isinstance(out, (str, Path)):
                        out_s = str(out)
                    if out_s:
                        enc = _suffix_encoder(out_s)
                        _say("VIDEO", f"{_tag}.main: output={out_s} encoder={enc}")
                    else:
                        _say("VIDEO", f"{_tag}.main: (no return value)")
                    return out
                except Exception as e:
                    _err("VIDEO", f"{_tag}.main error: {e}")
                    raise
            return _wrapped

        def _make_vid_wrapper(fn: Callable[..., Any], _t: str = tag) -> Callable[..., Any]:
            return _mk_vid_wrap(fn, _t)

        _wrap(m, "main", _make_vid_wrapper)

# ──────────────────────────────────────────────────────────────────────────────
# Progress package (attach our handler to its logger if any)
# ──────────────────────────────────────────────────────────────────────────────
def _patch_progress_logs() -> None:
    p = _try_import("panoptes.progress")
    if not p:
        return
    for attr in ("_LOG", "LOG", "logger"):
        lg = getattr(p, attr, None)
        if isinstance(lg, logging.Logger):
            try:
                lg.setLevel(logging.INFO)
                if _LOG.handlers and all(h is not _LOG.handlers[0] for h in lg.handlers):
                    lg.addHandler(_LOG.handlers[0])
            except Exception:
                pass

# ──────────────────────────────────────────────────────────────────────────────
# Global uncaught exception hook
# ──────────────────────────────────────────────────────────────────────────────
def _patch_excepthook() -> None:
    _orig = sys.excepthook

    def _hook(etype: type[BaseException], value: BaseException, tb: Optional[TracebackType]) -> None:
        try:
            lines = traceback.format_exception(etype, value, tb)
            for ln in lines:
                _err("ERROR", ln.rstrip("\n"))
        except Exception:
            pass
        _orig(etype, value, tb)

    sys.excepthook = _hook

# ──────────────────────────────────────────────────────────────────────────────
# Attach everything
# ──────────────────────────────────────────────────────────────────────────────
def _attach() -> None:
    _ensure_log_handler()
    _log_env_snapshot()
    _patch_traceback_printing()
    _patch_typer_io()
    _patch_model_registry()
    _patch_lambda_like_and_tasks()
    _patch_video_workers()
    _patch_progress_logs()
    _patch_bootstrap()
    _patch_excepthook()
    _say("ENV", "diagnostics hooks installed")

    # Ensure we flush the file at exit
    def _flush() -> None:
        try:
            for h in _LOG.handlers:
                try:
                    h.flush()  # type: ignore[call-arg]
                except Exception:
                    pass
        except Exception:
            pass

    atexit.register(_flush)

# ──────────────────────────────────────────────────────────────────────────────
# Activate if enabled
# ──────────────────────────────────────────────────────────────────────────────
if _ACTIVE:
    try:
        _attach()
    except Exception as e:
        # Never crash the host app due to diagnostics
        try:
            _err("INIT", f"failed to attach diagnostics: {e}")
        except Exception:
            pass
