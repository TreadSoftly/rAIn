# panoptes.live.cli — dedicated live/webcam entrypoint ("lv" / "live" / "livevideo")
from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import queue
import re
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Final, List, Optional, Tuple, Union, Literal

from multiprocessing.queues import Queue as MPQueue

try:
    from panoptes.runtime.venv_bootstrap import maybe_reexec_into_managed_venv # type: ignore[import]

    maybe_reexec_into_managed_venv()
except Exception:
    pass

import typer

from panoptes.logging_config import bind_context, setup_logging # type: ignore[import]
from panoptes.runtime.backend_probe import ort_available, torch_available # type: ignore[import]

try:
    from panoptes.support_bundle import write_support_bundle # type: ignore[import]
except ImportError:  # pragma: no cover - fallback for direct package execution
    from ..support_bundle import write_support_bundle  # type: ignore

setup_logging()
# Silence noisy FutureWarnings emitted by onnxscript when loading YOLO models.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"onnxscript(\.|$)")
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
LOGGER.setLevel(logging.ERROR)
_ERROR_TOKENS = ("error", "fail", "failed", "exception", "warning")
_ERROR_KEYS = ("error", "reason", "status")


def _log_event(event: str, **info: object) -> None:
    event_lower = event.lower()
    should_emit = any(token in event_lower for token in _ERROR_TOKENS)
    if not should_emit:
        for key in _ERROR_KEYS:
            value = info.get(key)
            if isinstance(value, str):
                if value and value.strip().lower() not in {"ok", "success"}:
                    should_emit = True
                    break
            elif value not in (None, 0, False):
                should_emit = True
                break
    if not should_emit:
        return
    detail = " ".join(f"{key}={info[key]}" for key in sorted(info) if info[key] is not None)
    if detail:
        LOGGER.error("%s %s", event, detail)
    else:
        LOGGER.error("%s", event)

from .pipeline import LivePipeline # noqa: E402
from . import tasks as live_tasks  # noqa: E402

# Ensure live-friendly progress behavior even when invoked via the console script.
os.environ.setdefault("PANOPTES_LIVE", "1")
os.environ.setdefault("PANOPTES_PROGRESS_TAIL", "none")          # hide [DONE] [PERCENT] tail
os.environ.setdefault("PANOPTES_PROGRESS_FINAL_NEWLINE", "0")    # keep line anchored
os.environ.setdefault("PANOPTES_NESTED_PROGRESS", "0")           # avoid nested spinners under live
os.environ.setdefault("OPENCV_VIDEOIO_ENABLE_OBSENSOR", "0")     # silence obsensor backend noise
TRACE_DISCOVERY = False

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

# ---------------------------------------------------------------------------
# Allowed task tokens (short and long spellings).
# NOTE: Map *aliases* to the canonical task names used by LivePipeline.
# ---------------------------------------------------------------------------
_TASK_CHOICES = {
    # detect
    "d": "detect",
    "detect": "detect",
    # heatmap
    "hm": "heatmap",
    "heatmap": "heatmap",
    # classify
    "clf": "classify",
    "classify": "classify",
    # pose
    "pse": "pose",      # ensure pse maps to pose (alias)
    "pose": "pose",
    # oriented bounding boxes
    "obb": "obb",
    "object": "obb",
}

# Tokens that indicate "live intent" and should be ignored as a source
_LIVE_MARKERS = {
    "lv",
    "livevideo",
    "live-video",
    "live_video",
    "ldv",
    "lvd",
    "live",
    "video",
    # spaced variants like "l d v" — treat 'l' and 'v' as noise here
    "l",
    "v",
}


@dataclass
class _ModelHint:
    source: str
    version: str
    size: str
    ext: Optional[str]


_ALLOWED_MODEL_SIZES: Final[set[str]] = {"n", "s", "m", "l", "x"}
_ALLOWED_MODEL_EXTS: Final[set[str]] = {"pt", "onnx"}
_MODEL_DEFAULT_SIZE: Final[str] = "x"
_TASK_WEIGHT_SUFFIX: Final[dict[str, str]] = {
    "detect": "",
    "heatmap": "-seg",
    "classify": "-cls",
    "pose": "-pose",
    "obb": "-obb",
}
_MODEL_TOKEN_RE = re.compile(
    r"""
    ^
    (?:yolo)?
    (?:v)?
    (?P<ver>\d{1,2})
    (?P<size>[nsmxl]?)       # optional size code
    (?:\.(?P<ext>[A-Za-z0-9]+))?
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _parse_model_token(token: str) -> Optional[_ModelHint]:
    raw = token.strip()
    if not raw:
        return None
    if any(sep in raw for sep in (os.sep, "/", "\\")):
        return None
    lowered = raw.lower()
    match = _MODEL_TOKEN_RE.fullmatch(lowered)
    if not match:
        return None

    size = (match.group("size") or "").lower()
    if size and size not in _ALLOWED_MODEL_SIZES:
        return None

    ext_raw = (match.group("ext") or "").lower()
    if ext_raw and ext_raw not in _ALLOWED_MODEL_EXTS:
        return None

    return _ModelHint(
        source=token,
        version=(match.group("ver") or "").lstrip("0") or "0",
        size=size,
        ext=(f".{ext_raw}" if ext_raw else None),
    )


def _model_hint_to_path(hint: _ModelHint, task: str) -> Optional[Path]:
    suffix = _TASK_WEIGHT_SUFFIX.get(task)
    if suffix is None:
        return None
    try:
        ver_int = int(hint.version)
    except ValueError:
        return None

    size = hint.size or _MODEL_DEFAULT_SIZE
    if size not in _ALLOWED_MODEL_SIZES:
        size = _MODEL_DEFAULT_SIZE

    prefix = "yolov" if ver_int <= 10 else "yolo"
    stem = f"{prefix}{ver_int}{size}{suffix}"

    from panoptes import model_registry as _model_registry  # type: ignore[import]  # defer heavy import

    candidates: list[str]
    if hint.ext is not None:
        candidates = [hint.ext.lower()]
    else:
        if task == "classify":
            candidates = [".pt", ".onnx"]
        else:
            candidates = [".onnx", ".pt"]

    for ext in candidates:
        path = _model_registry.MODEL_DIR / f"{stem}{ext}"
        if path.exists():
            return path

    # No candidate exists; return the first choice so the caller can surface a clear error.
    return _model_registry.MODEL_DIR / f"{stem}{candidates[0]}"


@dataclass
class _LiveSpec:
    task: str
    hint: Optional[_ModelHint]
    source: Optional[str]


@dataclass
class _ResolvedSpec:
    task: str
    source: Union[int, str]
    override: Optional[Path]
    prefer_small: bool


ResultMessage = tuple[Literal["ok", "err"], int, str]


def _clear_live_overrides() -> None:
    live_tasks.LIVE_DETECT_OVERRIDE = None
    live_tasks.LIVE_HEATMAP_OVERRIDE = None
    live_tasks.LIVE_CLASSIFY_OVERRIDE = None
    live_tasks.LIVE_POSE_OVERRIDE = None
    live_tasks.LIVE_PSE_OVERRIDE = None
    live_tasks.LIVE_OBB_OVERRIDE = None


def _extract_all_flag(tokens: List[str]) -> tuple[List[str], bool]:
    remaining: List[str] = []
    use_all = False
    for tok in tokens:
        if tok.lower() == "all":
            use_all = True
        else:
            remaining.append(tok)
    return remaining, use_all


def _parse_specs(tokens: List[str]) -> List[_LiveSpec]:
    specs: List[_LiveSpec] = []
    idx = 0
    total = len(tokens)
    pending_source: Optional[str] = None

    while idx < total:
        task_token = tokens[idx].strip()
        task_key = task_token.lower()
        if task_key not in _TASK_CHOICES:
            if (
                pending_source is None
                and (idx + 1) < total
                and tokens[idx + 1].strip().lower() in _TASK_CHOICES
            ):
                pending_source = task_token
                idx += 1
                task_token = tokens[idx].strip()
                task_key = task_token.lower()
            else:
                raise typer.BadParameter(f"Unexpected token {task_token!r}; specify a task before the camera/source.")

        task = _TASK_CHOICES[task_key]
        idx += 1

        hint: Optional[_ModelHint] = None
        if idx < total:
            model_token = tokens[idx].strip()
            model_hint = _parse_model_token(model_token)
            if model_hint is not None:
                hint = model_hint
                idx += 1

        source: Optional[str] = None
        if pending_source is not None:
            source = pending_source
            pending_source = None
        if source is None and idx < total:
            candidate_source = tokens[idx].strip()
            if candidate_source.lower() not in _TASK_CHOICES:
                source = candidate_source
                idx += 1

        specs.append(_LiveSpec(task=task, hint=hint, source=source))

    return specs


def _normalise_source_token(token: str) -> Union[int, str]:
    stripped = token.strip()
    low = stripped.lower()
    if low.startswith("synthetic"):
        return stripped
    for prefix in ("cam", "camera", "webcam", "webcamera"):
        if low.startswith(prefix):
            suffix = stripped[len(prefix):].strip()
            if suffix.lstrip("+-").isdigit():
                idx = int(suffix)
                if idx <= 0:
                    return 0
                return idx - 1
    if stripped.lstrip("+-").isdigit():
        idx = int(stripped)
        if idx <= 0:
            return 0
        return idx - 1
    return stripped


def _discover_cameras(max_index: int = 16) -> list[int]:
    from .camera import open_camera  # defer heavy import

    cv2_utils = None
    prev_log_level: Optional[int] = None
    try:
        import cv2  # type: ignore

        cv2_utils = getattr(cv2, "utils", None)
        cv2_logging = getattr(cv2_utils, "logging", None) if cv2_utils else None
        if cv2_logging is not None and hasattr(cv2_logging, "getLogLevel"):
            prev_log_level = cv2_logging.getLogLevel()
            target_level = (
                getattr(cv2_logging, "LOG_LEVEL_ERROR", None)
                or getattr(cv2_logging, "LOG_LEVEL_FATAL", None)
            )
            if target_level is not None:
                cv2_logging.setLogLevel(target_level)
    except Exception:
        cv2_utils = None
        prev_log_level = None

    indexes: list[int] = []
    last_found = -1
    gap_limit = 4
    devnull = None
    try:
        devnull = open(os.devnull, "w")
        with contextlib.redirect_stderr(devnull), contextlib.redirect_stdout(devnull):
            for idx in range(max_index):
                if last_found >= 0 and (idx - last_found) > gap_limit:
                    break
                if TRACE_DISCOVERY:
                    _log_event(
                        "live.cli.discover.probe",
                        index=str(idx),
                        last_found=str(last_found) if last_found >= 0 else None,
                    )
                try:
                    src = open_camera(idx)
                except Exception as exc:
                    if TRACE_DISCOVERY:
                        _log_event(
                            "live.cli.discover.fail",
                            index=str(idx),
                            error=type(exc).__name__,
                            message=str(exc),
                        )
                    if last_found >= 0 and (idx - last_found) >= gap_limit:
                        if TRACE_DISCOVERY:
                            _log_event(
                                "live.cli.discover.stop",
                                index=str(idx),
                                reason="gap-limit",
                                gap=str(idx - last_found),
                            )
                        break
                    continue
                if TRACE_DISCOVERY:
                    _log_event("live.cli.discover.ok", index=str(idx))
                indexes.append(idx)
                last_found = idx
                try:
                    src.release()
                except Exception:
                    pass
        if TRACE_DISCOVERY:
            _log_event(
                "live.cli.discover.complete",
                found=",".join(str(i) for i in indexes) if indexes else "none",
            )
        return indexes
    finally:
        if prev_log_level is not None and cv2_utils is not None:
            try:
                cv2_utils.logging.setLogLevel(prev_log_level)  # type: ignore[attr-defined]
            except Exception:
                pass
        if devnull is not None:
            try:
                devnull.close()
            except Exception:
                pass


def _build_resolved_spec(spec: _LiveSpec, source: Union[int, str], prefer_small_default: bool) -> _ResolvedSpec:
    prefer_small = prefer_small_default
    override: Optional[Path] = None
    if spec.hint is not None:
        path = _model_hint_to_path(spec.hint, spec.task)
        if path is None:
            raise typer.BadParameter(f"Model token {spec.hint.source!r} is not valid for task '{spec.task}'.")
        if not path.exists():
            raise typer.BadParameter(f"Model weight {path} (from token {spec.hint.source!r}) not found.")
        override = path
        prefer_small = False
    return _ResolvedSpec(task=spec.task, source=source, override=override, prefer_small=prefer_small)


def _resolve_specs(specs: List[_LiveSpec], use_all: bool, prefer_small_default: bool) -> List[_ResolvedSpec]:
    if not specs:
        return []

    if use_all:
        if len(specs) != 1:
            raise typer.BadParameter("When using 'all', specify exactly one task/model.")
        cameras = _discover_cameras()
        if not cameras:
            raise typer.BadParameter("No cameras detected.")
        _log_event("live.cli.cameras", cameras=",".join(_format_source_label(c) for c in cameras))
        base = specs[0]
        return [_build_resolved_spec(base, cam, prefer_small_default) for cam in cameras]

    resolved_pairs: List[tuple[int, _ResolvedSpec]] = []
    pending_defaults: List[tuple[int, _LiveSpec]] = []
    used_int_sources: set[int] = set()

    for idx, spec in enumerate(specs):
        if spec.source is None:
            pending_defaults.append((idx, spec))
            continue
        src = _normalise_source_token(spec.source)
        res = _build_resolved_spec(spec, src, prefer_small_default)
        resolved_pairs.append((idx, res))
        if isinstance(src, int):
            used_int_sources.add(src)

    if pending_defaults:
        cameras = _discover_cameras()
        if not cameras:
            raise typer.BadParameter("No cameras detected.")
        _log_event("live.cli.cameras", cameras=",".join(_format_source_label(c) for c in cameras))
        available = (cam for cam in cameras if cam not in used_int_sources)
        for idx, spec in pending_defaults:
            try:
                src = next(available)
            except StopIteration:
                raise typer.BadParameter("Not enough cameras detected to satisfy the request.") from None
            res = _build_resolved_spec(spec, src, prefer_small_default)
            resolved_pairs.append((idx, res))
            used_int_sources.add(src)

    resolved_pairs.sort(key=lambda item: item[0])
    return [res for _, res in resolved_pairs]


def _format_source_label(source: Union[int, str]) -> str:
    if isinstance(source, int):
        return str(source + 1)
    return str(source)


def _run_pipeline_worker(
    idx: int,
    task: str,
    source: Union[int, str],
    override_path: Optional[str],
    prefer_small: bool,
    fps: Optional[int],
    camera_auto_exposure: Optional[str],
    camera_exposure: Optional[float],
    size: Optional[Tuple[int, int]],
    headless: bool,
    conf: float,
    iou: float,
    duration: Optional[float],
    save_path: Optional[str],
    display_name: str,
    preprocess_device: str,
    warmup: bool,
    backend: str,
    ort_threads: Optional[int],
    ort_execution: Optional[str],
    nms_mode: str,
    result_queue: MPQueue[ResultMessage],
) -> None:
    try:
        override = Path(override_path) if override_path else None
        pipeline = LivePipeline(
            source=source,
            task=task,
            autosave=bool(save_path),
            out_path=save_path,
            prefer_small=prefer_small,
            fps=fps,
            camera_auto_exposure=camera_auto_exposure,
            camera_exposure=camera_exposure,
            size=size,
            headless=headless,
            conf=conf,
            iou=iou,
            duration=duration,
            override=override,
            display_name=display_name,
            preprocess_device=preprocess_device,
            warmup=warmup,
            backend=backend,
            ort_threads=ort_threads,
            ort_execution=ort_execution,
            nms_mode=nms_mode,
        )
        output = pipeline.run()
        result_queue.put(("ok", idx, output or ""))
    except Exception:
        result_queue.put(("err", idx, traceback.format_exc()))


@app.command()
def run(
    tokens: List[str] = typer.Argument(
        None,
        metavar="[TASK] [SOURCE]",
        help=(
            "Task (d|detect|hm|heatmap|clf|classify|pse|pose|obb|object) and source "
            "(camera index, path, or 'synthetic')."
        ),
    ),
    *,
    duration: Optional[float] = typer.Option(None, "--duration", help="Seconds to run; default until quit."),
    headless: bool = typer.Option(False, "--headless", help="Disable preview window."),
    save: Optional[str] = typer.Option(None, "--save", "-o", help="Optional MP4 output path."),
    fps: Optional[int] = typer.Option(None, "--fps", help="Target FPS for writer (default 30)."),
    camera_auto_exposure: Optional[str] = typer.Option(
        None,
        "--camera-auto-exposure",
        help="Override camera auto-exposure (auto/manual/off or numeric value).",
    ),
    camera_exposure: Optional[float] = typer.Option(
        None,
        "--camera-exposure",
        help="Explicit exposure value passed to the camera driver.",
    ),
    width: Optional[int] = typer.Option(None, "--width", help="Capture width hint."),
    height: Optional[int] = typer.Option(None, "--height", help="Capture height hint."),
    conf: float = typer.Option(0.25, "--conf", help="Detector confidence (detect/pose/obb where applicable)."),
    iou: float = typer.Option(0.45, "--iou", help="Detector IOU (detect/obb where applicable)."),
    small: bool = typer.Option(True, "--small/--no-small", help="Prefer small models for live."),
    support_bundle: bool = typer.Option(False, "--support-bundle", help="Write a support bundle zip after the session"),
    preprocess_device: str = typer.Option(
        "auto",
        "--preprocess-device",
        metavar="[cpu|gpu|auto]",
        help="Choose where preprocessing runs ('cpu', 'gpu', or 'auto').",
    ),
    warmup: bool = typer.Option(
        True,
        "--warmup/--no-warmup",
        help="Disable to skip dummy inferences that pre-warm the model.",
    ),
    backend: str = typer.Option(
        "auto",
        "--backend",
        metavar="[auto|torch|ort|tensorrt]",
        help="Select the preferred inference backend.",
    ),
    nms_mode: str = typer.Option(
        "auto",
        "--nms-mode",
        metavar="[auto|graph|torch]",
        help="Override non-max suppression handling: rely on the model graph, force Torch GPU NMS, or auto-detect.",
    ),
    ort_threads: Optional[int] = typer.Option(
        None,
        "--ort-threads",
        help="Override ONNX Runtime intra-op thread count.",
    ),
    ort_execution: Optional[str] = typer.Option(
        None,
        "--ort-execution",
        metavar="[sequential|parallel]",
        help="Set ONNX Runtime execution mode.",
    ),
) -> None:
    """Launch the live webcam/video pipeline."""

    _log_event("live.cli.start", tokens=",".join(tokens) if tokens else None, duration=duration, headless=headless, save=save)

    _clear_live_overrides()

    preprocess_device_norm = (preprocess_device or "auto").strip().lower()
    if preprocess_device_norm not in {"auto", "cpu", "gpu"}:
        raise typer.BadParameter("--preprocess-device must be one of cpu/gpu/auto")

    _log_event("live.cli.preprocess_device", device=preprocess_device_norm)

    backend_norm = (backend or "auto").strip().lower()
    if backend_norm not in {"auto", "torch", "ort", "tensorrt"}:
        raise typer.BadParameter("--backend must be one of auto/torch/ort/tensorrt")

    nms_mode_norm = (nms_mode or "auto").strip().lower()
    if nms_mode_norm not in {"auto", "graph", "torch"}:
        raise typer.BadParameter("--nms-mode must be one of auto/graph/torch")

    ort_execution_norm: Optional[str] = None
    if ort_execution is not None:
        val = ort_execution.strip().lower()
        if val not in {"sequential", "parallel"}:
            raise typer.BadParameter("--ort-execution must be sequential or parallel")
        ort_execution_norm = val

    _log_event("live.cli.backend", backend=backend_norm, ort_threads=ort_threads, ort_execution=ort_execution_norm)
    _log_event("live.cli.nms", mode=nms_mode_norm)

    # Some environments + Click variadic args can mis-route option values.
    # If Typer didn't bind --save/-o, fall back to parsing sys.argv directly.
    if save is None:
        argv = sys.argv[1:]
        for flag in ("--save", "-o"):
            if flag in argv:
                idx = argv.index(flag)
                if idx + 1 < len(argv):
                    val = argv[idx + 1]
                    if not val.startswith("-"):
                        save = val
                        break

    cleaned_tokens: List[str] = []
    for tok in (tokens or []):
        trimmed = tok.strip()
        if not trimmed:
            continue
        low = trimmed.lower()
        if low == "run":
            continue
        if low in _LIVE_MARKERS:
            continue
        cleaned_tokens.append(trimmed)

    cleaned_tokens, use_all = _extract_all_flag(cleaned_tokens)
    _log_event("live.cli.tokens", tokens=",".join(cleaned_tokens))
    specs = _parse_specs(cleaned_tokens)
    if not specs:
        specs = [_LiveSpec(task="detect", hint=None, source=None)]

    resolved_specs = _resolve_specs(specs, use_all, small)
    if not resolved_specs:
        raise typer.BadParameter("No live tasks were specified.")

    size: Optional[Tuple[int, int]] = (width, height) if (width and height) else None

    ort_status = ort_available()
    torch_ok = torch_available()
    try:
        import cv2  # type: ignore

        opencv_desc = getattr(cv2, "__version__", "unknown")
    except Exception as exc:
        opencv_desc = f"missing ({type(exc).__name__}: {exc})"

    allow_cpu_fallback = os.environ.get("ARGOS_ALLOW_CPU_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}

    capabilities = {  # type: ignore[var-annotated]
        "ort": {
            "ok": bool(ort_status.ok and ort_status.providers_ok),
            "version": ort_status.version,
            "providers": ort_status.providers,
            "reason": ort_status.reason,
            "expected": ort_status.expected_provider,
            "providers_ok": ort_status.providers_ok,
            "healed": ort_status.healed,
        },
        "torch": {"ok": bool(torch_ok)},
        "opencv": opencv_desc,
    }
    healed_provider_label: Optional[str] = None
    if ort_status.summary:
        capabilities["ort"]["summary"] = ort_status.summary  # type: ignore[index]
    _log_event("live.capabilities", **capabilities)  # type: ignore[arg-type]

    if ort_status.healed and ort_status.ok and ort_status.providers_ok and ort_status.expected_provider:
        healed_provider_label = ort_status.expected_provider
        os.environ["PANOPTES_ORT_HEALED"] = healed_provider_label

    if not ort_status.ok and not allow_cpu_fallback:
        _log_event(
            "live.capabilities.abort",
            reason=ort_status.reason or "onnxruntime unavailable",
            expected=ort_status.expected_provider,
            providers=",".join(ort_status.providers or []),
        )
        raise typer.Exit(code=1)
    if not ort_status.ok and allow_cpu_fallback:
        _log_event(
            "live.capabilities.fallback",
            reason=ort_status.reason or "onnxruntime unavailable",
            expected=ort_status.expected_provider,
            providers=",".join(ort_status.providers or []),
        )

    result_queue: MPQueue[ResultMessage] = mp.Queue()
    processes: list[mp.Process] = []
    try:
        for idx, spec in enumerate(resolved_specs):
            if healed_provider_label and idx == 1:
                os.environ.pop("PANOPTES_ORT_HEALED", None)
                healed_provider_label = None
            src_label = _format_source_label(spec.source)
            with bind_context(live_task=spec.task, source=src_label):
                _log_event("live.cli.selection", task=spec.task, source=src_label, small=spec.prefer_small, save=save, headless=headless)
            proc = mp.Process(
                target=_run_pipeline_worker,
                args=(
                    idx,
                    spec.task,
                    spec.source,
                    str(spec.override) if spec.override is not None else None,
                    spec.prefer_small,
                    fps,
                    camera_auto_exposure,
                    camera_exposure,
                    size,
                    headless,
                    conf,
                    iou,
                    duration,
                    save,
                    f"ARGOS Live ({spec.task}:{src_label})",
                    preprocess_device_norm,
                    warmup,
                    backend_norm,
                    ort_threads,
                    ort_execution_norm,
                    nms_mode_norm,
                    result_queue,
                ),
            )
            proc.start()
            processes.append(proc)

        outputs: list[tuple[int, str]] = []
        errors: list[tuple[int, str]] = []
        remaining = len(processes)
        while remaining > 0:
            try:
                kind, proc_idx, payload = result_queue.get(timeout=0.5)
            except queue.Empty:
                crashed = [p for p in processes if p.exitcode not in (None, 0)]
                if crashed:
                    break
                continue
            if kind == "ok":
                if payload:
                    outputs.append((proc_idx, payload))
            else:
                errors.append((proc_idx, payload))
            remaining -= 1

        for proc in processes:
            proc.join()

        for proc in processes:
            if proc.exitcode not in (0, None):
                errors.append((processes.index(proc), f"Process exited with code {proc.exitcode}"))

        if errors:
            detail = errors[0][1]
            raise RuntimeError(f"live pipeline failed: {detail}")

        for _, out in sorted(outputs):
            print(out)

        if support_bundle and outputs:
            try:
                bundle_path = write_support_bundle(extra_paths=[out for _, out in sorted(outputs)])
                if bundle_path:
                    typer.echo(f"Support bundle written: {bundle_path}")
            except Exception as exc:
                typer.secho(f"[bold red] failed to create support bundle: {exc}", err=True)

    except KeyboardInterrupt:
        for proc in processes:
            proc.terminate()
        raise
    finally:
        try:
            result_queue.close()
        except Exception:
            pass
        os.environ.pop("PANOPTES_ORT_HEALED", None)


def _prepend_argv(token: str) -> None:
    # Idempotently insert right before the first non-option token,
    # but only if it's not already present there.
    argv = sys.argv[1:]
    for a in argv:
        if a.startswith("-"):
            continue  # skip global options
        if a == token:
            return  # already present; do nothing
        sys.argv = sys.argv[:1] + [token] + sys.argv[1:]
        return
    # No positional args at all -> append at position 1
    sys.argv = sys.argv[:1] + [token] + sys.argv[1:]


def main() -> None:  # pragma: no cover
    try:
        # Make "python -m panoptes.live.cli hm synthetic ..." work without explicitly typing "run".
        argv = sys.argv[1:]
        if argv and not any(a in {"-h", "--help"} for a in argv):
            _prepend_argv("run")
        app()
    except KeyboardInterrupt:
        raise SystemExit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
