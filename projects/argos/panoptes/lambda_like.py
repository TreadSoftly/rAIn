# projects/argos/panoptes/lambda_like.py
from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import urllib.request
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol, TextIO, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .geo_sink import to_geojson
from .heatmap import heatmap_overlay
from .model_registry import load_detector, load_segmenter

_LOG = logging.getLogger("panoptes.lambda_like")
_LOG.addHandler(logging.NullHandler())
_LOG.setLevel(logging.ERROR)

def _say(msg: str) -> None:
    if _LOG.isEnabledFor(logging.DEBUG):
        _LOG.debug("[panoptes] %s", msg)

def set_log_level(level: int) -> None:
    _LOG.setLevel(level)

def set_verbose(enabled: bool = True) -> None:
    set_log_level(logging.INFO if enabled else logging.WARNING)

_BASE = Path(__file__).resolve().parent
ROOT  = _BASE.parent


def _results_dir() -> Path:
    """
    Return the results directory honoring the PANOPTES_RESULTS_DIR override.
    Tests and the CLI set this so that outputs land in the repo tree even
    when the CLI re-execs into the managed venv.
    """
    override = os.getenv("PANOPTES_RESULTS_DIR")
    if override:
        base = Path(override).expanduser()
    else:
        base = ROOT / "tests" / "results"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return base

try:
    from ultralytics import YOLO  # type: ignore
    _has_yolo = True
except ImportError:                # pragma: no cover
    YOLO      = None               # type: ignore
    _has_yolo = False

# Preload primary models once (respecting the central registry)
def _safe_load(
    loader: Callable[..., Any],
    *,
    task: str,
    small: Optional[bool] = None,
    override: Optional[Union[str, Path]] = None,
    fallback: Any = None,
) -> Any:
    try:
        kwargs: dict[str, Any] = {}
        if small is not None:
            kwargs["small"] = small
        if override is not None:
            kwargs["override"] = override
        model = loader(**kwargs)
        _say(f"loaded {task} model ({'small' if small else 'full'})")
        return model
    except Exception as exc:  # pragma: no cover - weight genuinely absent
        _LOG.warning("model load failed for %s: %s", task, exc)
        return fallback


_det_model: Any | None = _safe_load(load_detector, task="detect", small=False)
_seg_model: Any | None = _safe_load(load_segmenter, task="heatmap", small=True)


def reinit_models(
    *,
    detect_small: Optional[bool] = None,
    segment_small: Optional[bool] = None,
    det_override: Optional[Union[str, Path]] = None,
    seg_override: Optional[Union[str, Path]] = None,
) -> tuple[Any, Any]:
    """
    Reinitialize cached models based on small/override hints.
    No progress UI is created here; the single CLI spinner remains the only UI.
    """
    global _det_model, _seg_model
    if detect_small is not None or det_override is not None:
        _det_model = _safe_load(
            load_detector,
            task="detect",
            small=detect_small,
            override=det_override,
        )
    if segment_small is not None or seg_override is not None:
        _seg_model = _safe_load(
            load_segmenter,
            task="heatmap",
            small=segment_small,
            override=seg_override,
        )
    return _det_model, _seg_model


def run_inference(
    img: Image.Image,
    model: str = "primary",
    *,
    conf_thr: float = 0.40,
) -> NDArray[np.float32]:
    """
    Run detection using the preloaded detector. This function never opens
    its own progress UI and remains silent except for explicit logging.
    """
    if not _has_yolo or _det_model is None:
        raise RuntimeError("Ultralytics YOLO is not installed in this environment.")
    res_list = _det_model.predict(img, imgsz=640, conf=conf_thr, verbose=False)  # type: ignore
    if not res_list:
        return np.empty((0, 6), dtype=np.float32)

    res: Any = res_list[0]  # type: ignore[assignment]
    boxes = getattr(res, "boxes", None)  # type: ignore
    if boxes is None or not hasattr(boxes, "data"):
        return np.empty((0, 6), dtype=np.float32)

    boxes_data = res.boxes.data  # type: ignore
    try:
        if hasattr(boxes_data, "cpu"):
            boxes_arr = boxes_data.cpu().numpy().astype(np.float32)  # type: ignore[attr-defined]
        else:
            boxes_arr = np.asarray(boxes_data, dtype=np.float32)
    except Exception:
        boxes_arr = np.asarray(boxes_data, dtype=np.float32)

    return boxes_arr.reshape(-1, 6) if boxes_arr.size else np.empty((0, 6), dtype=np.float32)


def _is_remote(src: str) -> bool:
    return src.startswith(("http://", "https://", "data:"))


@contextmanager
def _silence_stdio(enabled: bool):
    """
    Redirect stdout/stderr to os.devnull while active when *enabled* is True.
    Spinner output uses sys.__stderr__ in the CLI, so it remains visible.
    """
    if not enabled:
        yield
        return
    devnull: TextIO = open(os.devnull, "w", buffering=1, encoding="utf-8", errors="ignore")
    try:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield
    finally:
        try:
            devnull.close()
        except Exception:
            pass


class SpinnerLike(Protocol):
    def update(self, **kwargs: Any) -> "SpinnerLike": ...


def _update_job(progress: Optional[SpinnerLike], text: str) -> None:
    """Update the shared spinner's job field (fallback to legacy current=...)."""
    if progress is None:
        return
    try:
        progress.update(job=str(text))
    except TypeError:
        try:
            progress.update(current=str(text))
        except Exception:
            pass
    except Exception:
        pass


def run_single(  # noqa: C901
    src: Union[str, os.PathLike[str]],
    *,
    model: str = "primary",
    task: Literal["detect", "heatmap", "geojson"] = "detect",
    progress: Optional[SpinnerLike] = None,
    quiet: bool = True,
    **hm_kwargs: Any,
) -> Optional[str]:
    """
    Execute a single image/URL workflow for detect/heatmap/geojson.

    PROGRESS POLICY:
    - We never instantiate progress ourselves.
    - We *only* update the single parent Halo/Rich spinner via `job=...`.
    - If a non-Halo object is passed, we degrade to `current=...` without
      creating any new UI.
    """

    src_str  = str(src)
    conf_thr = float(hm_kwargs.get("conf", 0.40))
    stem     = Path(src_str).stem or "image"

    # ---- load image ---------------------------------------------------------
    _update_job(progress, f"load: {Path(src_str).name if not _is_remote(src_str) else 'URL'}")

    # Silence libraries while loading regardless of spinner presence.
    with _silence_stdio(quiet):
        if src_str.startswith("data:"):
            _, b64data = src_str.split(",", 1)
            pil_img    = Image.open(io.BytesIO(base64.b64decode(b64data))).convert("RGB")
        elif src_str.lower().startswith(("http://", "https://")):
            req = urllib.request.Request(src_str, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                pil_img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        else:
            pil_img = Image.open(src_str).convert("RGB")

    # ---- inference ----------------------------------------------------------
    _update_job(progress, "run inference")
    with _silence_stdio(quiet):
        boxes = run_inference(pil_img, model=model, conf_thr=conf_thr)

    # ── GEOJSON --------------------------------------------------------------
    if task == "geojson":
        _update_job(progress, "compose geojson")
        try:
            geo = to_geojson(
                src_str,
                [list(b)[:5] for b in boxes.tolist()] if boxes.size else None,
            )
        except Exception:
            geo = to_geojson(src_str, None)

        import datetime as _dt
        geo["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

        if _is_remote(src_str):
            # stdout is the API for URLs
            sys.stdout.write(json.dumps(geo, separators=(",", ":")) + "\n")
            _update_job(progress, "done")
            return None
        else:
            out_path = _results_dir() / f"{stem}.geojson"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(geo, indent=2), encoding="utf-8")
            # only log when not in pinned-line mode
            if not quiet or progress is None:
                _say(f"wrote {out_path}")
            _update_job(progress, "done")
            return str(out_path)

    # ── HEATMAP --------------------------------------------------------------
    if task == "heatmap":
        _update_job(progress, "render heatmap")
        with _silence_stdio(quiet):
            alpha_val = float(hm_kwargs.get("alpha", 0.4))
            overlay_result = heatmap_overlay(
                pil_img,
                boxes=boxes if boxes.size else None,
                alpha=alpha_val,
            )

        if isinstance(overlay_result, np.ndarray):
            out_img = Image.fromarray(
                overlay_result.astype(np.uint8) if overlay_result.ndim == 2
                else overlay_result[:, :, ::-1].astype(np.uint8)
            )
        else:
            out_img = overlay_result

        out_ext  = Path(src_str).suffix.lower() if Path(src_str).suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
        out_path = _results_dir() / f"{stem}_heat{out_ext}"

    # ── DETECT ---------------------------------------------------------------
    else:
        _update_job(progress, "render boxes")
        with _silence_stdio(quiet):
            res     = _det_model.predict(pil_img, imgsz=640, conf=conf_thr, verbose=False)[0]  # type: ignore
            plotted: np.ndarray = res.plot()                                                   # type: ignore
        arr = np.asarray(plotted, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]
        out_img = Image.fromarray(arr)

        out_ext  = Path(src_str).suffix.lower() if Path(src_str).suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
        out_path = _results_dir() / f"{stem}_boxes{out_ext}"

    # ---- write result -------------------------------------------------------
    _update_job(progress, "write result")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with _silence_stdio(quiet):
        out_img.save(out_path)
    if not quiet or progress is None:
        _say(f"wrote {out_path}")
    _update_job(progress, "done")
    return str(out_path)
