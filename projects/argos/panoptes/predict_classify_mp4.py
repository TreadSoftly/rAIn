# projects/argos/panoptes/predict_classify_mp4.py
"""
predict_classify_mp4 — per-frame classification overlay for videos.

ONE PROGRESS SYSTEM
───────────────────
* Uses ONLY the Halo/Rich spinner from panoptes.progress.
* If a parent spinner is passed (recommended via CLI), we ONLY update its
  [File | Job | Model] fields (no 'current' writes) so the single-line
  format and colors exactly match Detect/Heatmap/GeoJSON.
* If no parent spinner is provided AND no global spinner is marked active
  (PANOPTES_PROGRESS_ACTIVE != "1"), a local Halo spinner is started so
  standalone runs still have a single consistent line.

Strict weights
──────────────
* Either pass an explicit *weights=* path, or load strictly via
  panoptes.model_registry.load_classifier().

Output
──────
* Prefers H.264 via FFmpeg → OpenCV mp4v fallback → keeps MJPG .avi last.
* Emits "<stem>_cls.mp4" (or ".avi" as last resort).
* Prints exactly one "Saved → <clickable>" line at the end (OSC-8).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Optional, Protocol, Sequence, Tuple, cast

import cv2  # type: ignore
import numpy as np
from numpy.typing import NDArray

from panoptes import ROOT  # type: ignore[import-not-found]
from .ffmpeg_utils import resolve_ffmpeg

# Strict model loader (central registry)
try:
    from panoptes.model_registry import load_classifier  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    load_classifier = None  # type: ignore[assignment]

# Halo/Rich spinner
try:
    from panoptes.progress import percent_spinner  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    percent_spinner = None  # type: ignore[assignment]

NDArrayU8 = NDArray[np.uint8]

_LOG = logging.getLogger("panoptes.predict_classify_mp4")
_LOG.addHandler(logging.NullHandler())
_LOG.setLevel(logging.ERROR)


def _say(msg: str) -> None:
    if _LOG.isEnabledFor(logging.INFO):
        _LOG.info("[panoptes] %s", msg)


class SpinnerLike(Protocol):
    def update(self, **kwargs: Any) -> "SpinnerLike": ...


class _NullCtx:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False


def _osc8(label: str, path: Path) -> str:
    try:
        uri = path.resolve().as_uri()
    except Exception:
        uri = "file:///" + str(path.resolve()).replace("\\", "/")
    esc = "\033"
    return f"{esc}]8;;{uri}{esc}\\{label}{esc}]8;;{esc}\\"


def _fourcc(code: str) -> int:
    fn: Callable[..., Any] = getattr(cv2, "VideoWriter_fourcc", getattr(cv2.VideoWriter, "fourcc"))
    return cast(int, fn(*code))


def _avi_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    vw = cv2.VideoWriter(str(path.with_suffix(".avi")), _fourcc("MJPG"), fps, size)
    if not vw.isOpened():
        raise RuntimeError("❌  OpenCV cannot open any MJPG writer on this system.")
    return vw


def _opencv_reencode_to_mp4(src_avi: Path, dst_mp4: Path, fps: float) -> bool:
    cap = cv2.VideoCapture(str(src_avi))
    if not cap.isOpened():
        return False
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(dst_mp4), _fourcc("mp4v"), fps, (w, h))
    if not out.isOpened():
        cap.release()
        return False
    ok_any = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
        ok_any = True
    cap.release()
    out.release()
    try:
        return ok_any and dst_mp4.exists() and dst_mp4.stat().st_size > 0
    except Exception:
        return False


def _topk_from_probs(model: Any, res: Any, k: int) -> list[tuple[str, float]]:
    names = getattr(model, "names", None) or getattr(res, "names", None)

    def _name_for(i: int) -> str:
        if isinstance(names, dict):
            d = cast(dict[int, str], names)
            return str(d.get(i, i))
        if isinstance(names, (list, tuple)):
            seq = cast(Sequence[str], names)
            return str(seq[i]) if 0 <= i < len(seq) else str(i)
        return str(i)

    probs = getattr(res, "probs", None)
    if probs is None:
        return []
    v = getattr(probs, "data", probs)
    if hasattr(v, "cpu"):
        try:
            v = v.cpu().numpy()  # type: ignore[assignment]
        except Exception:
            pass
    vec = np.asarray(v, dtype=np.float32).reshape(-1)
    idx = np.argsort(-vec)[: max(1, int(k))]
    return [(_name_for(int(i)), float(vec[int(i)])) for i in idx]


def _draw_topk_card_bgr(frame: NDArrayU8, pairs: Sequence[tuple[str, float]]) -> NDArrayU8:
    if not pairs:
        return frame
    x0, y0 = 8, 8
    pad = 8
    gap = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    maxw = 0
    totalh = 0
    for i, (name, prob) in enumerate(pairs):
        text = f"{i+1}. {name}  {prob:.2f}"
        (tw, th), _ = cv2.getTextSize(text, font, 0.6, 1)
        maxw = max(maxw, tw)
        totalh += th + gap
    totalh = max(0, totalh - gap)
    x1 = x0 + maxw + pad * 2
    y1 = y0 + totalh + pad * 2
    overlay: NDArrayU8 = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    frame = cast(NDArrayU8, cv2.addWeighted(overlay, 0.66, frame, 0.34, 0.0))

    y = y0 + pad
    for i, (name, prob) in enumerate(pairs):
        text = f"{i+1}. {name}  {prob:.2f}"
        cv2.putText(frame, text, (x0 + pad, y + 16), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20 + gap
    return frame


def _poke(sp: SpinnerLike | None, **fields: Any) -> None:
    """Best-effort spinner update (safe on both local and parent spinners)."""
    if sp is None:
        return
    try:
        sp.update(**fields)
    except Exception:
        pass


def _model_label(m: Any) -> str:
    """Human-friendly model label for the [Model: …] segment."""
    cand = (
        getattr(m, "name", None)
        or getattr(m, "model_name", None)
        or getattr(m, "weights", None)
        or getattr(getattr(m, "model", None), "name", None)
    )
    if cand:
        try:
            return Path(str(cand)).name
        except Exception:
            return str(cand)
    return m.__class__.__name__


def main(  # noqa: C901
    src: str | Path,
    *,
    out_dir: str | Path | None = None,
    weights: str | Path | None = None,
    small: bool = False,     # reserved; not used directly here
    conf: float | None = None,
    topk: int = 1,
    verbose: bool = False,
    progress: SpinnerLike | None = None,
    **kw: Any,
) -> Path:
    """
    Per‑frame classification overlay, writes <stem>_cls.mp4 (or .avi as last resort).
    """
    if verbose:
        _LOG.setLevel(logging.INFO)

    srcp = Path(src).expanduser().resolve()
    if not srcp.exists():
        raise FileNotFoundError(srcp)
    if out_dir is None:
        out_dir = ROOT / "tests" / "results"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model (strict)
    override: Optional[Path] = None
    if weights is not None:
        cand = Path(weights).expanduser().resolve()
        if not cand.exists():
            raise FileNotFoundError(f"override weight not found: {cand}")
        override = None if cand.is_dir() else cand
    if load_classifier is None:
        raise RuntimeError("Classifier loader is unavailable (model_registry missing).")
    cls_model: Any = load_classifier(override=override)  # type: ignore[call-arg]
    model_lbl = _model_label(cls_model)

    cap = cv2.VideoCapture(str(srcp))
    if not cap.isOpened():
        raise RuntimeError(f"❌  Cannot open video {srcp}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 1
    _say(f"video classify: src={srcp.name} fps={fps:.3f} size={w}x{h} frames~{total}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="dv_cls_"))
    avi = tmp_dir / f"{srcp.stem}.avi"
    vw = _avi_writer(avi, fps, (w, h))

    # Start a local Halo spinner ONLY if no parent spinner is passed AND no global spinner is active
    parent_active = (os.environ.get("PANOPTES_PROGRESS_ACTIVE") or "").strip() == "1"
    use_local = (progress is None) and (not parent_active) and (percent_spinner is not None)
    sp_ctx = percent_spinner(prefix="CLASSIFY-MP4", stream=sys.stderr) if use_local else _NullCtx()  # type: ignore[operator]

    i = 0
    file_lbl = srcp.name
    with sp_ctx:
        if use_local:
            # local spinner drives per-frame percent
            _poke(sp_ctx, total=float(total + 1), count=0.0, item=file_lbl, job=f"frame 0/{total}", model=model_lbl)  # type: ignore[arg-type]
        else:
            # parent spinner: keep File/Job/Model populated while also driving count
            _poke(progress, item=file_lbl, job=f"frame 0/{total}", model=model_lbl, total=total + 1, count=0)

        while True:
            ok, frame_raw = cap.read()
            if not ok:
                break
            frame: NDArrayU8 = np.asarray(frame_raw, dtype=np.uint8)

            # YOLO‑style predict accepts ndarray (BGR)
            res_list: list[Any] = cast(list[Any], cls_model.predict(frame, imgsz=640, conf=(conf or 0.0), verbose=False))
            res = res_list[0] if res_list else None
            pairs = _topk_from_probs(cls_model, res, max(1, int(topk))) if res is not None else []
            frame = _draw_topk_card_bgr(frame, pairs)
            vw.write(frame)
            i += 1

            if use_local:
                _poke(sp_ctx, count=float(i), item=file_lbl, job=f"frame {min(i, total)}/{total}", model=model_lbl)  # type: ignore[arg-type]
            else:
                _poke(
                    progress,
                    item=file_lbl,
                    job=f"frame {min(i, total)}/{total}",
                    model=model_lbl,
                    total=total + 1,
                    count=min(i, total),
                )

        cap.release()
        vw.release()

        preferred = out_dir / f"{srcp.stem}_cls.mp4"
        # encode step
        if use_local:
            _poke(sp_ctx, item=file_lbl, job="encode mp4", model=model_lbl)  # type: ignore[arg-type]
        else:
            _poke(progress, item=file_lbl, job="encode mp4", model=model_lbl, total=total + 1, count=total)

        ffmpeg_path, ffmpeg_source = resolve_ffmpeg()
        try:
            if not ffmpeg_path:
                raise FileNotFoundError("ffmpeg not found")
            subprocess.run(
                [ffmpeg_path, "-y", "-i", str(avi), "-c:v", "libx264", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(preferred)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
            avi.unlink(missing_ok=True)
            _say(f"FFmpeg re-encode successful ({ffmpeg_source})")
            final = preferred
        except (FileNotFoundError, subprocess.CalledProcessError):
            ok_mp4 = _opencv_reencode_to_mp4(avi, preferred, fps=fps)
            if ok_mp4:
                avi.unlink(missing_ok=True)
                final = preferred
            else:
                final = out_dir / f"{srcp.stem}_cls.avi"
                shutil.move(str(avi), str(final))

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Saved → {_osc8(final.name, final)}")

        # done
        if use_local:
            _poke(sp_ctx, count=float(total + 1), item=file_lbl, job="done", model=model_lbl)  # type: ignore[arg-type]
        else:
            _poke(progress, item=file_lbl, job="done", model=model_lbl, total=total + 1, count=total + 1)

        return final
