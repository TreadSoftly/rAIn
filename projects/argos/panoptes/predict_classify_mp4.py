# projects/argos/panoptes/predict_classify_mp4.py
"""
predict_classify_mp4 — per-frame classification overlay for videos.

Lock‑down
─────────
* A **classifier** weight is mandatory — either:
    1) explicit *weights=* (path to .pt/.onnx), or
    2) `panoptes.model_registry.load_classifier()` strict selection.

* No env probing; the model registry controls weights.

Progress & Output
─────────────────
* If *progress* is provided, we only update its `current` label (frame i/N → encode mp4 → done).
* Otherwise, we use a local percent spinner (suppressed when a parent spinner is active).
* Emits `<stem>_cls.mp4` (H.264 via FFmpeg when available; OpenCV fallback; as a last resort keeps MJPG `.avi`).
* Prints exactly one `Saved → <clickable>` line at the end (OSC‑8).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Final, Protocol, Sequence, Tuple, cast

import cv2  # type: ignore
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# ROOT resolution (single assignment; avoids "constant redefined" diagnostics)
# ────────────────────────────────────────────────────────────────────────────────

def _resolve_root() -> Path:
    try:
        from panoptes import ROOT as _ROOT  # type: ignore[import-not-found]
        return _ROOT
    except Exception:
        return Path.cwd()

ROOT: Final[Path] = _resolve_root()

# ────────────────────────────────────────────────────────────────────────────────
# Type aliases (compile‑time only)
# ────────────────────────────────────────────────────────────────────────────────

if TYPE_CHECKING:
    import numpy as _np
    import numpy.typing as npt

    NDArrayU8 = npt.NDArray[_np.uint8]
else:
    NDArrayU8 = Any  # type: ignore[assignment]

# ────────────────────────────────────────────────────────────────────────────────
# Strict model loader (central registry)
# ────────────────────────────────────────────────────────────────────────────────

try:
    from panoptes.model_registry import load_classifier  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    load_classifier = None  # type: ignore[assignment]

# optional progress (safe off‑TTY)
try:
    from panoptes.progress import ProgressEngine  # type: ignore[import-not-found]
    from panoptes.progress.bridges import live_percent  # type: ignore[import-not-found]
    from panoptes.progress.progress_ux import simple_status  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    ProgressEngine = None  # type: ignore[assignment]
    live_percent = None  # type: ignore[assignment]
    simple_status = None  # type: ignore[assignment]

_LOG = logging.getLogger("panoptes.predict_classify_mp4")
if not _LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
_LOG.setLevel(logging.WARNING)


def _say(msg: str) -> None:
    _LOG.info(f"[panoptes] {msg}")


class SpinnerLike(Protocol):
    def update(self, **kwargs: Any) -> "SpinnerLike": ...


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


def main(  # noqa: C901
    src: str | Path,
    *,
    out_dir: str | Path | None = None,
    weights: str | Path | None = None,
    small: bool = False,     # reserved for registry; not used directly here
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
    override: Path | None = None
    if weights is not None:
        cand = Path(weights).expanduser().resolve()
        if not cand.exists():
            raise FileNotFoundError(f"override weight not found: {cand}")
        override = None if cand.is_dir() else cand
    if load_classifier is None:
        raise RuntimeError("Classifier loader is unavailable (model_registry missing).")
    cls_model: Any = load_classifier(override=override)  # type: ignore[call-arg]

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

    # progress orchestration
    ps_progress_active = (os.environ.get("PANOPTES_PROGRESS_ACTIVE") or "") == "1"
    use_local = (progress is None) and (ProgressEngine is not None) and (live_percent is not None) and (not ps_progress_active)  # type: ignore[truthy-bool]
    if use_local:
        eng = ProgressEngine()  # type: ignore[call-arg]
        ctx: ContextManager[None] = cast(ContextManager[None], live_percent(eng, prefix="CLASSIFY-MP4"))  # type: ignore[arg-type]
    else:
        eng = None

        class _Null:
            def __enter__(self) -> None:  # noqa: D401
                return None
            def __exit__(self, et: type[BaseException] | None, ex: BaseException | None, tb: TracebackType | None) -> bool:
                return False
        ctx = _Null()

    i = 0
    with ctx:
        if eng:
            eng.set_total(float(total + 1))
            eng.set_current("frame 0")
        elif progress is not None:
            progress.update(current=f"frame 0/{total}")

        while True:
            ok, frame_raw = cap.read()
            if not ok:
                break
            # Ensure uint8 for type-checker and OpenCV
            if getattr(frame_raw, "dtype", None) is not None and frame_raw.dtype != np.uint8:
                frame_raw = frame_raw.astype(np.uint8)
            frame: NDArrayU8 = cast(NDArrayU8, frame_raw)

            # YOLO‑style predict accepts ndarray (BGR)
            res_list: list[Any] = cast(list[Any], cls_model.predict(frame, imgsz=640, conf=(conf or 0.0), verbose=False))
            res = res_list[0] if res_list else None
            pairs = _topk_from_probs(cls_model, res, max(1, int(topk))) if res is not None else []
            frame = _draw_topk_card_bgr(frame, pairs)
            vw.write(frame)
            i += 1
            if eng:
                eng.add(1.0, current_item=f"frame {min(i, total)}/{total}")
            elif progress is not None:
                progress.update(current=f"frame {min(i, total)}/{total}")

        cap.release()
        vw.release()

        # Re‑encode to MP4
        preferred = out_dir / f"{srcp.stem}_cls.mp4"
        if eng:
            eng.set_current("encode mp4")
        elif progress is not None:
            progress.update(current="encode mp4")

        try:
            if simple_status is not None and use_local:
                sp: ContextManager[None] = cast(ContextManager[None], simple_status("FFmpeg re-encode"))
            else:
                class _Null2:
                    def __enter__(self) -> None: return None
                    def __exit__(self, et: type[BaseException] | None, ex: BaseException | None, tb: TracebackType | None) -> bool: return False
                sp = _Null2()
            with sp:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(avi), "-c:v", "libx264", "-crf", "23",
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(preferred)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                )
            avi.unlink(missing_ok=True)
            final = preferred
        except (FileNotFoundError, subprocess.CalledProcessError):
            # OpenCV fallback or keep AVI
            ok_mp4 = _opencv_reencode_to_mp4(avi, preferred, fps=fps)
            if ok_mp4:
                avi.unlink(missing_ok=True)
                final = preferred
            else:
                final = out_dir / f"{srcp.stem}_cls.avi"
                shutil.move(str(avi), str(final))

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Saved → {_osc8(final.name, final)}")
        if eng:
            eng.add(1.0, current_item="done")
        elif progress is not None:
            progress.update(current="done")
        return final
