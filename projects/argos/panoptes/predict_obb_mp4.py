# projects/argos/panoptes/predict_obb_mp4.py
"""
predict_obb_mp4 — per‑frame oriented bounding boxes overlay for videos.

ONE PROGRESS SYSTEM
───────────────────
* Uses ONLY the Halo/Rich spinner from panoptes.progress.
* If a parent spinner is passed (recommended via CLI), we ONLY update its
  [File | Job | Model] fields (no 'current' writes) so the single-line
  format/colors match Detect/Heatmap/GeoJSON.
* If no parent spinner is provided AND no global spinner is marked active
  (PANOPTES_PROGRESS_ACTIVE != "1"), a local Halo spinner is started for
  standalone runs — still a single consistent line.

Strict weights
──────────────
* Either pass explicit *weights=* or load strictly via model_registry.load_obb().

Output ladder
─────────────
* FFmpeg H.264 preferred → OpenCV mp4v fallback → keep MJPG .avi as last resort.
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
from typing import Any, Callable, Optional, Protocol, Tuple, cast

import cv2  # type: ignore
import numpy as np
from numpy.typing import NDArray

from panoptes import ROOT  # type: ignore[import-not-found]
from panoptes.model_registry import load_obb  # type: ignore[import-not-found]
from .ffmpeg_utils import resolve_ffmpeg

# ---- progress spinner (Halo-based) ------------------------------------------
try:
    from panoptes.progress import percent_spinner  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    percent_spinner = None  # type: ignore[assignment]

# ---- logging ----------------------------------------------------------------
_LOG = logging.getLogger("panoptes.predict_obb_mp4")
_LOG.addHandler(logging.NullHandler())
_LOG.setLevel(logging.ERROR)


def _say(msg: str) -> None:
    if _LOG.isEnabledFor(logging.INFO):
        _LOG.info("[panoptes] %s", msg)


# ---- helpers ----------------------------------------------------------------
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
    w = int(cv2.CAP_PROP_FRAME_WIDTH if False else cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # keep type-checkers happy
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


NDArrayU8 = NDArray[np.uint8]
NDArrayF32 = NDArray[np.float32]


def _extract_polys(res: Any) -> list[NDArrayF32]:
    """
    Return list of (4,2) float32 polygons if available; else [].
    Supports Ultralytics OBB results (xyxyxyxy) and variants; falls back to AABB.
    """
    polys: list[NDArrayF32] = []

    obb = getattr(res, "obb", None)
    if obb is not None:
        for key in ("xyxyxyxy", "xyxy", "xy"):
            pts = getattr(obb, key, None)
            if pts is None:
                continue
            if hasattr(pts, "cpu"):
                try:
                    pts = pts.cpu().numpy()
                except Exception:
                    pass
            arr = np.asarray(pts, dtype=np.float32)
            if arr.size == 0:
                continue

            # Normalize to shape (-1, 8) then → (-1, 4, 2)
            try:
                flat = arr.reshape(-1, 8)
            except Exception:
                # Some reps may already be (N,4,2)
                if arr.ndim == 3 and arr.shape[-2:] == (4, 2):
                    flat = arr.reshape(-1, 8)
                else:
                    continue
            shaped = flat.reshape(-1, 4, 2).astype(np.float32, copy=False)
            for poly in shaped:
                polys.append(np.asarray(poly, dtype=np.float32))
            if polys:
                return polys  # prefer first found representation

    # AABB fallback
    boxes = getattr(getattr(res, "boxes", None), "xyxy", None)
    if boxes is not None:
        if hasattr(boxes, "cpu"):
            try:
                boxes = boxes.cpu().numpy()
            except Exception:
                pass
        arr2 = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        for x1, y1, x2, y2 in arr2:
            polys.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
    return polys


def _poke(sp: SpinnerLike | None, **fields: Any) -> None:
    if sp is None:
        return
    try:
        sp.update(**fields)
    except Exception:
        pass


def _model_label(m: Any) -> str:
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


# ---- main worker -------------------------------------------------------------
def main(  # noqa: C901
    src: str | Path,
    *,
    out_dir: str | Path | None = None,
    weights: str | Path | None = None,
    conf: float | None = None,
    iou: float | None = None,
    verbose: bool = False,
    progress: SpinnerLike | None = None,
    **kw: Any,
) -> Path:
    if verbose:
        _LOG.setLevel(logging.INFO)

    srcp = Path(src).expanduser().resolve()
    if not srcp.exists():
        raise FileNotFoundError(srcp)
    if out_dir is None:
        out_dir = ROOT / "tests" / "results"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # strict model
    override: Optional[Path] = None
    if weights is not None:
        cand = Path(weights).expanduser().resolve()
        if not cand.exists():
            raise FileNotFoundError(f"override weight not found: {cand}")
        override = None if cand.is_dir() else cand
    obb_model: Any = load_obb(override=override)  # type: ignore[call-arg]
    model_lbl = _model_label(obb_model)

    cap = cv2.VideoCapture(str(srcp))
    if not cap.isOpened():
        raise RuntimeError(f"❌  Cannot open video {srcp}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 1
    _say(f"video obb: src={srcp.name} fps={fps:.3f} size={w}x{h} frames~{total}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="dv_obb_"))
    avi = tmp_dir / f"{srcp.stem}.avi"
    vw = _avi_writer(avi, fps, (w, h))

    # local spinner only if parent not provided AND no global spinner is active
    parent_active = (os.environ.get("PANOPTES_PROGRESS_ACTIVE") or "").strip() == "1"
    use_local = (progress is None) and (not parent_active) and (percent_spinner is not None)
    sp_ctx = percent_spinner(prefix="OBB-MP4", stream=sys.stderr) if use_local else _NullCtx()  # type: ignore[operator]

    i = 0
    file_lbl = srcp.name
    with sp_ctx:
        if use_local:
            _poke(sp_ctx, total=float(total + 1), count=0.0, item=file_lbl, job=f"frame 0/{total}", model=model_lbl)  # type: ignore[arg-type]
        else:
            _poke(progress, item=file_lbl, job=f"frame 0/{total}", model=model_lbl, total=total + 1, count=0)

        while True:
            ok, frame_raw = cap.read()
            if not ok:
                break
            frame: NDArrayU8 = np.asarray(frame_raw, dtype=np.uint8)

            res_list: list[Any] = cast(
                list[Any],
                obb_model.predict(frame, imgsz=640, conf=(conf or 0.25), iou=(iou or 0.45), verbose=False)
            )
            res = res_list[0] if res_list else None
            if res is not None:
                polys = _extract_polys(res)
                for poly in polys:
                    pts = np.asarray(poly, dtype=np.float32).reshape(4, 2).astype(np.int32)
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 210, 255), thickness=2)
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

        preferred = out_dir / f"{srcp.stem}_obb.mp4"
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
                final = out_dir / f"{srcp.stem}_obb.avi"
                shutil.move(str(avi), str(final))

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Saved → {_osc8(final.name, final)}")

        if use_local:
            _poke(sp_ctx, count=float(total + 1), item=file_lbl, job="done", model=model_lbl)  # type: ignore[arg-type]
        else:
            _poke(progress, item=file_lbl, job="done", model=model_lbl, total=total + 1, count=total + 1)

        return final
