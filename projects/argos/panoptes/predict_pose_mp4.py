# projects/argos/panoptes/predict_pose_mp4.py
"""
predict_pose_mp4 — per‑frame pose overlay for videos.

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
* Either pass explicit *weights=* or load strictly via model_registry.load_pose().

Output
──────
* FFmpeg H.264 preferred → OpenCV mp4v fallback → keeps MJPG .avi last.
* Emits "<stem>_pose.mp4" (or ".avi" fallback).
* Prints exactly one "Saved → <clickable>" line at the end (OSC‑8).
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
from panoptes.model_registry import load_pose  # type: ignore[import-not-found]

# ---- progress spinner (Halo-based) ------------------------------------------
try:
    from panoptes.progress import percent_spinner  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    percent_spinner = None  # type: ignore[assignment]

# ---- logging ----------------------------------------------------------------
_LOG = logging.getLogger("panoptes.predict_pose_mp4")
if not _LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
_LOG.setLevel(logging.WARNING)


def _say(msg: str) -> None:
    _LOG.info(f"[panoptes] {msg}")


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


# COCO-17 skeleton edges
_COCO_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 3), (0, 2), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

NDArrayU8 = NDArray[np.uint8]
NDArrayF32 = NDArray[np.float32]


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
    pose_model: Any = load_pose(override=override)  # type: ignore[call-arg]
    model_lbl = _model_label(pose_model)

    cap = cv2.VideoCapture(str(srcp))
    if not cap.isOpened():
        raise RuntimeError(f"❌  Cannot open video {srcp}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 1
    _say(f"video pose: src={srcp.name} fps={fps:.3f} size={w}x{h} frames~{total}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="dv_pose_"))
    avi = tmp_dir / f"{srcp.stem}.avi"
    vw = _avi_writer(avi, fps, (w, h))

    # local spinner only if parent not provided AND no global spinner is active
    parent_active = (os.environ.get("PANOPTES_PROGRESS_ACTIVE") or "").strip() == "1"
    use_local = (progress is None) and (not parent_active) and (percent_spinner is not None)
    sp_ctx = percent_spinner(prefix="POSE-MP4", stream=sys.stderr) if use_local else _NullCtx()  # type: ignore[operator]

    i = 0
    file_lbl = srcp.name
    with sp_ctx:
        if use_local:
            _poke(sp_ctx, total=float(total + 1), count=0.0, item=file_lbl, job=f"frame 0/{total}", model=model_lbl)  # type: ignore[arg-type]
        else:
            _poke(progress, item=file_lbl, job=f"frame 0/{total}", model=model_lbl)

        while True:
            ok, frame_raw = cap.read()
            if not ok:
                break
            frame: NDArrayU8 = np.asarray(frame_raw, dtype=np.uint8)

            res_list: list[Any] = cast(
                list[Any],
                pose_model.predict(frame, imgsz=640, conf=(conf or 0.25), iou=(iou or 0.45), verbose=False)
            )
            res = res_list[0] if res_list else None
            if res is not None and getattr(res, "keypoints", None) is not None:
                kp_obj = res.keypoints
                try:
                    pts = getattr(kp_obj, "xy", None)
                    if pts is None:
                        pts = getattr(kp_obj, "data", None)
                    if pts is not None and hasattr(pts, "cpu"):
                        try:
                            pts = pts.cpu().numpy()
                        except Exception:
                            pass
                    pts_np: NDArrayF32 = np.asarray(pts if pts is not None else [], dtype=np.float32)
                except Exception:
                    pts_np = np.zeros((0, 0, 2), dtype=np.float32)

                if pts_np.size and pts_np.ndim == 3 and pts_np.shape[2] >= 2:
                    n, k, _ = pts_np.shape
                    color_pt = (0, 255, 255)
                    color_ln = (0, 210, 255)
                    for b in range(n):
                        p = pts_np[b, :, :2]
                        if k == 17:
                            for a, d in _COCO_EDGES:
                                ax, ay = int(p[a, 0]), int(p[a, 1])
                                dx, dy = int(p[d, 0]), int(p[d, 1])
                                cv2.line(frame, (ax, ay), (dx, dy), color_ln, 2)
                        for j in range(k):
                            x, y = int(p[j, 0]), int(p[j, 1])
                            cv2.circle(frame, (x, y), 3, color_pt, -1)

            vw.write(frame)
            i += 1

            if use_local:
                _poke(sp_ctx, count=float(i), item=file_lbl, job=f"frame {min(i, total)}/{total}", model=model_lbl)  # type: ignore[arg-type]
            else:
                _poke(progress, item=file_lbl, job=f"frame {min(i, total)}/{total}", model=model_lbl)

        cap.release()
        vw.release()

        preferred = out_dir / f"{srcp.stem}_pose.mp4"
        if use_local:
            _poke(sp_ctx, item=file_lbl, job="encode mp4", model=model_lbl)  # type: ignore[arg-type]
        else:
            _poke(progress, item=file_lbl, job="encode mp4", model=model_lbl)

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(avi), "-c:v", "libx264", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(preferred)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
            avi.unlink(missing_ok=True)
            final = preferred
        except (FileNotFoundError, subprocess.CalledProcessError):
            ok_mp4 = _opencv_reencode_to_mp4(avi, preferred, fps=fps)
            if ok_mp4:
                avi.unlink(missing_ok=True)
                final = preferred
            else:
                final = out_dir / f"{srcp.stem}_pose.avi"
                shutil.move(str(avi), str(final))

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Saved → {_osc8(final.name, final)}")

        if use_local:
            _poke(sp_ctx, count=float(total + 1), item=file_lbl, job="done", model=model_lbl)  # type: ignore[arg-type]
        else:
            _poke(progress, item=file_lbl, job="done", model=model_lbl)

        return final
