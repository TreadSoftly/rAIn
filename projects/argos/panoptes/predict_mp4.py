# \rAIn\projects\argos\panoptes\predict_mp4.py
"""
predict_mp4.py – per-frame object-detection overlay for videos.

Lock-down (2025-08-07)
────────────────────────────────────────────────────────────────────
* Model selection is **strict**:
      1. an explicit *weights=* argument (override path)
      2. `panoptes.model_registry.load_detector()` – which enforces
         the hard-coded `WEIGHT_PRIORITY["detect"]`.

* If the detector weight is missing (or Ultralytics is not installed) the
  script aborts with a clear error; it no longer copies the video verbatim.

* No environment variables are consulted.

Progress
────────
* If a `progress` handle is provided, we update its `current` label
  (frame i/N → encode → done) and DO NOT create a local spinner.
* Otherwise: live percent spinner (frames processed / total) using our
  local ProgressEngine (suppressed automatically if a parent spinner is active).

Output noise
────────────
* Quiet by default (WARNING). Pass `verbose=true` on the CLI to get INFO logs.
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
from typing import Any, ContextManager, Protocol, cast

import cv2
import numpy as np

from panoptes import ROOT  # type: ignore
from panoptes.model_registry import load_detector  # type: ignore

# optional progress (safe off-TTY)
try:
    from panoptes.progress import ProgressEngine  # type: ignore
    from panoptes.progress.bridges import live_percent  # type: ignore
    from panoptes.progress.progress_ux import simple_status  # type: ignore
except Exception:  # pragma: no cover
    ProgressEngine = None  # type: ignore
    live_percent = None  # type: ignore
    simple_status = None  # type: ignore

# Ultralytics may be absent in some minimal environments
try:
    from ultralytics import YOLO  # type: ignore
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore

# ───────────────────────── logging ──────────────────────────
_LOG = logging.getLogger("panoptes.predict_mp4")
if not _LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
# Quiet by default; flipped to INFO if verbose=True is passed to main()
_LOG.setLevel(logging.WARNING)


def _say(msg: str) -> None:
    _LOG.info(f"[panoptes] {msg}")


# ───────────────────────── helpers ──────────────────────────
class SpinnerLike(Protocol):
    def update(self, **kwargs: Any) -> "SpinnerLike": ...


def _osc8(label: str, path: Path) -> str:
    """
    Render a clickable filename label that links to the absolute file path (OSC-8).
    If the terminal doesn't support OSC-8, the raw label still renders harmlessly.
    """
    try:
        uri = path.resolve().as_uri()
    except Exception:
        uri = "file:///" + str(path.resolve()).replace("\\", "/")
    esc = "\033"
    return f"{esc}]8;;{uri}{esc}\\{label}{esc}]8;;{esc}\\"


def _avi_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    vw = cv2.VideoWriter(
        str(path.with_suffix(".avi")),
        cv2.VideoWriter_fourcc(*"MJPG"),  # type: ignore[attr-defined]
        fps,
        size,
    )
    if not vw.isOpened():
        raise RuntimeError("❌  OpenCV cannot open any MJPG writer on this system.")
    return vw


# ───────────────────────── main worker ─────────────────────
def main(  # noqa: C901 – CLI glue
    src: str | Path,
    *,
    weights: str | Path | None = None,
    conf: float = 0.40,
    imgsz: int = 640,
    out_dir: str | Path | None = None,
    verbose: bool = False,  # NEW: quiet by default, INFO when True
    progress: SpinnerLike | None = None,  # NEW: nudge parent spinner if provided
    **kw: Any,
) -> Path:
    """
    Overlay YOLO detections on *src* video and emit <stem>_det.mp4> (or .avi if FFmpeg unavailable).
    If *progress* is passed, updates its `current` label with "frame i/N", "encode mp4", "done".
    """
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is not installed in this environment.")

    if verbose:
        _LOG.setLevel(logging.INFO)

    # ── normalise paths ──────────────────────────────────────────────
    src = Path(src).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    if out_dir is None:
        out_dir = ROOT / "tests" / "results"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── interpret *weights* argument ─────────────────────────────────
    override_path: Path | None = None
    if weights is not None:
        cand = Path(weights).expanduser().resolve()
        if not cand.exists():
            raise FileNotFoundError(f"override weight not found: {cand}")
        override_path = None if cand.is_dir() else cand

    # ── acquire the detector model (hard-fail if missing) ────────────
    det_model = load_detector(override=override_path)  # registry logs selection

    # ── open video stream & temp writer ──────────────────────────────
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"❌  Cannot open video {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        total_frames = 1
    _say(f"video detect: src={src.name} fps={fps:.3f} size={w}x{h} frames~{total_frames}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="dv_det_"))
    avi = tmp_dir / f"{src.stem}.avi"
    vw = _avi_writer(avi, float(fps), (w, h))

    # progress: prefer parent spinner if provided; else local engine (unless suppressed)
    ps_progress_active = os.environ.get("PANOPTES_PROGRESS_ACTIVE") == "1"
    use_local = (progress is None) and (not ps_progress_active)
    eng = ProgressEngine() if (ProgressEngine is not None and use_local) else None  # type: ignore[truthy-bool]
    if use_local and live_percent is not None and eng is not None:  # type: ignore[truthy-bool]
        ctx: ContextManager[None] = cast(ContextManager[None], live_percent(eng, prefix="DETECT-MP4"))
    else:
        class _Null:
            def __enter__(self) -> None: return None
            def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> bool: return False
        ctx: ContextManager[None] = _Null()

    frame_idx = 0
    with ctx:
        if eng:
            eng.set_total(float(total_frames + 1))  # +1 for encode
            eng.set_current("frame 0")
        elif progress is not None:
            progress.update(current=f"frame 0/{total_frames}")

        # ── frame loop ───────────────────────────────────────────────────
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            try:
                res = det_model.predict(frame, imgsz=imgsz, conf=conf, verbose=False, **kw)  # type: ignore[arg-type]
                plotted = res[0].plot()  # type: ignore[index]
                out_bgr = np.asarray(plotted, dtype=np.uint8)
            except Exception as exc:  # pragma: no cover
                cap.release()
                vw.release()
                raise RuntimeError(f"YOLO inference failed: {exc}") from exc

            vw.write(out_bgr)

            frame_idx += 1
            if eng:
                eng.add(1.0, current_item=f"frame {min(frame_idx, total_frames)}/{total_frames}")
            elif progress is not None:
                progress.update(current=f"frame {min(frame_idx, total_frames)}/{total_frames}")

        cap.release()
        vw.release()

        # ── re-encode MJPG → H.264 MP4 via FFmpeg ────────────────────────
        preferred = out_dir / f"{src.stem}_det.mp4"
        final: Path
        if eng:
            eng.set_current("encode mp4")
        elif progress is not None:
            progress.update(current="encode mp4")
        try:
            if simple_status is not None and not ps_progress_active and use_local:
                sp: ContextManager[None] = cast(ContextManager[None], simple_status("FFmpeg re-encode"))
            else:
                class _Null2:
                    def __enter__(self) -> None: return None
                    def __exit__(self, et: type[BaseException] | None, ex: BaseException | None, tb: TracebackType | None) -> bool: return False
                sp = _Null2()
            with sp:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i", str(avi),
                        "-c:v", "libx264",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        str(preferred),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
            avi.unlink(missing_ok=True)
            final = preferred
        except (FileNotFoundError, subprocess.CalledProcessError):
            # FFmpeg missing or failed – emit AVI instead
            final = out_dir / f"{src.stem}_det.avi"
            shutil.move(str(avi), str(final))

        shutil.rmtree(tmp_dir, ignore_errors=True)

        # One clean summary line with a clickable filename label
        print(f"Saved → {_osc8(final.name, final)}")
        if eng:
            eng.add(1.0, current_item="done")
        elif progress is not None:
            progress.update(current="done")
        return final
