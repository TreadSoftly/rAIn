# C:\Users\MrDra\OneDrive\Desktop\rAIn\projects\argos\panoptes\predict_mp4.py
"""
predict_mp4.py – per-frame object-detection overlay for videos.

Lock-down (2025-08-07)
────────────────────────────────────────────────────────────────────
* Model selection is **strict**:
      1. an explicit *weights=* argument (override path)
      2. `panoptes.model_registry.load_detector()`
        – which itself enforces the hard-coded `WEIGHT_PRIORITY["detect"]`.

* If the detector weight is missing (or Ultralytics is not installed) the
  script aborts with a clear error; it no longer copies the video verbatim.

* No environment variables are consulted.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from panoptes import ROOT # type: ignore
from panoptes.model_registry import load_detector # type: ignore

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
_LOG.setLevel(logging.INFO)
def _say(msg: str) -> None:
    _LOG.info(f"[panoptes] {msg}")

# ───────────────────────── helpers ──────────────────────────
def _auto(v: str) -> object:
    """Tiny literal-to-Python converter for CLI k=v pairs."""
    lv = v.lower()
    if lv in {"true", "false"}:
        return lv == "true"
    try:
        return int(v) if v.isdigit() else float(v)
    except ValueError:
        return v

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
    **kw: Any,
) -> Path:
    """
    Overlay YOLO detections on *src* video and emit <stem>_det.mp4>.
    """
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is not installed in this environment.")

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
    _say(f"video detect: src={src.name} fps={fps:.3f} size={w}x{h}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="dv_det_"))
    avi = tmp_dir / f"{src.stem}.avi"
    vw = _avi_writer(avi, fps, (w, h))

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

    cap.release()
    vw.release()

    # ── re-encode MJPG → H.264 MP4 via FFmpeg ────────────────────────
    final = out_dir / f"{src.stem}_det.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(avi),
                "-c:v",
                "libx264",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(final),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        avi.unlink(missing_ok=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        shutil.move(str(avi), str(final))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    _say(f"Saved → {final}")
    return final

# ───────────────────────── CLI entry-point ──────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m panoptes.predict_mp4 <video> [k=v …]")

    kv_pairs = (arg.split("=", 1) for arg in sys.argv[2:])
    args = {k: _auto(v) for k, v in kv_pairs}

    weights_arg = args.pop("weights", None)
    if weights_arg is not None:
        weights_arg = str(weights_arg)

    out_dir_arg = args.pop("out_dir", None)
    if out_dir_arg is not None:
        out_dir_arg = str(out_dir_arg)

    conf_arg = float(str(args.pop("conf", 0.40)))
    imgsz_arg = int(str(args.pop("imgsz", 640)))

    main(
        sys.argv[1],
        weights=weights_arg,
        conf=conf_arg,
        imgsz=imgsz_arg,
        out_dir=out_dir_arg,
        **args,
    )
