# C:\Users\MrDra\OneDrive\Desktop\rAIn\projects\argos\panoptes\predict_heatmap_mp4.py
"""
predict_heatmap_mp4.py – per-frame segmentation / heat-map overlay for videos.

Lock-down (2025-08-07)
────────────────────────────────────────────────────────────────────
* A **segmentation** weight is mandatory – either:
      1. an explicit *weights=* argument whose filename contains “-seg”
      2. the hard-coded  WEIGHT_PRIORITY["heatmap"]  entry.

* No detector fall-back: if a segmentation model cannot be loaded the
  script aborts immediately.

* Environment variables are ignored; weight control lives exclusively
  in `panoptes.model_registry`.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
from PIL import Image

import panoptes.heatmap as _hm # type: ignore
from panoptes import ROOT # type: ignore
from panoptes.heatmap import heatmap_overlay # type: ignore
from panoptes.model_registry import load_segmenter # type: ignore

# Ultralytics may be absent in some minimal environments
try:
    from ultralytics import YOLO  # type: ignore
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore

# ───────────────────────── logging ──────────────────────────
_LOG = logging.getLogger("panoptes.predict_heatmap_mp4")
if not _LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
_LOG.setLevel(logging.INFO)
def _say(msg: str) -> None:
    _LOG.info(f"[panoptes] {msg}")

# ───────────────────────── helpers ──────────────────────────
def _auto(v: str) -> object:
    """Literal-string → Python tiny converter for CLI k=v pairs."""
    lv = v.lower()
    if lv in {"true", "false"}:
        return lv == "true"
    try:
        return int(v) if v.isdigit() else float(v)
    except ValueError:
        return v

def _avi_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
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
def main(  # noqa: C901 – unavoidable CLI glue
    src: str | Path,
    *,
    weights: str | Path | None = None,
    cmap: str = "COLORMAP_JET",
    alpha: float = 0.4,
    kernel_scale: float = 5.0,  # unused, kept for flag-compat
    out_dir: str | Path | None = None,
    **kw: Any,
) -> Path:
    """
    Overlay segmentation heat-map on *src* video and emit <stem>_heat.mp4>.
    """
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is not installed in this environment.")

    # ── path normalisation ────────────────────────────────────────────
    src_path = Path(src).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    if out_dir is None:
        out_dir = ROOT / "tests" / "results"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load segmentation model (strict) ─────────────────────────────
    override_path: Path | None = None
    if weights:
        override_path = Path(weights).expanduser().resolve()
        if not override_path.exists():
            raise FileNotFoundError(f"override weight not found: {override_path}")
    seg_model = load_segmenter(override=override_path)  # registry logs weight

    # propagate into heatmap module so subsequent calls are instant
    _hm._seg_model = seg_model  # type: ignore[attr-defined]

    # strip YOLO-specific keys that users might pass by copy-paste
    STRIP = {"weights", "conf", "imgsz", "iou", "classes", "max_det"}
    for k in STRIP:
        kw.pop(k, None)

    # ── open video stream & temp MJPG writer ─────────────────────────
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"❌  Cannot open video {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _say(f"video heatmap: src={src_path.name} fps={fps:.3f} size={w}x{h}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="dv_hm_"))
    avi = tmp_dir / f"{src_path.stem}.avi"
    vw = _avi_writer(avi, fps, (w, h))

    # ── frame loop ───────────────────────────────────────────────────
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ov = heatmap_overlay(
            Image.fromarray(frame[:, :, ::-1]),
            boxes=None,  # true seg masks only
            alpha=alpha,
            cmap=cmap,
            kernel_scale=kernel_scale,
            **kw,
        )

        if isinstance(ov, Image.Image):
            out_bgr = np.asarray(ov)[:, :, ::-1]
        else:  # ndarray returned (already BGR)
            out_bgr = ov
        vw.write(np.asarray(out_bgr, dtype=np.uint8))

    cap.release()
    vw.release()

    # ── re-encode MJPG → H.264 MP4 via FFmpeg ─────────────────────────
    final = out_dir / f"{src_path.stem}_heat.mp4"
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
        sys.exit("Usage: python -m panoptes.predict_heatmap_mp4 <video> [k=v …]")

    kv_pairs = (arg.split("=", 1) for arg in sys.argv[2:])
    args = {k: _auto(v) for k, v in kv_pairs}

    weights_arg = args.pop("weights", None)
    if weights_arg is not None and not isinstance(weights_arg, (str, Path)):
        weights_arg = str(weights_arg)

    main(
        sys.argv[1],
        weights=weights_arg,
        cmap=str(args.pop("cmap", "COLORMAP_JET")),
        alpha=float(str(args.pop("alpha", 0.4))),
        kernel_scale=float(str(args.pop("kernel_scale", 5.0))),
        out_dir=str(args.pop("out_dir", None)) if args.get("out_dir", None) is not None else None,
        **args,
    )
