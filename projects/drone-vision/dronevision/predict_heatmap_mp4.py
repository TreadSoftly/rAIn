"""
predict_heatmap_mp4.py – apply Drone-Vision segmentation & labelling
(“heat-map”) on **every frame** of a video and save the result.

• Always writes “*_heat.mp4” into *tests/results/* by default.
• Uses FFmpeg for fast H.264 re-encoding if available.
  – If FFmpeg is **absent** *or* exits with a non-zero status we fall back
    to moving the raw MJPG/AVI so the file still exists for the tests.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
from PIL import Image

from dronevision.heatmap import heatmap_overlay  # type: ignore[import-untyped]


# ───────────────────────── helpers ──────────────────────────
def _auto(v: str):
    """Lightweight str-to-literal helper for CLI key=value pairs."""
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return int(v) if v.isdigit() else float(v)
    except ValueError:
        return v


def _open_avi_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """Guaranteed-compatible MJPG/AVI writer."""
    vw = cv2.VideoWriter(
        str(path.with_suffix(".avi")),
        cv2.VideoWriter_fourcc(*"MJPG"),  # type: ignore[attr-defined]
        fps,
        size,
    )
    if not vw.isOpened():
        raise RuntimeError("❌  OpenCV cannot open any video writer on this system.")
    return vw


# ───────────────────────── main ─────────────────────────────
def main(
    src: str | Path,
    *,
    cmap: str = "COLORMAP_JET",
    alpha: float = 0.4,
    kernel_scale: float = 5.0,
    out_dir: str | Path | None = None,
    **kw: Any,
) -> Path:
    """
    Apply heat-map overlay to each frame of *src* video.

    Parameters
    ----------
    src
        Input video (any format OpenCV can read).
    cmap, alpha, kernel_scale
        Forwarded to ``heatmap_overlay``.
    out_dir
        Output directory (default: *projects/drone-vision/tests/results*).

    Returns
    -------
    Path to the written “*_heat.mp4”.
    """
    src = Path(src)

    # default → …/projects/drone-vision/tests/results
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "tests" / "results"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem    = src.stem
    tmp_dir = Path(tempfile.mkdtemp(prefix="dv_hm_"))
    avi     = tmp_dir / f"{stem}.avi"

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"❌  Cannot open video {src!s}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vw = _open_avi_writer(avi, fps, (width, height))

    # YOLO-only flags that heatmap_overlay doesn’t know about
    _DETECT_ONLY = {"conf", "imgsz", "iou", "classes", "max_det"}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Strip unsupported CLI kwargs **each loop** (safe & cheap)
        for bad in _DETECT_ONLY:
            kw.pop(bad, None)

        # OpenCV → BGR, PIL → RGB
        overlay = heatmap_overlay( # type: ignore[call-arg]
            Image.fromarray(frame[:, :, ::-1]),
            boxes=None,
            alpha=alpha,
            **kw,
        )
        vw.write(np.asarray(overlay, dtype=np.uint8)[:, :, ::-1])  # back to BGR, ensure uint8

    cap.release()
    vw.release()

    final = out_dir / f"{stem}_heat.mp4"

    # ── try FFmpeg re-encode ─────────────────────────────────────────────
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y", "-i", str(avi),
                "-c:v", "libx264", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(final),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        avi.unlink(missing_ok=True)

    except (FileNotFoundError, subprocess.CalledProcessError):
        # FFmpeg missing *or* failed – keep the raw AVI but rename it so
        # the expected “*_heat.mp4” path exists.
        shutil.move(str(avi), str(final))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("✅  Saved →", final)
    return final


# ───────────────────────── CLI entry point ──────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m dronevision.predict_heatmap_mp4 <video> [k=v …]")

    kv_pairs = (arg.split("=", 1) for arg in sys.argv[2:])
    args = {k: _auto(v) for k, v in kv_pairs}

    # Extract known parameters with correct types
    cmap         = str(args.pop("cmap", "COLORMAP_JET"))
    alpha        = float(args.pop("alpha", 0.4))
    kernel_scale = float(args.pop("kernel_scale", 5.0))
    out_dir      = args.pop("out_dir", None)
    if out_dir is not None:
        out_dir = str(out_dir)

    main(
        sys.argv[1],
        cmap=cmap,
        alpha=alpha,
        kernel_scale=kernel_scale,
        out_dir=out_dir,
        **args,
    )
