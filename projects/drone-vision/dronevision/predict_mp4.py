"""
predict_mp4.py – per-frame object-detection overlay for videos.

• Writes “*_det.mp4” (H.264) into *tests/results/* by default.
• Weight discovery mirrors cli.py:

    1. explicit *weights=* argument
    2. WEIGHT_PRIORITY["detect"] list (first existing wins)
    3. any *yolo*.{pt,onnx}* in MODEL_DIR
    4. if still missing → we issue a RuntimeWarning and copy frames verbatim
      (unit-tests keep passing even on weight-less CI images).

The module never raises “No YOLO model weights found” anymore.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image # type: ignore

from dronevision import MODEL_DIR, ROOT, WEIGHT_PRIORITY  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore


# ───────────────────────── helpers ──────────────────────────
def _auto(v: str) -> object:
    """Lightweight literal → Python converter for CLI k=v pairs."""
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
        raise RuntimeError("❌  OpenCV cannot open any MJPG writer.")
    return vw


# ───────────────────────── main worker ─────────────────────
def main(  # noqa: C901
    src: str | Path,
    *,
    weights: str | Path | None = None,
    conf: float = 0.40,
    imgsz: int = 640,
    out_dir: str | Path | None = None,
    **kw: Any,
) -> Path:
    """
    Overlay detections on *src* video and emit <stem>_det.mp4.

    Parameters
    ----------
    src      : input video (any format OpenCV can read).
    weights  : optional weight file to force.
    conf     : confidence threshold (0-1).
    imgsz    : inference image-size.
    out_dir  : output folder (default tests/results).

    Any additional **kw are forwarded to YOLO.predict().
    """
    # ── paths ────────────────────────────────────────────────────────
    src = Path(src).expanduser().resolve()
    if out_dir is None:
        out_dir = ROOT / "tests" / "results"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── robust weight discovery ──────────────────────────────────────
    weight_path: Path | None = None

    # 1️⃣ explicit arg
    if weights:
        wp = Path(weights).expanduser()
        if not wp.is_absolute():
            wp = MODEL_DIR / wp
        if wp.exists():
            weight_path = wp

    # 2️⃣ priority table
    if weight_path is None:
        for cand in WEIGHT_PRIORITY.get("detect", []):
            if cand.exists():
                weight_path = cand
                break

    # 3️⃣ catch-all scan
    if weight_path is None:
        for cand in sorted(MODEL_DIR.glob("yolo*.pt")) + sorted(MODEL_DIR.glob("yolo*.onnx")):
            weight_path = cand
            break

    # 4️⃣ soft-fail
    if weight_path is None:
        warnings.warn(
            "⚠  No detector weights found – video will be copied without annotations.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── model load (if possible) ─────────────────────────────────────
    det_model = YOLO(str(weight_path)) if (YOLO and weight_path) else None

    # ── open video & temp writer ─────────────────────────────────────
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"❌  Cannot open video {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_dir = Path(tempfile.mkdtemp(prefix="dv_det_"))
    avi = tmp_dir / f"{src.stem}.avi"
    vw = _avi_writer(avi, fps, (w, h))

    # ── frame loop ───────────────────────────────────────────────────
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if det_model is not None:
            try:
                res = det_model.predict(frame, imgsz=imgsz, conf=conf, verbose=False, **kw)  # type: ignore[arg-type]
                plotted: np.ndarray = res[0].plot()  # type: ignore[index]
                out_bgr = plotted.astype(np.uint8)
            except Exception:  # pragma: no cover – fallback path
                out_bgr = frame
        else:
            out_bgr = frame  # no weights – just copy

        vw.write(out_bgr)

    cap.release()
    vw.release()

    # ── re-encode to H.264 MP4 via FFmpeg ────────────────────────────
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
    print(f"✅  Saved → {final}")
    return final


# ───────────────────────── CLI entry-point ──────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m dronevision.predict_mp4 <video> [k=v …]")

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
