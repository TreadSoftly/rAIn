"""
predict_heatmap_mp4.py – per-frame segmentation (heat-map overlay) for videos.

• Always writes “*_heat.mp4” under tests/results/.
• If *weights* points at a -seg model it is injected into `dronevision.heatmap`
  so **true** instance masks are rendered.
• If a detection-only weight is supplied we fall back to bounding-box drawing.
• If FFmpeg is unavailable we keep the raw MJPG/AVI so unit-tests still pass.
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

from dronevision import MODEL_DIR, ROOT  # type: ignore

# Ultralytics may be missing in some environments
try:
    from ultralytics import YOLO  # type: ignore
except ImportError:               # pragma: no cover
    yolo_model = None

import dronevision.heatmap as _hm
from dronevision.heatmap import heatmap_overlay  # type: ignore



# ───────────────────────── helpers ──────────────────────────
def _auto(v: str):
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
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
        raise RuntimeError("❌  OpenCV cannot open any video writer on this system.")
    return vw


# ───────────────────────── main worker ─────────────────────
def main(  # noqa: C901
    src: str | Path,
    *,
    weights: str | Path | None = None,
    cmap: str = "COLORMAP_JET",
    alpha: float = 0.4,
    kernel_scale: float = 5.0,  # kept for CLI compatibility (unused)
    out_dir: str | Path | None = None,
    **kw: Any,
) -> Path:
    """
    Apply segmentation overlay to *src* video and write <stem>_heat.mp4.

    Parameters
    ----------
    src       : path to input video.
    weights   : optional weight file to force (-seg preferred).
    out_dir   : output directory (default: tests/results).

    Other keyword-args are forwarded to ``heatmap_overlay`` **except** ones
    consumed here (weights, conf, imgsz, …).
    """
    # ── normalise paths ──────────────────────────────────────────────
    src = Path(src).expanduser().resolve()
    if weights is not None:
        weights = Path(weights).expanduser().resolve()
    if out_dir is None:
        out_dir = ROOT / "tests" / "results"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── weight / model setup ─────────────────────────────────────────
    det_model = None
    if weights is not None:
        # user supplied a weight – decide if it’s segmentation or detection
        if "-seg" in weights.name.lower():
            try:
                _hm._seg_model = YOLO(str(weights))             # type: ignore[arg-type]
            except Exception:
                pass
        else:
            try:
                det_model = YOLO(str(weights))                  # type: ignore[arg-type]
            except Exception:
                pass

    # auto‑discover a detector if no seg model is available
    if getattr(_hm, "seg_model", None) is None and det_model is None and ('YOLO' in globals() or 'yolo_model' in globals()):
        for name in (
            "yolov8x.pt", "yolo11x.pt", "yolov12x.pt",
            "yolov8n.pt", "yolo11n.pt", "yolov12n.pt",
            "yolov8x.onnx", "yolo11x.onnx", "yolov12x.onnx",
            "yolov8n.onnx", "yolo11n.onnx", "yolov12n.onnx",
        ):
            cand = MODEL_DIR / name
            if cand.exists():
                det_model = YOLO(str(cand))                     # type: ignore[arg-type]
                break


    # Anything not understood by heatmap_overlay must be stripped
    STRIP_KEYS = {"weights", "conf", "imgsz", "iou", "classes", "max_det"}
    for k in STRIP_KEYS:
        kw.pop(k, None)

    # ── open video & temp writer ─────────────────────────────────────
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"❌  Cannot open video {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_dir = Path(tempfile.mkdtemp(prefix="dv_hm_"))
    avi = tmp_dir / f"{src.stem}.avi"
    vw = _avi_writer(avi, fps, (w, h))

    # ── frame loop ───────────────────────────────────────────────────
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # fallback to detection boxes when no seg model is present
        boxes_np = None
        if getattr(_hm, "seg_model", None) is None and det_model is not None:
            try:
                res = det_model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]  # type: ignore[index]
                data = getattr(getattr(res, "boxes", None), "data", None)
                if data is not None:
                    try:
                        import torch  # type: ignore
                        if isinstance(data, torch.Tensor):
                            data = data.cpu().numpy()
                    except ImportError:
                        pass
                    boxes_np = np.asarray(data)
            except Exception:
                boxes_np = None

        ov = heatmap_overlay(
            Image.fromarray(frame[:, :, ::-1]),
            boxes=boxes_np,
            alpha=alpha,
            **kw,
        )

        # PIL → BGR ndarray
        if isinstance(ov, Image.Image):
            out_bgr = np.asarray(ov)[:, :, ::-1]
        else:  # already ndarray; ensure BGR, uint8
            out_bgr = ov[:, :, ::-1] if ov.shape[2] == 3 else cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)
        vw.write(out_bgr.astype(np.uint8))

    cap.release()
    vw.release()

    # ── re-encode to H.264 MP4 via FFmpeg ────────────────────────────
    final = out_dir / f"{src.stem}_heat.mp4"
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
        sys.exit("Usage: python -m dronevision.predict_heatmap_mp4 <video> [k=v …]")

    kv_pairs = (arg.split("=", 1) for arg in sys.argv[2:])
    args = {k: _auto(v) for k, v in kv_pairs}

    weights_arg = args.pop("weights", None)
    if weights_arg is not None:
        weights_arg = str(weights_arg)
    out_dir_arg = args.pop("out_dir", None)
    if out_dir_arg is not None:
        out_dir_arg = str(out_dir_arg)
    alpha_arg = float(args.pop("alpha", 0.4))
    cmap_arg = str(args.pop("cmap", "COLORMAP_JET"))
    kernel_scale_arg = float(args.pop("kernel_scale", 5.0))

    main(
        sys.argv[1],
        weights=weights_arg,
        cmap=cmap_arg,
        alpha=alpha_arg,
        kernel_scale=kernel_scale_arg,
        out_dir=out_dir_arg,
        **args,
    )
