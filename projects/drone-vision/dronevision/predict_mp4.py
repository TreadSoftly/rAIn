"""
predict_mp4.py – YOLO helper (library **and** CLI)

• Annotates **one** source video (MP4 / MOV / …).
• Writes a H.264 MP4 into *tests/results/* (or a custom *save_dir*).
• Leaves no runs-/detect-/predict- folders behind.

The module is *safe to import*: all work happens inside ``main()`` or under a
``if __name__ == "__main__"`` guard, so other code can simply

    from dronevision.predict_mp4 import main as predict_video

without triggering any argument parsing.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple
from typing import Any

import cv2
from ultralytics import YOLO # type: ignore[import-untyped]

# ─────────────────────────────────────────────────────────────
# ❶  Helpers
# ─────────────────────────────────────────────────────────────
def _auto(v: str):  # tiny literal-to-type helper
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return int(v) if v.isdigit() else float(v)
    except ValueError:
        return v


def _open_avi_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """Guaranteed-to-work MJPG/AVI writer (every OpenCV build supports this)."""
    vw = cv2.VideoWriter(
        str(path.with_suffix(".avi")),
        cv2.VideoWriter_fourcc(*"MJPG"),  # type: ignore[attr-defined]
        fps,
        size,
    )
    if not vw.isOpened():
        raise RuntimeError("❌  OpenCV cannot open any video writer on this system.")
    return vw


# ─────────────────────────────────────────────────────────────
# ❷  Main worker (can be imported)
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # …/projects/drone-vision
DEFAULT_SAVE_DIR = PROJECT_ROOT / "tests" / "results"


def main(src: str | Path, **overrides: Any) -> Path:
    """
    Parameters
    ----------
    src        : str | Path
        Path to *one* video file.
    **overrides
        Any ``ultralytics.YOLO.predict`` keyword plus an optional
        *save_dir* (folder to write the final MP4).

    Returns
    -------
    Path
        The path of the MP4 that was written.
    """
    src_path = Path(src).expanduser()

    save_dir_val = overrides.pop("save_dir", DEFAULT_SAVE_DIR)
    save_dir = Path(save_dir_val) if isinstance(save_dir_val, (str, Path)) else Path(str(save_dir_val))
    save_dir = save_dir.expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Model & capture ────────────────────────────────────────────
    model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))  # swap to custom weights if needed

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"❌  Could not open source video: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_dir = Path(tempfile.mkdtemp())
    avi_path = tmp_dir / f"{src_path.stem}.avi"
    writer = _open_avi_writer(avi_path, fps, (w, h))

    # ── Predict frame-by-frame ────────────────────────────────────
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict( # type: ignore[call-arg]
            frame,
            imgsz=overrides.get("imgsz", 640),
            **{k: v for k, v in overrides.items() if k not in ("imgsz", "stream")},
        )[0]
        img = result.plot()  # type: ignore
        writer.write(img)    # type: ignore

    cap.release()
    writer.release()

    # ── ffmpeg → proper MP4 ───────────────────────────────────────
    final_mp4 = save_dir / f"{src_path.stem}.mp4"
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(avi_path),
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(final_mp4),
    ]
    try:
        subprocess.run(
            ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        avi_path.unlink(missing_ok=True)
    except FileNotFoundError:
        # No ffmpeg – keep the AVI so at least something is produced.
        shutil.move(str(avi_path), final_mp4.with_suffix(".avi"))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"✅  Saved → {final_mp4}")
    return final_mp4


# ─────────────────────────────────────────────────────────────
# ❸  CLI entry-point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m dronevision.predict_mp4 <source> [key=value …]")

    kv_pairs = (a.split("=", 1) for a in sys.argv[2:])
    cli_overrides = {k: _auto(v) for k, v in kv_pairs}
    main(sys.argv[1], **cli_overrides)
