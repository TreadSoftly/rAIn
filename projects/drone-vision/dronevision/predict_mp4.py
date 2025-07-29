"""
predict_mp4.py – YOLO helper that

• takes ONE source video (MP4 / MOV / …)
• writes ONE annotated **real** MP4 into `save_dir`  (default: tests/results)
• leaves no runs‑/detect‑/predict‑ folders behind

Example
-------

python -m dronevision.predict_mp4 `
       projects/drone-vision/tests/raw/bunny.mp4 `
       save_dir=projects/drone-vision/tests/results `
       conf=0.3 imgsz=640
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
from ultralytics import YOLO


# ────────────────────────────────────────────────────────────
# ❶  Helpers
# ────────────────────────────────────────────────────────────
def _auto(v: str):                # tiny literal‑to‑type helper
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return int(v) if v.isdigit() else float(v)
    except ValueError:
        return v


def _open_avi_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """Guaranteed‑to‑work MJPG/AVI writer (every OpenCV build supports this)."""
    vw = cv2.VideoWriter(
        str(path.with_suffix(".avi")),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        size,
    )
    if not vw.isOpened():
        raise RuntimeError("❌  OpenCV cannot open any video writer on this system.")
    return vw


# ────────────────────────────────────────────────────────────
# ❷  Parse CLI
# ────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    sys.exit("Usage: python -m dronevision.predict_mp4 <source> [key=value …]")

src = Path(sys.argv[1]).expanduser()
overrides = {k: _auto(v) for k, v in (a.split("=", 1) for a in sys.argv[2:])}
save_dir = Path(overrides.pop("save_dir", "projects/drone-vision/tests/results")).expanduser()
save_dir.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────
# ❸  Model + capture
# ────────────────────────────────────────────────────────────
model = YOLO("projects/drone-vision/yolov8n.pt")      # swap to custom weights if needed

cap = cv2.VideoCapture(str(src))
if not cap.isOpened():
    sys.exit(f"❌  Could not open source video: {src}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

tmp_dir = Path(tempfile.mkdtemp())
avi_path = tmp_dir / f"{src.stem}.avi"
writer = _open_avi_writer(avi_path, fps, (w, h))

# ────────────────────────────────────────────────────────────
# ❹  Predict frame‑by‑frame
# ────────────────────────────────────────────────────────────
while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = model.predict(
        frame,
        imgsz=overrides.get("imgsz", 640),
        **{k: v for k, v in overrides.items() if k != "imgsz"},
    )[0]
    writer.write(result.plot())

cap.release()
writer.release()

# ────────────────────────────────────────────────────────────
# ❺  ffmpeg → real MP4 (H.264 + faststart)
# ────────────────────────────────────────────────────────────
final_mp4 = save_dir / f"{src.stem}.mp4"

ffmpeg_cmd = [
    "ffmpeg", "-y", "-i", str(avi_path),
    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    str(final_mp4)
]
try:
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
except FileNotFoundError:
    shutil.move(str(avi_path), save_dir / avi_path.name)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    sys.exit("❌  ffmpeg not found on PATH – copied AVI instead (VS Code preview cannot play it).")

# clean up
avi_path.unlink(missing_ok=True)
shutil.rmtree(tmp_dir, ignore_errors=True)

print(f"✅  Saved → {final_mp4}")
