"""
run_all_tests.py – bulk-annotate everything in tests/raw/

• Videos  → calls predict_mp4 (the H.264 version you just fixed)
• Images  → saves <stem>.jpg beside tests/results/
• No runs/, predict/ junk is left anywhere.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLO # type: ignore[import-untyped]

# ─────────────────────────────────────────────────────────────
# ❶  Paths & global config
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]          # ← projects/drone-vision
RAW  = ROOT / "tests" / "raw"
RES  = ROOT / "tests" / "results"
RES.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS = ROOT / "yolov8n.pt"                 # adjust if you use custom weights
YOLO_KW: dict[str, float | int] = dict(conf=0.3, imgsz=640)

# ─────────────────────────────────────────────────────────────
# ❷  Helpers
# ─────────────────────────────────────────────────────────────
def _is_video(p: Path) -> bool:
    return p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}


def _annotate_image(img_path: Path) -> None:
    model = YOLO(str(YOLO_WEIGHTS))
    result = model.predict(str(img_path), **YOLO_KW, save=False)[0]
    out_path = RES / img_path.with_suffix(".jpg").name
    cv2.imwrite(str(out_path), result.plot())


def _annotate_video(vid_path: Path) -> None:
    """Call predict_mp4 as a normal module:  python -m dronevision.predict_mp4 …"""
    cmd = [
        sys.executable,
        "-m",
        "dronevision.predict_mp4",
        str(vid_path),
        f"save_dir={RES}",
        *(f"{k}={v}" for k, v in YOLO_KW.items()),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "predict_mp4 failed")


# ─────────────────────────────────────────────────────────────
# ❸  Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    if not RAW.exists():
        sys.exit(f"❌  RAW folder not found: {RAW}")

    files: List[Path] = sorted(p for p in RAW.iterdir() if p.is_file())
    if not files:
        sys.exit(f"❌  No files in {RAW}")

    print(f"→ Annotating {len(files)} file(s)…\n")

    for f in files:
        print(f" • {f.name}", end="  ")
        try:
            _annotate_video(f) if _is_video(f) else _annotate_image(f)
            print("✅")
        except Exception as e:
            print(f"❌  {e}")

    # clean up any detect/ folder Ultralytics may have left for images
    detect_dir = RAW / "detect"
    if detect_dir.exists():
        shutil.rmtree(detect_dir, ignore_errors=True)

    print(f"\nAll done → outputs in {RES}")


if __name__ == "__main__":
    main()
