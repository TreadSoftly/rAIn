"""
run_all_tests.py – bulk-annotate the sample corpus in *tests/raw/*

This version **removes the hard dependency** on the missing
``dronevision.predict_heatmap`` and ``dronevision.predict_geojson`` modules.
Instead, it simply calls the **target** CLI (already exercised by your unit
tests) to generate heat‑maps and GeoJSON, so *no extra files are required*.

Output files and folder layout are unchanged – everything still lands under
*tests/results/* with exactly the same names as before.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLO  # type: ignore
from ultralytics.engine.results import Results  # type: ignore

# ──────────────────────────────────────────────────────────────
# ❶  Paths & global config
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]  # …/projects/drone-vision
RAW = ROOT / "tests" / "raw"
RES = ROOT / "tests" / "results"
RES.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS = ROOT / "model" / "yolov8x.pt"  # primary detector
YOLO_KW: dict[str, object] = dict(conf=0.3, imgsz=640)

_VIDEO_SKIP = {"bunny.mp4", "city5s.mp4"}  # videos that would blow run‑time

# ──────────────────────────────────────────────────────────────
# ❷  Low‑level helpers
# ──────────────────────────────────────────────────────────────
def _is_video(p: Path) -> bool:
    return p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}


def _annotate_image(img_path: Path) -> None:
    """Save a JPEG with YOLOv8 rectangles under tests/results/."""
    model = YOLO(str(YOLO_WEIGHTS))
    results: List[Results] = model.predict(  # type: ignore
        str(img_path), conf=YOLO_KW["conf"], imgsz=YOLO_KW["imgsz"], save=False
    )
    out_path = RES / img_path.with_suffix(".jpg").name
    cv2.imwrite(str(out_path), results[0].plot())  # type: ignore


def _annotate_video(vid_path: Path) -> None:
    """Delegate to the existing predict_mp4 helper (rectangle overlay)."""
    cmd = [
        sys.executable,
        "-m",
        "dronevision.predict_mp4",
        str(vid_path),
        f"out_dir={RES}",
        *(f"{k}={v}" for k, v in YOLO_KW.items()),
    ]
    subprocess.check_call(cmd)


# ──────────────────────────────────────────────────────────────
# ❸  Single‑frame helpers via the *target* CLI
# ──────────────────────────────────────────────────────────────
def _run_cli(img_path: Path, *, task: str, model: str = "airplane") -> None:
    """
    Call the fully‑featured drone‑vision CLI (entry‑point ``target``) so we
    don't have to maintain shadow wrapper modules for each task.
    """
    cmd = [
        "target",
        str(img_path),
        "--task",
        task,
        "--model",
        model,
        "--conf",
        str(YOLO_KW["conf"]),
    ]
    if task == "heatmap":
        cmd += ["--alpha", "0.4"]
    subprocess.check_call(cmd)


# ──────────────────────────────────────────────────────────────
# ❹  Main driver
# ──────────────────────────────────────────────────────────────
def main() -> None:
    if not RAW.exists():
        sys.exit(f"❌  RAW folder not found: {RAW}")

    items = sorted(p for p in RAW.iterdir() if p.is_file())
    print(f"→ Annotating {len(items)} file(s)…\n")

    for src in items:
        if _is_video(src):
            if src.name in _VIDEO_SKIP:
                print(f" • {src.name}  ⚠ skipped (not whitelisted)")
                continue
            print(f" • {src.name}", end="")
            try:
                _annotate_video(src)
                print("")
            except Exception as exc:  # pragma: no cover
                print(f"\n    video error: {exc}")
        else:
            print(f"  {src.name}", end="")
            errs: list[str] = []

            try:
                _annotate_image(src)
            except Exception as exc:  # pragma: no cover
                errs.append(f"detect error: {exc}")

            for tsk in ("heatmap", "geojson"):
                try:
                    _run_cli(src, task=tsk)
                except Exception as exc:  # pragma: no cover
                    errs.append(f"{tsk} error: {exc}")

            if errs:
                print("\n    " + "\n    ".join(errs))
            else:
                print("")

    # Ultralytics tends to leave a stray detect/ folder – clean it.
    detect_dir = RAW / "detect"
    if detect_dir.exists():
        shutil.rmtree(detect_dir, ignore_errors=True)

    print(f"\nAll done → outputs in {RES}")


if __name__ == "__main__":  # pragma: no cover
    main()
