# C:\Users\MrDra\OneDrive\Desktop\rAIn\projects\argos\panoptes\model\_fetch_models.py
from __future__ import annotations

import glob
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

# --- Ultralytics (quiet logging, version-agnostic) ---
has_yolo: bool = False
try:
    from ultralytics import YOLO  # type: ignore[reportMissingTypeStubs]
    try:
        from ultralytics.utils import LOGGER as _ULTRA_LOGGER  # type: ignore
        _rem = getattr(_ULTRA_LOGGER, "remove", None)
        if callable(_rem):
            _rem()  # silence Ultralytics banner/logging
    except Exception:
        pass
    has_yolo = True
except Exception:
    YOLO = None  # type: ignore[assignment]
    has_yolo = False

MODEL_DIR = Path(__file__).resolve().parent  # panoptes/model


@contextmanager
def _cd(path: Path):
    prev = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev)


def _latest_exported_onnx() -> Optional[Path]:
    hits = [Path(p) for p in glob.glob("runs/**/**/*.onnx", recursive=True)]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def _synonyms(name: str) -> List[str]:
    """
    Try Ultralytics spelling variants:
      yolo11 ↔ yolov11, yolo12 ↔ yolov12 (suffixes like -seg/-pose kept).
    """
    base = Path(name).name
    alts = {base}
    for fam in ("11", "12"):
        alts.add(base.replace(f"yolov{fam}", f"yolo{fam}"))
        alts.add(base.replace(f"yolo{fam}", f"yolov{fam}"))
    return [a for a in alts if a]


def _fetch_one(name: str, dst: Path) -> Tuple[str, str]:
    """
    Obtain *name* into *dst* and return (basename, action).
    Actions: present | download | copied | exported | failed
    """
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / Path(name).name

    # Already present
    if target.exists():
        return (target.name, "present")

    if not has_yolo or YOLO is None:
        return (target.name, "failed")

    # 1) Try to fetch directly by name and synonyms (handles yolo12/yolov12)
    for nm in _synonyms(name):
        try:
            with _cd(dst):
                m = YOLO(nm)  # may download into CWD
                p = Path(getattr(m, "ckpt_path", nm)).expanduser()
                if not p.exists():
                    p = Path(nm).expanduser()
                if p.exists():
                    if p.resolve() == target.resolve():
                        return (target.name, "download")
                    shutil.copy2(p, target)
                    return (target.name, "copied")
        except Exception:
            pass  # try next strategy

    # 2) If ONNX requested, export from matching .pt (also try synonyms)
    if name.endswith(".onnx"):
        canonical_pt = name[:-5] + ".pt"
        for pt_nm in _synonyms(canonical_pt):
            try:
                with _cd(dst):
                    # ensure a .pt exists in dst (download if needed)
                    m_pt = YOLO(pt_nm)
                    p_pt = Path(getattr(m_pt, "ckpt_path", pt_nm)).expanduser()
                    if not p_pt.exists():
                        p_pt = Path(pt_nm).expanduser()

                    # normalize filename -> canonical_pt inside dst
                    canon_path = dst / Path(canonical_pt).name
                    if p_pt.exists() and p_pt.resolve() != canon_path.resolve():
                        shutil.copy2(p_pt, canon_path)

                    # export; sometimes Ultralytics writes to CWD (not runs/)
                    try:
                        YOLO(str(canon_path)).export(
                            format="onnx", dynamic=True, simplify=True, imgsz=640, opset=12, device="cpu"
                        )
                    except Exception:
                        YOLO(str(canon_path)).export(
                            format="onnx", dynamic=True, simplify=False, imgsz=640, opset=12, device="cpu"
                        )

                    # success path (a): file saved as target in CWD
                    if target.exists():
                        return (target.name, "exported")

                    # success path (b): file under runs/
                    cand = _latest_exported_onnx()
                    if cand and cand.exists():
                        shutil.copy2(cand, target)
                        try:
                            shutil.rmtree(dst / "runs", ignore_errors=True)
                        except Exception:
                            pass
                        if target.exists():
                            return (target.name, "exported")
            except Exception:
                continue

    return (target.name, "failed")


def main(argv: List[str]) -> int:
    if not argv:
        print("usage: _fetch_models.py <name.pt|name.onnx|hub-name> [...]")
        return 2

    results: List[Tuple[str, str]] = []
    for a in argv:
        base, action = _fetch_one(a, MODEL_DIR)
        results.append((base, action))

    icons = {
        "present": "↺",
        "download": "↓",
        "copied": "⇢",
        "exported": "⎘",
        "failed": "✗",
    }
    for base, action in results:
        if action == "present":
            print(f"{icons[action]} {base}  (already present)")
        elif action == "download":
            print(f"{icons[action]} {base}  (downloaded)")
        elif action == "copied":
            print(f"{icons[action]} {base}  (downloaded → copied into model dir)")
        elif action == "exported":
            print(f"{icons[action]} {base}  (exported from matching .pt)")
        else:
            print(f"{icons[action]} {base}  (failed)")

    failed = [b for b, a in results if a == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
