# \rAIn\projects\argos\panoptes\model\_export_onnx.py
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
            _rem()
    except Exception:
        pass
    has_yolo = True
except Exception:
    YOLO = None  # type: ignore[assignment]
    has_yolo = False

MODEL_DIR = Path(__file__).resolve().parent  # panoptes/model


@contextmanager
def _cd(p: Path):
    prev = Path.cwd()
    try:
        os.chdir(p)
        yield
    finally:
        os.chdir(prev)


def _latest_exported_onnx() -> Optional[Path]:
    hits = [Path(p) for p in glob.glob("runs/**/**/*.onnx", recursive=True)]
    return max(hits, key=lambda pp: pp.stat().st_mtime) if hits else None


def _synonyms(name: str) -> List[str]:
    """Try YOLO’s common spelling variants (yolo12 ↔ yolov12; etc.)."""
    base = Path(name).name
    alts = {base}
    for ver in ("11", "12"):
        alts.add(base.replace(f"yolov{ver}", f"yolo{ver}"))
        alts.add(base.replace(f"yolo{ver}",  f"yolov{ver}"))
    return [a for a in alts if a]


def _export_one(arg: str) -> Tuple[Path, str]:
    """
    Export ONNX for *arg*.
    Returns (final_path, action) where action ∈ {present|exported|failed}.
    """
    if not has_yolo or YOLO is None:
        return (MODEL_DIR / Path(arg).with_suffix(".onnx").name, "failed")

    p = Path(arg)
    # Decide target directory + canonical ONNX name
    if p.suffix.lower() == ".pt":
        dst = (p.parent if p.parent != Path() else MODEL_DIR)
        onnx_name = p.with_suffix(".onnx").name
        pt_name = p.name
    elif p.suffix.lower() == ".onnx":
        dst = (p.parent if p.parent != Path() else MODEL_DIR)
        onnx_name = p.name
        pt_name = Path(onnx_name).with_suffix(".pt").name
    else:
        # bare name → export .onnx into MODEL_DIR
        dst = MODEL_DIR
        root = p.name
        if not root.endswith((".pt", ".onnx")):
            root += ".onnx"
        onnx_name = Path(root).with_suffix(".onnx").name
        pt_name = Path(root).with_suffix(".pt").name

    dst.mkdir(parents=True, exist_ok=True)
    target = dst / onnx_name
    if target.exists():
        return (target, "present")

    # Work inside *dst* so Ultralytics writes runs/ there (not repo root)
    with _cd(dst):
        # Ensure we have the .pt inside dst (copy if given as a real file path)
        src_pt_path: Optional[Path] = None
        if (p.suffix.lower() == ".pt") and p.exists():
            src_pt_path = p.resolve()
        # Try the provided spelling + synonyms for hub fetch
        tried_names: List[str] = [pt_name, *{nm for nm in _synonyms(pt_name) if nm != pt_name}]
        for nm in tried_names:
            try:
                m = YOLO(str(src_pt_path or nm))  # may download into CWD=dst
                # normalise local path of the .pt
                ck = Path(getattr(m, "ckpt_path", nm)).expanduser()
                if not ck.exists():
                    ck = Path(nm).expanduser()
                canon_pt = dst / pt_name  # canonical local filename
                if ck.exists() and ck.resolve() != canon_pt.resolve():
                    shutil.copy2(ck, canon_pt)
                # Export ONNX (with safe fallback)
                try:
                    YOLO(str(canon_pt)).export(
                        format="onnx", dynamic=True, simplify=True, imgsz=640, opset=12, device="cpu"
                    )
                except Exception:
                    YOLO(str(canon_pt)).export(
                        format="onnx", dynamic=True, simplify=False, imgsz=640, opset=12, device="cpu"
                    )
                # (a) direct save as target
                if target.exists():
                    action = "exported"
                else:
                    # (b) under runs/ → copy to target
                    cand = _latest_exported_onnx()
                    if cand and cand.exists():
                        shutil.copy2(cand, target)
                    action = "exported" if target.exists() else "failed"
                # Tidy transient folder; ignore errors
                try:
                    shutil.rmtree(dst / "runs", ignore_errors=True)
                except Exception:
                    pass
                return (target, action)
            except Exception:
                continue

    return (target, "failed")


def main(argv: List[str]) -> int:
    if not argv:
        print("usage: _export_onnx.py <weight.pt|weight.onnx|name> [...]")
        return 2

    results: List[Tuple[Path, str]] = []
    for a in argv:
        path, action = _export_one(a)
        results.append((path, action))

    icons = {"present": "↺", "exported": "⎘", "failed": "✗"}
    for path, action in results:
        print(f"{icons.get(action, '•')} {path} ({action})")

    failed: List[str] = [str(p) for p, act in results if act == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

