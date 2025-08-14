# \rAIn\projects\argos\panoptes\model\_export_onnx.py
from __future__ import annotations

import glob
import os
import shutil
import sys
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable, ContextManager, Generator, List, Optional, Tuple, cast

# Optional progress (auto-disables off-TTY / in CI)
try:
    from panoptes.progress import ( # type: ignore[reportMissingTypeStubs]
        ProgressEngine,  # type: ignore[reportMissingTypeStubs]; type: ignore
        live_percent,
        simple_status,
    )
except Exception:
    ProgressEngine = None  # type: ignore[assignment]
    live_percent = None    # type: ignore[assignment]
    simple_status = None   # type: ignore[assignment]

# --- Ultralytics (quiet logging, version-agnostic) ---
has_yolo: bool = False
try:
    from ultralytics import YOLO  # type: ignore[reportMissingTypeStubs]
    try:
        from ultralytics.utils import LOGGER as _ULTRA_LOGGER  # type: ignore
        _rem = getattr(_ULTRA_LOGGER, "remove", None)
        if callable(_rem):
            _rem()
        else:
            for h in list(getattr(_ULTRA_LOGGER, "handlers", [])):
                try:
                    _ULTRA_LOGGER.removeHandler(h)  # type: ignore[attr-defined]
                except Exception:
                    pass
    except Exception:
        pass
    has_yolo = True
except Exception:
    YOLO = None  # type: ignore[assignment]
    has_yolo = False

MODEL_DIR = Path(__file__).resolve().parent  # panoptes/model


def _osc8(label: str, path: Path) -> str:
    try:
        uri = path.resolve().as_uri()
    except Exception:
        uri = "file:///" + str(path.resolve()).replace("\\", "/")
    return f"\x1b]8;;{uri}\x1b\\{label}\x1b]8;;\x1b\\"


@contextmanager
def _cd(p: Path) -> Generator[None, None, None]:
    prev = Path.cwd()
    try:
        os.chdir(p)
        yield
    finally:
        os.chdir(prev)


def _latest_exported_onnx() -> Optional[Path]:
    hits = [Path(p) for p in glob.glob("runs/**/**/*.onnx", recursive=True)]
    return max(hits, key=lambda pp: pp.stat().st_mtime) if hits else None


def _looks_like_text_header(bs: bytes) -> bool:
    s = bs.lstrip()
    if not s:
        return False
    return s.startswith(b"<") or s.startswith(b"{") or b"AccessDenied" in s or b"Error" in s or b"<!DOCTYPE" in s


def _validate_weight(p: Path) -> bool:
    """Basic sanity for .onnx: exists, ≥1MB, not an HTML/JSON error."""
    try:
        if not p.exists():
            return False
        if p.stat().st_size < 1_000_000:
            return False
        with p.open("rb") as fh:
            head = fh.read(512)
        if _looks_like_text_header(head):
            return False
        return True
    except Exception:
        return False


def _synonyms(name: str) -> List[str]:
    """Try YOLO’s common spelling variants (yolo8/11/12 ↔ yolov8/11/12; suffixes preserved)."""
    base = Path(name).name
    alts = {base}
    for ver in ("8", "11", "12"):
        alts.add(base.replace(f"yolov{ver}", f"yolo{ver}"))
        alts.add(base.replace(f"yolo{ver}",  f"yolov{ver}"))
    return [a for a in alts if a]


@contextmanager
def _silence_stdio() -> Generator[None, None, None]:
    buf_out, buf_err = StringIO(), StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        yield

def _status_cm(label: str) -> ContextManager[Any]:
    """Typed wrapper for optional simple_status."""
    enabled = (os.environ.get("PANOPTES_PROGRESS_ACTIVE") != "1")
    if simple_status is not None:
        fn = cast(Callable[..., ContextManager[Any]], simple_status)
        return fn(label, enabled=enabled)  # type: ignore[misc]
    return cast(ContextManager[Any], nullcontext())


def _export_one(arg: str) -> Tuple[Path, str]:
    """
    Export ONNX for *arg*.
    Returns (final_path, action) where action ∈ {"present" | "exported" | "failed"}.
    """
    # Decide canonical target path first (works even if YOLO missing)
    p = Path(arg)
    if p.suffix.lower() == ".pt":
        dst = (p.parent if p.parent != Path() else MODEL_DIR)
        onnx_name = p.with_suffix(".onnx").name
        pt_name = p.name
    elif p.suffix.lower() == ".onnx":
        dst = (p.parent if p.parent != Path() else MODEL_DIR)
        onnx_name = p.name
        pt_name = Path(onnx_name).with_suffix(".pt").name
    else:
        dst = MODEL_DIR
        root = p.name
        if not root.endswith((".pt", ".onnx")):
            root += ".onnx"
        onnx_name = Path(root).with_suffix(".onnx").name
        pt_name = Path(root).with_suffix(".pt").name

    dst.mkdir(parents=True, exist_ok=True)
    target = dst / onnx_name

    # Already present (only if valid)
    if target.exists():
        if _validate_weight(target):
            return (target, "present")
        try:
            target.unlink()
        except Exception:
            pass

    if not has_yolo or YOLO is None:
        return (target, "failed")

    # Work inside *dst* so Ultralytics writes runs/ here (not repo root)
    with _cd(dst):
        # Ensure we have the .pt inside dst if a real file path was provided
        src_pt_path: Optional[Path] = None
        if (p.suffix.lower() == ".pt") and p.exists():
            src_pt_path = p.resolve()

        # Try the provided spelling + synonyms for hub fetch
        tried_names: List[str] = [pt_name, *{nm for nm in _synonyms(pt_name) if nm != pt_name}]
        for nm in tried_names:
            try:
                # Load/Fetch the .pt
                with _status_cm("fetch weights"), _silence_stdio():
                    m = YOLO(str(src_pt_path or nm))  # may download into CWD=dst
                    ck = Path(getattr(m, "ckpt_path", nm)).expanduser()
                    if not ck.exists():
                        ck = Path(nm).expanduser()
                    canon_pt = dst / pt_name  # canonical local filename
                    if ck.exists() and ck.resolve() != canon_pt.resolve():
                        shutil.copy2(ck, canon_pt)

                # Export ONNX (with safe fallbacks)
                with _status_cm("export onnx"), _silence_stdio():
                    try:
                        YOLO(str(canon_pt)).export(  # type: ignore[call-arg]
                            format="onnx", dynamic=True, simplify=True, imgsz=640, opset=12, device="cpu"
                        )
                    except Exception:
                        try:
                            YOLO(str(canon_pt)).export(  # type: ignore[call-arg]
                                format="onnx", dynamic=True, simplify=False, imgsz=640, opset=12, device="cpu"
                            )
                        except Exception:
                            YOLO(str(canon_pt)).export(  # last ditch: drop dynamic
                                format="onnx", simplify=False, imgsz=640, opset=12, device="cpu"  # type: ignore[call-arg]
                            )

                # (a) direct save as target
                if target.exists() and _validate_weight(target):
                    action = "exported"
                else:
                    # (b) under runs/ → copy to target (then validate)
                    cand = _latest_exported_onnx()
                    if cand and cand.exists():
                        shutil.copy2(cand, target)
                    action = "exported" if (target.exists() and _validate_weight(target)) else "failed"

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

    # Nice percent progress if available
    if ProgressEngine is not None and live_percent is not None:
        eng = ProgressEngine()  # type: ignore[call-arg]
        with live_percent(eng, prefix="ONNX"):  # type: ignore[misc]
            eng.set_total(len(argv))
            for a in argv:
                eng.set_current(a)
                path, action = _export_one(a)
                results.append((path, action))
                eng.add(1)
    else:
        for a in argv:
            path, action = _export_one(a)
            results.append((path, action))

    icons = {"present": "↺", "exported": "⎘", "failed": "✗"}
    for path, action in results:
        base = path.name
        if action == "exported":
            print(f"{icons.get(action, '•')} {_osc8(base, path)} ({action})")
        else:
            print(f"{icons.get(action, '•')} {path} ({action})")

    failed: List[str] = [str(p) for p, act in results if act == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
