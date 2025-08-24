# projects/argos/panoptes/model/_export_onnx.py
from __future__ import annotations

import glob
import importlib.util
import os
import re
import shutil
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import TracebackType
from typing import Any, ContextManager, Generator, List, Optional, Tuple, cast


# ---------------------------------------------------------------------
# Minimal no-op spinner (works with "with ... as sp: sp.update(...)")
# ---------------------------------------------------------------------
class _NoopSpinner:
    def __enter__(self) -> "_NoopSpinner":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return None

    def update(self, **_: Any) -> "_NoopSpinner":
        return self


# ---------------------------------------------------------------------
# Panoptes progress (Halo-based, single-line) + Path-safe osc8 wrapper
# ---------------------------------------------------------------------
try:
    from panoptes.progress import osc8 as _osc8_raw  # type: ignore
    from panoptes.progress import percent_spinner as _percent_spinner  # type: ignore
    from panoptes.progress import simple_status as _simple_status  # type: ignore

    def percent_spinner(*args: Any, **kwargs: Any) -> ContextManager[Any]:  # type: ignore[misc]
        return _percent_spinner(*args, **kwargs)  # type: ignore[misc]

    def simple_status(*args: Any, **kwargs: Any) -> ContextManager[Any]:  # type: ignore[misc]
        return _simple_status(*args, **kwargs)  # type: ignore[misc]

except Exception:  # pragma: no cover

    def percent_spinner(*_: Any, **__: Any) -> ContextManager[Any]:  # type: ignore[no-redef]
        return _NoopSpinner()

    def simple_status(*_: Any, **__: Any) -> ContextManager[Any]:  # type: ignore[no-redef]
        return _NoopSpinner()

    def _osc8_raw(label: str, target: str) -> str:  # type: ignore[no-redef]
        return str(target)


def osc8_link(label: str, target: str | Path) -> str:
    try:
        return _osc8_raw(label, str(target))
    except Exception:
        return str(target)


# ---------------------------------------------------------------------
# Auto-install export deps (quiet)
# ---------------------------------------------------------------------
def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def _pip_quiet(*pkgs: str) -> None:
    if not pkgs:
        return
    try:
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-input", "--quiet", *pkgs],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _parse_ver_tuple(s: str) -> Tuple[int, int, int]:
    nums = [int(x) for x in re.findall(r"\d+", s)[:3]]
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def _ensure_export_toolchain() -> None:
    # Ultralytics pinned to range that works for this project
    need_ultra = False
    if not _have("ultralytics"):
        need_ultra = True
    else:
        try:
            from importlib.metadata import version as _ver  # type: ignore
        except Exception:  # pragma: no cover
            try:
                from importlib_metadata import version as _ver  # type: ignore
            except Exception:
                _ver = None  # type: ignore

        try:
            v: Optional[str] = cast(Optional[str], _ver("ultralytics") if _ver else None)
            if v is not None:
                vt = _parse_ver_tuple(v)
                if not ((8, 3, 0) <= vt < (8, 6, 0)):
                    need_ultra = True
            else:
                need_ultra = True
        except Exception:
            need_ultra = True

    if need_ultra:
        _pip_quiet("ultralytics>=8.3,<8.6")

    # ONNX dependencies – ensure correct versions
    if not _have("onnx"):
        _pip_quiet("onnx>=1.14,<1.18")

    if not _have("onnxruntime"):
        if sys.version_info < (3, 10):
            _pip_quiet("onnxruntime==1.19.2")        # use older ORT on Py3.9
        elif os.name == "nt":
            _pip_quiet("onnxruntime>=1.22,<1.23")    # Windows Py3.10+ (avoid 1.23+ on Win)
        else:
            _pip_quiet("onnxruntime>=1.22,<1.24")    # Linux/macOS Py3.10+

    # Pin ONNX-Slim to safe version range (avoid newer incompatible builds)
    _pip_quiet("onnxslim>=0.1.59,<0.1.60")


# Ensure deps before we try to import YOLO
_ensure_export_toolchain()

# --- Ultralytics (quiet logging, version-agnostic) ---
has_yolo: bool = False
try:
    from ultralytics import YOLO  # type: ignore
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
    return (
        s.startswith(b"<")
        or s.startswith(b"{")
        or b"AccessDenied" in s
        or b"Error" in s
        or b"<!DOCTYPE" in s
    )


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
        alts.add(base.replace(f"yolo{ver}", f"yolov{ver}"))
    return [a for a in alts if a]


@contextmanager
def _silence_stdio() -> Generator[None, None, None]:
    buf_out, buf_err = StringIO(), StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        yield


def _status_cm(label: str) -> ContextManager[Any]:
    enabled = (os.environ.get("PANOPTES_PROGRESS_ACTIVE") != "1")
    try:
        return cast(ContextManager[Any], simple_status(label, enabled=enabled))  # type: ignore[misc]
    except Exception:
        return _NoopSpinner()


def _export_one(arg: str) -> Tuple[Path, str]:
    """
    Export ONNX for *arg*.
    Returns (final_path, action) where action ∈ {"present" | "exported" | "failed"}.
    """
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

    if target.exists():
        if _validate_weight(target):
            return (target, "present")
        try:
            target.unlink()
        except Exception:
            pass

    if not has_yolo or YOLO is None:
        return (target, "failed")

    with _cd(dst):
        src_pt_path: Optional[Path] = None
        if (p.suffix.lower() == ".pt") and p.exists():
            src_pt_path = p.resolve()

        tried_names: List[str] = [pt_name, *{nm for nm in _synonyms(pt_name) if nm != pt_name}]
        for nm in tried_names:
            try:
                with _status_cm("fetch weights"), _silence_stdio():
                    m = YOLO(str(src_pt_path or nm))  # may download into CWD=dst
                    ck = Path(getattr(m, "ckpt_path", nm)).expanduser()
                    if not ck.exists():
                        ck = Path(nm).expanduser()
                    canon_pt = dst / pt_name
                    if ck.exists() and ck.resolve() != canon_pt.resolve():
                        shutil.copy2(ck, canon_pt)

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
                            YOLO(str(canon_pt)).export(  # type: ignore[call-arg]
                                format="onnx", dynamic=False, simplify=False, imgsz=640, opset=12, device="cpu"
                            )

                if target.exists() and _validate_weight(target):
                    action = "exported"
                else:
                    cand = _latest_exported_onnx()
                    if cand and cand.exists():
                        shutil.copy2(cand, target)
                    action = "exported" if (target.exists() and _validate_weight(target)) else "failed"

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

    with percent_spinner(prefix="ONNX") as sp:
        sp.update(total=len(argv), count=0)
        for i, a in enumerate(argv, start=1):
            try:
                model_name = Path(a).name
                sp.update(current=a, job="export", model=model_name)
            except Exception:
                pass
            path, action = _export_one(a)
            results.append((path, action))
            try:
                sp.update(count=i, job="", model=Path(path).name)
            except Exception:
                pass

    icons = {"present": "↺", "exported": "⎘", "failed": "✗"}
    for path, action in results:
        base = path.name
        if action == "exported":
            print(f"{icons.get(action, '•')} {osc8_link(base, path)} ({action})")
        else:
            print(f"{icons.get(action, '•')} {path} ({action})")

    failed: List[str] = [str(p) for p, act in results if act == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
