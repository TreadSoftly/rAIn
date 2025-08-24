# projects/argos/panoptes/model/_export_onnx.py
from __future__ import annotations

import glob
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import TracebackType
from typing import Any, ContextManager, Generator, List, Optional, Tuple, cast

# ---------------------------------------------------------------------
# One-time ONNX preflight/repair flags & log
# ---------------------------------------------------------------------
_onnx_preflight_done: bool = False
_onnx_usable: Optional[bool] = None
_DIAG_LOG_NAME = "_onnx_diagnostics.txt"

os.environ.setdefault("PIP_ONLY_BINARY", ":all:")
os.environ.setdefault("PIP_NO_BUILD_ISOLATION", "1")
os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

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


def _pip_quiet(*pkgs: str, force_reinstall: bool = False) -> None:
    if not pkgs:
        return
    args = [sys.executable, "-m", "pip", "install", "--no-input", "--quiet"]
    if force_reinstall:
        args.append("--force-reinstall")
    args.extend(pkgs)
    env = os.environ.copy()
    env.setdefault("PIP_ONLY_BINARY", ":all:")
    env.setdefault("PIP_NO_BUILD_ISOLATION", "1")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    try:
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    except Exception:
        pass


def _parse_ver_tuple(s: str) -> Tuple[int, int, int]:
    nums = [int(x) for x in re.findall(r"\d+", s)[:3]]
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def _decide_opset(torch_version_str: Optional[str]) -> int:
    tv = _parse_ver_tuple(torch_version_str or "0.0.0")
    if tv >= (2, 4, 0):
        return 19
    if tv >= (2, 2, 0):
        return 17
    return 12


def _candidate_opsets() -> List[int]:
    """
    Determine the candidate opset order from installed torch if available.
    Prefer the torch-based recommendation, then fall back to 19/17/12.
    """
    preferred = 12
    try:
        import torch  # type: ignore
        preferred = _decide_opset(getattr(torch, "__version__", "0.0.0"))
    except Exception:
        pass
    out: List[int] = []
    for o in (preferred, 19, 17, 12):
        if o not in out:
            out.append(o)
    return out


def _diag_log_path() -> Path:
    return Path(__file__).resolve().parent / _DIAG_LOG_NAME


def _write_diag(header: str, lines: List[str]) -> None:
    try:
        log = _diag_log_path()
        with log.open("a", encoding="utf-8", errors="ignore") as fh:
            fh.write(f"\n=== {header} ===\n")
            for ln in lines:
                fh.write(ln.rstrip() + "\n")
    except Exception:
        pass


def _snapshot_env() -> None:
    try:
        lines: List[str] = []
        lines.append(f"Time: {__import__('time').ctime()}")
        lines.append(f"OS: {platform.platform()}  ({os.name})")
        lines.append(f"Machine/Arch: {platform.machine()}  Python: {sys.version.split()[0]}  Bits: {platform.architecture()[0]}")
        lines.append(f"Executable: {sys.executable}")
        if os.name == "nt":
            win = os.environ.get("WINDIR", r"C:\Windows")
            sys32 = Path(win) / "System32"
            for nm in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
                lines.append(f"{nm}: {'present' if (sys32 / nm).exists() else 'missing'}")
        for p in ("onnx", "onnxruntime", "numpy", "torch", "protobuf", "ultralytics"):
            try:
                out = subprocess.check_output([sys.executable, "-m", "pip", "show", p], stderr=subprocess.STDOUT).decode("utf-8", "ignore")
                ver = ""
                for ln in out.splitlines():
                    if ln.lower().startswith("version:"):
                        ver = ln.split(":", 1)[1].strip()
                        break
                lines.append(f"{p}: {ver or 'not installed'}")
            except Exception:
                lines.append(f"{p}: not installed")
        _write_diag("ENV SNAPSHOT", lines)
    except Exception:
        pass


def _ensure_export_toolchain() -> None:
    """
    Ensure a working ONNX export toolchain across clean machines.

    - numpy < 2
    - protobuf < 5
    - ultralytics >=8.3,<8.6
    - onnx >=1.16,<1.18
    - onnxruntime pinned by Python version and OS
    - onnxsim >=0.4.17,<0.5
    - onnxslim >=0.1.59,<0.1.60
    """
    try:
        import numpy as _np  # type: ignore
        nvt = _parse_ver_tuple(getattr(_np, "__version__", "0.0.0"))
        if nvt >= (2, 0, 0):
            _pip_quiet("numpy<2", force_reinstall=True)
    except Exception:
        _pip_quiet("numpy<2")

    _pip_quiet("protobuf<5")

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

    if not _have("onnx"):
        _pip_quiet("onnx>=1.16,<1.18")
    if not _have("onnxruntime"):
        if sys.version_info < (3, 10):
            _pip_quiet("onnxruntime==1.19.2")
        elif os.name == "nt":
            _pip_quiet("onnxruntime>=1.22,<1.23")
        else:
            _pip_quiet("onnxruntime>=1.22,<1.24")
    if not _have("onnxsim"):
        _pip_quiet("onnxsim>=0.4.17,<0.5")
    _pip_quiet("onnxslim>=0.1.59,<0.1.60")


def _try_import_onnx() -> Tuple[bool, str]:
    try:
        import onnx  # type: ignore
        _ = getattr(onnx, "version", None)
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _repair_onnx_stack() -> None:
    _write_diag("REPAIR START", ["Reinstall numpy<2, protobuf<5, onnx>=1.16,<1.18; refresh onnxruntime"])
    _pip_quiet("numpy<2", "protobuf<5", "onnx>=1.16,<1.18", force_reinstall=True)
    if sys.version_info < (3, 10):
        _pip_quiet("onnxruntime==1.19.2", force_reinstall=True)
    elif os.name == "nt":
        _pip_quiet("onnxruntime>=1.22,<1.23", force_reinstall=True)
    else:
        _pip_quiet("onnxruntime>=1.22,<1.24", force_reinstall=True)
    _write_diag("REPAIR END", ["done"])


def _preflight_and_repair_onnx_once() -> bool:
    global _onnx_preflight_done, _onnx_usable
    if _onnx_preflight_done:
        return bool(_onnx_usable)
    _onnx_preflight_done = True
    _snapshot_env()
    # Shadowing
    here = Path.cwd()
    bad: List[str] = []
    for p in (here / "onnx.py", here / "onnx"):
        if p.exists():
            bad.append(str(p))
    if bad:
        _write_diag("LOCAL SHADOWING", bad)
    _ensure_export_toolchain()
    ok, err = _try_import_onnx()
    if not ok:
        _write_diag("FIRST IMPORT FAILURE", [err])
        _repair_onnx_stack()
        ok, err2 = _try_import_onnx()
        if not ok:
            _write_diag("SECOND IMPORT FAILURE", [err2])
    _onnx_usable = ok
    if ok:
        print("ONNX environment: OK (validated)")
    else:
        print(f"ONNX environment not usable after auto-repair. See log: {osc8_link(_DIAG_LOG_NAME, _diag_log_path())}")
    return ok


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
    """Try YOLO spelling variants (yolo8/11/12 ↔ yolov8/11/12; suffixes preserved)."""
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

    # One-time preflight + auto-repair
    if not _preflight_and_repair_onnx_once():
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
                    exported = False
                    # Try robust matrix: opset (torch‑guided), dynamic/simplify fallbacks.
                    for ops in _candidate_opsets():
                        for (dyn, simp) in ((True, True), (True, False), (False, False)):
                            try:
                                YOLO(str(canon_pt)).export(  # type: ignore[call-arg]
                                    format="onnx",
                                    dynamic=dyn,
                                    simplify=simp,
                                    imgsz=640,
                                    opset=ops,
                                    device="cpu",
                                )
                                if target.exists() and _validate_weight(target):
                                    exported = True
                                    break
                                cand = _latest_exported_onnx()
                                if cand and cand.exists():
                                    shutil.copy2(cand, target)
                                if target.exists() and _validate_weight(target):
                                    exported = True
                                    break
                            except Exception:
                                continue
                        if exported:
                            break

                if target.exists() and _validate_weight(target):
                    action = "exported"
                else:
                    action = "failed"

                # Tidy transient outputs
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

    # One-time doctor/repair (summary line only)
    _preflight_and_repair_onnx_once()

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
