# projects/argos/panoptes/model/_export_onnx.py
from __future__ import annotations

import glob
import os
import shutil
import sys
import subprocess
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    ContextManager,
    Generator,
    List,
    Dict,
    Optional,
    Tuple,
    Sequence,
    Union,
    Protocol,
    runtime_checkable,
    cast,
)

import typer

# ---------------------------------------------------------------------
# Environment safety for reliable ONNX export on fresh installs
# (project-driven, no terminal steps required by end users)
# ---------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("ORT_DISABLE_CPU_CAPABILITY_CHECK", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

MODEL_DIR = Path(__file__).resolve().parent  # panoptes/model
DEFAULT_IMGSZ = int(os.environ.get("ARGOS_ONNX_IMGSZ", "640"))

# ---------------------------------------------------------------------
# Minimal quiet stdio helpers
# ---------------------------------------------------------------------
@contextmanager
def _silence_stdio() -> Generator[None, None, None]:
    buf_out, buf_err = StringIO(), StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        yield


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
# Panoptes progress (Halo-based, single-line) with safe fallbacks
# ---------------------------------------------------------------------
try:
    from panoptes.progress import osc8 as _osc8_raw  # type: ignore[reportMissingTypeStubs]
    from panoptes.progress import (  # type: ignore[reportMissingTypeStubs]
        percent_spinner as _percent_spinner,
    )
    from panoptes.progress import simple_status as _simple_status  # type: ignore[reportMissingTypeStubs]

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
# CWD switcher
# ---------------------------------------------------------------------
@contextmanager
def _cd(p: Path) -> Generator[None, None, None]:
    prev = Path.cwd()
    try:
        os.chdir(p)
        yield
    finally:
        os.chdir(prev)


# ---------------------------------------------------------------------
# Light validation/utils
# ---------------------------------------------------------------------
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
    """
    Basic sanity for .onnx:
      - exists
      - ≥ 1 MB (heuristic; avoids HTML error bodies)
      - not an HTML/JSON error
    """
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
    """
    YOLO spelling variants: yolo8/11/12 ↔ yolov8/11/12; preserve suffixes.
    E.g., 'yolo11s-seg.pt' <-> 'yolov11s-seg.pt'
    """
    base = Path(name).name
    alts = {base}
    for ver in ("8", "11", "12"):
        alts.add(base.replace(f"yolov{ver}", f"yolo{ver}"))
        alts.add(base.replace(f"yolo{ver}", f"yolov{ver}"))
    return [a for a in alts if a]


# ---------------------------------------------------------------------
# Runtime ensure for export: torch + onnx + onnxscript + friends
# ---------------------------------------------------------------------
def _pip_install(pkgs: List[str], extra_args: Optional[List[str]] = None) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-input"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(pkgs)
    try:
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _try_import(name: str) -> tuple[bool, Optional[str]]:
    try:
        __import__(name)
        return True, None
    except Exception as e:
        return False, f"{e.__class__.__name__}: {e}"


def _ensure_export_runtime() -> None:
    """
    Ensure the *local venv* has the packages needed to export ONNX.
    Silent best-effort; callers still have fallbacks.
    """
    # 1) torch (CPU) — required to load Ultralytics weights and to export
    ok, _ = _try_import("torch")
    if not ok:
        torch_ver = os.environ.get("ARGOS_TORCH_VERSION", "2.4.1")
        extra: List[str] = []
        # Prefer CPU wheels for Windows/Linux to avoid CUDA downloads; macOS uses default (MPS wheels)
        if os.name == "nt" or sys.platform.startswith("linux"):
            idx = os.environ.get("ARGOS_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cpu")
            extra = ["--index-url", idx]
        _pip_install([f"torch=={torch_ver}"], extra_args=extra)
        _try_import("torch")  # retry once; ignore result (export will handle errors)

    # 2) onnx + protobuf + onnxscript + onnxsim (ranges aligned with project constraints)
    wants = [
        "onnx>=1.18,<1.19",
        "protobuf>=4.23,<5",
        "onnxscript>=0.1.0,<0.2",
        "onnxsim>=0.4.36,<0.5",
    ]
    _pip_install(wants)

    # 3) Sanity: import onnx; if we hit the Windows DLL error, pin a known-good set
    ok, err = _try_import("onnx")
    if not ok and err and ("onnx_cpp2py_export" in err or "DLL load failed" in err):
        _pip_install(["onnx==1.18.0", "protobuf==4.25.3"], extra_args=["--force-reinstall", "--no-deps"])
        _try_import("onnx")  # try again; final failure will be surfaced by exporter


# ---------------------------------------------------------------------
# YOLO typing helpers (to quiet editors without stubs)
# ---------------------------------------------------------------------
@runtime_checkable
class _YOLOClassLike(Protocol):
    def __call__(self, source: str) -> Any: ...


ExportOut = Union[
    str,
    os.PathLike[str],  # type: ignore[name-defined]
    Sequence[Union[str, os.PathLike[str]]],  # type: ignore[name-defined]
    None,
]


# ---------------------------------------------------------------------
# Lazy Ultralytics import (after ensuring torch)
# ---------------------------------------------------------------------
def _get_yolo_class() -> _YOLOClassLike | None:
    try:
        from ultralytics import YOLO as _YOLO  # type: ignore
        # Silence Ultralytics’ global logger to keep build output clean
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
        return cast(_YOLOClassLike, _YOLO)
    except Exception:
        return None


def _status_cm(label: str) -> ContextManager[Any]:
    """
    Typed wrapper over progress.simple_status; auto-disables if nested.
    """
    enabled = (os.environ.get("PANOPTES_PROGRESS_ACTIVE") != "1")
    try:
        return cast(ContextManager[Any], simple_status(label, enabled=enabled))  # type: ignore[misc]
    except Exception:
        return _NoopSpinner()


def _parse_ops_candidates() -> List[int]:
    """
    Read ARGOS_ONNX_OPSETS or default to a modern cascade.
    """
    raw = os.environ.get("ARGOS_ONNX_OPSETS", "").strip()
    if raw:
        out: List[int] = []
        for part in raw.replace(",", " ").split():
            try:
                out.append(int(part))
            except Exception:
                pass
        if out:
            return out
    # default cascade: newest → older, wide coverage for YOLOv8/11/12
    return [21, 20, 19, 18, 17]


def _try_ultralytics_export(YOLO: _YOLOClassLike, pt: Path, target: Path) -> Tuple[bool, Optional[str]]:
    """
    Use Ultralytics' exporter across a cascade of opsets and flags.
    Returns (ok, last_error_message).
    """
    opsets = _parse_ops_candidates()
    last_err: Optional[str] = None
    with _status_cm("export onnx (ultralytics)"), _silence_stdio():
        m: Any = YOLO(str(pt))  # runtime-provided class; methods are dynamically typed
        outp_path: Optional[Path] = None

        for opset in opsets:
            for dynamic in (True, False):
                for simplify_flag in (True, False):
                    try:
                        export_kwargs: Dict[str, Any] = {
                            "format": "onnx",
                            "dynamic": dynamic,
                            "simplify": simplify_flag,
                            "imgsz": DEFAULT_IMGSZ,
                            "opset": opset,
                            "device": "cpu",
                        }
                        try:
                            export_kwargs["postprocess"] = "graph"
                            out_any: ExportOut = cast(
                                ExportOut,
                                m.export(**export_kwargs),  # type: ignore[attr-defined]
                            )
                        except TypeError:
                            export_kwargs.pop("postprocess", None)
                            out_any = cast(
                                ExportOut,
                                m.export(**export_kwargs),  # type: ignore[attr-defined]
                            )
                        # Some versions return str, some a path-like, some a list
                        if isinstance(out_any, (list, tuple)) and out_any:
                            outp_path = Path(str(out_any[0]))
                        elif out_any:
                            outp_path = Path(str(out_any))
                        else:
                            outp_path = _latest_exported_onnx()

                            if outp_path and outp_path.exists():
                                if outp_path.resolve() != target.resolve():
                                    shutil.copy2(outp_path, target)
                                if _validate_weight(target):
                                    try:
                                        from panoptes.model.artifact_metadata import analyse_artifact  # type: ignore[import]

                                        meta = analyse_artifact(target)
                                        if not meta.get("nms_in_graph"):
                                            typer.secho(
                                                f"Warning: ONNX export for {target.name} lacks in-graph NMS.",
                                                fg="yellow",
                                            )
                                    except Exception:
                                        pass
                                    return True, None
                            try:
                                target.unlink(missing_ok=True)
                            except TypeError:
                                if target.exists():
                                    target.unlink()
                    except Exception as e:
                        last_err = f"opset={opset}, dynamic={dynamic}, simplify={simplify_flag}: {e}"
    return False, last_err


def _try_torch_export_fallback(pt: Path, target: Path) -> Tuple[bool, Optional[str]]:
    """
    Optional fallback: use torch.onnx exporter.
    Tries the modern `dynamo=True` path first (PyTorch ≥ 2.5), then legacy if needed.
    Returns (ok, error_message).
    """
    try:
        import torch  # type: ignore
        from ultralytics import YOLO as _Y  # type: ignore
    except Exception as e:
        return False, f"torch/ultralytics import failed: {e}"

    try:
        m: Any = _Y(str(pt))
        model: Any = getattr(m, "model", m)
        try:
            model.eval()
        except Exception:
            pass

        dummy = torch.zeros(1, 3, DEFAULT_IMGSZ, DEFAULT_IMGSZ)  # type: ignore[attr-defined]
        opsets = _parse_ops_candidates()
        opver = max(min(opsets), 12)
        tmp = target.with_suffix(".dynamo.onnx")

        # 1) Try dynamo exporter
        try:
            with _status_cm("export onnx (torch dynamo)"), _silence_stdio():
                torch.onnx.export(  # type: ignore[attr-defined]
                    model,
                    (dummy,),
                    str(tmp),
                    opset_version=opver,
                    input_names=["images"],
                    output_names=["output"],
                    dynamic_axes={"images": {0: "batch", 2: "h", 3: "w"}, "output": {0: "batch"}},
                    export_params=True,
                    do_constant_folding=True,
                    verbose=False,
                    dynamo=True,  # PyTorch ≥ 2.5
                )
        except TypeError:
            # 2) Fallback to legacy exporter (no `dynamo` kw)
            with _status_cm("export onnx (torch legacy)"), _silence_stdio():
                torch.onnx.export(  # type: ignore[attr-defined]
                    model,
                    (dummy,),
                    str(tmp),
                    opset_version=opver,
                    input_names=["images"],
                    output_names=["output"],
                    dynamic_axes={"images": {0: "batch", 2: "h", 3: "w"}, "output": {0: "batch"}},
                    export_params=True,
                    do_constant_folding=True,
                    verbose=False,
                )

        if tmp.exists():
            if tmp.resolve() != target.resolve():
                shutil.copy2(tmp, target)
            if _validate_weight(target):
                return True, None
            try:
                target.unlink(missing_ok=True)
            except TypeError:
                if target.exists():
                    target.unlink()
            return False, "torch exporter produced non-valid file"
        return False, "torch exporter produced no output"
    except Exception as e:
        return False, f"torch export failed: {e}"


def _export_one(arg: str) -> Tuple[Path, str]:
    """
    Export ONNX for *arg*.
    Returns (final_path, action) where action ∈ {"present" | "exported" | "failed"}.
    """
    # Decide canonical target path first
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

    # Ensure runtime first (torch + onnx + onnxscript + protobuf + onnxsim)
    _ensure_export_runtime()

    # Acquire Ultralytics class after torch is available
    YOLO = _get_yolo_class()
    if YOLO is None:
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
                # 1) Fetch or locate the .pt
                with _status_cm("fetch weights"), _silence_stdio():
                    m: Any = YOLO(str(src_pt_path or nm))  # may download into CWD=dst
                    ck = Path(getattr(m, "ckpt_path", nm)).expanduser()
                    if not ck.exists():
                        ck = Path(nm).expanduser()
                    canon_pt = dst / pt_name  # canonical local filename
                    if ck.exists() and ck.resolve() != canon_pt.resolve():
                        shutil.copy2(ck, canon_pt)

                # 2) Export via Ultralytics (multi-try)
                ok, _ = _try_ultralytics_export(YOLO, canon_pt, target)

                # 3) Fallback: torch exporter (dynamo or legacy)
                if not ok:
                    ok, _ = _try_torch_export_fallback(canon_pt, target)

                # 4) Finalize outcome
                if ok and target.exists() and _validate_weight(target):
                    # Tidy transient folder; ignore errors
                    try:
                        shutil.rmtree(dst / "runs", ignore_errors=True)
                    except Exception:
                        pass
                    return (target, "exported")
            except Exception:
                continue

    return (target, "failed")


def main(argv: List[str]) -> int:
    if not argv:
        print("usage: _export_onnx.py <weight.pt|weight.onnx|name> [...]")
        return 2

    results: List[Tuple[Path, str]] = []

    # Halo spinner progress (single-line, fixed width)
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
