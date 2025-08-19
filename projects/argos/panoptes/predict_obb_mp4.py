# projects/argos/panoptes/predict_obb_mp4.py
"""
predict_obb_mp4 — per-frame oriented bounding boxes overlay for videos.

Strict model selection via panoptes.model_registry.load_obb().
FFmpeg→OpenCV→AVI encode ladder; single "Saved → …" line; progress policy like other workers.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Final, Protocol, Tuple, cast

import cv2  # type: ignore
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# ROOT resolution (single assignment)
# ────────────────────────────────────────────────────────────────────────────────

def _resolve_root() -> Path:
    try:
        from panoptes import ROOT as _ROOT  # type: ignore[import-not-found]
        return _ROOT
    except Exception:
        return Path.cwd()

ROOT: Final[Path] = _resolve_root()

# ────────────────────────────────────────────────────────────────────────────────
# Type aliases
# ────────────────────────────────────────────────────────────────────────────────

if TYPE_CHECKING:
    import numpy as _np
    import numpy.typing as npt

    NDArrayU8 = npt.NDArray[_np.uint8]
    NDArrayF32 = npt.NDArray[_np.float32]
    Poly42F32 = NDArrayF32  # (4,2) float32 polygon; shape not enforced by checker
else:
    NDArrayU8 = Any  # type: ignore[assignment]
    NDArrayF32 = Any  # type: ignore[assignment]
    Poly42F32 = Any  # type: ignore[assignment]

# ────────────────────────────────────────────────────────────────────────────────
# Strict model loader (central registry)
# ────────────────────────────────────────────────────────────────────────────────

try:
    from panoptes.model_registry import load_obb  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    load_obb = None  # type: ignore[assignment]

# progress (optional)
try:
    from panoptes.progress import ProgressEngine  # type: ignore[import-not-found]
    from panoptes.progress.bridges import live_percent  # type: ignore[import-not-found]
    from panoptes.progress.progress_ux import simple_status  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    ProgressEngine = None  # type: ignore[assignment]
    live_percent = None  # type: ignore[assignment]
    simple_status = None  # type: ignore[assignment]

_LOG = logging.getLogger("panoptes.predict_obb_mp4")
if not _LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
_LOG.setLevel(logging.WARNING)


def _say(msg: str) -> None:
    _LOG.info(f"[panoptes] {msg}")


class SpinnerLike(Protocol):
    def update(self, **kwargs: Any) -> "SpinnerLike": ...


def _osc8(label: str, path: Path) -> str:
    try:
        uri = path.resolve().as_uri()
    except Exception:
        uri = "file:///" + str(path.resolve()).replace("\\", "/")
    esc = "\033"
    return f"{esc}]8;;{uri}{esc}\\{label}{esc}]8;;{esc}\\"


def _fourcc(code: str) -> int:
    fn: Callable[..., Any] = getattr(cv2, "VideoWriter_fourcc", getattr(cv2.VideoWriter, "fourcc"))
    return cast(int, fn(*code))


def _avi_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    vw = cv2.VideoWriter(str(path.with_suffix(".avi")), _fourcc("MJPG"), fps, size)
    if not vw.isOpened():
        raise RuntimeError("❌  OpenCV cannot open any MJPG writer on this system.")
    return vw


def _extract_polys(res: Any) -> list[Poly42F32]:
    """
    Return list of (4,2) float32 polygons if available; else [].
    Supports Ultralytics OBB results (xyxyxyxy) and a few common variants.
    """
    polys: list[Poly42F32] = []

    obb = getattr(res, "obb", None)
    if obb is not None:
        for key in ("xyxyxyxy", "xyxy", "xy"):
            pts = getattr(obb, key, None)
            if pts is None:
                continue
            if hasattr(pts, "cpu"):
                try:
                    pts = pts.cpu().numpy()
                except Exception:
                    pass
            arr = np.asarray(pts, dtype=np.float32)
            if arr.size == 0:
                continue
            # Normalize to (N, 8) then to (N, 4, 2)
            arr = arr.reshape(-1, 8)
            p = arr.reshape(-1, 4, 2).astype(np.float32)
            for poly in p:
                polys.append(cast(Poly42F32, poly))
            return polys  # prefer first successful representation

    # Fallback: axis-aligned boxes
    boxes = getattr(getattr(res, "boxes", None), "xyxy", None)
    if boxes is not None:
        if hasattr(boxes, "cpu"):
            try:
                boxes = boxes.cpu().numpy()
            except Exception:
                pass
        arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        for x1, y1, x2, y2 in arr:
            polys.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
    return polys


def main(  # noqa: C901
    src: str | Path,
    *,
    out_dir: str | Path | None = None,
    weights: str | Path | None = None,
    small: bool = False,
    conf: float = 0.25,
    iou: float = 0.45,
    verbose: bool = False,
    progress: SpinnerLike | None = None,
    **kw: Any,
) -> Path:
    if verbose:
        _LOG.setLevel(logging.INFO)

    srcp = Path(src).expanduser().resolve()
    if not srcp.exists():
        raise FileNotFoundError(srcp)
    if out_dir is None:
        out_dir = ROOT / "tests" / "results"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model (strict)
    override: Path | None = None
    if weights is not None:
        cand = Path(weights).expanduser().resolve()
        if not cand.exists():
            raise FileNotFoundError(f"override weight not found: {cand}")
        override = None if cand.is_dir() else cand
    if load_obb is None:
        raise RuntimeError("OBB loader is unavailable (model_registry missing).")
    obb_model: Any = load_obb(override=override)  # type: ignore[call-arg]

    cap = cv2.VideoCapture(str(srcp))
    if not cap.isOpened():
        raise RuntimeError(f"❌  Cannot open video {srcp}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 1
    _say(f"video obb: src={srcp.name} fps={fps:.3f} size={w}x{h} frames~{total}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="dv_obb_"))
    avi = tmp_dir / f"{srcp.stem}.avi"
    vw = _avi_writer(avi, fps, (w, h))

    ps_progress_active = (os.environ.get("PANOPTES_PROGRESS_ACTIVE") or "") == "1"
    use_local = (progress is None) and (ProgressEngine is not None) and (live_percent is not None) and (not ps_progress_active)  # type: ignore[truthy-bool]
    if use_local:
        eng = ProgressEngine()  # type: ignore
        ctx: ContextManager[None] = cast(ContextManager[None], live_percent(eng, prefix="OBB-MP4"))  # type: ignore
    else:
        eng = None

        class _Null:
            def __enter__(self) -> None:  # noqa: D401
                return None
            def __exit__(self, et: type[BaseException] | None, ex: BaseException | None, tb: TracebackType | None) -> bool:
                return False
        ctx = _Null()

    i = 0
    with ctx:
        if eng:
            eng.set_total(float(total + 1))
            eng.set_current("frame 0")
        elif progress is not None:
            progress.update(current=f"frame 0/{total}")

        while True:
            ok, frame_raw = cap.read()
            if not ok:
                break
            if getattr(frame_raw, "dtype", None) is not None and frame_raw.dtype != np.uint8:
                frame_raw = frame_raw.astype(np.uint8)
            frame: NDArrayU8 = cast(NDArrayU8, frame_raw)

            res_list: list[Any] = cast(list[Any], obb_model.predict(frame, imgsz=640, conf=conf, iou=iou, verbose=False))
            res = res_list[0] if res_list else None
            if res is not None:
                polys = _extract_polys(res)
                for poly in polys:
                    pts = np.asarray(poly, dtype=np.float32).reshape(4, 2).astype(np.int32)
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 210, 255), thickness=2)
            vw.write(frame)
            i += 1
            if eng:
                eng.add(1.0, current_item=f"frame {min(i, total)}/{total}")
            elif progress is not None:
                progress.update(current=f"frame {min(i, total)}/{total}")

        cap.release()
        vw.release()

        preferred = out_dir / f"{srcp.stem}_obb.mp4"
        if eng:
            eng.set_current("encode mp4")
        elif progress is not None:
            progress.update(current="encode mp4")

        try:
            if simple_status is not None and use_local:
                sp: ContextManager[None] = cast(ContextManager[None], simple_status("FFmpeg re-encode"))
            else:
                class _Null2:
                    def __enter__(self) -> None: return None
                    def __exit__(self, et: type[BaseException] | None, ex: BaseException | None, tb: TracebackType | None) -> bool: return False
                sp = _Null2()
            with sp:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(avi), "-c:v", "libx264", "-crf", "23",
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(preferred)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                )
            avi.unlink(missing_ok=True)
            final = preferred
        except (FileNotFoundError, subprocess.CalledProcessError):
            # OpenCV fallback
            ok_mp4 = False
            cap2 = cv2.VideoCapture(str(avi))
            if cap2.isOpened():
                w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(str(preferred), _fourcc("mp4v"), fps, (w2, h2))
                if out.isOpened():
                    ok_any = False
                    while True:
                        ok, fr = cap2.read()
                        if not ok:
                            break
                        out.write(fr)
                        ok_any = True
                    out.release()
                    ok_mp4 = ok_any and preferred.exists() and preferred.stat().st_size > 0
                cap2.release()
            if ok_mp4:
                avi.unlink(missing_ok=True)
                final = preferred
            else:
                final = out_dir / f"{srcp.stem}_obb.avi"
                shutil.move(str(avi), str(final))

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Saved → {_osc8(final.name, final)}")
        if eng:
            eng.add(1.0, current_item="done")
        elif progress is not None:
            progress.update(current="done")
        return final
