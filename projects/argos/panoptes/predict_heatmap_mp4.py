# \rAIn\projects\argos\panoptes\predict_heatmap_mp4.py
"""
predict_heatmap_mp4.py - per-frame segmentation / heat-map overlay for videos.

Lock-down (2025-08-07)
────────────────────────────────────────────────────────────────────
* A **segmentation** weight is mandatory - either:
      1. an explicit *weights=* argument whose filename contains “-seg”
      2. the hard-coded  WEIGHT_PRIORITY["heatmap"]  entry.

* No detector fall-back: if a segmentation model cannot be loaded the
  script aborts immediately.

* Environment variables are ignored; weight control lives exclusively
  in `panoptes.model_registry`.

Progress
────────
* If a `progress` handle is provided, we update its `current` label
  (frame i/N → encode → done) and DO NOT create a local spinner.
* Otherwise: live percent spinner (frames processed / total) via local engine.

Output noise
────────────
* Quiet by default (WARNING). Pass `verbose=true` on the CLI to get INFO logs.

Compatibility
─────────────
* FFmpeg is preferred for robust MP4 output.
* If FFmpeg is missing or fails, we **fall back to OpenCV** to re-encode the
  temporary MJPG stream into an `.mp4`. As a final last-resort we keep `.avi`.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, ContextManager, Protocol, Tuple, Union, cast

import cv2
import numpy as np
from PIL import Image

import panoptes.heatmap as _hm  # type: ignore
from panoptes import ROOT  # type: ignore
from panoptes.heatmap import heatmap_overlay  # type: ignore
from panoptes.model_registry import load_segmenter  # type: ignore

# optional progress (safe off-TTY)
try:
    from panoptes.progress import ProgressEngine  # type: ignore
    from panoptes.progress.bridges import live_percent  # type: ignore
    from panoptes.progress.progress_ux import simple_status  # type: ignore
except Exception:  # pragma: no cover
    ProgressEngine = None  # type: ignore
    live_percent = None  # type: ignore
    simple_status = None  # type: ignore

# NOTE: Ultralytics is not required for segmentation-only heatmaps.
# We keep this import guarded so environments with it don't fail.
try:  # pragma: no cover
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

# ───────────────────────── logging ──────────────────────────
_LOG = logging.getLogger("panoptes.predict_heatmap_mp4")
if not _LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(h)
# Quiet by default; flipped to INFO if verbose=True is passed to main()
_LOG.setLevel(logging.WARNING)


def _say(msg: str) -> None:
    _LOG.info(f"[panoptes] {msg}")


# ───────────────────────── helpers ──────────────────────────
CLIVal = Union[bool, int, float, str]

class SpinnerLike(Protocol):
    def update(self, **kwargs: Any) -> "SpinnerLike": ...


def _osc8(label: str, path: Path) -> str:
    """
    Render a clickable filename label that links to the absolute file path (OSC-8).
    If the terminal doesn't support OSC-8, the raw label still renders harmlessly.
    """
    try:
        uri = path.resolve().as_uri()
    except Exception:
        # best-effort file:// URL on odd platforms
        uri = "file:///" + str(path.resolve()).replace("\\", "/")
    esc = "\033"
    return f"{esc}]8;;{uri}{esc}\\{label}{esc}]8;;{esc}\\"


def _fourcc(code: str) -> int:
    """
    Return an int FOURCC code while keeping type-checkers happy.
    Falls back to cv2.VideoWriter.fourcc if VideoWriter_fourcc is missing in stubs.
    """
    fn: Callable[..., Any] = getattr(cv2, "VideoWriter_fourcc", getattr(cv2.VideoWriter, "fourcc"))
    return cast(int, fn(*code))


def _avi_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    vw = cv2.VideoWriter(
        str(path.with_suffix(".avi")),
        _fourcc("MJPG"),
        fps,
        size,
    )
    if not vw.isOpened():
        raise RuntimeError("❌  OpenCV cannot open any MJPG writer on this system.")
    return vw


def _opencv_reencode_to_mp4(src_avi: Path, dst_mp4: Path, fps: float) -> bool:
    """
    Best-effort re-encode AVI → MP4 using OpenCV only (for systems without FFmpeg).
    Returns True on success (MP4 created), False otherwise.
    """
    cap = cv2.VideoCapture(str(src_avi))
    if not cap.isOpened():
        return False
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # mp4v is widely supported in OpenCV builds; avc1 may be unavailable
    fourcc: int = _fourcc("mp4v")
    out = cv2.VideoWriter(str(dst_mp4), fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release()
        return False

    ok_any = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ok_any = True
        out.write(frame)

    cap.release()
    out.release()
    try:
        return ok_any and dst_mp4.exists() and dst_mp4.stat().st_size > 0
    except Exception:
        return False


# ───────────────────────── main worker ─────────────────────
def main(  # noqa: C901 - unavoidable CLI glue
    src: str | Path,
    *,
    weights: str | Path | None = None,
    cmap: str = "COLORMAP_JET",
    alpha: float = 0.4,
    kernel_scale: float = 5.0,  # unused, kept for flag-compat
    out_dir: str | Path | None = None,
    verbose: bool = False,       # NEW: quiet by default, INFO when True
    progress: SpinnerLike | None = None,  # NEW: nudge parent spinner if provided
    **kw: Any,
) -> Path:
    """
    Overlay segmentation heat-map on *src* video and emit <stem>_heat.mp4> (or .avi if everything else fails).
    If *progress* is passed, updates its `current` label with "frame i/N", "encode mp4", "done".
    """
    # Ultralytics YOLO is optional; segmentation path does not require it.
    # (Do NOT raise here; tests/CI may run without ultralytics installed.)

    if verbose:
        _LOG.setLevel(logging.INFO)

    # ── path normalisation ────────────────────────────────────────────
    src_path = Path(src).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    if out_dir is None:
        out_dir = ROOT / "tests" / "results"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load segmentation model (strict) ─────────────────────────────
    override_path: Path | None = None
    if weights:
        override_path = Path(weights).expanduser().resolve()
        if not override_path.exists():
            raise FileNotFoundError(f"override weight not found: {override_path}")
    seg_model = load_segmenter(override=override_path)  # registry logs weight

    # propagate into heatmap module so subsequent calls are instant
    _hm._seg_model = seg_model  # type: ignore[attr-defined]

    # strip YOLO-specific keys that users might pass by copy-paste
    STRIP = {"weights", "conf", "imgsz", "iou", "classes", "max_det"}
    for k in STRIP:
        kw.pop(k, None)

    # ── open video stream & temp MJPG writer ─────────────────────────
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"❌  Cannot open video {src_path}")

    # OpenCV getters are typed as float in stubs; guard values at runtime
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    fps: float = fps_raw if fps_raw and fps_raw > 0 else 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tf_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = int(tf_raw) if tf_raw and tf_raw > 0 else 1  # approximate when unknown
    _say(f"video heatmap: src={src_path.name} fps={fps:.3f} size={w}x{h} frames~{total_frames}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="dv_hm_"))
    avi = tmp_dir / f"{src_path.stem}.avi"
    vw = _avi_writer(avi, fps, (w, h))

    # progress: prefer parent spinner if provided; else local engine (unless suppressed)
    ps_progress_active = os.environ.get("PANOPTES_PROGRESS_ACTIVE") == "1"
    use_local = (progress is None) and (not ps_progress_active)
    eng = ProgressEngine() if (ProgressEngine is not None and use_local) else None  # type: ignore[truthy-bool]
    if use_local and live_percent is not None and eng is not None:  # type: ignore[truthy-bool]
        ctx: ContextManager[None] = cast(ContextManager[None], live_percent(eng, prefix="HEATMAP-MP4"))
    else:
        class _Null:
            def __enter__(self) -> None: return None
            def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> bool: return False
        ctx: ContextManager[None] = _Null()

    frame_idx = 0
    with ctx:
        if eng:
            # +1 step reserved for the encode phase
            eng.set_total(float(total_frames + 1))
            eng.set_current("frame 0")
        elif progress is not None:
            progress.update(current=f"frame 0/{total_frames}")

        # ── frame loop ───────────────────────────────────────────────────
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ov = heatmap_overlay(
                Image.fromarray(frame[:, :, ::-1]),
                boxes=None,  # true seg masks only
                alpha=alpha,
                cmap=cmap,
                kernel_scale=kernel_scale,
                **kw,
            )

            if isinstance(ov, Image.Image):
                out_bgr = np.asarray(ov)[:, :, ::-1]
            else:  # ndarray returned (already BGR)
                out_bgr = ov
            vw.write(np.asarray(out_bgr, dtype=np.uint8))

            frame_idx += 1
            if eng:
                eng.add(1.0, current_item=f"frame {min(frame_idx, total_frames)}/{total_frames}")
            elif progress is not None:
                progress.update(current=f"frame {min(frame_idx, total_frames)}/{total_frames}")

        cap.release()
        vw.release()

        # ── re-encode MJPG → H.264 MP4 ───────────────────────────────────
        preferred = out_dir / f"{src_path.stem}_heat.mp4"
        final: Path
        if eng:
            eng.set_current("encode mp4")
        elif progress is not None:
            progress.update(current="encode mp4")

        # Try FFmpeg first (if available)
        ffmpeg_exists = shutil.which("ffmpeg") is not None
        try:
            if simple_status is not None and not ps_progress_active and use_local:
                sp: ContextManager[None] = cast(ContextManager[None], simple_status("FFmpeg re-encode"))
            else:
                class _Null2:
                    def __enter__(self) -> None: return None
                    def __exit__(self, et: type[BaseException] | None, ex: BaseException | None, tb: TracebackType | None) -> bool: return False
                sp = _Null2()

            if ffmpeg_exists:
                with sp:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i", str(avi),
                            "-c:v", "libx264",
                            "-crf", "23",
                            "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart",
                            str(preferred),
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                    )
                # Success path
                avi.unlink(missing_ok=True)
                final = preferred
            else:
                raise FileNotFoundError("ffmpeg not found")
        except (FileNotFoundError, subprocess.CalledProcessError):
            # FFmpeg missing or failed - try OpenCV re-encode to MP4
            ok_mp4 = _opencv_reencode_to_mp4(avi, preferred, fps=fps)
            if ok_mp4:
                avi.unlink(missing_ok=True)
                final = preferred
            else:
                # Final fallback: ship AVI instead of lying with .mp4
                final = out_dir / f"{src_path.stem}_heat.avi"
                shutil.move(str(avi), str(final))

        shutil.rmtree(tmp_dir, ignore_errors=True)

        # One clean summary line with a clickable filename label
        print(f"Saved → {_osc8(final.name, final)}")
        if eng:
            eng.add(1.0, current_item="done")
        elif progress is not None:
            progress.update(current="done")
        return final


# ───────────────────────── CLI entry-point ──────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m panoptes.predict_heatmap_mp4 <video> [k=v …]")

    # simple k=v parsing to stay dependency-free
    def _auto_cli(s: str) -> CLIVal:
        s = s.strip()
        if s.lower() in {"true", "false"}:
            return s.lower() == "true"
        try:
            return int(s) if s.isdigit() else float(s)
        except ValueError:
            return s

    kv_pairs = (arg.split("=", 1) for arg in sys.argv[2:])
    args: dict[str, CLIVal] = {k: _auto_cli(v) for k, v in kv_pairs}

    weights_arg = args.pop("weights", None)
    out_dir_arg = args.pop("out_dir", None)
    cmap_arg = str(args.pop("cmap", "COLORMAP_JET"))
    alpha_arg = float(args.pop("alpha", 0.4))
    kscale_arg = float(args.pop("kernel_scale", 5.0))
    verbose_arg = bool(args.pop("verbose", False))

    # Remove unsupported/typed-only CLI keys to avoid binding to typed parameters.
    args.pop("progress", None)

    # Note: CLI `__main__` path doesn’t have a spinner to pass; local engine is used.
    main(
        sys.argv[1],
        weights=str(weights_arg) if weights_arg is not None else None,
        cmap=cmap_arg,
        alpha=alpha_arg,
        kernel_scale=kscale_arg,
        out_dir=str(out_dir_arg) if out_dir_arg is not None else None,
        verbose=verbose_arg,
        progress=None,
        **args,
    )
