# projects/argos/panoptes/predict_classify_mp4.py
'''
predict_classify_mp4 - per-frame classification overlay for videos.

ONE PROGRESS SYSTEM
-------------------
* Uses ONLY the Halo/Rich spinner from panoptes.progress.
* If a parent spinner is passed (recommended via CLI), we ONLY update its
  [File | Job | Model] fields (no 'current' writes) so the single-line
  format and colors exactly match Detect/Heatmap/GeoJSON.
* If no parent spinner is provided AND no global spinner is marked active
  (PANOPTES_PROGRESS_ACTIVE != "1"), a local Halo spinner is started so
  standalone runs still have a single consistent line.

Strict weights
--------------
* Loads strictly via live.tasks.build_classify (which honours panoptes.model_registry overrides).

Output
------
* Prefers H.264 via FFmpeg -> OpenCV mp4v fallback -> keeps MJPG .avi last.
* Emits "<stem>_cls.mp4" (or ".avi" as last resort).
* Prints exactly one "Saved -> <clickable>" line at the end (OSC-8).
'''

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Optional, Protocol, Sequence, Tuple, Union, cast

import cv2  # type: ignore
import numpy as np
from numpy.typing import NDArray

from panoptes import ROOT  # type: ignore[import-not-found]
from .ffmpeg_utils import resolve_ffmpeg
from .live import tasks as live_tasks

NDArrayU8 = NDArray[np.uint8]

try:
    from panoptes.progress import percent_spinner  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    percent_spinner = None  # type: ignore[assignment]

_LOG = logging.getLogger("panoptes.predict_classify_mp4")
_LOG.addHandler(logging.NullHandler())
_LOG.setLevel(logging.ERROR)


def _say(msg: str) -> None:
    if _LOG.isEnabledFor(logging.INFO):
        _LOG.info("[panoptes] %s", msg)


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
    return fn(*code)


def _avi_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    vw = cv2.VideoWriter(str(path.with_suffix(".avi")), _fourcc("MJPG"), fps, size)
    if not vw.isOpened():
        raise RuntimeError("?  OpenCV cannot open any MJPG writer on this system.")
    return vw


def _opencv_reencode_to_mp4(src_avi: Path, dst_mp4: Path, fps: float) -> bool:
    cap = cv2.VideoCapture(str(src_avi))
    if not cap.isOpened():
        return False
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(dst_mp4), _fourcc("mp4v"), fps, (w, h))
    if not out.isOpened():
        cap.release()
        return False
    ok_any = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
        ok_any = True
    cap.release()
    out.release()
    try:
        return ok_any and dst_mp4.exists() and dst_mp4.stat().st_size > 0
    except Exception:
        return False


def _poke(sp: SpinnerLike | None, **fields: Any) -> None:
    if sp is None:
        return
    try:
        sp.update(**fields)
    except Exception:
        pass


def _parse_class_list(raw: Optional[str]) -> Optional[Tuple[str, ...]]:
    if raw is None:
        return None
    parts = [item.strip() for item in raw.split(",")]
    filtered = [item for item in parts if item]
    return tuple(filtered) if filtered else None


def load_classifier(
    *,
    small: bool = False,
    override: Optional[Path] = None,
    topk: int = 1,
    smooth_probs: bool = False,
    class_whitelist: Optional[Sequence[str]] = None,
    class_blacklist: Optional[Sequence[str]] = None,
) -> Any:
    """
    Compatibility loader for tests and legacy callers.

    Internally delegates to ``live.tasks.build_classify``.
    """
    whitelist_seq = tuple(class_whitelist) if class_whitelist else None
    blacklist_seq = tuple(class_blacklist) if class_blacklist else None
    return live_tasks.build_classify(
        small=small,
        topk=topk,
        override=override,
        smooth_probs=smooth_probs,
        class_whitelist=whitelist_seq,
        class_blacklist=blacklist_seq,
    )


def main(
    src: str | Path,
    *,
    out_dir: str | Path | None = None,
    weights: str | Path | None = None,
    small: bool = False,     # reserved; not used directly here
    conf: float | None = None,
    topk: int = 1,
    smooth_probs: bool = False,
    class_whitelist: Optional[Sequence[str]] = None,
    class_blacklist: Optional[Sequence[str]] = None,
    verbose: bool = False,
    progress: SpinnerLike | None = None,
    **_kw: Any,
) -> Path:
    if verbose:
        _LOG.setLevel(logging.INFO)

    src_path = Path(src).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    if out_dir is None:
        out_dir = ROOT / "tests" / "results"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    override_path: Optional[Path] = None
    if weights is not None:
        cand = Path(weights).expanduser().resolve()
        if not cand.exists():
            raise FileNotFoundError(f"override weight not found: {cand}")
        override_path = None if cand.is_dir() else cand

    classify_adapter = load_classifier(
        small=small,
        topk=topk,
        override=override_path,
        smooth_probs=smooth_probs,
        class_whitelist=class_whitelist,
        class_blacklist=class_blacklist,
    )
    model_lbl = classify_adapter.current_label()

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"?  Cannot open video {src_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 1
    _say(f"video classify: src={src_path.name} fps={fps:.3f} size={w}x{h} frames~{total}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="dv_cls_"))
    avi = tmp_dir / f"{src_path.stem}.avi"
    vw = _avi_writer(avi, fps, (w, h))

    parent_active = (os.environ.get("PANOPTES_PROGRESS_ACTIVE") or "").strip() == "1"
    use_local = (progress is None) and (not parent_active) and (percent_spinner is not None)
    spinner_ctx: ContextManager[SpinnerLike | None]
    if use_local and percent_spinner is not None:
        spinner_ctx = cast(ContextManager[SpinnerLike], percent_spinner(prefix="CLASSIFY-MP4", stream=sys.stderr))
    else:
        spinner_ctx = nullcontext(None)

    frame_idx = 0
    file_lbl = src_path.name
    with spinner_ctx as spinner:
        if use_local and spinner is not None:
            _poke(spinner, total=float(total + 1), count=0.0, item=file_lbl, job=f"frame 0/{total}", model=model_lbl)
        else:
            _poke(progress, item=file_lbl, job=f"frame 0/{total}", model=model_lbl, total=total + 1, count=0)

        while True:
            ok, frame_raw = cap.read()
            if not ok:
                break
            frame_bgr: NDArrayU8 = np.asarray(frame_raw, dtype=np.uint8)
            results = classify_adapter.infer(frame_bgr)
            annotated = classify_adapter.render(frame_bgr, results)
            vw.write(np.asarray(annotated, dtype=np.uint8))
            frame_idx += 1

            if use_local and spinner is not None:
                _poke(
                    spinner,
                    count=float(frame_idx),
                    item=file_lbl,
                    job=f"frame {min(frame_idx, total)}/{total}",
                    model=model_lbl,
                )
            else:
                _poke(
                    progress,
                    item=file_lbl,
                    job=f"frame {min(frame_idx, total)}/{total}",
                    model=model_lbl,
                    total=total + 1,
                    count=min(frame_idx, total),
                )

        cap.release()
        vw.release()

        preferred = out_dir / f"{src_path.stem}_cls.mp4"
        if use_local and spinner is not None:
            _poke(spinner, item=file_lbl, job="encode mp4", model=model_lbl)
        else:
            _poke(progress, item=file_lbl, job="encode mp4", model=model_lbl, total=total + 1, count=total)

        ffmpeg_path, ffmpeg_source = resolve_ffmpeg()
        try:
            if not ffmpeg_path:
                raise FileNotFoundError("ffmpeg not found")
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    str(avi),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(preferred),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            avi.unlink(missing_ok=True)
            _say(f"FFmpeg re-encode successful ({ffmpeg_source})")
            final = preferred
        except (FileNotFoundError, subprocess.CalledProcessError):
            ok_mp4 = _opencv_reencode_to_mp4(avi, preferred, fps=fps)
            if ok_mp4:
                avi.unlink(missing_ok=True)
                final = preferred
            else:
                final = out_dir / f"{src_path.stem}_cls.avi"
                shutil.move(str(avi), str(final))

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Saved -> {_osc8(final.name, final)}")

        if use_local and spinner is not None:
            _poke(spinner, count=float(total + 1), item=file_lbl, job="done", model=model_lbl)
        else:
            _poke(progress, item=file_lbl, job="done", model=model_lbl, total=total + 1, count=total + 1)

        try:
            classify_adapter.reset_temporal_state()  # type: ignore[attr-defined]
        except Exception:
            pass

        return final


CLIVal = Union[bool, int, float, str]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m panoptes.predict_classify_mp4 <video> [k=v ...]")

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

    out_dir_arg = args.pop("out_dir", None)
    weights_arg = args.pop("weights", None)
    topk_arg = int(args.pop("topk", 1))
    smooth_probs_arg = bool(args.pop("smooth_probs", False))
    whitelist_arg = args.pop("class_whitelist", None)
    blacklist_arg = args.pop("class_blacklist", None)
    verbose_arg = bool(args.pop("verbose", False))

    whitelist_seq = _parse_class_list(str(whitelist_arg)) if whitelist_arg is not None else None
    blacklist_seq = _parse_class_list(str(blacklist_arg)) if blacklist_arg is not None else None

    main(
        sys.argv[1],
        out_dir=str(out_dir_arg) if out_dir_arg is not None else None,
        weights=str(weights_arg) if weights_arg is not None else None,
        topk=topk_arg,
        smooth_probs=smooth_probs_arg,
        class_whitelist=whitelist_seq,
        class_blacklist=blacklist_seq,
        verbose=verbose_arg,
        progress=None,
    )


