# projects/argos/tests/unit-tests/test_cli_tasks_new_videos.py
from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
import pytest

# OpenCV is required for reading/writing frames and for text overlays
cv2 = pytest.importorskip("cv2")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _fake_cls_model() -> Any:
    class Adapter:
        names = {0: "cat", 1: "dog"}
        label = "fake-classifier"

        def current_label(self) -> str:
            return self.label

        def infer(self, frame_bgr: npt.NDArray[np.uint8]) -> list[tuple[str, float]]:
            return [("cat", 0.7), ("dog", 0.3)]

        def render(self, frame_bgr: npt.NDArray[np.uint8], result: list[tuple[str, float]]) -> npt.NDArray[np.uint8]:
            # Return the frame unchanged (tests only verify output file exists).
            return frame_bgr

        def reset_temporal_state(self) -> None:
            pass

    return Adapter()


def _fake_pose_model() -> Any:
    class M:
        names: dict[int, str] = {}

        def predict(
            self,
            frame: npt.NDArray[np.uint8] | Any,
            imgsz: int = 640,
            conf: float = 0.25,
            iou: float = 0.45,
            verbose: bool = False,
        ) -> list[types.SimpleNamespace]:
            kps: npt.NDArray[np.float32] = np.zeros((1, 17, 2), dtype=np.float32)
            h = int(frame.shape[0])
            w = int(frame.shape[1])
            kps[0, :, 0] = np.linspace(8, w - 8, 17, dtype=np.float32)
            kps[0, :, 1] = np.linspace(8, h - 8, 17, dtype=np.float32)
            return [types.SimpleNamespace(keypoints=types.SimpleNamespace(xy=kps))]

    return M()


def _fake_obb_model() -> Any:
    class M:
        names: dict[int, str] = {}

        def predict(
            self,
            frame: npt.NDArray[np.uint8] | Any,
            imgsz: int = 640,
            conf: float = 0.25,
            iou: float = 0.45,
            verbose: bool = False,
        ) -> list[types.SimpleNamespace]:
            h = int(frame.shape[0])
            w = int(frame.shape[1])
            poly: npt.NDArray[np.float32] = np.array(
                [[10, 10, w - 10, 10, w - 12, h - 10, 12, h - 12]], dtype=np.float32
            )
            obb = types.SimpleNamespace(xyxyxyxy=poly)
            return [types.SimpleNamespace(obb=obb)]

    return M()


def _fourcc(code: str) -> int:
    fn = getattr(cv2, "VideoWriter_fourcc", getattr(cv2.VideoWriter, "fourcc"))
    return int(fn(*code))


def _make_video(
    tmp_path: Path, name: str = "toy.mp4", size: Tuple[int, int] = (96, 72), frames: int = 12
) -> Path:
    path = tmp_path / name
    vw = cv2.VideoWriter(str(path), _fourcc("mp4v"), 10.0, size)
    assert vw.isOpened(), "OpenCV cannot open a writer"
    for i in range(frames):
        img: npt.NDArray[np.uint8] = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.rectangle(img, (4 + i, 8), (size[0] - 4, size[1] - 8), (0, 255, 0), 1)
        vw.write(img)
    vw.release()
    assert path.exists()
    return path


def _mk_fake_cls_loader(**kw: Any) -> Any:
    return _fake_cls_model()


def _mk_fake_pose_loader(**kw: Any) -> Any:
    return _fake_pose_model()


def _mk_fake_obb_loader(**kw: Any) -> Any:
    return _fake_obb_model()


# ── Tests: call the new MP4 workers directly (fast & stubbed) ─────────────────
def test_predict_classify_mp4_saves_cls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from panoptes import predict_classify_mp4 as mod  # type: ignore[import-not-found]
    monkeypatch.setattr(mod, "load_classifier", _mk_fake_cls_loader)  # type: ignore[arg-type]

    src = _make_video(tmp_path, "bunny.mp4")
    out = mod.main(src, out_dir=tmp_path, topk=1, verbose=False)
    assert out.suffix in {".mp4", ".avi"}
    assert out.name.startswith("bunny_cls")
    assert out.exists()


def test_predict_pose_mp4_saves_pose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from panoptes import predict_pose_mp4 as mod  # type: ignore[import-not-found]
    monkeypatch.setattr(mod, "load_pose", _mk_fake_pose_loader)  # type: ignore[arg-type]

    src = _make_video(tmp_path, "bunny.mp4")
    out = mod.main(src, out_dir=tmp_path, verbose=False)
    assert out.suffix in {".mp4", ".avi"}
    assert out.name.startswith("bunny_pose")
    assert out.exists()


def test_predict_obb_mp4_saves_obb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from panoptes import predict_obb_mp4 as mod  # type: ignore[import-not-found]
    monkeypatch.setattr(mod, "load_obb", _mk_fake_obb_loader)  # type: ignore[arg-type]

    src = _make_video(tmp_path, "bunny.mp4")
    out = mod.main(src, out_dir=tmp_path, verbose=False)
    assert out.suffix in {".mp4", ".avi"}
    assert out.name.startswith("bunny_obb")
    assert out.exists()
