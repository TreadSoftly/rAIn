# projects/argos/tests/unit-tests/test_cli_tasks_new_images.py
from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

# We exercise the *image* overlays directly (fast) and stub the model loaders.


# ── Helpers: tiny fake models ──────────────────────────────────────────────────
def _fake_cls_model() -> Any:
    class M:
        names = {0: "cat", 1: "dog"}

        def predict(
            self,
            img: npt.NDArray[np.uint8] | Any,
            imgsz: int = 640,
            conf: float = 0.0,
            verbose: bool = False,
        ) -> list[types.SimpleNamespace]:
            # Return one result with "probs" vector (top-1 = cat)
            probs = types.SimpleNamespace(data=np.array([0.85, 0.15], dtype=np.float32))
            return [types.SimpleNamespace(probs=probs)]

    return M()


def _fake_pose_model() -> Any:
    class M:
        names: dict[int, str] = {}

        def predict(
            self,
            img: npt.NDArray[np.uint8] | Any,
            imgsz: int = 640,
            conf: float = 0.25,
            iou: float = 0.45,
            verbose: bool = False,
        ) -> list[types.SimpleNamespace]:
            # 1 instance, 17 keypoints (COCO-like topology)
            kps: npt.NDArray[np.float32] = np.zeros((1, 17, 2), dtype=np.float32)
            kps[0, :, 0] = np.linspace(8, 80, 17)  # x series
            kps[0, :, 1] = np.linspace(8, 60, 17)  # y series
            return [types.SimpleNamespace(keypoints=types.SimpleNamespace(xy=kps))]

    return M()


def _fake_obb_model() -> Any:
    class M:
        names: dict[int, str] = {}

        def predict(
            self,
            img: npt.NDArray[np.uint8] | Any,
            imgsz: int = 640,
            conf: float = 0.25,
            iou: float = 0.45,
            verbose: bool = False,
        ) -> list[types.SimpleNamespace]:
            # 1 oriented quad in "xyxyxyxy" format
            poly: npt.NDArray[np.float32] = np.array(
                [[20, 15, 75, 20, 70, 55, 15, 50]], dtype=np.float32
            )
            obb = types.SimpleNamespace(xyxyxyxy=poly)
            return [types.SimpleNamespace(obb=obb)]

    return M()


# ── Tests: run each overlay on a solid 96x72 RGB image ─────────────────────────
def _make_image(tmp_path: Path) -> Path:
    p = tmp_path / "toy.png"
    Image.new("RGB", (96, 72), "black").save(p)
    return p


def _mk_fake_cls_loader(**kw: Any) -> Any:
    return _fake_cls_model()


def _mk_fake_pose_loader(**kw: Any) -> Any:
    return _fake_pose_model()


def _mk_fake_obb_loader(**kw: Any) -> Any:
    return _fake_obb_model()


def test_classify_image_overlay_saves_suffix_cls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patch the strict registry entry point used by panoptes.classify
    import panoptes.classify as mod  # type: ignore[import-not-found]

    monkeypatch.setattr(mod, "load_classifier", _mk_fake_cls_loader)  # type: ignore[arg-type]

    src = _make_image(tmp_path)
    out = mod.run_image(src, out_dir=tmp_path, topk=1)
    assert out.name == f"{src.stem}_cls{out.suffix}"
    assert out.exists()


def test_pose_image_overlay_saves_suffix_pose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cv2 = pytest.importorskip("cv2")  # drawing uses OpenCV  # noqa: F401
    import panoptes.pose as mod  # type: ignore[import-not-found]

    monkeypatch.setattr(mod, "load_pose", _mk_fake_pose_loader)  # type: ignore[arg-type]

    src = _make_image(tmp_path)
    out = mod.run_image(src, out_dir=tmp_path)
    assert out.name == f"{src.stem}_pose{out.suffix}"
    assert out.exists()


def test_obb_image_overlay_saves_suffix_obb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cv2 = pytest.importorskip("cv2")  # drawing uses OpenCV  # noqa: F401
    import panoptes.obb as mod  # type: ignore[import-not-found]

    monkeypatch.setattr(mod, "load_obb", _mk_fake_obb_loader)  # type: ignore[arg-type]

    src = _make_image(tmp_path)
    out = mod.run_image(src, out_dir=tmp_path)
    assert out.name == f"{src.stem}_obb{out.suffix}"
    assert out.exists()
