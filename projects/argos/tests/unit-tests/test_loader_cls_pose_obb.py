# projects/argos/tests/unit-tests/test_loader_cls_pose_obb.py
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest


class _DummyYOLO:
    """Ultra-lightweight stand-in for ultralytics.YOLO (no torch)."""

    def __init__(self, weights: str | Path):
        self.weights = str(weights)
        self.names = {0: "x", 1: "y"}

    def predict(self, *a: Any, **kw: Any) -> list[types.SimpleNamespace]:
        # Return a minimal result shape that's friendly to our overlays
        probs = types.SimpleNamespace(data=np.array([0.6, 0.4], dtype=np.float32))
        obb = types.SimpleNamespace(
            xyxyxyxy=np.array([[10, 10, 50, 10, 50, 40, 10, 40]], dtype=np.float32)
        )
        keypoints = types.SimpleNamespace(xy=np.zeros((1, 17, 2), dtype=np.float32))
        return [types.SimpleNamespace(probs=probs, obb=obb, keypoints=keypoints)]


def _install_fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    m = types.ModuleType("ultralytics")
    m.YOLO = _DummyYOLO  # type: ignore[attr-defined]
    sys.modules["ultralytics"] = m


def test_model_registry_loaders_accept_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ultralytics(monkeypatch)
    from panoptes import model_registry as mr  # type: ignore[import-not-found]

    # Create innocent "weight files"
    w_cls = tmp_path / "w-cls.pt"
    w_cls.write_bytes(b"")
    w_pose = tmp_path / "w-pose.pt"
    w_pose.write_bytes(b"")
    w_obb = tmp_path / "w-obb.pt"
    w_obb.write_bytes(b"")

    m_cls = mr.load_classifier(override=w_cls)
    m_pose = mr.load_pose(override=w_pose)
    m_obb = mr.load_obb(override=w_obb)

    assert isinstance(m_cls, _DummyYOLO) and m_cls.weights == str(w_cls)
    assert isinstance(m_pose, _DummyYOLO) and m_pose.weights == str(w_pose)
    assert isinstance(m_obb, _DummyYOLO) and m_obb.weights == str(w_obb)


def test_classify_run_image_uses_registry_when_model_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # We don't need PIL files on disk; function accepts a numpy array too.
    img: npt.NDArray[np.uint8] = np.full((72, 96, 3), 10, dtype=np.uint8)  # BGR-ish; safe

    # Count calls to the strict registry loader
    calls: dict[str, int] = {"n": 0}

    import panoptes.classify as mod  # type: ignore[import-not-found]

    def _mk_cls_loader(**kw: Any) -> Any:
        calls["n"] = calls["n"] + 1
        return _DummyYOLO("X")

    monkeypatch.setattr(mod, "load_classifier", _mk_cls_loader)  # type: ignore[arg-type]

    out = mod.run_image(img, out_dir=tmp_path, topk=1)
    assert out.name.endswith("_cls.jpg") or out.name.endswith("_cls.png")
    assert out.exists()
    assert calls["n"] == 1


def test_pose_and_obb_image_runners_use_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cv2 = pytest.importorskip("cv2")  # overlays use cv2  # noqa: F401
    img: npt.NDArray[np.uint8] = np.full((72, 96, 3), 20, dtype=np.uint8)

    import panoptes.pose as pose_mod  # type: ignore[import-not-found]
    import panoptes.obb as obb_mod  # type: ignore[import-not-found]

    calls: dict[str, int] = {"pose": 0, "obb": 0}

    def _mk_pose_loader(**kw: Any) -> Any:
        calls["pose"] = calls["pose"] + 1
        return _DummyYOLO("P")

    def _mk_obb_loader(**kw: Any) -> Any:
        calls["obb"] = calls["obb"] + 1
        return _DummyYOLO("B")

    monkeypatch.setattr(pose_mod, "load_pose", _mk_pose_loader)  # type: ignore[arg-type]
    monkeypatch.setattr(obb_mod, "load_obb", _mk_obb_loader)  # type: ignore[arg-type]

    p1 = pose_mod.run_image(img, out_dir=tmp_path)
    p2 = obb_mod.run_image(img, out_dir=tmp_path)
    assert p1.name.endswith("_pose.jpg") or p1.name.endswith("_pose.png")
    assert p2.name.endswith("_obb.jpg") or p2.name.endswith("_obb.png")
    assert p1.exists() and p2.exists()
    assert calls["pose"] == 1 and calls["obb"] == 1
