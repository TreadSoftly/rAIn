from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Type

from panoptes.runtime.resilient_yolo import ResilientYOLO  # type: ignore[import]
from pytest import MonkeyPatch


def _install_dummy_ultralytics(monkeypatch: MonkeyPatch, cls: Type[Any]) -> None:
    module = types.SimpleNamespace(YOLO=cls)
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def test_resilient_fallback_on_init_failure(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    class DummyYOLO:
        def __init__(self, weight: Path, *args: Any, **kwargs: Any):
            self.weight = Path(weight)
            if self.weight.suffix == ".onnx":
                raise ImportError("onnxruntime missing")
            self.names = {0: "dummy"}
            self.device = "cpu"

        def predict(self, frame: Any, **kwargs: Any):
            return ["ok"]

    _install_dummy_ultralytics(monkeypatch, DummyYOLO)

    bad = tmp_path / "bad.onnx"
    bad.write_text("x", encoding="utf-8")
    good = tmp_path / "good.pt"
    good.write_text("x", encoding="utf-8")

    toasts: list[str] = []
    wrapper = ResilientYOLO([bad, good], task="detect", conf=0.25, on_switch=toasts.append)
    wrapper.prepare()

    assert wrapper.backend == "torch"
    assert wrapper.weight_label == good.name
    assert toasts and "Switched backend" in toasts[0]


def test_resilient_runtime_retry(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    class DummyYOLO:
        def __init__(self, weight: Path, *args: Any, **kwargs: Any):
            self.weight = Path(weight)
            self.names = {0: "dummy"}
            self.device = "cpu"

        def predict(self, frame: Any, **kwargs: Any):
            if self.weight.suffix == ".onnx":
                raise ImportError("onnxruntime missing")
            return ["ok"]

    _install_dummy_ultralytics(monkeypatch, DummyYOLO)

    bad = tmp_path / "bad.onnx"
    bad.write_text("x", encoding="utf-8")
    good = tmp_path / "good.pt"
    good.write_text("x", encoding="utf-8")

    toasts: list[str] = []
    wrapper = ResilientYOLO([bad, good], task="detect", conf=0.25, on_switch=toasts.append)
    wrapper.prepare()

    # First backend is ONNX, but predict should fall back to Torch
    assert wrapper.backend == "onnxruntime"
    wrapper.predict("frame")
    assert wrapper.backend == "torch"
    assert toasts and any("Switched backend" in msg for msg in toasts)
