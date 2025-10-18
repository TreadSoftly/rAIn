from pathlib import Path

import panoptes.model_registry as mr  # type: ignore[import]
from pytest import MonkeyPatch


class DummyModel:
    pass


def test_load_detector_falls_back_to_next_weight(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    bad_weight = tmp_path / "bad.onnx"
    bad_weight.write_text("x", encoding="utf-8")
    good_weight = tmp_path / "good.pt"
    good_weight.write_text("x", encoding="utf-8")

    monkeypatch.setitem(mr.WEIGHT_PRIORITY, "detect_small", [bad_weight])
    monkeypatch.setitem(mr.WEIGHT_PRIORITY, "detect", [good_weight])

    calls: list[tuple[Path, str]] = []

    def fake_load(weight: Path, *, task: str):
        calls.append((Path(weight), task))
        if str(weight).endswith("bad.onnx"):
            raise RuntimeError("onnxruntime missing")
        return DummyModel()

    monkeypatch.setattr(mr, "_load", fake_load)

    model = mr.load_detector(small=True)

    assert isinstance(model, DummyModel)
    assert [p.resolve() for p, _ in calls] == [bad_weight.resolve(), good_weight.resolve()]
    assert [t for _, t in calls] == ["detect", "detect"]
