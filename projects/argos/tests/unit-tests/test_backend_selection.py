from pathlib import Path

import panoptes.model_registry as mr


def _write_weight(path: Path) -> Path:
    path.write_text("stub", encoding="utf-8")
    return path


def test_candidate_weights_skips_onnx_when_ort_unavailable(monkeypatch, tmp_path):
    onnx = _write_weight(tmp_path / "model.onnx")
    pt = _write_weight(tmp_path / "model.pt")

    monkeypatch.setitem(mr.WEIGHT_PRIORITY, "pose_small", [onnx, pt])
    monkeypatch.setattr(mr, "ort_available", lambda: (False, "ImportError: missing"))

    candidates = mr.candidate_weights("pose", small=True)
    assert candidates == [pt]


def test_candidate_weights_prefers_pt_when_env_disables_onnx(monkeypatch, tmp_path):
    onnx = _write_weight(tmp_path / "model.onnx")
    pt = _write_weight(tmp_path / "model.pt")

    monkeypatch.setitem(mr.WEIGHT_PRIORITY, "detect", [onnx, pt])
    monkeypatch.setattr(mr, "ort_available", lambda: (True, "CPUExecutionProvider"))
    monkeypatch.setenv("ARGOS_PREFER_ONNX", "0")

    candidates = mr.candidate_weights("detect", small=False)
    assert candidates[0] == pt


def test_candidate_weights_prefers_onnx_by_default(monkeypatch, tmp_path):
    onnx = _write_weight(tmp_path / "model.onnx")
    pt = _write_weight(tmp_path / "model.pt")

    monkeypatch.setitem(mr.WEIGHT_PRIORITY, "heatmap_small", [onnx, pt])
    monkeypatch.setattr(mr, "ort_available", lambda: (True, "CPUExecutionProvider"))
    monkeypatch.delenv("ARGOS_PREFER_ONNX", raising=False)

    candidates = mr.candidate_weights("heatmap", small=True)
    assert candidates[0] == onnx
