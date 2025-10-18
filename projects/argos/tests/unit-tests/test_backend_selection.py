import platform
import sys
from pathlib import Path

import panoptes.model_registry as mr
from panoptes.runtime.onnx_spec import desired_ort_spec


def _write_weight(path: Path) -> Path:
    path.write_text("stub", encoding="utf-8")
    return path


def test_candidate_weights_skips_onnx_when_ort_unavailable(monkeypatch, tmp_path):
    onnx = _write_weight(tmp_path / "model.onnx")
    pt = _write_weight(tmp_path / "model.pt")

    monkeypatch.setitem(mr.WEIGHT_PRIORITY, "pose_small", [onnx, pt])
    monkeypatch.setattr(mr, "ort_available", lambda: (False, None, None, "ImportError: missing"))

    candidates = mr.candidate_weights("pose", small=True)
    assert candidates == [pt]


def test_candidate_weights_prefers_pt_when_env_disables_onnx(monkeypatch, tmp_path):
    onnx = _write_weight(tmp_path / "model.onnx")
    pt = _write_weight(tmp_path / "model.pt")

    monkeypatch.setitem(mr.WEIGHT_PRIORITY, "detect", [onnx, pt])
    monkeypatch.setattr(mr, "ort_available", lambda: (True, "1.22.1", ["CPUExecutionProvider"], None))
    monkeypatch.setenv("ARGOS_PREFER_ONNX", "0")

    candidates = mr.candidate_weights("detect", small=False)
    assert candidates[0] == pt


def test_candidate_weights_prefers_onnx_by_default(monkeypatch, tmp_path):
    onnx = _write_weight(tmp_path / "model.onnx")
    pt = _write_weight(tmp_path / "model.pt")

    monkeypatch.setitem(mr.WEIGHT_PRIORITY, "heatmap_small", [onnx, pt])
    monkeypatch.setattr(mr, "ort_available", lambda: (True, "1.22.1", ["CPUExecutionProvider"], None))
    monkeypatch.delenv("ARGOS_PREFER_ONNX", raising=False)

    candidates = mr.candidate_weights("heatmap", small=True)
    assert candidates[0] == onnx


def test_ort_available_failure(monkeypatch):
    import panoptes.runtime.backend_probe as bp

    monkeypatch.delenv("ARGOS_DISABLE_ONNX", raising=False)
    monkeypatch.setattr(bp, "_try_import_ort", lambda: (False, None, None, "ImportError: missing"))
    monkeypatch.setattr(bp, "_BOOTSTRAP", None)

    ok, version, providers, reason = bp.ort_available()
    assert ok is False
    assert version is None
    assert providers is None
    assert reason == "ImportError: missing"


def test_desired_ort_spec_matches_packaging():
    spec = desired_ort_spec()
    if sys.version_info >= (3, 10):
        expected = "onnxruntime>=1.22,<1.23" if platform.system() == "Windows" else "onnxruntime>=1.22,<1.24"
    else:
        expected = "onnxruntime==1.19.2"
    assert spec == expected
