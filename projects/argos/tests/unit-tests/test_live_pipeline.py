from __future__ import annotations

import types
from typing import Any, cast

import pytest
import typer

from projects.argos.panoptes.live import config as live_config
from projects.argos.panoptes.live.cli import parse_imgsz
from projects.argos.panoptes.live.pipeline import LivePipeline


def _stub_hardware() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        backend="auto",
        preprocess_device="auto",
        prefer_small=True,
        input_size=None,
        ort_threads=None,
        ort_execution=None,
        nms_mode=None,
        fingerprint={},
        arch="x86_64",
        gpu=None,
        ram_gb=None,
    )


def _select_models_for_live(task: str, hw: Any) -> dict[str, str]:
    return {"label": "stub"}


def test_parse_imgsz_variants() -> None:
    assert parse_imgsz("640") == (640, 640)
    assert parse_imgsz("640x512") == (640, 512)
    assert parse_imgsz("416,320") == (416, 320)


def test_parse_imgsz_invalid() -> None:
    with pytest.raises(typer.BadParameter):
        parse_imgsz("foo")


def test_live_pipeline_autodownscale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_config, "probe_hardware", _stub_hardware)
    monkeypatch.setattr(live_config, "select_models_for_live", _select_models_for_live)
    pipeline = LivePipeline(source="synthetic", task="detect", warmup=False)
    pipeline_any = cast(Any, pipeline)
    pipeline_any._build_task = lambda: types.SimpleNamespace(label="resized")
    pipeline_any._fps_history.clear()
    max_len = pipeline_any._fps_history.maxlen or 0
    if max_len:
        pipeline_any._fps_history.extend([25.0] * max_len)
    pipeline_any._frames_since_size_change = pipeline_any._downscale_warmup_frames + 5

    class _Spinner:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def update(self, **kwargs: object) -> None:
            self.calls.append(kwargs)

    spinner = _Spinner()
    task = types.SimpleNamespace(label="original")
    new_task, new_label = pipeline_any._maybe_adjust_resolution(task, "original", spinner)

    assert pipeline.size == (512, 512)
    assert pipeline_any._current_size_index == 1
    assert len(pipeline_any._fps_history) == 0
    assert new_label == "resized"
    assert new_task is not task


def test_live_pipeline_manual_size_disables_autoscale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_config, "probe_hardware", _stub_hardware)
    monkeypatch.setattr(live_config, "select_models_for_live", _select_models_for_live)
    pipeline = LivePipeline(source="synthetic", task="detect", warmup=False, size=(800, 600))
    pipeline_any = cast(Any, pipeline)
    assert pipeline.size == (800, 600)
    assert pipeline_any._auto_schedule_enabled is False
    assert pipeline_any._resolution_schedule[0] == (800, 600)
