from __future__ import annotations

import numpy as np

from geoseg.pipeline.segment import run_segmentation_stage


def test_run_segmentation_stage_uses_stage_helpers(monkeypatch):
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "geoseg.pipeline.segment.maybe_skip_tiny_image",
        lambda img_rgb, review_warnings: calls.append(("tiny", img_rgb.shape)) or None,
    )
    monkeypatch.setattr(
        "geoseg.pipeline.segment.classify_figure_stage",
        lambda *args, **kwargs: ({"figure_type": "conceptual_model"}, None),
    )
    monkeypatch.setattr(
        "geoseg.pipeline.segment.detect_panels_stage",
        lambda img_rgb: calls.append(("detect", img_rgb.shape)) or [{"id": 0, "bbox": (0, 0, 40, 40)}],
    )
    monkeypatch.setattr(
        "geoseg.pipeline.segment.review_figure_stage",
        lambda *args, **kwargs: (
            [{"id": 0, "bbox": (0, 0, 40, 40)}],
            [],
            False,
            -1,
        ),
    )
    monkeypatch.setattr(
        "geoseg.pipeline.segment.resolve_target_panel_stage",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        "geoseg.pipeline.segment.segment_panel_stage",
        lambda *args, **kwargs: (
            {
                "panel_id": 0,
                "bbox": [0, 0, 40, 40],
                "classification": {"figure_type": "conceptual_model"},
                "segmentation": {
                    "labels": np.ones((40, 40), dtype=np.int32),
                    "overlay": None,
                    "meta": {"engine": "mock"},
                },
                "review": {"n_layers_found": 1, "is_target_panel": True},
            },
            1,
            "mock",
        ),
    )
    monkeypatch.setattr(
        "geoseg.pipeline.segment.summarize_pipeline_result",
        lambda *args, **kwargs: {
            "classification": {"figure_type": "conceptual_model"},
            "panels": [],
            "summary": {"status": "ok"},
        },
    )

    result = run_segmentation_stage(
        img,
        caption="caption",
        text_blocks=[{"text": "caption"}],
        n_layers=3,
        skip_non_velocity_model=False,
        use_vlm=False,
    )

    assert result["summary"]["status"] == "ok"
    assert calls[0][0] == "tiny"
    assert calls[1][0] == "detect"


def test_run_segmentation_stage_forwards_boundary_mode(monkeypatch):
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    received: list[str] = []

    monkeypatch.setattr(
        "geoseg.pipeline.segment.maybe_skip_tiny_image", lambda *args: None
    )
    monkeypatch.setattr(
        "geoseg.pipeline.segment.classify_figure_stage",
        lambda *args, **kwargs: ({"figure_type": "conceptual_model"}, None),
    )
    monkeypatch.setattr(
        "geoseg.pipeline.segment.detect_panels_stage",
        lambda *args: [{"id": 0, "bbox": (0, 0, 40, 40)}],
    )
    monkeypatch.setattr(
        "geoseg.pipeline.segment.review_figure_stage",
        lambda *args, **kwargs: (
            [{"id": 0, "bbox": (0, 0, 40, 40)}],
            [],
            False,
            0,
        ),
    )
    monkeypatch.setattr(
        "geoseg.pipeline.segment.resolve_target_panel_stage",
        lambda *args, **kwargs: 0,
    )

    def fake_segment(*args, **kwargs):
        received.append(kwargs["boundary_mode"])
        return None, 0, None

    monkeypatch.setattr("geoseg.pipeline.segment.segment_panel_stage", fake_segment)
    monkeypatch.setattr(
        "geoseg.pipeline.segment.summarize_pipeline_result",
        lambda *args, **kwargs: {"summary": {"status": "ok"}},
    )

    run_segmentation_stage(
        img,
        skip_non_velocity_model=False,
        use_vlm=False,
        boundary_mode="red",
    )

    assert received == ["red"]
