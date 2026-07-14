from __future__ import annotations

import numpy as np

from geoseg.preprocessing import segmentation as prep_seg
from geoseg.modules.segment_engines.runner import run_engine


def test_preprocessing_segmentation_facade_forwards(monkeypatch):
    calls: dict[str, object] = {}

    def fake_run_engine(engine, image_rgb, reps, colorbar_rgb, n_layers):
        calls.setdefault("engines", []).append(
            (engine, image_rgb.shape, reps, None if colorbar_rgb is None else colorbar_rgb.shape, n_layers)
        )
        return {"labels": np.zeros((2, 2), dtype=np.int32), "overlay": None, "seeds": []}

    def fake_create_overlay_with_legend(image_rgb, labels):
        calls["legend"] = (image_rgb.shape, labels.shape)
        return np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(
        prep_seg,
        "run_engine",
        fake_run_engine,
    )
    monkeypatch.setattr(
        prep_seg,
        "create_overlay_with_legend",
        fake_create_overlay_with_legend,
    )

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    cb = np.zeros((2, 2, 3), dtype=np.uint8)
    labels = np.zeros((4, 4), dtype=np.int32)

    assert prep_seg.segment_artifact_baseline(img, n_layers=3)["labels"].shape == (2, 2)
    assert prep_seg.segment_artifact_colorbar_guided(img, cb, n_layers=4)["labels"].shape == (2, 2)
    assert prep_seg.create_audit_overlay(img, labels).shape == (2, 2, 3)

    assert calls["engines"] == [
        ("v4_kmeans", (4, 4, 3), None, None, 3),
        ("v4_kmeans_colorbar", (4, 4, 3), None, (2, 2, 3), 4),
    ]
    assert calls["legend"] == ((4, 4, 3), (4, 4))


def test_run_engine_preserves_seed_palette(monkeypatch):
    from geoseg.modules.segment_engines import v4_kmeans

    seeds = [[10, 20, 30], [40, 50, 60]]

    def fake_segment(image_rgb, reps, colorbar_rgb, n_layers, n_color_zones=0):
        return {
            "labels": np.zeros(image_rgb.shape[:2], dtype=np.int32),
            "overlay": None,
            "seeds": seeds,
            "meta": {"engine": "v4_kmeans"},
        }

    monkeypatch.setattr(v4_kmeans, "segment", fake_segment)

    result = run_engine(
        "v4_kmeans",
        np.zeros((3, 4, 3), dtype=np.uint8),
        None,
        None,
        2,
    )

    assert result["seeds"] == seeds
