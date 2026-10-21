"""Tests for image-edge clipping and open-interface endpoint handling."""

from __future__ import annotations

import numpy as np

from geoseg.modules.curve_editor import (
    clip_polygon_to_rect,
    extend_open_curve,
    model_from_labels,
    self_intersects,
    snap_closed_curve,
)


def _bands() -> np.ndarray:
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[20:50] = 1
    labels[50:80] = 2
    labels[80:] = 3
    return labels


def test_clip_polygon_to_rect_stays_inside_and_keeps_border():
    polygon = np.array(
        [[-20, 30], [60, -20], [140, 40], [70, 120], [-10, 80]], dtype=float
    )
    clipped = clip_polygon_to_rect(polygon, (100, 120))

    assert (clipped[:, 0] >= 0).all() and (clipped[:, 0] <= 119).all()
    assert (clipped[:, 1] >= 0).all() and (clipped[:, 1] <= 99).all()
    assert ((clipped[:, 0] == 0) | (clipped[:, 1] == 0)).any()
    assert len(clipped) >= 3


def test_extend_open_curve_inside_region_reaches_boundaries():
    labels = _bands()
    adjusted = extend_open_curve(np.array([[50, 30], [50, 45]], dtype=float), labels)

    assert len(adjusted) == 2
    assert abs(adjusted[0, 1] - 19.5) <= 2.0
    assert abs(adjusted[-1, 1] - 49.5) <= 2.0


def test_extend_open_curve_trims_overshoot_to_primary_region():
    labels = _bands()
    adjusted = extend_open_curve(np.array([[50, 10], [50, 60]], dtype=float), labels)

    # Most of the stroke lies in band 1; both ends are trimmed to that band.
    assert adjusted[0, 1] < 25
    assert adjusted[-1, 1] < 55


def test_extend_open_curve_uses_panel_edges_without_labels():
    labels = np.zeros((100, 100), dtype=np.int32)
    adjusted = extend_open_curve(np.array([[10, 30], [90, 30]], dtype=float), labels)

    assert adjusted[0, 0] == 0.0
    assert adjusted[-1, 0] == 99.0


def test_open_split_interface_seeds_the_new_side():
    labels = _bands()
    model = model_from_labels(labels)
    adjusted = extend_open_curve(np.array([[50, 30], [50, 45]], dtype=float), labels)

    interface = model.add_split_interface(adjusted, basis="B3")
    output = model.to_labels()
    region_id = interface.metadata.get("split_region_id")

    assert region_id is not None
    assert region_id in set(np.unique(output).tolist())
    assert output[35, 40] != output[35, 60]
    assert output[65, 40] == 2


def test_self_intersects_detects_bowtie():
    bowtie = np.array([[10, 10], [90, 90], [10, 90], [90, 10]], dtype=float)
    square = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=float)

    assert self_intersects(bowtie)
    assert not self_intersects(square)


def test_snap_closed_curve_clips_stroke_outside_panel():
    h, w = 120, 140
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[30:90, 40:100] = [220, 220, 220]
    stroke = np.array(
        [[-20, 60], [70, -20], [160, 60], [70, 140], [-20, 60]], dtype=float
    )

    snapped = snap_closed_curve(image, stroke, search_radius=12)

    assert (snapped[:, 0] >= 0).all() and (snapped[:, 0] <= w - 1).all()
    assert (snapped[:, 1] >= 0).all() and (snapped[:, 1] <= h - 1).all()
