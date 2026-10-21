"""Tests for label-pair and curvature constrained boundary refinement."""

from __future__ import annotations

import numpy as np

from geoseg.modules.curve_editor.extract import ExtractedCurve
from geoseg.modules.curve_editor.refine import (
    BoundaryRefinementConfig,
    adjacent_labels,
    refine_curve,
    refine_curves,
)


def _two_region_labels() -> np.ndarray:
    labels = np.zeros((100, 140), dtype=np.int32)
    labels[:50] = 1
    labels[50:] = 2
    return labels


def _horizontal(length: float = 100.0, y: float = 50.0) -> ExtractedCurve:
    points = np.column_stack(
        [np.linspace(20, 20 + length, 10), np.full(10, y)]
    )
    return ExtractedCurve(points=[tuple(p) for p in points], closed=False, length=length)


def test_adjacent_labels_returns_foreground_pair():
    labels = _two_region_labels()
    curve = _horizontal()
    pair = adjacent_labels(labels, np.asarray(curve.points), closed=False)
    assert pair is not None
    assert set(pair) == {1, 2}


def test_background_boundary_is_dropped():
    labels = _two_region_labels()
    labels[48:53, 70:75] = 0
    hole = ExtractedCurve(
        points=[(69, 47), (76, 47), (76, 53), (69, 53)],
        closed=True,
        length=28.0,
    )
    assert set(adjacent_labels(labels, np.asarray(hole.points), closed=True)) == {0, 2}
    assert refine_curve(hole, labels) is None


def test_short_curve_is_dropped():
    labels = _two_region_labels()
    short = _horizontal(length=10.0, y=50.0)
    assert refine_curve(short, labels) is None


def test_refinement_preserves_endpoints_and_reduces_curvature():
    labels = _two_region_labels()
    # A wavy interface between the two regions.
    xs = np.linspace(20, 120, 80)
    ys = 50 + 3 * np.sin(xs / 2.0)
    raw = ExtractedCurve(
        points=[(float(x), float(y)) for x, y in zip(xs, ys)],
        closed=False,
        length=float(np.hypot(*np.diff(np.column_stack([xs, ys]), axis=0).T).sum()),
    )
    config = BoundaryRefinementConfig(smoothing=8.0, max_curvature=0.2)
    refined = refine_curve(raw, labels, config=config)

    assert refined is not None
    assert set(refined.labels) == {1, 2}
    assert np.allclose(refined.points[0], raw.points[0], atol=1e-6)
    assert np.allclose(refined.points[-1], raw.points[-1], atol=1e-6)
    assert refined.max_curvature <= config.max_curvature + 1e-9


def test_refine_curves_filters_and_keeps_valid_interfaces():
    labels = _two_region_labels()
    labels[48:53, 70:75] = 0
    valid = _horizontal(length=100)
    hole = ExtractedCurve(
        points=[(69, 47), (76, 47), (76, 53), (69, 53)],
        closed=True,
        length=28.0,
    )
    short = _horizontal(length=8, y=50)
    refined = refine_curves([valid, hole, short], labels)

    assert len(refined) == 1
    assert set(refined[0].labels) == {1, 2}
