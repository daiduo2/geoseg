"""Tests for conservative boundary keep/review decisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geoseg import api
from geoseg.modules.curve_editor.decisions import (
    background_roles,
    decide_boundaries,
    refine_decisions,
)
from geoseg.modules.curve_editor.extract import ExtractedCurve


def _curve(points, *, closed=False):
    points = np.asarray(points, dtype=float)
    length = float(np.hypot(*np.diff(points, axis=0).T).sum())
    return ExtractedCurve(
        points=[tuple(point) for point in points], closed=closed, length=length
    )


def test_large_background_is_exterior_and_long_surface_is_kept():
    labels = np.zeros((100, 140), dtype=np.int32)
    labels[10:90, 10:130] = 1
    # Large central background region (not necessarily touching the border).
    labels[20:80, 30:110] = 0
    roles = background_roles(labels)
    assert roles[1] == "exterior" or "exterior" in roles.values()

    curve = _curve([(30, 20), (70, 20), (110, 20)])
    decision = decide_boundaries([curve], labels, min_exterior_length=50)[0]
    assert decision.background_role == "exterior"
    assert decision.action == "keep"


def test_short_exterior_boundary_goes_to_review_not_remove():
    labels = np.zeros((100, 140), dtype=np.int32)
    labels[10:90, 10:130] = 1
    labels[20:80, 30:110] = 0
    curve = _curve([(50, 20), (55, 21), (58, 24)])
    decision = decide_boundaries([curve], labels, min_exterior_length=50)[0]
    assert decision.action == "review"


def test_foreground_boundary_is_kept_and_short_foreground_is_review():
    labels = np.zeros((100, 140), dtype=np.int32)
    labels[:50] = 1
    labels[50:] = 2
    long_curve = _curve([(20, 50), (60, 50), (100, 50)])
    short_curve = _curve([(20, 50), (28, 50)])
    decisions = decide_boundaries([long_curve, short_curve], labels)
    assert decisions[0].action == "keep"
    assert decisions[1].action == "review"


def test_internal_hole_boundary_is_review():
    labels = np.zeros((100, 140), dtype=np.int32)
    labels[:50] = 1
    labels[50:] = 2
    labels[48:53, 70:75] = 0  # internal hole
    hole = _curve([(69, 47), (76, 47), (76, 53), (69, 53)], closed=True)
    decision = decide_boundaries([hole], labels)[0]
    assert decision.background_role == "internal_hole"
    assert decision.action == "review"
    assert not decision.reasons or "hole" in " ".join(decision.reasons)


def test_newimage_p0_no_false_deletion():
    root = Path(__file__).resolve().parents[3] / "validation" / "curated"
    record = next(
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if "newimage_p0" in line
    )
    labels = np.load(root / record["files"]["labels"])["labels"]
    curves = api.curves_from_labels(labels)
    decisions = decide_boundaries(curves, labels)

    decisions_by_length = sorted(decisions, key=lambda item: item.curve.length, reverse=True)
    assert decisions_by_length[0].action == "keep"
    assert decisions_by_length[0].background_role == "exterior"
    assert all(item.action != "remove" for item in decisions)
    assert any(item.action == "review" for item in decisions)

    result = refine_decisions(decisions, labels)
    assert len(result.kept) == 3
    assert len(result.review) == 2
    assert result.removed == []
