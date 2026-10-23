"""Tests for exact partition-closure boundaries and conservative decisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geoseg.modules.curve_editor.exact_boundaries import (
    boundary_union,
    closed_boundary_union,
    decide_exact_boundaries,
    draw_exact_decisions,
    extract_exact_boundaries,
)


def _bands_with_hole():
    labels = np.zeros((100, 140), dtype=np.int32)
    labels[:50] = 1
    labels[50:] = 2
    labels[48:53, 70:75] = 0
    return labels


def test_exact_boundaries_are_pixel_closures_not_splines():
    labels = _bands_with_hole()
    boundaries = extract_exact_boundaries(labels)

    assert boundaries
    assert all(boundary.length == len(boundary.pixels) for boundary in boundaries)
    assert all(np.issubdtype(boundary.pixels.dtype, np.floating) for boundary in boundaries)
    assert any(boundary.labels_pair == (1, 2) for boundary in boundaries)


def test_boundary_union_is_the_raw_4_neighbour_set():
    labels = np.array(
        [
            [1, 1, 2],
            [1, 2, 2],
            [3, 3, 3],
        ],
        dtype=np.int32,
    )
    expected = np.array(
        [
            [False, True, True],
            [True, True, True],
            [True, True, True],
        ]
    )

    union = boundary_union(labels)

    assert np.array_equal(union, expected)
    assert union.dtype == bool


def test_boundary_union_lower_side_matches_exact_objects():
    labels = np.array(
        [
            [1, 1, 2],
            [1, 2, 2],
            [3, 3, 3],
        ],
        dtype=np.int32,
    )
    expected_lower = np.array(
        [
            [False, True, False],
            [True, True, True],
            [False, False, False],
        ]
    )

    assert np.array_equal(boundary_union(labels, side="lower"), expected_lower)


def test_boundary_union_equals_the_union_of_extracted_objects():
    labels = _bands_with_hole()
    boundaries = extract_exact_boundaries(labels)
    from_objects = np.zeros(labels.shape, dtype=bool)
    for boundary in boundaries:
        rows = np.rint(boundary.pixels[:, 1]).astype(int)
        cols = np.rint(boundary.pixels[:, 0]).astype(int)
        from_objects[rows, cols] = True

    assert np.array_equal(
        boundary_union(labels, side="lower"),
        from_objects,
    )
    assert np.all(from_objects <= boundary_union(labels))


def test_boundary_union_rejects_non_2d_labels():
    with pytest.raises(ValueError):
        boundary_union(np.zeros((2, 2, 2), dtype=np.int32))


def test_closed_boundary_union_adds_domain_rectangle_for_foreground():
    labels = np.ones((7, 9), dtype=np.int32)
    closed = closed_boundary_union(labels)

    assert closed[0, :].all()
    assert closed[-1, :].all()
    assert closed[:, 0].all()
    assert closed[:, -1].all()
    assert not closed[1:-1, 1:-1].any()


def test_closed_boundary_union_skips_background_border_when_requested():
    labels = np.zeros((7, 9), dtype=np.int32)
    labels[2:5, 3:6] = 1
    raw = boundary_union(labels)
    closed = closed_boundary_union(labels, include_background=False)

    assert np.array_equal(closed, raw)
    assert not closed[0, :].any()


def test_closed_boundary_union_includes_full_rectangle_by_default():
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[2, 2] = 1
    closed = closed_boundary_union(labels)

    assert closed[0, :].all()
    assert closed[-1, :].all()
    assert closed[:, 0].all()
    assert closed[:, -1].all()


def test_foreground_closure_is_kept_and_hole_is_review_no_auto_remove():
    labels = _bands_with_hole()
    boundaries = extract_exact_boundaries(labels)
    decisions = decide_exact_boundaries(labels, boundaries)

    foreground = [item for item in decisions if item.boundary.labels_pair == (1, 2)]
    hole = [
        item
        for item in decisions
        if 0 in item.boundary.labels_pair
    ]
    assert foreground and all(item.action == "keep" for item in foreground)
    assert all(item.action in {"review", "keep"} for item in decisions)
    assert all(item.action != "remove" for item in decisions)
    assert hole


def test_draw_exact_decisions_preserves_image_shape():
    labels = _bands_with_hole()
    image = np.full((100, 140, 3), 180, dtype=np.uint8)
    decisions = decide_exact_boundaries(labels)
    drawn = draw_exact_decisions(image, decisions)

    assert drawn.shape == image.shape
    assert (drawn != image).any()


def test_newimage_p0_keeps_long_exterior_closure():
    root = Path(__file__).resolve().parents[3] / "validation" / "curated"
    record = next(
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if "newimage_p0" in line
    )
    labels = np.load(root / record["files"]["labels"])["labels"]
    boundaries = extract_exact_boundaries(labels)
    decisions = decide_exact_boundaries(labels, boundaries)

    exterior = [
        item
        for item in decisions
        if item.boundary.labels_pair == (0, 2) and item.boundary.length > 1000
    ]
    assert exterior
    assert exterior[0].action == "keep"
    assert all(item.action != "remove" for item in decisions)
    assert any(item.action == "review" for item in decisions)
