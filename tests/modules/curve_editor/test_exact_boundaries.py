"""Tests for exact partition-closure boundaries and conservative decisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geoseg.modules.curve_editor.exact_boundaries import (
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
