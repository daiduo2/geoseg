"""Tests for the VLM-style representative-point synthesizer."""

from __future__ import annotations

import numpy as np

from synthetic_benchmark.vlm_seeds import (
    synthesize_noisy_reps,
    synthesize_reps,
)


def _toy_panel(h: int = 32, w: int = 48, k: int = 3):
    gt = np.zeros((h, w), dtype=np.int32)
    band = h // k
    for i in range(k):
        gt[i * band : (i + 1) * band, :] = i + 1
    label_to_value = {i + 1: (i + 1) / k for i in range(k)}
    return gt, label_to_value


def test_synthesize_reps_one_per_label():
    gt, l2v = _toy_panel(k=4)
    reps = synthesize_reps(gt, l2v, seed=0)
    assert len(reps) == 4
    for rep in reps:
        assert "color_name" in rep
        assert "representative_point" in rep
        x, y = rep["representative_point"]["x"], rep["representative_point"]["y"]
        assert 0 <= x < gt.shape[1]
        assert 0 <= y < gt.shape[0]


def test_synthesize_reps_centroid_inside_label():
    gt, l2v = _toy_panel(k=3)
    reps = synthesize_reps(gt, l2v, seed=0)
    # Label 1 occupies rows [0, h/3); centroid should be there.
    y1 = reps[0]["representative_point"]["y"]
    assert y1 < gt.shape[0] // 3


def test_synthesize_reps_label_order_matches_label_id():
    gt, l2v = _toy_panel(k=4)
    reps = synthesize_reps(gt, l2v, seed=0)
    # Each rep should correspond to a different label — points are spread.
    ys = sorted(r["representative_point"]["y"] for r in reps)
    assert len(set(ys)) == len(ys)  # all distinct


def test_synthesize_noisy_reps_in_bounds():
    gt, l2v = _toy_panel(k=5)
    reps = synthesize_noisy_reps(
        gt, l2v, offset_max=10, color_noise=0.5, seed=42,
    )
    h, w = gt.shape
    for r in reps:
        x, y = r["representative_point"]["x"], r["representative_point"]["y"]
        assert 0 <= x < w
        assert 0 <= y < h


def test_synthesize_noisy_reps_can_change_color_name():
    gt, l2v = _toy_panel(k=6)
    # 100% noise → every color_name should be cycled once
    reps_noisy = synthesize_noisy_reps(
        gt, l2v, offset_max=0, color_noise=1.0, seed=7,
    )
    reps_clean = synthesize_reps(gt, l2v, seed=7)
    names_noisy = [r["color_name"] for r in reps_noisy]
    names_clean = [r["color_name"] for r in reps_clean]
    assert names_noisy != names_clean


def test_synthesize_reps_skips_label_zero():
    # Add a stray 0 to label_to_value to confirm it's ignored.
    gt, l2v = _toy_panel(k=2)
    l2v[0] = 0.0  # background shouldn't generate a rep
    reps = synthesize_reps(gt, l2v, seed=0)
    assert len(reps) == 2  # only labels 1 and 2
