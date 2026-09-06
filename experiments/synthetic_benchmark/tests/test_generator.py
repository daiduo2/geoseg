"""Tests for the synthetic panel generator."""

from __future__ import annotations

import numpy as np

from synthetic_benchmark.configs import PanelConfig
from synthetic_benchmark.generator import generate_panel


def test_generate_panel_basic_shape():
    cfg = PanelConfig(height=64, width=96, n_layers=3, seed=7)
    rgb, gt, meta = generate_panel(cfg)
    assert rgb.shape == (64, 96, 3)
    assert rgb.dtype == np.uint8
    assert gt.shape == (64, 96)
    assert gt.dtype == np.int32
    assert meta["n_layers"] == 3
    assert meta["seed"] == 7
    # Background may be 0; layers use 1..n_layers
    uniq = sorted(set(gt.flatten().tolist()))
    assert 0 in uniq or len(uniq) == cfg.n_layers
    assert max(uniq) <= cfg.n_layers


def test_generate_panel_with_fault():
    cfg = PanelConfig(height=64, width=96, n_layers=4, has_fault=True, seed=8)
    rgb, gt, meta = generate_panel(cfg)
    kinds = [c["kind"] for c in meta["components"]]
    assert "layers" in kinds
    assert "fault" in kinds
    # Fault should introduce a left/right asymmetry in the label distribution
    left_uniq = set(gt[:, : int(0.6 * 96)].flatten().tolist())
    right_uniq = set(gt[:, int(0.6 * 96):].flatten().tolist())
    assert left_uniq != right_uniq


def test_generate_panel_with_channel():
    cfg = PanelConfig(
        height=96,
        width=128,
        n_layers=3,
        has_channel=True,
        channel_amplitude=30,
        channel_thickness=12,
        seed=9,
    )
    rgb, gt, meta = generate_panel(cfg)
    kinds = [c["kind"] for c in meta["components"]]
    assert "channel" in kinds
    # Channel adds one extra label
    assert gt.max() >= cfg.n_layers + 1


def test_generate_panel_with_unconformity():
    cfg = PanelConfig(
        height=128,
        width=128,
        n_layers=4,
        has_unconformity=True,
        unconformity_y_frac=0.5,
        seed=10,
    )
    rgb, gt, meta = generate_panel(cfg)
    kinds = [c["kind"] for c in meta["components"]]
    assert "unconformity" in kinds


def test_generate_panel_is_deterministic():
    cfg = PanelConfig(height=48, width=64, n_layers=3, seed=42)
    rgb1, gt1, _ = generate_panel(cfg)
    rgb2, gt2, _ = generate_panel(cfg)
    np.testing.assert_array_equal(rgb1, rgb2)
    np.testing.assert_array_equal(gt1, gt2)


def test_generate_panel_different_seeds_differ():
    cfg_a = PanelConfig(height=48, width=64, n_layers=3, seed=1)
    cfg_b = PanelConfig(height=48, width=64, n_layers=3, seed=2)
    rgb_a, gt_a, _ = generate_panel(cfg_a)
    rgb_b, gt_b, _ = generate_panel(cfg_b)
    assert not np.array_equal(rgb_a, rgb_b)
    assert not np.array_equal(gt_a, gt_b)


def test_colormap_round_trip_in_meta():
    cfg = PanelConfig(height=32, width=48, n_layers=3, colormap="viridis", seed=1)
    _, gt, meta = generate_panel(cfg)
    # Every non-background label must map to a scalar in [0, 1]
    for lbl, v in meta["label_to_value"].items():
        assert 0.0 <= v <= 1.0, f"label {lbl} -> value {v} out of range"
