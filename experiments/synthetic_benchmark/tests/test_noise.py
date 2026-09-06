"""Tests for the noise simulator."""

from __future__ import annotations

import numpy as np

from synthetic_benchmark.configs import PanelConfig
from synthetic_benchmark.generator import generate_panel
from synthetic_benchmark.noise import apply_noise


def _panel() -> tuple[np.ndarray, np.ndarray]:
    cfg = PanelConfig(height=64, width=96, n_layers=4, seed=11)
    return generate_panel(cfg)[:2]


def test_apply_noise_keeps_shape():
    rgb, gt = _panel()
    art = apply_noise(
        rgb,
        gt,
        jpeg_std=0.0,
        n_text_blocks=0,
        n_contours=0,
        add_colorbar=False,
        seed=1,
    )
    assert art["rgb"].shape == rgb.shape
    np.testing.assert_array_equal(art["gt_labels"], gt)
    assert art["colorbar_rgb"] is None
    assert not art["text_mask"].any()


def test_apply_noise_text_blocks_create_mask():
    rgb, gt = _panel()
    art = apply_noise(
        rgb,
        gt,
        jpeg_std=0.0,
        n_text_blocks=3,
        n_contours=0,
        add_colorbar=False,
        seed=1,
    )
    assert art["text_mask"].sum() > 0
    # Black-painted regions should be 0,0,0
    painted = art["rgb"][art["text_mask"]]
    np.testing.assert_array_equal(painted, np.zeros((int(art["text_mask"].sum()), 3)))


def test_apply_noise_colorbar_appended():
    rgb, gt = _panel()
    h, w = rgb.shape[:2]
    art = apply_noise(
        rgb,
        gt,
        jpeg_std=0.0,
        n_text_blocks=0,
        n_contours=0,
        add_colorbar=True,
        seed=1,
    )
    assert art["colorbar_rgb"] is not None
    assert art["colorbar_rgb"].shape == (h, 24, 3)
    # Width of the final RGB should be wider if we appended on the right
    assert art["rgb"].shape[1] >= w


def test_apply_noise_is_deterministic():
    rgb, gt = _panel()
    a = apply_noise(rgb, gt, jpeg_std=3.0, n_text_blocks=4, n_contours=5, add_colorbar=True, seed=99)
    b = apply_noise(rgb, gt, jpeg_std=3.0, n_text_blocks=4, n_contours=5, add_colorbar=True, seed=99)
    np.testing.assert_array_equal(a["rgb"], b["rgb"])
    np.testing.assert_array_equal(a["text_mask"], b["text_mask"])
