"""Tests for geoseg.modules.visual_audit.color_residual."""
from __future__ import annotations

import numpy as np
import pytest

from geoseg.modules.visual_audit.color_residual import (
    compute_color_residual_map,
    compute_palette_match_residuals,
    compute_label_representative_colors,
    compute_label_residual_stats,
    create_color_residual_overlay,
    estimate_text_mask,
    find_high_deviation_regions,
)


def test_compute_palette_match_residuals_exact_and_ambiguous() -> None:
    palette = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    image = np.array([[[255, 0, 0], [0, 0, 255], [128, 0, 128]]], dtype=np.uint8)

    result = compute_palette_match_residuals(image, palette, chunk_size=2)

    assert result["labels"].shape == (1, 3)
    assert result["labels"][0, :2].tolist() == [0, 1]
    assert result["rgb_residual"][0, :2].tolist() == [0.0, 0.0]
    assert result["delta_e"][0, :2].tolist() == [0.0, 0.0]
    assert result["margin_delta_e"][0, 2] < result["margin_delta_e"][0, 0]


def test_compute_palette_match_residuals_validates_palette() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="K >= 2"):
        compute_palette_match_residuals(image, np.zeros((1, 3), dtype=np.uint8))


def _make_solid_image(h: int, w: int, color: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def test_compute_label_representative_colors_uses_median():
    h, w = 20, 20
    panel = _make_solid_image(h, w, (100, 100, 100))
    labels = np.ones((h, w), dtype=np.int32)

    # Inject a few bright outlier pixels (simulated text).
    panel[:2, :2] = (255, 255, 255)

    reps = compute_label_representative_colors(labels, panel)
    assert 1 in reps
    np.testing.assert_array_equal(reps[1]["median_rgb"], np.array([100, 100, 100]))


def test_compute_color_residual_map_shape_and_background():
    h, w = 30, 40
    panel = _make_solid_image(h, w, (120, 120, 120))
    labels = np.zeros((h, w), dtype=np.int32)
    labels[:, : w // 2] = 1
    labels[:, w // 2 :] = 2

    residual = compute_color_residual_map(labels, panel)
    assert residual.shape == (h, w)
    assert residual.dtype == np.float64
    assert np.all(residual[labels == 0] == 0.0)
    # Same color across labels should give near-zero residuals.
    assert residual.max() < 1.0


def test_find_high_deviation_regions_detects_embedded_blob():
    h, w = 60, 60
    panel = _make_solid_image(h, w, (100, 100, 100))
    labels = np.ones((h, w), dtype=np.int32)

    # Embedded blob of different color inside label 1.
    panel[20:40, 20:40] = (200, 50, 50)

    residual = compute_color_residual_map(labels, panel)
    candidates = find_high_deviation_regions(
        labels, residual, min_area_frac=0.005, deviation_percentile=85.0
    )

    assert len(candidates) >= 1
    assert candidates[0]["area"] >= 100


def test_create_color_residual_overlay_with_candidates():
    h, w = 40, 40
    panel = _make_solid_image(h, w, (100, 100, 100))
    labels = np.ones((h, w), dtype=np.int32)
    panel[15:25, 15:25] = (200, 50, 50)

    residual = compute_color_residual_map(labels, panel)
    candidates = find_high_deviation_regions(labels, residual)
    overlay = create_color_residual_overlay(
        residual, panel, labels, candidates=candidates, alpha=0.5
    )

    assert overlay.shape == (h, w, 3)
    assert overlay.dtype == np.uint8


def test_compute_label_residual_stats_shape_error():
    labels = np.ones((10, 10), dtype=np.int32)
    panel = _make_solid_image(10, 10, (100, 100, 100))
    bad_residual = np.zeros((12, 12), dtype=np.float64)

    with pytest.raises(ValueError):
        compute_label_residual_stats(labels, panel, bad_residual)


def test_inputs_not_mutated():
    h, w = 20, 20
    panel = _make_solid_image(h, w, (100, 100, 100))
    labels = np.ones((h, w), dtype=np.int32)
    panel_copy = panel.copy()
    labels_copy = labels.copy()

    compute_label_representative_colors(labels, panel)
    compute_color_residual_map(labels, panel)

    np.testing.assert_array_equal(panel, panel_copy)
    np.testing.assert_array_equal(labels, labels_copy)


def test_shape_mismatch_raises():
    labels = np.ones((10, 10), dtype=np.int32)
    panel = _make_solid_image(12, 12, (100, 100, 100))

    with pytest.raises(ValueError):
        compute_color_residual_map(labels, panel)


def test_estimate_text_mask_finds_bright_text():
    h, w = 60, 60
    panel = _make_solid_image(h, w, (100, 100, 100))
    # Draw bright text-like strokes.
    panel[10:12, 10:30] = (255, 255, 255)
    panel[20:22, 15:35] = (255, 255, 255)

    mask = estimate_text_mask(panel)
    assert mask.shape == (h, w)
    assert mask.dtype == bool
    # Most text pixels should be detected.
    text_pixels = (panel[:, :, 0] == 255).sum()
    detected_pixels = mask.sum()
    assert detected_pixels > text_pixels * 0.5


def test_estimate_text_mask_shape_error():
    with pytest.raises(ValueError):
        estimate_text_mask(np.zeros((10, 10), dtype=np.uint8))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
