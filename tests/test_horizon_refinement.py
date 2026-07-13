"""Unit tests for horizon_refinement engine.

Covers:
- Label-0 semantic protection (editor separator vs pipeline layer)
- Protocol compliance (segment() returns SegmentationResult dict)
- Internal helper correctness (refine_label_blur, _adjust_boundaries, _repartition_columns)
"""

from __future__ import annotations

import numpy as np
import pytest

from geoseg.modules.segment_engines.horizon_refinement import (
    _adjust_boundaries,
    _coarse_segment,
    _repartition_columns,
    _separator_mask,
    refine_boundaries,
    refine_label_blur,
    segment,
)


# ---------------------------------------------------------------------------
# _separator_mask
# ---------------------------------------------------------------------------


def test_separator_mask_detects_thin_line() -> None:
    """Label 0 with tiny area (< 2%) is treated as separator."""
    labels = np.ones((100, 100), dtype=np.int32)
    labels[50, :] = 0  # 1-pixel horizontal line = 1% area
    mask = _separator_mask(labels)
    assert mask.sum() == 100
    assert mask[50, 0]
    assert not mask[0, 0]


def test_separator_mask_ignores_large_layer() -> None:
    """Label 0 with large area (>= 2%) is treated as a real layer."""
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[:50, :] = 1
    # label 0 covers 50% of image
    mask = _separator_mask(labels)
    assert not mask.any()


# ---------------------------------------------------------------------------
# refine_label_blur
# ---------------------------------------------------------------------------


def test_refine_label_blur_preserves_separator() -> None:
    """When label 0 is a thin separator line, it must survive label-blur."""
    h, w = 100, 100
    coarse = np.ones((h, w), dtype=np.int32)
    coarse[:40, :] = 2
    coarse[40, :] = 0  # separator line at row 40
    coarse[41:, :] = 3

    result = refine_label_blur(coarse, sigma=5.0)

    # Separator line should still be 0
    assert np.all(result[40, :] == 0)
    # Layers should remain distinct
    assert np.all(result[:35, :] == 2)
    assert np.all(result[45:, :] == 3)


def test_refine_label_blur_includes_label_zero_as_layer() -> None:
    """When label 0 is a large layer, it participates in competition."""
    h, w = 100, 100
    coarse = np.zeros((h, w), dtype=np.int32)
    coarse[:50, :] = 1  # label 0 = bottom half (50%), label 1 = top half

    result = refine_label_blur(coarse, sigma=5.0)

    # Both labels should still exist (Gaussian blur at sigma=5 won't merge
    # two 50% halves)
    assert 0 in result
    assert 1 in result


# ---------------------------------------------------------------------------
# _adjust_boundaries
# ---------------------------------------------------------------------------


def test_adjust_boundaries_preserves_separator() -> None:
    """Boundary adjustment must not overwrite separator pixels."""
    h, w = 100, 100
    coarse = np.ones((h, w), dtype=np.int32)
    coarse[:50, :] = 2
    coarse[50, :] = 0  # separator

    # Fit a boundary at row 45 (moved up from 50)
    boundary = np.full(w, 45.0, dtype=np.float32)

    result = _adjust_boundaries(
        coarse, [boundary], [(2, 1)], blend_width=3
    )

    # Separator row must stay 0
    assert np.all(result[50, :] == 0)


# ---------------------------------------------------------------------------
# _repartition_columns
# ---------------------------------------------------------------------------


def test_repartition_columns_preserves_separator() -> None:
    """Global repartitioning must restore separator pixels."""
    h, w = 100, 100
    coarse = np.ones((h, w), dtype=np.int32)
    coarse[:, 50] = 0  # 1-pixel vertical separator line (1% area < 2% threshold)

    boundaries = [np.full(w, 50.0, dtype=np.float32)]
    spatial_order = [1, 2]

    result = _repartition_columns(coarse, spatial_order, boundaries)

    # Vertical separator line should remain 0
    assert np.all(result[:, 50] == 0)
    # Above boundary = layer 1, below = layer 2
    assert np.all(result[:50, 0] == 1)
    assert np.all(result[50:, 0] == 2)


# ---------------------------------------------------------------------------
# refine_boundaries (integration)
# ---------------------------------------------------------------------------


def test_refine_boundaries_with_separator_topology() -> None:
    """End-to-end: editor-style labels with separator survive refinement."""
    h, w = 100, 100
    panel_rgb = np.ones((h, w, 3), dtype=np.uint8) * 200
    panel_rgb[:40, :] = [180, 60, 60]
    panel_rgb[41:, :] = [60, 60, 180]

    # Editor topology: label 0 = boundary between layers 1 and 2
    coarse = np.ones((h, w), dtype=np.int32)
    coarse[:40, :] = 2
    coarse[40, :] = 0  # thin separator
    coarse[41:, :] = 3

    labels, boundaries = refine_boundaries(
        panel_rgb, coarse_labels=coarse, n_layers=2, method="savgol"
    )

    # Separator should survive (label-space blur path preserves it)
    assert np.all(labels[40, :] == 0)
    # Layers on either side should remain distinct
    assert np.all(labels[:35, :] == 2) or np.all(labels[:35, :] == 3)
    assert np.all(labels[45:, :] == 2) or np.all(labels[45:, :] == 3)


def test_refine_boundaries_with_pipeline_topology() -> None:
    """End-to-end: pipeline-style labels (label 0 = layer) work correctly."""
    h, w = 100, 100
    panel_rgb = np.ones((h, w, 3), dtype=np.uint8) * 200
    panel_rgb[:30, :] = [180, 60, 60]
    panel_rgb[30:60, :] = [60, 180, 60]
    panel_rgb[60:, :] = [60, 60, 180]

    # Pipeline topology: labels 0, 1, 2 are three layers (no separator)
    coarse = np.zeros((h, w), dtype=np.int32)
    coarse[:30, :] = 0
    coarse[30:60, :] = 1
    coarse[60:, :] = 2

    labels, boundaries = refine_boundaries(
        panel_rgb, coarse_labels=coarse, n_layers=3, method="savgol"
    )

    # All three layers should still exist
    unique = set(labels.flatten())
    assert unique >= {0, 1, 2}
    # No separator semantics -> label 0 should not be treated specially
    assert (labels == 0).sum() > 0


# ---------------------------------------------------------------------------
# segment() — Protocol compliance
# ---------------------------------------------------------------------------


def test_segment_returns_protocol_dict() -> None:
    """segment() must return {"labels", "overlay", "meta"}."""
    h, w = 100, 100
    panel_rgb = np.ones((h, w, 3), dtype=np.uint8) * 200
    panel_rgb[:50, :] = [180, 60, 60]
    panel_rgb[50:, :] = [60, 60, 180]

    result = segment(panel_rgb, n_layers=2)

    assert "labels" in result
    assert "overlay" in result
    assert "meta" in result
    assert result["meta"]["engine"] == "horizon_refinement"
    assert isinstance(result["labels"], np.ndarray)
    assert isinstance(result["overlay"], np.ndarray)


def test_segment_with_coarse_labels() -> None:
    """segment() accepts pre-computed coarse_labels."""
    h, w = 100, 100
    panel_rgb = np.ones((h, w, 3), dtype=np.uint8) * 200
    coarse = np.zeros((h, w), dtype=np.int32)
    coarse[:40, :] = 1
    coarse[40:80, :] = 2
    coarse[80:, :] = 3

    result = segment(panel_rgb, n_layers=3, coarse_labels=coarse)

    assert result["meta"]["n_layers"] >= 2
    assert result["labels"].shape == (h, w)


# ---------------------------------------------------------------------------
# _coarse_segment
# ---------------------------------------------------------------------------


def test_coarse_segment_uses_row_median() -> None:
    """_coarse_segment should use anisotropic row-median filtering."""
    h, w = 100, 100
    panel_rgb = np.ones((h, w, 3), dtype=np.uint8) * 200
    # Add horizontal noise stripes (simulating text)
    panel_rgb[20, :] = 0
    panel_rgb[50, :] = 255
    panel_rgb[80, :] = 0

    labels = _coarse_segment(panel_rgb, n_layers=2, blur_sigma=2.0)

    assert labels.shape == (h, w)
    assert len(set(labels.flatten())) >= 2
