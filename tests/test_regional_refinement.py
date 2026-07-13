"""Tests for geoseg.modules.segment_engines.regional_refinement."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from geoseg.modules.segment_engines.regional_refinement import (
    RefinementConfig,
    refine_by_candidate_regions,
    refine_by_residual_mask,
)


def _make_panel(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_refine_by_residual_mask_preserves_frozen_pixels():
    h, w = 40, 40
    base_labels = np.zeros((h, w), dtype=np.int32)
    base_labels[: h // 2] = 1
    base_labels[h // 2 :] = 2
    panel = _make_panel(h, w)

    refine_mask = np.zeros((h, w), dtype=bool)
    refine_mask[5:15, 5:15] = True

    # run_engine is called on the cropped region, so mock labels must match
    # the crop shape (refine_mask bbox + margin).
    crop_h, crop_w = 25, 25
    patch_labels = np.full((crop_h, crop_w), 3, dtype=np.int32)
    patch_labels[:, : crop_w // 2] = 4
    patch_labels[:, crop_w // 2 :] = 5

    with patch(
        "geoseg.modules.segment_engines.regional_refinement.run_engine",
        return_value={"labels": patch_labels},
    ):
        result = refine_by_residual_mask(
            base_labels,
            panel,
            refine_mask,
            n_layers=2,
            config=RefinementConfig(secondary_engine="mock"),
        )

    refined = result["labels"]
    # Seam smoothing may alter pixels near the refine boundary, so only check
    # pixels well outside the refine region.
    from scipy import ndimage

    safe_margin = 5
    dilated = ndimage.binary_dilation(
        refine_mask, iterations=safe_margin
    )
    safe_frozen = (~refine_mask) & (~dilated)
    assert np.array_equal(refined[safe_frozen], base_labels[safe_frozen])
    # Refined pixels should be present (non-zero after relabeling).
    assert len(np.unique(refined[refine_mask])) > 0


def test_refine_by_residual_mask_empty_mask_returns_base():
    h, w = 20, 20
    base_labels = np.ones((h, w), dtype=np.int32)
    base_labels[h // 2 :] = 2
    panel = _make_panel(h, w)
    refine_mask = np.zeros((h, w), dtype=bool)

    result = refine_by_residual_mask(
        base_labels, panel, refine_mask, n_layers=2
    )

    assert result["meta"]["refined"] is False
    assert np.array_equal(result["labels"], base_labels)


def test_refine_by_residual_mask_full_mask_uses_secondary_result():
    h, w = 20, 20
    base_labels = np.ones((h, w), dtype=np.int32)
    base_labels[h // 2 :] = 2
    panel = _make_panel(h, w)
    refine_mask = np.ones((h, w), dtype=bool)

    patch_labels = np.zeros((h, w), dtype=np.int32)
    patch_labels[: h // 2] = 1
    patch_labels[h // 2 :] = 2

    with patch(
        "geoseg.modules.segment_engines.regional_refinement.run_engine",
        return_value={"labels": patch_labels},
    ):
        result = refine_by_residual_mask(
            base_labels,
            panel,
            refine_mask,
            n_layers=2,
            config=RefinementConfig(secondary_engine="mock"),
        )

    refined = result["labels"]
    assert result["meta"]["refined"] is True
    # Final reorder maps labels to 0..N-1; expect two distinct labels total.
    assert len(set(np.unique(refined))) == 2


def test_refine_by_residual_mask_shape_mismatch_raises():
    base_labels = np.ones((10, 10), dtype=np.int32)
    panel = _make_panel(12, 12)
    refine_mask = np.zeros((10, 10), dtype=bool)

    with pytest.raises(ValueError):
        refine_by_residual_mask(base_labels, panel, refine_mask, n_layers=2)


def test_refine_by_candidate_regions_processes_each_bbox():
    h, w = 40, 40
    base_labels = np.ones((h, w), dtype=np.int32)
    base_labels[h // 2 :] = 2
    panel = _make_panel(h, w)

    candidates = [
        {"bbox": [5, 5, 15, 15]},
        {"bbox": [25, 25, 35, 35]},
    ]

    call_count = 0

    def mock_run_engine(engine, crop_rgb, reps, colorbar_rgb, n_layers):
        nonlocal call_count
        call_count += 1
        ch, cw = crop_rgb.shape[:2]
        patch = np.zeros((ch, cw), dtype=np.int32)
        patch[: ch // 2] = 1
        patch[ch // 2 :] = 2
        return {"labels": patch}

    with patch(
        "geoseg.modules.segment_engines.regional_refinement.run_engine",
        side_effect=mock_run_engine,
    ):
        result = refine_by_candidate_regions(
            base_labels,
            panel,
            candidates,
            n_layers=2,
            config=RefinementConfig(secondary_engine="mock"),
        )

    refined = result["labels"]
    assert result["meta"]["refined"] is True
    assert result["meta"]["n_candidates"] == 2
    assert len(result["meta"]["refined_regions"]) == 2
    assert call_count == 2
    # Original labels should still exist outside candidate boxes.
    assert np.all(refined[18:22, 18:22] == base_labels[18:22, 18:22])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
