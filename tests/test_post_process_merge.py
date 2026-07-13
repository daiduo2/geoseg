"""Tests for post_process merge utilities."""
from __future__ import annotations

import numpy as np
import pytest

from geoseg.modules.post_process.merge import merge_labels_by_ids, merge_warm_labels


class TestMergeLabelsByIds:
    def test_merge_two_labels(self):
        labels = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int32)
        result = merge_labels_by_ids(labels, [1, 2], target_id=1)
        expected = np.array([[1, 1, 1], [1, 3, 3]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_merge_all_nonzero(self):
        labels = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.int32)
        result = merge_labels_by_ids(labels, [1, 2, 3], target_id=5)
        expected = np.array([[0, 5, 5], [5, 0, 5]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_no_op_when_single_label(self):
        labels = np.array([[1, 1], [1, 0]], dtype=np.int32)
        result = merge_labels_by_ids(labels, [1], target_id=1)
        np.testing.assert_array_equal(result, labels)

    def test_does_not_mutate_input(self):
        labels = np.array([[1, 2], [2, 1]], dtype=np.int32)
        original = labels.copy()
        merge_labels_by_ids(labels, [1, 2], target_id=5)
        np.testing.assert_array_equal(labels, original)


class TestMergeWarmLabels:
    def test_shape_mismatch_raises(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            merge_warm_labels(labels, img)

    def test_all_cool_labels_no_merge(self):
        """Pure blue image — no warm labels to merge."""
        labels = np.array([[1, 1, 2], [2, 2, 1]], dtype=np.int32)
        img = np.full((2, 3, 3), [0, 0, 255], dtype=np.uint8)  # pure blue
        result = merge_warm_labels(labels, img, warm_threshold=0.5)
        # All labels are cool, so no merge should happen
        np.testing.assert_array_equal(result, labels)

    def test_warm_labels_get_merged(self):
        """Orange image — all labels should merge into label 1."""
        h, w = 4, 4
        labels = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ], dtype=np.int32)
        # Orange-ish RGB
        img = np.full((h, w, 3), [230, 140, 60], dtype=np.uint8)
        result = merge_warm_labels(labels, img, warm_threshold=0.5)
        # Labels 1, 2, 3 are all warm — should all become label 1
        expected = np.ones_like(labels)
        np.testing.assert_array_equal(result, expected)

    def test_mixed_warm_and_cool(self):
        """Top half orange (warm), bottom half blue (cool)."""
        h, w = 4, 4
        labels = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ], dtype=np.int32)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:2] = [230, 140, 60]   # orange
        img[2:] = [40, 60, 200]    # blue
        result = merge_warm_labels(labels, img, warm_threshold=0.5)
        # Labels 1, 2 are warm; 3, 4 are cool
        expected = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_small_fragments_absorbed(self):
        """Small non-warm specks inside warm region get absorbed."""
        h, w = 6, 6
        labels = np.ones((h, w), dtype=np.int32)
        # Place a tiny cool label inside the warm region
        labels[2, 2] = 2
        img = np.full((h, w, 3), [230, 140, 60], dtype=np.uint8)
        # Make the tiny speck a different (cool) colour
        img[2, 2] = [40, 60, 200]
        result = merge_warm_labels(labels, img, warm_threshold=0.5, min_component_size=10)
        # The single-pixel component should be absorbed into label 1
        assert result[2, 2] == 1
        assert (result == 1).sum() == h * w

    def test_does_not_mutate_input(self):
        labels = np.array([[1, 2], [2, 1]], dtype=np.int32)
        img = np.full((2, 2, 3), [230, 140, 60], dtype=np.uint8)
        original = labels.copy()
        merge_warm_labels(labels, img)
        np.testing.assert_array_equal(labels, original)

    def test_fill_background_recover_warm_bg(self):
        """Background pixels colour-similar to plume get filled when requested."""
        h, w = 6, 6
        labels = np.ones((h, w), dtype=np.int32)
        labels[2:4, 2:4] = 0  # background hole inside warm region
        img = np.full((h, w, 3), [230, 140, 60], dtype=np.uint8)
        # Without fill_background, hole stays as background
        result_no_fill = merge_warm_labels(labels, img, warm_threshold=0.5, fill_background=False)
        assert (result_no_fill == 0).sum() == 4
        # With fill_background, hole is recovered
        result_fill = merge_warm_labels(labels, img, warm_threshold=0.5, fill_background=True)
        assert (result_fill == 0).sum() == 0
        assert (result_fill == 1).sum() == h * w


class TestPanel3EndToEnd:
    """Run merge_warm_labels on the real panel 3 data."""

    def test_panel3_plume_merge(self):
        from pathlib import Path
        from PIL import Image

        base = Path("runs/3d_schematic_correct_e2e/panel_3_front")
        if not base.exists():
            pytest.skip("panel 3 e2e data not available")

        labels = np.load(base / "labels_primary.npz")["labels"]
        img = np.array(Image.open(base / "00_enhanced.jpg").convert("RGB"))

        # Panel 3: label 1 contains warm clouds at the top that must not be
        # merged into the plume.  Label 3 is a wide warm band that needs
        # aggressive centre-cropping to avoid swallowing side crust.
        result = merge_warm_labels(
            labels,
            img,
            warm_threshold=0.5,
            fill_background=True,
            max_width_ratio=0.85,
            center_column_ratio=0.5,
            fill_max_dist=30,
            exclude_labels=[1],
            per_label_crop={3: 0.30, 4: 1.0},
        )
        unique_before = set(labels.flatten()) - {0}
        unique_after = set(result.flatten()) - {0}

        # Should have fewer labels after merging
        assert len(unique_after) <= len(unique_before)
        # Label 1 (merged plume) should be present
        assert (result == 1).sum() > 0
        # Plume coverage should be in a reasonable range (manual ground truth ~27%)
        coverage = (result == 1).sum() / result.size
        assert 0.20 < coverage < 0.35, f"Plume coverage {coverage:.2%} outside expected 20-35%"

        # Save for manual inspection if needed
        out = Path("runs/tubular_panel3")
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "test_merge_result.npz", labels=result)
