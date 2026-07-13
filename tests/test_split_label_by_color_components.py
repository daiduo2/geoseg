"""Tests for post_process split utilities."""
from __future__ import annotations

import numpy as np
import pytest

from geoseg.modules.post_process.split import split_label_by_color_components


class TestSplitLabelByColorComponents:
    def test_two_color_regions_split(self):
        """A label containing two distinct colours should split into two."""
        h, w = 20, 20
        labels = np.zeros((h, w), dtype=np.int32)
        # Label 1 covers two horizontally-separated colour blocks
        labels[:, :10] = 1
        labels[:, 10:] = 1

        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :10] = [200, 50, 50]   # reddish
        img[:, 10:] = [50, 200, 50]   # greenish

        result = split_label_by_color_components(
            labels, img, target_label=1, k=2, min_component_area=50, seed=42
        )

        # Original background preserved
        assert (result == 0).sum() == 0
        # Label 1 was replaced by two new compact labels
        unique = set(result.flatten())
        assert len(unique) == 2
        # Each new label should match one of the colour blocks
        assert (result[:, :10] == result[:, :10][0, 0]).all()
        assert (result[:, 10:] == result[:, 10:][0, 0]).all()
        assert result[0, 0] != result[0, 10]

    def test_shape_mismatch_raises(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Shape mismatch"):
            split_label_by_color_components(labels, img, target_label=1)

    def test_invalid_color_space_raises(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported color_space"):
            split_label_by_color_components(labels, img, target_label=1, color_space="HSV")

    def test_invalid_k_raises(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="k must be"):
            split_label_by_color_components(labels, img, target_label=1, k=0)

    def test_invalid_min_component_area_raises(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="min_component_area"):
            split_label_by_color_components(labels, img, target_label=1, min_component_area=-1)

    def test_no_target_label_returns_copy(self):
        labels = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int32)
        img = np.random.default_rng(0).integers(0, 255, size=(2, 3, 3), dtype=np.uint8)
        result = split_label_by_color_components(labels, img, target_label=9, k=2)
        np.testing.assert_array_equal(result, labels)

    def test_does_not_mutate_input(self):
        labels = np.array([[1, 1, 2], [2, 2, 1]], dtype=np.int32)
        img = np.full((2, 3, 3), [100, 100, 100], dtype=np.uint8)
        original_labels = labels.copy()
        original_img = img.copy()
        split_label_by_color_components(labels, img, target_label=1, k=1)
        np.testing.assert_array_equal(labels, original_labels)
        np.testing.assert_array_equal(img, original_img)

    def test_small_components_dropped_and_reassigned(self):
        """Tiny colour components are absorbed into the nearest large one."""
        h, w = 20, 20
        labels = np.ones((h, w), dtype=np.int32)
        img = np.full((h, w, 3), [200, 50, 50], dtype=np.uint8)
        # One tiny region of a different colour
        img[5:7, 5:7] = [50, 200, 50]

        result = split_label_by_color_components(
            labels, img, target_label=1, k=2, min_component_area=50, seed=42
        )

        unique = set(result.flatten())
        # The small 2x2 component should be merged into the large one
        assert len(unique) == 1

    def test_preserved_labels_unchanged(self):
        """Non-target labels keep their original values."""
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:5, :] = 1
        labels[5:, :] = 2

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:5, :] = [200, 50, 50]
        img[5:, :] = [50, 200, 50]

        result = split_label_by_color_components(
            labels, img, target_label=1, k=2, min_component_area=10, seed=42
        )

        # Label 2 region should be untouched
        assert (result[5:, :] == 2).all()
        # Original label 1 region should no longer contain label 1
        assert (result[:5, :] != 1).all()

    def test_k_larger_than_pixels_reduces_gracefully(self):
        """If k exceeds the number of masked pixels, it is capped."""
        labels = np.ones((2, 2), dtype=np.int32)
        img = np.full((2, 2, 3), [100, 100, 100], dtype=np.uint8)
        result = split_label_by_color_components(
            labels, img, target_label=1, k=10, min_component_area=1, seed=42
        )
        # Should still produce a valid label map
        assert result.shape == labels.shape
        assert (result != 1).all() or (result == 1).all()

    def test_spatial_split_when_k_equals_one(self):
        """With k=1 the function still splits by spatial connectivity."""
        labels = np.zeros((10, 10), dtype=np.int32)
        # Two disconnected blobs with same colour
        labels[1:4, 1:4] = 1
        labels[6:9, 6:9] = 1

        img = np.full((10, 10, 3), [120, 120, 120], dtype=np.uint8)

        result = split_label_by_color_components(
            labels, img, target_label=1, k=1, min_component_area=1, seed=42
        )

        unique = set(result.flatten()) - {0}
        assert len(unique) == 2
