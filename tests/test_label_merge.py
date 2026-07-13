"""Tests for geoseg.preprocessing.label_merge."""
from __future__ import annotations

import numpy as np
import pytest

from geoseg.preprocessing.label_merge import merge_artifact_labels


def _three_label_map() -> np.ndarray:
    """Background 75%, region 1 (20%), region 2 (5%)."""
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[10:30, :] = 1
    labels[90:95, :] = 2
    return labels


def test_merge_small_label_by_area():
    labels = _three_label_map()

    merged = merge_artifact_labels(labels, min_area_frac=0.11)
    assert 2 not in np.unique(merged)


def test_merge_dark_label_by_brightness():
    labels = _three_label_map()
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    image[90:95, :] = 30  # dark artifact region

    merged = merge_artifact_labels(
        labels, image_rgb=image, max_mean_brightness=80, min_area_frac=0.11
    )
    assert 2 not in np.unique(merged)


def test_explicit_artifact_labels_ignore_area_and_brightness():
    labels = _three_label_map()
    image = np.full((100, 100, 3), 200, dtype=np.uint8)

    # min_area_frac is set so that label 2 (5%) is below it but label 1 (20%)
    # and the background are not.  Label 2 is explicitly merged anyway.
    merged = merge_artifact_labels(
        labels,
        image_rgb=image,
        max_mean_brightness=180,
        min_area_frac=0.1,
        artifact_labels=[2],
    )
    assert 2 not in np.unique(merged)
    assert 0 in np.unique(merged)
    assert 1 in np.unique(merged)


def test_no_merge_when_criteria_not_met():
    labels = _three_label_map()

    # No region is smaller than 1% and there is no brightness threshold.
    merged = merge_artifact_labels(labels, min_area_frac=0.01)
    assert set(np.unique(merged)) == {0, 1, 2}


def test_mask_overlap_merges_only_masked_portion():
    """A large label with a small artifact bump overlapping the mask.

    Only the masked portion should be merged; the unmasked bulk stays as label 1.
    """
    labels = np.zeros((100, 100), dtype=np.int32)
    # Large real region (label 1) on the left, neighbor (label 2) on the right.
    labels[10:90, 10:50] = 1
    labels[10:90, 50:90] = 2
    # A bump of label 1 that extends into label 2.
    labels[45:52, 48:55] = 1

    mask = np.zeros((100, 100), dtype=bool)
    # Mask covers only the right half of the bump (the part inside label 2).
    mask[45:52, 52:55] = True

    merged = merge_artifact_labels(labels, artifact_mask=mask, mask_overlap_frac=0.9)
    # Masked right part of the bump should become label 2.
    assert (merged[45:52, 52:55] == 2).all()
    # The unmasked left part of the bump and the bulk of label 1 must stay.
    assert (merged[45:52, 48:51] == 1).all()
    assert (merged[15:40, 15:40] == 1).all()


def test_explicit_label_merges_whole_label():
    labels = _three_label_map()
    merged = merge_artifact_labels(labels, artifact_labels=[1])
    assert 1 not in np.unique(merged)
    assert 0 in np.unique(merged)
    assert 2 in np.unique(merged)
