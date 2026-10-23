"""Tests for deterministic small-component cleanup of a partition."""

from __future__ import annotations

import numpy as np

from geoseg.modules.curve_editor.partition import prune_small_components


def test_small_island_is_merged_but_thin_layer_is_kept():
    labels = np.ones((30, 40), dtype=np.int32)
    labels[10:13, 10:13] = 2      # 3x3 island, area 9, radius ~2
    labels[20:21, 4:34] = 3       # 1x30 thin layer, area 30

    cleaned, merges = prune_small_components(
        labels,
        min_area=20,
        max_thickness=3.0,
    )

    assert 2 not in np.unique(cleaned)
    assert 3 in np.unique(cleaned)
    assert len(merges) == 1
    assert merges[0].label == 2
    assert merges[0].target == 1
    assert merges[0].area == 9


def test_equal_shared_boundaries_use_smaller_label_as_tie_break():
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[:, :10] = 5
    labels[:, 10:] = 3
    labels[9:11, 9:11] = 9

    cleaned, merges = prune_small_components(
        labels,
        min_area=10,
        max_thickness=2.0,
    )

    assert 9 not in np.unique(cleaned)
    assert len(merges) == 1
    assert merges[0].target == 3
    assert merges[0].shared_boundary == 4


def test_background_components_are_protected_by_default():
    labels = np.ones((20, 20), dtype=np.int32)
    labels[5:7, 5:7] = 0

    cleaned, merges = prune_small_components(
        labels,
        min_area=10,
        max_thickness=3.0,
    )

    assert np.array_equal(cleaned, labels)
    assert merges == []


def test_foreground_fragment_does_not_merge_into_background_by_default():
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[5:7, 5:7] = 4

    cleaned, merges = prune_small_components(
        labels,
        min_area=10,
        max_thickness=3.0,
    )

    assert 4 in np.unique(cleaned)
    assert merges == []

    cleaned, merges = prune_small_components(
        labels,
        min_area=10,
        max_thickness=3.0,
        allow_background_merge=True,
    )

    assert 4 not in np.unique(cleaned)
    assert len(merges) == 1
    assert merges[0].target == 0
