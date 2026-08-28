from __future__ import annotations

import numpy as np
from scipy import ndimage

from geoseg.modules.post_process.split import split_labels_by_red_boundaries
from geoseg.preprocessing.detectors import detect_red_boundaries


def test_detect_red_boundaries_keeps_line_and_rejects_compact_mark():
    image = np.full((100, 120, 3), 180, dtype=np.uint8)
    image[10:90, 58:62] = [220, 30, 30]
    image[20:28, 20:28] = [220, 30, 30]

    mask = detect_red_boundaries(image, closing_radius=0)

    assert mask[50, 60]
    assert not mask[24, 24]


def test_red_boundary_erosion_closes_endpoint_gaps_and_splits_region():
    image = np.full((100, 120, 3), [180, 190, 120], dtype=np.uint8)
    labels = np.ones((100, 120), dtype=np.int32)
    boundary = np.zeros_like(labels, dtype=bool)
    boundary[5:95, 59:62] = True

    refined, parent_map, returned_boundary = split_labels_by_red_boundaries(
        labels,
        image,
        boundary_mask=boundary,
        erosion_radius=6,
        min_component_area_frac=0.01,
    )

    assert parent_map == {1: 1, 2: 1}
    np.testing.assert_array_equal(returned_boundary, boundary)
    assert set(np.unique(refined)) == {0, 1, 2}
    assert refined[50, 20] != refined[50, 100]
    for label_id in (1, 2):
        _, count = ndimage.label(refined == label_id)
        assert count == 1


def test_red_boundary_split_validates_boundary_shape():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    labels = np.ones((20, 30), dtype=np.int32)

    with np.testing.assert_raises_regex(ValueError, "boundary_mask"):
        split_labels_by_red_boundaries(
            labels,
            image,
            boundary_mask=np.zeros((10, 10), dtype=bool),
        )
