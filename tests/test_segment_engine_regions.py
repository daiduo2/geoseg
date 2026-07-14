from __future__ import annotations

import numpy as np

from geoseg.modules.segment_engines.regions import reorder_labels_top_to_bottom


def test_reorder_labels_by_median_y_orders_top_to_bottom():
    labels = np.array(
        [
            [7, 7, 7],
            [3, 3, 3],
            [5, 5, 5],
        ],
        dtype=np.int32,
    )

    reordered = reorder_labels_top_to_bottom(labels)

    assert reordered.tolist() == [
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
    ]
