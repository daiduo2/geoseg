from __future__ import annotations

import numpy as np
import pytest

from geoseg.core.models import (
    coerce_bbox_xywh,
    empty_segmentation_result,
    make_panel_input,
    validate_segmentation_result,
)


def test_make_panel_input_normalizes_bbox_values():
    panel = make_panel_input(2, np.array([1, 2, 30, 40]), source="cv", confidence=0.8)

    assert panel == {
        "id": 2,
        "bbox": (1, 2, 30, 40),
        "source": "cv",
        "confidence": 0.8,
    }


def test_coerce_bbox_xywh_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="width and height"):
        coerce_bbox_xywh((1, 2, 0, 4))


def test_validate_segmentation_result_accepts_minimum_contract():
    result = empty_segmentation_result((5, 7, 3))

    validated = validate_segmentation_result(result, image_shape=(5, 7, 3))

    assert validated["meta"]["engine"] == "empty"
    assert validated["labels"].shape == (5, 7)


def test_validate_segmentation_result_rejects_bad_overlay_shape():
    result = {
        "labels": np.zeros((5, 7), dtype=np.int32),
        "overlay": np.zeros((4, 7, 3), dtype=np.uint8),
        "meta": {"engine": "mock"},
    }

    with pytest.raises(ValueError, match="overlay shape"):
        validate_segmentation_result(result)
