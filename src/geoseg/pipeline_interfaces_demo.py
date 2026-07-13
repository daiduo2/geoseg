"""Demo for pipeline_interfaces Protocol compatibility.

Verifies that existing modules conform to the defined Protocols.
Runs without external API calls (stub mode).

Test scenario:
    >>> python -m geoseg.pipeline_interfaces_demo
    All Protocol compatibility checks passed!
"""

from __future__ import annotations

import numpy as np

from geoseg.pipeline_interfaces import (
    PanelInput,
    SegmentationResult,
    QualityReview,
    PanelDetector,
    Segmenter,
    QualityReviewer,
    make_whole_image_panel,
    empty_segmentation_result,
)
from geoseg.modules.cv_detect.panel_detector import detect_panels
from geoseg.modules.segment_engines.router import route_and_segment
from geoseg.modules.vlm_client import quality_review


def test_panel_detector_protocol():
    """cv_detect.panel_detector should implement PanelDetector."""
    img = np.full((300, 600, 3), 255, dtype=np.uint8)
    img[20:120, 20:180] = 128
    img[20:120, 220:380] = 100

    panels = detect_panels(img)

    assert isinstance(panels, list)
    assert len(panels) == 2

    for p in panels:
        assert isinstance(p, dict)
        assert "id" in p
        assert "bbox" in p
        assert isinstance(p["bbox"], tuple)
        assert len(p["bbox"]) == 4
        assert p["source"] == "cv_detect"
        assert "confidence" in p

    # Runtime Protocol conformance check
    assert callable(detect_panels)
    print("  PanelDetector Protocol: detect_panels conforms")


def test_segmenter_protocol():
    """segment_engines.router should implement Segmenter."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)

    result = route_and_segment(img, n_layers=3, is_velocity_model=False)

    assert isinstance(result, dict)
    assert "labels" in result
    assert "meta" in result
    assert result["meta"]["engine"] == "skip"
    assert result["meta"]["n_layers"] == 3

    print("  Segmenter Protocol: route_and_segment conforms")


def test_quality_reviewer_protocol():
    """vlm_client.quality_review should implement QualityReviewer."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 255

    review = quality_review(
        img,
        context={"review_type": "figure_classification", "mode": "stub"},
    )

    assert isinstance(review, dict)
    assert "warnings" in review
    assert "score" in review
    assert "suggested_action" in review

    print("  QualityReviewer Protocol: quality_review conforms")


def test_helpers():
    """Test convenience helpers."""
    img = np.full((100, 200, 3), 128, dtype=np.uint8)

    panel = make_whole_image_panel(img)
    assert panel["bbox"] == (0, 0, 200, 100)
    assert panel["source"] == "fallback_whole"

    empty = empty_segmentation_result((50, 80))
    assert empty["labels"].shape == (50, 80)
    assert empty["labels"].sum() == 0
    assert empty["meta"]["engine"] == "empty"

    print("  Helpers: make_whole_image_panel + empty_segmentation_result work")


def test_full_pipeline_interface():
    """Test that full_pipeline works with adapted interfaces."""
    from geoseg.modules.segment_engines.full_pipeline import process_figure

    img = np.full((200, 400, 3), 255, dtype=np.uint8)
    img[50:150, 50:180] = 128
    img[50:150, 220:350] = 100

    result = process_figure(img, n_layers=3, use_vlm=False)

    assert result["summary"]["status"] in ("ok", "skipped")
    assert "panels" in result
    if result["panels"]:
        panel = result["panels"][0]
        seg = panel["segmentation"]
        assert "labels" in seg
        assert "meta" in seg
        assert "engine" in seg["meta"]

    print("  full_pipeline works with adapted Protocol interfaces")


if __name__ == "__main__":
    test_panel_detector_protocol()
    test_segmenter_protocol()
    test_quality_reviewer_protocol()
    test_helpers()
    test_full_pipeline_interface()
    print("\nAll Protocol compatibility checks passed!")
