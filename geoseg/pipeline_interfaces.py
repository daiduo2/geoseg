"""Pipeline interface contracts.

Defines shared data structures and Protocol interfaces for the dual-pipeline
architecture (Manual Pipeline A + Agent Pipeline B).

Any module that produces or consumes panels, segmentations, or reviews
must use these types. Schema changes require updating ALL consumers.

Test scenario:
    >>> from geoseg.pipeline_interfaces import PanelInput, SegmentationResult
    >>> panel: PanelInput = {"id": 0, "bbox": (0, 0, 100, 100), "source": "cv_detect"}
    >>> panel["bbox"]
    (0, 0, 100, 100)
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Data contracts (TypedDict — JSON-serializable except ndarray fields)
# ---------------------------------------------------------------------------

class PanelInput(TypedDict, total=False):
    """A detected or manually-drawn panel bounding box.

    Universal input to segmentation engines, regardless of whether the panel
    came from CV detection, VLM hint, or user manual selection.

    Fields:
        id: Panel identifier (0-based).
        bbox: (x, y, w, h) in pixel coordinates.
        source: Provenance — "cv_detect", "manual", "vlm_hint", "fallback_whole".
        confidence: Detection confidence 0.0-1.0 (None for manual).
    """
    id: int
    bbox: tuple[int, int, int, int]
    source: str
    confidence: float | None


class SegmentationMeta(TypedDict, total=False):
    """Metadata describing how a segmentation was produced.

    Carries provenance so downstream steps know whether this came from an
    automatic engine, manual trace, or hybrid approach.
    """
    engine: str
    color_names: list[str]
    n_layers: int
    quality_score: float | None


class SegmentationResult(TypedDict, total=False):
    """Universal output of any segmentation step.

    ``labels`` is the single source of truth. ``overlay`` is display-only.
    ``meta`` carries provenance for audit trails.
    """
    labels: np.ndarray
    overlay: np.ndarray | None
    meta: SegmentationMeta


class QualityReview(TypedDict, total=False):
    """Quality review result from VLM or human reviewer.

    ``suggested_action`` tells the controller what to do next.
    """
    warnings: list[str]
    score: float
    can_auto_fix: bool
    suggested_action: str


class FigureClassification(TypedDict, total=False):
    """VLM figure classification output (M1)."""
    figure_type: str
    confidence: float
    reason: str


class PageOverview(TypedDict, total=False):
    """VLM page overview output (M1a) — figure-level review."""
    page_idx: int
    image_size: dict
    figure_type: str
    panels: list[dict]
    target_panel_id: int
    has_colorbar: bool
    color_zones: list[dict]
    confidence: float


# ---------------------------------------------------------------------------
# Protocol interfaces (runtime-checkable, for pluggable implementations)
# ---------------------------------------------------------------------------

@runtime_checkable
class PanelDetector(Protocol):
    """Detect panels in a figure image.

    Implementations:
    - Agent: ``cv_detect.panel_detector.detect_panels``
    - Manual: GUI user-drawn rectangles → wrapped as PanelInput list
    """

    def detect(self, img_rgb: np.ndarray) -> list[PanelInput]:
        """Return panels found in the image.

        Empty list signals "no distinct panels detected"; the caller should
        fall back to a whole-image panel (bbox = full image).
        """


@runtime_checkable
class Segmenter(Protocol):
    """Segment a figure or panel into labeled regions.

    Implementations:
    - Agent: ``segment_engines.router.route_and_segment``
    - Manual: trace-based or interactive segmentation (future)
    """

    def segment(
        self,
        img_rgb: np.ndarray,
        *,
        n_layers: int = 5,
        reps: list[tuple[int, int, int]] | None = None,
        colorbar_rgb: np.ndarray | None = None,
        **kwargs: object,
    ) -> SegmentationResult:
        """Segment the image into labeled regions."""


@runtime_checkable
class QualityReviewer(Protocol):
    """Review segmentation or panel detection quality.

    Implementations:
    - Agent: ``vlm_client.classify_figure``, ``vlm_client.review_page_overview``
    - Manual: human expert review dialog (future)
    """

    def review(self, img_rgb: np.ndarray, context: dict | None = None) -> QualityReview:
        """Review the image and return quality assessment."""


@runtime_checkable
class PipelineStep(Protocol):
    """Single step in either Pipeline A (manual) or Pipeline B (agent).

    All pipeline steps share this interface so they can be composed,
    logged, and swapped between pipelines.
    """

    def run(self, input_data: dict, context: dict | None = None) -> dict:
        """Execute the step and return standardized output."""


# ---------------------------------------------------------------------------
# Convenience helpers (stateless, immutable)
# ---------------------------------------------------------------------------

def make_whole_image_panel(img_rgb: np.ndarray) -> PanelInput:
    """Create a fallback PanelInput covering the entire image.

    Used when no panels are detected or when VLM review suggests whole-image
    segmentation (e.g. panel mismatch fallback).
    """
    h, w = img_rgb.shape[:2]
    return {
        "id": 0,
        "bbox": (0, 0, w, h),
        "source": "fallback_whole",
        "confidence": 1.0,
    }


def empty_segmentation_result(img_shape: tuple[int, ...]) -> SegmentationResult:
    """Create an empty SegmentationResult (all background).

    Useful as a safe default when segmentation fails or is skipped.
    """
    return {
        "labels": np.zeros(img_shape[:2], dtype=np.int32),
        "overlay": None,
        "meta": {
            "engine": "empty",
            "color_names": [],
            "n_layers": 0,
            "quality_score": 0.0,
        },
    }


__all__ = [
    "PanelInput",
    "SegmentationMeta",
    "SegmentationResult",
    "QualityReview",
    "FigureClassification",
    "PageOverview",
    "PanelDetector",
    "Segmenter",
    "QualityReviewer",
    "PipelineStep",
    "make_whole_image_panel",
    "empty_segmentation_result",
]
