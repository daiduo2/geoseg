"""Stable data contracts and protocols for geoseg pipelines."""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

import numpy as np


class PanelInput(TypedDict, total=False):
    """A detected or manually-drawn panel bounding box."""

    id: int
    bbox: tuple[int, int, int, int]
    source: str
    confidence: float | None


class RegionalAudit(TypedDict, total=False):
    """Agent regional audit result."""

    frozen_labels: list[int]
    retry_labels: list[int]
    notes: str
    iteration: int


class SegmentationMeta(TypedDict, total=False):
    """Metadata describing how a segmentation was produced."""

    engine: str
    color_names: list[str]
    n_layers: int
    quality_score: float | None
    edited: bool
    editor_version: str | None
    parent_engine: str | None
    fusion_applied: bool
    primary_engine: str | None
    secondary_engine: str | None
    frozen_labels: list[int]
    retry_labels: list[int]
    fusion_notes: str
    iteration: int


class SegmentationResult(TypedDict, total=False):
    """Universal output of any segmentation step."""

    labels: np.ndarray
    overlay: np.ndarray | None
    meta: SegmentationMeta


class QualityReview(TypedDict, total=False):
    """Quality review result from agent or human reviewer."""

    warnings: list[str]
    score: float
    can_auto_fix: bool
    suggested_action: str


class FigureClassification(TypedDict, total=False):
    """Figure classification output."""

    figure_type: str
    confidence: float
    reason: str


class PageOverview(TypedDict, total=False):
    """Figure-level page overview output."""

    page_idx: int
    image_size: dict
    figure_type: str
    panels: list[dict]
    target_panel_id: int
    has_colorbar: bool
    color_zones: list[dict]
    confidence: float


@runtime_checkable
class PanelDetector(Protocol):
    """Detect panels in a figure image."""

    def detect(self, img_rgb: np.ndarray) -> list[PanelInput]:
        """Return panels found in the image."""


@runtime_checkable
class Segmenter(Protocol):
    """Segment a figure or panel into labeled regions."""

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
    """Review segmentation or panel detection quality."""

    def review(self, img_rgb: np.ndarray, context: dict | None = None) -> QualityReview:
        """Review the image and return quality assessment."""


@runtime_checkable
class PipelineStep(Protocol):
    """Single pipeline step contract."""

    def run(self, input_data: dict, context: dict | None = None) -> dict:
        """Execute the step and return standardized output."""


def make_whole_image_panel(img_rgb: np.ndarray) -> PanelInput:
    """Create a fallback PanelInput covering the entire image."""
    h, w = img_rgb.shape[:2]
    return {
        "id": 0,
        "bbox": (0, 0, w, h),
        "source": "fallback_whole",
        "confidence": 1.0,
    }


def empty_segmentation_result(img_shape: tuple[int, ...]) -> SegmentationResult:
    """Create an empty SegmentationResult with all pixels set to background."""
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
    "FigureClassification",
    "PageOverview",
    "PanelDetector",
    "PanelInput",
    "PipelineStep",
    "QualityReview",
    "QualityReviewer",
    "RegionalAudit",
    "Segmenter",
    "SegmentationMeta",
    "SegmentationResult",
    "empty_segmentation_result",
    "make_whole_image_panel",
]
