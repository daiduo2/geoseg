"""Stable data contracts and protocols for geoseg pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NotRequired, Protocol, TypedDict, runtime_checkable

import numpy as np


BBoxXYWH = tuple[int, int, int, int]
RGBColor = tuple[int, int, int]


class PanelInput(TypedDict):
    """A detected or manually-drawn panel bounding box."""

    id: int
    bbox: BBoxXYWH
    source: NotRequired[str]
    confidence: NotRequired[float | None]


class RegionalAudit(TypedDict, total=False):
    """Agent regional audit result."""

    frozen_labels: list[int]
    retry_labels: list[int]
    notes: str
    iteration: int


class SegmentationMeta(TypedDict):
    """Metadata describing how a segmentation was produced."""

    engine: str
    color_names: NotRequired[list[str]]
    n_layers: NotRequired[int]
    quality_score: NotRequired[float | None]
    edited: NotRequired[bool]
    editor_version: NotRequired[str | None]
    parent_engine: NotRequired[str | None]
    fusion_applied: NotRequired[bool]
    primary_engine: NotRequired[str | None]
    secondary_engine: NotRequired[str | None]
    frozen_labels: NotRequired[list[int]]
    retry_labels: NotRequired[list[int]]
    fusion_notes: NotRequired[str]
    iteration: NotRequired[int]
    boundary_mode: NotRequired[str]
    boundary_pixels: NotRequired[int]
    parent_labels: NotRequired[dict[int, int]]


class SegmentationResult(TypedDict):
    """Universal output of any segmentation step."""

    labels: np.ndarray
    meta: SegmentationMeta
    overlay: NotRequired[np.ndarray | None]
    seeds: NotRequired[list | np.ndarray]
    color_partition: NotRequired[np.ndarray]
    boundary_mask: NotRequired[np.ndarray]


class QualityReview(TypedDict, total=False):
    """Quality review result from agent or human reviewer."""

    warnings: list[str]
    score: float
    can_auto_fix: bool
    suggested_action: str


class FigureClassification(TypedDict):
    """Figure classification output."""

    figure_type: str
    confidence: NotRequired[float]
    reason: NotRequired[str]


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
        reps: list[RGBColor] | None = None,
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


def coerce_bbox_xywh(bbox: Sequence[int]) -> BBoxXYWH:
    """Validate and normalize an ``x, y, width, height`` bbox."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain 4 values, got {len(bbox)}")

    x, y, w, h = (int(v) for v in bbox)
    if x < 0 or y < 0:
        raise ValueError(f"bbox origin must be non-negative, got {(x, y)}")
    if w <= 0 or h <= 0:
        raise ValueError(f"bbox width and height must be positive, got {(w, h)}")
    return x, y, w, h


def make_panel_input(
    panel_id: int,
    bbox: Sequence[int],
    *,
    source: str | None = None,
    confidence: float | None = None,
) -> PanelInput:
    """Create a validated PanelInput."""
    panel: PanelInput = {
        "id": int(panel_id),
        "bbox": coerce_bbox_xywh(bbox),
    }
    if source is not None:
        panel["source"] = source
    if confidence is not None:
        panel["confidence"] = float(confidence)
    return panel


def make_whole_image_panel(img_rgb: np.ndarray) -> PanelInput:
    """Create a fallback PanelInput covering the entire image."""
    h, w = img_rgb.shape[:2]
    return make_panel_input(0, (0, 0, w, h), source="fallback_whole", confidence=1.0)


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


def validate_segmentation_result(
    result: dict,
    *,
    image_shape: tuple[int, ...] | None = None,
) -> SegmentationResult:
    """Validate the minimum runtime contract for a segmentation result."""
    labels = result.get("labels")
    if not isinstance(labels, np.ndarray):
        raise TypeError("segmentation labels must be a numpy.ndarray")
    if labels.ndim != 2:
        raise ValueError(f"segmentation labels must be 2D, got shape {labels.shape}")
    if image_shape is not None and labels.shape != tuple(image_shape[:2]):
        raise ValueError(
            f"segmentation labels shape {labels.shape} "
            f"does not match image shape {image_shape[:2]}"
        )

    meta = result.get("meta")
    if not isinstance(meta, dict):
        raise TypeError("segmentation meta must be a dict")
    if not isinstance(meta.get("engine"), str) or not meta["engine"]:
        raise ValueError("segmentation meta.engine must be a non-empty string")

    overlay = result.get("overlay")
    if overlay is not None:
        if not isinstance(overlay, np.ndarray):
            raise TypeError("segmentation overlay must be a numpy.ndarray or None")
        if overlay.shape[:2] != labels.shape:
            raise ValueError(
                f"segmentation overlay shape {overlay.shape[:2]} does not match labels {labels.shape}"
            )

    return result  # type: ignore[return-value]


__all__ = [
    "BBoxXYWH",
    "FigureClassification",
    "PageOverview",
    "PanelDetector",
    "PanelInput",
    "PipelineStep",
    "QualityReview",
    "QualityReviewer",
    "RegionalAudit",
    "RGBColor",
    "Segmenter",
    "SegmentationMeta",
    "SegmentationResult",
    "coerce_bbox_xywh",
    "empty_segmentation_result",
    "make_panel_input",
    "make_whole_image_panel",
    "validate_segmentation_result",
]
