"""HTTP API schema models for the geoseg FastAPI server."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PanelInput(BaseModel):
    """Panel detection/selection minimal contract."""

    id: int = Field(..., description="Panel identifier (0-based)")
    bbox: tuple[int, int, int, int] = Field(
        ..., description="(x, y, width, height) in pixel coordinates"
    )
    source: str = Field(
        ..., description='Provenance: "cv_detect" | "manual" | "vlm_hint" | "fallback_whole"'
    )
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class SegmentationMeta(BaseModel):
    """Metadata describing how a segmentation was produced."""

    engine: str
    color_names: list[str]
    n_layers: int
    quality_score: float | None = None


class SegmentationResult(BaseModel):
    """Universal segmentation output for the HTTP API."""

    labels_base64: str | None = Field(
        None, description="Base64-encoded compressed NPZ (debug only)"
    )
    contours: list[list[dict[str, int]]] = Field(
        ..., description="List of contour polygons, each is [{x:int, y:int}, ...]"
    )
    overlay_base64: str | None = Field(None, description="Base64 PNG overlay")
    meta: SegmentationMeta


class FigureClassificationOut(BaseModel):
    """Figure classification result."""

    figure_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str


class PanelReviewOut(BaseModel):
    """Per-panel review info."""

    panel_id: int
    bbox: tuple[int, int, int, int]
    classification: dict[str, Any]
    segmentation: SegmentationResult
    review: dict[str, Any]


class AgentProcessFigureResponse(BaseModel):
    """Response from POST /api/agent/process-figure."""

    classification: dict[str, Any]
    panels: list[PanelReviewOut]
    summary: dict[str, Any]


class QualityReviewDialog(BaseModel):
    """Quality review for frontend dialog display."""

    warnings: list[str]
    score: float
    can_auto_fix: bool
    suggested_action: str = Field(
        ..., pattern=r"^(continue|retry|manual_intervention|skip)$"
    )


class ExportSpecfemResponse(BaseModel):
    """Response from POST /api/export/specfem."""

    tomo_xyz: str = Field(..., description="Tomography file content or download URL")
    parfile_snippet: str = Field(..., description="Par_file snippet content")


class PdfImportResponse(BaseModel):
    """Response from POST /api/pdf/import."""

    job_id: str
    status: str = "accepted"


class PdfStatusResponse(BaseModel):
    """Response from GET /api/pdf/status/{job_id}."""

    status: str = Field(..., pattern=r"^(pending|done|error)$")
    figures: list[dict[str, Any]] = []
    message: str = ""


__all__ = [
    "AgentProcessFigureResponse",
    "ExportSpecfemResponse",
    "FigureClassificationOut",
    "PanelInput",
    "PanelReviewOut",
    "PdfImportResponse",
    "PdfStatusResponse",
    "QualityReviewDialog",
    "SegmentationMeta",
    "SegmentationResult",
]
