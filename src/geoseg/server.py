"""FastAPI HTTP server — contract layer between TS frontend and Python backend.

This file is the **contract landing** for DESIGN.md v0.7 §4.1 HTTP API schema.
All request/response models are defined here; business logic is delegated to
existing modules (controller, full_pipeline, etc.).

Start:
    python -m geoseg.server

Test:
    curl -X POST http://localhost:8000/api/agent/detect-panels -F image=@test.png
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field
from PIL import Image
from skimage import measure

app = FastAPI(title="geoseg", version="2.0.0")


# ---------------------------------------------------------------------------
# Shared schema models (match pipeline_interfaces.py + DESIGN.md §4.1)
# ---------------------------------------------------------------------------

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
    """Universal segmentation output (labels as base64 NPZ; contours as JSON)."""

    labels_base64: str | None = Field(
        None, description="Base64-encoded compressed NPZ (debug only)"
    )
    contours: list[list[dict[str, int]]] = Field(
        ..., description="List of contour polygons, each is [{x:int, y:int}, ...]"
    )
    overlay_base64: str | None = Field(None, description="Base64 PNG overlay")
    meta: SegmentationMeta


class FigureClassificationOut(BaseModel):
    """Figure classification result (wrapper around pipeline_interfaces.FigureClassification)."""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _upload_to_ndarray(upload: UploadFile) -> np.ndarray:
    """Convert an uploaded image file to RGB ndarray."""
    image = Image.open(upload.file).convert("RGB")
    return np.array(image)


def _labels_to_contours(labels: np.ndarray) -> list[list[dict[str, int]]]:
    """Convert label map to list of contour polygons for frontend rendering.

    Each unique non-zero label produces one or more contours.
    """
    contours_out: list[list[dict[str, int]]] = []
    for idx in sorted(set(labels.flatten()) - {0}):
        mask = labels == idx
        if not mask.any():
            continue
        contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
        for cnt in contours:
            if len(cnt) < 4:
                continue
            poly = [{"x": int(round(p[1])), "y": int(round(p[0]))} for p in cnt]
            contours_out.append(poly)
    return contours_out


def _labels_to_base64(labels: np.ndarray) -> str:
    """Compress labels ndarray to base64-encoded NPZ bytes."""
    buf = io.BytesIO()
    np.savez_compressed(buf, labels=labels)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _overlay_to_base64(overlay: np.ndarray | None) -> str | None:
    """Encode overlay RGB ndarray to base64 PNG."""
    if overlay is None:
        return None
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _segmentation_to_api(seg: dict[str, Any]) -> SegmentationResult:
    """Convert internal SegmentationResult (with ndarray) to API response."""
    labels = seg["labels"]
    overlay = seg.get("overlay")
    meta = seg.get("meta", {})
    return SegmentationResult(
        labels_base64=_labels_to_base64(labels),
        contours=_labels_to_contours(labels),
        overlay_base64=_overlay_to_base64(overlay),
        meta=SegmentationMeta(
            engine=meta.get("engine", "unknown"),
            color_names=meta.get("color_names", []),
            n_layers=meta.get("n_layers", 0),
            quality_score=meta.get("quality_score"),
        ),
    )


# ---------------------------------------------------------------------------
# PDF endpoints
# ---------------------------------------------------------------------------

@app.post("/api/pdf/import", response_model=PdfImportResponse)
async def import_pdf(pdf: UploadFile = File(...)) -> PdfImportResponse:
    """Upload a PDF and start MinerU extraction.

    Returns a job_id for polling status.
    """
    # TODO: delegate to batch_processor or mineru_client
    raise NotImplementedError("import_pdf stub — implement in Week 5")


@app.get("/api/pdf/status/{job_id}", response_model=PdfStatusResponse)
async def pdf_status(job_id: str) -> PdfStatusResponse:
    """Poll extraction status and get figure list."""
    # TODO: query job store
    raise NotImplementedError("pdf_status stub — implement in Week 5")


# ---------------------------------------------------------------------------
# Pipeline B — Agent endpoints
# ---------------------------------------------------------------------------

@app.post("/api/agent/process-figure", response_model=AgentProcessFigureResponse)
async def process_figure_agent(
    image: UploadFile = File(...),
    caption: str = Form(""),
    text_blocks: str = Form("[]"),  # JSON string
    n_layers: int = Form(5),
    quality_preference: str = Form("balanced"),
) -> AgentProcessFigureResponse:
    """Run the full Agent pipeline on a figure image.

    Steps: classify → review_page_overview → detect_panels → segment.
    """
    img_rgb = _upload_to_ndarray(image)
    _tb: list[dict] = json.loads(text_blocks)

    from geoseg.controller import run_pipeline

    result = run_pipeline(
        img_rgb,
        caption=caption,
        text_blocks=_tb,
        n_layers=n_layers,
        quality_preference=quality_preference,
        skip_non_velocity_model=True,
        use_vlm=True,
        save_intermediates=False,
    )

    # Build response
    panels_out: list[PanelReviewOut] = []
    for p in result.get("panels", []):
        seg = p.get("segmentation")
        panels_out.append(
            PanelReviewOut(
                panel_id=p["panel_id"],
                bbox=tuple(p["bbox"]),  # type: ignore[arg-type]
                classification=p.get("classification", {}),
                segmentation=_segmentation_to_api(seg) if seg else _segmentation_to_api({"labels": np.zeros((10, 10), dtype=np.int32), "meta": {"engine": "empty", "color_names": [], "n_layers": 0}}),
                review=p.get("review", {}),
            )
        )

    return AgentProcessFigureResponse(
        classification=result.get("classification", {}),
        panels=panels_out,
        summary=result.get("summary", {}),
    )


@app.post("/api/agent/detect-panels", response_model=list[PanelInput])
async def detect_panels_agent(image: UploadFile = File(...)) -> list[PanelInput]:
    """Detect panels in a figure image using CV."""
    img_rgb = _upload_to_ndarray(image)

    from geoseg.modules.cv_detect.panel_detector import detect_panels

    bboxes = detect_panels(img_rgb)
    if not bboxes:
        from geoseg.pipeline_interfaces import make_whole_image_panel

        bboxes = [make_whole_image_panel(img_rgb)]

    return [
        PanelInput(
            id=pb["id"],
            bbox=pb["bbox"],
            source=pb.get("source", "cv_detect"),
            confidence=pb.get("confidence"),
        )
        for pb in bboxes
    ]


@app.post("/api/agent/segment", response_model=SegmentationResult)
async def segment_agent(
    image: UploadFile = File(...),
    n_layers: int = Form(5),
    reps: str | None = Form(None),  # JSON string or None
) -> SegmentationResult:
    """Segment a panel image using the automatic router."""
    img_rgb = _upload_to_ndarray(image)

    from geoseg.modules.segment_engines.router import route_and_segment

    kwargs: dict[str, Any] = {"n_layers": n_layers}
    if reps:
        kwargs["reps"] = json.loads(reps)

    seg = route_and_segment(img_rgb, **kwargs)
    return _segmentation_to_api(seg)


# ---------------------------------------------------------------------------
# Pipeline A — Manual endpoints
# ---------------------------------------------------------------------------

@app.post("/api/manual/segment-from-polygon", response_model=SegmentationResult)
async def segment_from_polygon(
    image: UploadFile = File(...),
    polygon: str = Form(...),  # JSON: [{"x":int, "y":int}, ...]
    n_layers: int = Form(5),
) -> SegmentationResult:
    """Segment using a user-drawn polygon mask.

    Strategy:
    1. Create binary mask from polygon.
    2. Run auto-segmentation on the full image.
    3. Mask out pixels outside the polygon (set to 0).
    4. Renumber remaining labels contiguously.
    """
    img_rgb = _upload_to_ndarray(image)
    _poly: list[dict[str, int]] = json.loads(polygon)

    from PIL import ImageDraw

    h, w = img_rgb.shape[:2]

    # Build binary mask from polygon
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon([(p["x"], p["y"]) for p in _poly], fill=255)
    mask = np.array(mask_img) > 0

    # Run auto-segmentation
    from geoseg.modules.segment_engines.router import route_and_segment

    seg = route_and_segment(img_rgb, n_layers=n_layers)
    labels = seg["labels"].copy()

    # Mask out pixels outside polygon
    labels[~mask] = 0

    # Renumber remaining labels contiguously starting from 1
    unique = sorted(set(labels.flatten()) - {0})
    relabel_map = {old: new for new, old in enumerate(unique, start=1)}
    new_labels = np.zeros_like(labels)
    for old, new in relabel_map.items():
        new_labels[labels == old] = new

    # Build result
    meta = dict(seg.get("meta", {}))
    meta["engine"] = "manual_polygon"
    meta["n_layers"] = len(unique)

    return _segmentation_to_api({"labels": new_labels, "overlay": seg.get("overlay"), "meta": meta})


@app.post("/api/manual/segment-from-rect", response_model=SegmentationResult)
async def segment_from_rect(
    image: UploadFile = File(...),
    bbox: str = Form(...),  # JSON: [x, y, w, h]
    n_layers: int = Form(5),
) -> SegmentationResult:
    """Segment inside a user-drawn bbox using grab-cut / graph-cut shrink."""
    img_rgb = _upload_to_ndarray(image)
    _bbox: tuple[int, int, int, int] = tuple(json.loads(bbox))  # type: ignore[assignment]

    # TODO: implement grab-cut / graph-cut in bbox
    raise NotImplementedError("segment_from_rect stub — implement in Week 4")


@app.post("/api/manual/segment-from-stroke", response_model=SegmentationResult)
async def segment_from_stroke(
    image: UploadFile = File(...),
    strokes: str = Form(...),  # JSON: [{"x":int, "y":int, "label":int}, ...]
    n_layers: int = Form(5),
) -> SegmentationResult:
    """Segment using brush strokes as seeds for region growing / watershed."""
    img_rgb = _upload_to_ndarray(image)
    _strokes: list[dict[str, int]] = json.loads(strokes)

    # TODO: implement region growing / watershed from stroke seeds
    raise NotImplementedError("segment_from_stroke stub — implement in Week 4")


# ---------------------------------------------------------------------------
# Shared — Export endpoint
# ---------------------------------------------------------------------------

@app.post("/api/export/specfem", response_model=ExportSpecfemResponse)
async def export_specfem(
    labels: UploadFile = File(...),
    color_names: str = Form(...),  # JSON: ["layer_1", ...]
) -> ExportSpecfemResponse:
    """Export labels to SPECFEM tomography file + Par_file snippet."""
    import tempfile

    from geoseg.modules.exporter.specfem import (
        labels_to_grids,
        write_parfile_snippet,
        write_tomography_file,
    )
    from geoseg.modules.post_process.properties import (
        assign_properties,
        generate_properties_for_layers,
    )

    # Load labels from NPZ
    npz = np.load(labels.file)
    labels_arr: np.ndarray = npz["labels"].astype(np.int32)
    _color_names: list[str] = json.loads(color_names)

    # Assign properties
    try:
        props = assign_properties(_color_names)
    except ValueError:
        props = generate_properties_for_layers(_color_names)

    # Build grids
    vp, vs, rho = labels_to_grids(labels_arr, props, color_names=_color_names)

    h, w = labels_arr.shape
    x_coords = np.linspace(0, w - 1, w)
    z_coords = np.linspace(0, h - 1, h)

    # Write to temp files and read back as strings
    with tempfile.TemporaryDirectory() as tmpdir:
        tomo_path = Path(tmpdir) / "tomo.xyz"
        parfile_path = Path(tmpdir) / "parfile_snippet.txt"

        write_tomography_file(vp, vs, rho, x_coords, z_coords, tomo_path)
        write_parfile_snippet(_color_names, props, parfile_path, nx=w, nz=h)

        tomo_content = tomo_path.read_text(encoding="utf-8")
        parfile_content = parfile_path.read_text(encoding="utf-8")

    return ExportSpecfemResponse(
        tomo_xyz=tomo_content,
        parfile_snippet=parfile_content,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
