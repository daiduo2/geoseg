"""FastAPI app and route handlers for geoseg."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from geoseg.api.schemas import (
    AgentProcessFigureResponse,
    ExportSpecfemResponse,
    PanelInput,
    PanelReviewOut,
    PdfImportResponse,
    PdfStatusResponse,
    SegmentationResult,
)
from geoseg.api.serialization import segmentation_to_api, upload_to_ndarray


app = FastAPI(title="geoseg", version="2.0.0")


@app.post("/api/pdf/import", response_model=PdfImportResponse)
async def import_pdf(pdf: UploadFile = File(...)) -> PdfImportResponse:
    """Upload a PDF and start MinerU extraction."""
    raise NotImplementedError("import_pdf stub - implement in Week 5")


@app.get("/api/pdf/status/{job_id}", response_model=PdfStatusResponse)
async def pdf_status(job_id: str) -> PdfStatusResponse:
    """Poll extraction status and get figure list."""
    raise NotImplementedError("pdf_status stub - implement in Week 5")


@app.post("/api/agent/process-figure", response_model=AgentProcessFigureResponse)
async def process_figure_agent(
    image: UploadFile = File(...),
    caption: str = Form(""),
    text_blocks: str = Form("[]"),
    n_layers: int = Form(5),
    quality_preference: str = Form("balanced"),
) -> AgentProcessFigureResponse:
    """Run the full Agent pipeline on a figure image."""
    img_rgb = upload_to_ndarray(image)
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

    panels_out: list[PanelReviewOut] = []
    empty_seg = {
        "labels": np.zeros((10, 10), dtype=np.int32),
        "meta": {"engine": "empty", "color_names": [], "n_layers": 0},
    }
    for p in result.get("panels", []):
        seg = p.get("segmentation")
        panels_out.append(
            PanelReviewOut(
                panel_id=p["panel_id"],
                bbox=tuple(p["bbox"]),  # type: ignore[arg-type]
                classification=p.get("classification", {}),
                segmentation=segmentation_to_api(seg if seg else empty_seg),
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
    img_rgb = upload_to_ndarray(image)

    from geoseg.modules.cv_detect.panel_detector import detect_panels

    bboxes = detect_panels(img_rgb)
    if not bboxes:
        from geoseg.core.models import make_whole_image_panel

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
    reps: str | None = Form(None),
) -> SegmentationResult:
    """Segment a panel image using the automatic router."""
    img_rgb = upload_to_ndarray(image)

    from geoseg.modules.segment_engines import route_and_segment

    kwargs: dict[str, Any] = {"n_layers": n_layers}
    if reps:
        kwargs["reps"] = json.loads(reps)

    seg = route_and_segment(img_rgb, **kwargs)
    return segmentation_to_api(seg)


@app.post("/api/manual/segment-from-polygon", response_model=SegmentationResult)
async def segment_from_polygon(
    image: UploadFile = File(...),
    polygon: str = Form(...),
    n_layers: int = Form(5),
) -> SegmentationResult:
    """Segment using a user-drawn polygon mask."""
    img_rgb = upload_to_ndarray(image)
    _poly: list[dict[str, int]] = json.loads(polygon)

    from PIL import ImageDraw

    h, w = img_rgb.shape[:2]
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon([(p["x"], p["y"]) for p in _poly], fill=255)
    mask = np.array(mask_img) > 0

    from geoseg.modules.segment_engines import route_and_segment

    seg = route_and_segment(img_rgb, n_layers=n_layers)
    labels = seg["labels"].copy()
    labels[~mask] = 0

    unique = sorted(set(labels.flatten()) - {0})
    relabel_map = {old: new for new, old in enumerate(unique, start=1)}
    new_labels = np.zeros_like(labels)
    for old, new in relabel_map.items():
        new_labels[labels == old] = new

    meta = dict(seg.get("meta", {}))
    meta["engine"] = "manual_polygon"
    meta["n_layers"] = len(unique)

    return segmentation_to_api(
        {"labels": new_labels, "overlay": seg.get("overlay"), "meta": meta}
    )


@app.post("/api/manual/segment-from-rect", response_model=SegmentationResult)
async def segment_from_rect(
    image: UploadFile = File(...),
    bbox: str = Form(...),
    n_layers: int = Form(5),
) -> SegmentationResult:
    """Segment inside a user-drawn bbox using grab-cut / graph-cut shrink."""
    _img_rgb = upload_to_ndarray(image)
    _bbox: tuple[int, int, int, int] = tuple(json.loads(bbox))  # type: ignore[assignment]
    _n_layers = n_layers
    raise NotImplementedError("segment_from_rect stub - implement in Week 4")


@app.post("/api/manual/segment-from-stroke", response_model=SegmentationResult)
async def segment_from_stroke(
    image: UploadFile = File(...),
    strokes: str = Form(...),
    n_layers: int = Form(5),
) -> SegmentationResult:
    """Segment using brush strokes as seeds for region growing / watershed."""
    _img_rgb = upload_to_ndarray(image)
    _strokes: list[dict[str, int]] = json.loads(strokes)
    _n_layers = n_layers
    raise NotImplementedError("segment_from_stroke stub - implement in Week 4")


@app.post("/api/export/specfem", response_model=ExportSpecfemResponse)
async def export_specfem(
    labels: UploadFile = File(...),
    color_names: str = Form(...),
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

    npz = np.load(labels.file)
    labels_arr: np.ndarray = npz["labels"].astype(np.int32)
    _color_names: list[str] = json.loads(color_names)

    try:
        props = assign_properties(_color_names)
    except ValueError:
        props = generate_properties_for_layers(_color_names)

    vp, vs, rho = labels_to_grids(labels_arr, props, color_names=_color_names)

    h, w = labels_arr.shape
    x_coords = np.linspace(0, w - 1, w)
    z_coords = np.linspace(0, h - 1, h)

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


__all__ = ["app"]
