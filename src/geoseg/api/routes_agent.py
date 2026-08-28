"""Agent API routes."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, UploadFile

from geoseg.api.schemas import (
    AgentProcessFigureResponse,
    PanelInput,
    PanelReviewOut,
    SegmentationResult,
)
from geoseg.api.serialization import segmentation_to_api, upload_to_ndarray

router = APIRouter(prefix="/api/agent")


@router.post("/process-figure", response_model=AgentProcessFigureResponse)
async def process_figure_agent(
    image: UploadFile = File(...),
    caption: str = Form(""),
    text_blocks: str = Form("[]"),
    n_layers: int = Form(5),
    quality_preference: str = Form("balanced"),
    boundary_mode: str = Form("none"),
) -> AgentProcessFigureResponse:
    """Run the full Agent pipeline on a figure image."""
    img_rgb = upload_to_ndarray(image)
    text_block_data: list[dict] = json.loads(text_blocks)

    from geoseg.controller import run_pipeline

    result = run_pipeline(
        img_rgb,
        caption=caption,
        text_blocks=text_block_data,
        n_layers=n_layers,
        quality_preference=quality_preference,
        boundary_mode=boundary_mode,
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


@router.post("/detect-panels", response_model=list[PanelInput])
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


@router.post("/segment", response_model=SegmentationResult)
async def segment_agent(
    image: UploadFile = File(...),
    n_layers: int = Form(5),
    reps: str | None = Form(None),
    boundary_mode: str = Form("none"),
) -> SegmentationResult:
    """Segment a panel image using the automatic router."""
    img_rgb = upload_to_ndarray(image)

    from geoseg.modules.segment_engines import route_and_segment

    kwargs: dict[str, Any] = {"n_layers": n_layers}
    if reps:
        kwargs["reps"] = json.loads(reps)

    seg = route_and_segment(img_rgb, **kwargs)
    if boundary_mode == "red":
        from geoseg.modules.post_process.split import (
            split_label_by_color_components,
            split_labels_by_red_boundaries,
        )
        from geoseg.preprocessing.absorption import absorb_artifacts
        from geoseg.preprocessing.detectors import detect_red_boundaries, detect_text

        boundary_mask = detect_red_boundaries(img_rgb)
        text_mask = detect_text(
            img_rgb, min_area=20, min_width=8, min_aspect=2.0
        )
        cleaned = absorb_artifacts(
            img_rgb, boundary_mask | text_mask, inpaint_radius=5, dilate_iters=1
        )
        color_labels = split_label_by_color_components(
            np.ones(img_rgb.shape[:2], dtype=np.int32),
            cleaned,
            target_label=1,
            k=max(2, n_layers),
            min_component_area=max(300, int(img_rgb.shape[0] * img_rgb.shape[1] * 0.005)),
        )
        labels, parent_map, boundary_mask = split_labels_by_red_boundaries(
            color_labels, img_rgb, boundary_mask=boundary_mask
        )
        seg["labels"] = labels
        seg["color_partition"] = color_labels
        seg["boundary_mask"] = boundary_mask
        seg["meta"]["boundary_mode"] = "red"
        seg["meta"]["boundary_pixels"] = int(boundary_mask.sum())
        seg["meta"]["parent_labels"] = parent_map
    elif boundary_mode != "none":
        raise ValueError(f"Unsupported boundary_mode: {boundary_mode!r}")
    return segmentation_to_api(seg)


__all__ = [
    "detect_panels_agent",
    "process_figure_agent",
    "router",
    "segment_agent",
]
