"""Manual segmentation API routes."""

from __future__ import annotations

import json

import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image, ImageDraw

from geoseg.api.schemas import SegmentationResult
from geoseg.api.serialization import segmentation_to_api, upload_to_ndarray

router = APIRouter(prefix="/api/manual")


@router.post("/segment-from-polygon", response_model=SegmentationResult)
async def segment_from_polygon(
    image: UploadFile = File(...),
    polygon: str = Form(...),
    n_layers: int = Form(5),
) -> SegmentationResult:
    """Segment using a user-drawn polygon mask."""
    img_rgb = upload_to_ndarray(image)
    polygon_points: list[dict[str, int]] = json.loads(polygon)

    h, w = img_rgb.shape[:2]
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon([(p["x"], p["y"]) for p in polygon_points], fill=255)
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


@router.post("/segment-from-rect", response_model=SegmentationResult)
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


@router.post("/segment-from-stroke", response_model=SegmentationResult)
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


__all__ = [
    "router",
    "segment_from_polygon",
    "segment_from_rect",
    "segment_from_stroke",
]
