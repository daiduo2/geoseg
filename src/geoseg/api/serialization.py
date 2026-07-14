"""HTTP serialization helpers for image and segmentation payloads."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from fastapi import UploadFile
from PIL import Image
from skimage import measure

from geoseg.api.schemas import SegmentationMeta, SegmentationResult


def upload_to_ndarray(upload: UploadFile) -> np.ndarray:
    """Convert an uploaded image file to RGB ndarray."""
    image = Image.open(upload.file).convert("RGB")
    return np.array(image)


def labels_to_contours(labels: np.ndarray) -> list[list[dict[str, int]]]:
    """Convert label map to list of contour polygons for frontend rendering."""
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


def labels_to_base64(labels: np.ndarray) -> str:
    """Compress labels ndarray to base64-encoded NPZ bytes."""
    buf = io.BytesIO()
    np.savez_compressed(buf, labels=labels)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def overlay_to_base64(overlay: np.ndarray | None) -> str | None:
    """Encode overlay RGB ndarray to base64 PNG."""
    if overlay is None:
        return None
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def segmentation_to_api(seg: dict[str, Any]) -> SegmentationResult:
    """Convert internal segmentation dict to API response."""
    labels = seg["labels"]
    overlay = seg.get("overlay")
    meta = seg.get("meta", {})
    return SegmentationResult(
        labels_base64=labels_to_base64(labels),
        contours=labels_to_contours(labels),
        overlay_base64=overlay_to_base64(overlay),
        meta=SegmentationMeta(
            engine=meta.get("engine", "unknown"),
            color_names=meta.get("color_names", []),
            n_layers=meta.get("n_layers", 0),
            quality_score=meta.get("quality_score"),
        ),
    )


__all__ = [
    "labels_to_base64",
    "labels_to_contours",
    "overlay_to_base64",
    "segmentation_to_api",
    "upload_to_ndarray",
]
