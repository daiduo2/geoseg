"""Panel preflight: crop to the coloured data area before segmentation.

Axes, tick labels, tick marks and margin text are not geology.  Cropping them
before any engine prevents the audit loop from "repairing" a panel that should
never have entered the engine with axes in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CropResult:
    image: np.ndarray
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (x1/y1 exclusive)
    method: str
    saturation_fraction: float

    @property
    def offset(self) -> tuple[int, int]:
        return (self.bbox[0], self.bbox[1])

    def crop_labels(self, labels: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.bbox
        return labels[y0:y1, x0:x1].copy()


def _as_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("image must be HxW or HxWx3")
    return array[..., :3]


def detect_panel_bbox(
    image: np.ndarray,
    *,
    saturation_threshold: int = 25,
    min_area_fraction: float = 0.02,
) -> tuple[int, int, int, int]:
    """Bounding box of colourful panel content, excluding grey/white axes."""
    rgb = _as_rgb(image)
    saturation = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    mask = saturation >= saturation_threshold
    if mask.mean() < min_area_fraction:
        # Low-saturation/grayscale panels: use non-white content instead.
        mask = np.min(rgb, axis=2) < 245
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        h, w = rgb.shape[:2]
        return (0, 0, w, h)
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    return (x0, y0, x1, y1)


def crop_panel(
    image: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
    *,
    auto: bool = True,
    saturation_threshold: int = 25,
) -> CropResult:
    """Crop the panel to its data area, optionally using an explicit bbox."""
    rgb = _as_rgb(image)
    h, w = rgb.shape[:2]
    if bbox is not None:
        x0, y0, x1, y1 = (int(value) for value in bbox)
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"invalid crop bbox {bbox} for image {w}x{h}")
        method = "manual"
    elif auto:
        x0, y0, x1, y1 = detect_panel_bbox(
            rgb, saturation_threshold=saturation_threshold
        )
        method = "saturation_bbox"
    else:
        x0, y0, x1, y1 = 0, 0, w, h
        method = "none"

    saturation = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    return CropResult(
        image=rgb[y0:y1, x0:x1].copy(),
        bbox=(x0, y0, x1, y1),
        method=method,
        saturation_fraction=float((saturation >= saturation_threshold).mean()),
    )


__all__ = ["CropResult", "crop_panel", "detect_panel_bbox"]
