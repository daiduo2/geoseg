"""Panel detection stage helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from geoseg.core.image_ops import saturation_ratio
from geoseg.core.models import make_whole_image_panel
from geoseg.modules.cv_detect.panel_detector import detect_panels


def detect_panels_stage(img_rgb: np.ndarray) -> list[dict[str, Any]]:
    """Detect panels, falling back to a whole-image panel."""
    panel_bboxes = detect_panels(img_rgb)
    if not panel_bboxes:
        panel_bboxes = [make_whole_image_panel(img_rgb)]
    return panel_bboxes


def panel_complexity_score(panel_rgb: np.ndarray) -> float:
    """Score a panel by structural complexity to avoid simple gradients."""
    from skimage.color import rgb2gray
    from skimage.filters import sobel

    gray = rgb2gray(panel_rgb)
    h, w = gray.shape
    if h < 10 or w < 10:
        return 0.0

    edges = sobel(gray)
    edge_dens = float((np.abs(edges) > 0.05).mean())

    gy, gx = np.gradient(gray.astype(np.float64))
    mag = np.sqrt(gx**2 + gy**2)
    mean_mag = mag.mean()
    if mean_mag < 1e-9:
        grad_uniformity = 1.0
    else:
        grad_uniformity = float(np.clip(mag.std() / mean_mag / 3.0, 0.0, 1.0))

    sat = saturation_ratio(panel_rgb)
    return edge_dens * 0.5 + sat * 0.3 + (1.0 - grad_uniformity) * 0.2


__all__ = ["detect_panels_stage", "panel_complexity_score"]
