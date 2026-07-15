"""Top-level v4 K-Means path dispatch."""

from __future__ import annotations

import numpy as np

from geoseg.modules.segment_engines.internal.color import saturation_ratio
from geoseg.modules.segment_engines.v4.colorbar_guided import segment_colorbar_guided
from geoseg.modules.segment_engines.v4.jet_vivid import segment_jet_vivid
from geoseg.modules.segment_engines.v4.pastel import segment_pastel_faded

JET_VIVID_RATIO = 0.05


def segment(
    panel_rgb: np.ndarray,
    reps: list[dict] | None = None,
    colorbar_rgb: np.ndarray | None = None,
    n_layers: int = 5,
    max_auto_k: int = 0,
    n_color_zones: int = 0,
) -> dict:
    """Dispatcher: pick jet_vivid or colorbar-guided by saturation ratio.

    Routing logic:
    - sat >= JET_VIVID_RATIO AND reps present -> jet_vivid (VLM rep points)
    - colorbar_rgb present -> colorbar_guided
    - fallback -> pastel_faded (legacy K-means + shape filter)

    Args:
        panel_rgb: RGB uint8 array (H, W, 3).
        reps: VLM representative points, each with color_name and representative_point {x, y}.
        colorbar_rgb: Optional colorbar strip image.
        n_layers: Number of layers to extract.
        max_auto_k: Maximum extra seeds to auto-detect for jet_vivid path.
        n_color_zones: Number of color zones detected by VLM (tunes k when >= 3).

    Returns:
        dict with keys: labels, seeds, overlay, meta.
    """
    ratio = saturation_ratio(panel_rgb)
    if ratio >= JET_VIVID_RATIO and reps:
        return segment_jet_vivid(panel_rgb, reps, max_auto_k=max_auto_k)
    if colorbar_rgb is not None and colorbar_rgb.size > 0:
        return segment_colorbar_guided(panel_rgb, colorbar_rgb, n_layers=n_layers, n_color_zones=n_color_zones)
    return segment_pastel_faded(panel_rgb, colorbar_rgb, n_layers=n_layers, n_color_zones=n_color_zones)

__all__ = [
    "JET_VIVID_RATIO",
    "segment",
    "segment_colorbar_guided",
    "segment_jet_vivid",
    "segment_pastel_faded",
]
