"""Convert VLM color_zones into segmentation engine reps.

VLM returns semantic color descriptions (color_name + colorbar_value 0-100)
but no pixel coordinates. This module samples the actual colorbar strip and
finds matching pixels in the panel to build rep points.

Also includes a vertical-scan fallback that auto-detects horizontal layers
from the panel itself — critical when VLM color_zones are incomplete.
"""

from __future__ import annotations

import numpy as np

from geoseg.modules.segment_engines._shared import (
    _estimate_background_color,
    _find_pixel_for_color,
)


def _sample_colorbar_at_value(colorbar_rgb: np.ndarray, colorbar_value: int) -> np.ndarray:
    """Sample RGB from a colorbar strip at a given value (0-100).

    Maps 0-100 to the 5%-95% span of the strip to avoid edge artifacts.
    Handles both horizontal and vertical orientations.
    """
    h, w = colorbar_rgb.shape[:2]
    frac = 0.05 + (np.clip(colorbar_value, 0, 100) / 100.0) * 0.9

    if h >= w:
        # Vertical colorbar: value increases top -> bottom
        y = int(frac * (h - 1))
        return colorbar_rgb[y, w // 2]
    else:
        # Horizontal colorbar: value increases left -> right
        x = int(frac * (w - 1))
        return colorbar_rgb[h // 2, x]


def _auto_reps_from_colorbar(
    panel_rgb: np.ndarray,
    colorbar_rgb: np.ndarray,
    n_layers: int = 5,
) -> list[dict]:
    """Generate reps by sampling the colorbar at evenly-spaced positions.

    Used as fallback when VLM returns no color_zones.
    """
    h, w = colorbar_rgb.shape[:2]
    bg_rgb = _estimate_background_color(panel_rgb)
    reps: list[dict] = []
    used_points: set[tuple[int, int]] = set()

    if h >= w:
        # Vertical: sample top -> bottom
        ys = np.linspace(int(0.05 * h), int(0.95 * h) - 1, n_layers).astype(int)
        cx = w // 2
        sampled = [(colorbar_rgb[y, cx], i) for i, y in enumerate(ys)]
    else:
        # Horizontal: sample left -> right
        xs = np.linspace(int(0.05 * w), int(0.95 * w) - 1, n_layers).astype(int)
        cy = h // 2
        sampled = [(colorbar_rgb[cy, x], i) for i, x in enumerate(xs)]

    for target_rgb, idx in sampled:
        found = _find_pixel_for_color(
            panel_rgb, target_rgb, bg_rgb, color_tol=45.0, bg_tol=50.0
        )
        if found is None:
            continue
        cx, cy = found
        if (cx, cy) in used_points:
            continue
        used_points.add((cx, cy))
        reps.append({
            "color_name": f"layer_{idx + 1}",
            "representative_point": {"x": cx, "y": cy},
        })

    return reps


def vertical_scan_reps(
    panel_rgb: np.ndarray,
    n_layers_hint: int = 5,
    min_layer_height: int = 8,
) -> list[dict]:
    """Auto-detect horizontal-layer reps by scanning vertically.

    Takes advantage of velocity-model panels having roughly horizontal layers.
    Samples the central column, smooths the colour profile, detects layer
    boundaries from LAB-colour change peaks, and generates one rep per layer.

    Args:
        panel_rgb: RGB uint8 array of the panel.
        n_layers_hint: Expected number of layers (guides peak-detection
                       sensitivity; actual reps may differ).
        min_layer_height: Minimum vertical thickness of a layer (pixels).

    Returns:
        List of rep dicts, top-to-bottom order.
    """
    from skimage.color import rgb2lab
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    h, w = panel_rgb.shape[:2]
    bg_rgb = _estimate_background_color(panel_rgb)

    # Average 3 central columns for robustness
    cx = w // 2
    cols = [max(0, min(cx + i, w - 1)) for i in range(-1, 2)]
    column_rgb = panel_rgb[:, cols, :].mean(axis=1).astype(np.uint8)
    column_lab = rgb2lab(column_rgb)

    # Smooth LAB channels vertically
    column_lab_smooth = gaussian_filter1d(column_lab, sigma=max(1.5, h / 300), axis=0)

    # Colour-change magnitude along vertical axis
    diffs = np.linalg.norm(np.diff(column_lab_smooth, axis=0), axis=1)

    # Adaptive peak threshold: at least 70th percentile, but require a
    # minimum absolute jump so near-uniform regions do not split.
    # For panels with lots of near-uniform background (low median diffs)
    # we raise abs_min to avoid false boundaries; for vivid panels we keep
    # the lower threshold so thin layers are not missed.
    median_diff = float(np.median(diffs))
    if median_diff < 1.0:
        abs_min = 5.0  # strict: low-saturation / mostly-background panels
        rel_min = np.percentile(diffs, 80)
    else:
        abs_min = 3.0  # standard: vivid velocity models
        rel_min = np.percentile(diffs, 70)
    threshold = max(abs_min, rel_min)

    # Safety: if threshold is above every diff value, no peaks can ever be
    # found (e.g. smooth high-saturation panels like wise_fwi). Lower it to
    # catch the strongest transitions present.
    max_diff = float(diffs.max())
    if threshold > max_diff:
        threshold = max(min(abs_min, 3.0), max_diff * 0.85, median_diff * 2.0, 1.5)

    # Minimum distance between peaks scales with expected layer count.
    # Also enforce a proportional minimum so thin artifacts on large panels
    # are not treated as layers.
    effective_min_layer_height = max(min_layer_height, h // 50)
    min_dist = max(effective_min_layer_height, h // (n_layers_hint * 2))

    peaks, _ = find_peaks(diffs, height=threshold, distance=min_dist)

    # Layer boundaries: image edges + detected peaks
    boundaries = [0] + sorted(peaks.tolist()) + [h - 1]

    reps: list[dict] = []

    for i in range(len(boundaries) - 1):
        y0, y1 = boundaries[i], boundaries[i + 1]
        if y1 - y0 < effective_min_layer_height:
            continue

        # Use the layer's mid-point on the central scan line as the rep.
        # For horizontally-layered velocity models this is the most
        # reliable seed position.
        ym = (y0 + y1) // 2
        px = w // 2
        py = ym

        reps.append({
            "color_name": f"layer_{i + 1}",
            "representative_point": {"x": px, "y": py},
        })

    # Ensure top-to-bottom order
    reps.sort(key=lambda r: r["representative_point"]["y"])
    return reps


def color_zones_to_reps(
    panel_rgb: np.ndarray,
    color_zones: list[dict],
    colorbar_rgb: np.ndarray | None = None,
    n_layers: int = 5,
) -> list[dict]:
    """Convert VLM color_zones to reps for segmentation engines.

    Falls back to vertical-scan auto-detection if VLM returns no color_zones
    or if the resulting reps are insufficient (< 2).  The vertical-scan
    fallback is tuned for velocity-model panels with horizontal layers.

    Args:
        panel_rgb: RGB uint8 array of the panel.
        color_zones: List of {"color_name": str, "colorbar_value": int} from VLM.
        colorbar_rgb: Optional extracted colorbar strip.
        n_layers: Number of layers for fallback auto-sampling.

    Returns:
        List of rep dicts: {"color_name": str, "representative_point": {"x": int, "y": int}}.
        May be empty if no matching pixels are found.
    """
    # ── VLM path ──────────────────────────────────────────────────────
    if color_zones:
        bg_rgb = _estimate_background_color(panel_rgb)
        reps: list[dict] = []
        used_points: set[tuple[int, int]] = set()

        for zone in color_zones:
            color_name = zone.get("color_name", "unknown")
            cb_val = zone.get("colorbar_value", 50)

            if colorbar_rgb is not None and colorbar_rgb.size > 0:
                target_rgb = _sample_colorbar_at_value(colorbar_rgb, cb_val)
            else:
                continue

            found = _find_pixel_for_color(
                panel_rgb, target_rgb, bg_rgb, color_tol=45.0, bg_tol=50.0
            )
            if found is None:
                continue

            cx, cy = found
            if (cx, cy) in used_points:
                continue
            used_points.add((cx, cy))

            reps.append({
                "color_name": color_name,
                "representative_point": {"x": cx, "y": cy},
            })

        # Fallback if VLM reps are clearly insufficient for the expected layer count
        if len(reps) >= max(2, min(n_layers, 4)):
            return reps

    # ── Fallback 1: auto-sample colorbar ──────────────────────────────
    if colorbar_rgb is not None and colorbar_rgb.size > 0:
        reps = _auto_reps_from_colorbar(panel_rgb, colorbar_rgb, n_layers=n_layers)
        if len(reps) >= 2:
            return reps

    # ── Fallback 2: vertical scan (panel-internal layer detection) ────
    reps = vertical_scan_reps(panel_rgb, n_layers_hint=n_layers)
    if len(reps) >= 2:
        return reps

    return []
