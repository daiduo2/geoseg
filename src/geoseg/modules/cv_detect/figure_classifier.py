"""Classify extracted figure images by type.

Distinguishes:
- conceptual_model: hand-drawn schematic / cartoon diagrams
- observational_data: data-driven maps / cross-sections with colormaps
- multi_panel: composite figures with multiple sub-panels
- other: unclear or unsupported

Used upstream of segmentation to route images to appropriate engines.

Test scenario:
    >>> import numpy as np
    >>> img = np.full((100, 200, 3), 128, dtype=np.uint8)
    >>> r = classify(img)
    >>> assert r["figure_type"] in ("conceptual_model", "observational_data", "multi_panel", "other")
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import sobel
from skimage.measure import label, regionprops


def _edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels with significant edge magnitude."""
    edges = sobel(gray)
    return float((np.abs(edges) > 0.05).mean())


def _color_quantization_score(panel_rgb: np.ndarray) -> float:
    """Measure how quantized (blocky) colors are vs smooth gradients.

    High score = few distinct colors with sharp boundaries (conceptual).
    Low score = many continuous tones (observational).
    """
    # Downsample for speed
    h, w = panel_rgb.shape[:2]
    if max(h, w) > 400:
        scale = 400 / max(h, w)
        from PIL import Image
        small = np.array(
            Image.fromarray(panel_rgb).resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        )
    else:
        small = panel_rgb

    # Online color grouping
    pixels = small.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pixels), 5000, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    groups: list[tuple[np.ndarray, int]] = []
    tol_sq = 60.0 * 60.0
    for px in sample:
        matched = False
        for i, (mean, count) in enumerate(groups):
            diff = px - mean
            if np.dot(diff, diff) <= tol_sq:
                new_mean = (mean * count + px) / (count + 1)
                groups[i] = (new_mean, count + 1)
                matched = True
                break
        if not matched:
            groups.append((px.copy(), 1))

    if not groups:
        return 0.0

    # Quantization = ratio of dominant groups to total
    total = sum(g[1] for g in groups)
    groups.sort(key=lambda g: g[1], reverse=True)
    top5 = sum(g[1] for g in groups[:5])
    return float(top5 / total) if total > 0 else 0.0


def _gradient_uniformity(gray: np.ndarray) -> float:
    """Measure how uniform the gradient magnitude distribution is.

    Observational data (smooth gradients) has more uniform distribution.
    Conceptual models (sharp lines) has highly skewed distribution.
    """
    gy, gx = np.gradient(gray.astype(np.float64))
    mag = np.sqrt(gx**2 + gy**2)
    # Coefficient of variation = std / mean
    mean_mag = mag.mean()
    if mean_mag < 1e-9:
        return 1.0
    cv = mag.std() / mean_mag
    # Normalize: high CV = conceptual, low CV = observational
    return float(np.clip(cv / 3.0, 0.0, 1.0))


def _has_colorbar_region(panel_rgb: np.ndarray) -> bool:
    """Heuristic: check edge strips for rich color variation (colorbar)."""
    h, w = panel_rgb.shape[:2]
    if w < 100 or h < 100:
        return False

    strip_w = max(20, w // 10)
    strip_h = max(20, h // 10)

    # Check right edge
    right_strip = panel_rgb[:, -strip_w:]
    right_lab = rgb2lab(right_strip)
    right_ab = right_lab[:, :, 1:].reshape(-1, 2).std(axis=0).mean()
    right_l = right_lab[:, :, 0].std()

    # Check bottom edge
    bottom_strip = panel_rgb[-strip_h:, :]
    bottom_lab = rgb2lab(bottom_strip)
    bottom_ab = bottom_lab[:, :, 1:].reshape(-1, 2).std(axis=0).mean()
    bottom_l = bottom_lab[:, :, 0].std()

    # Relaxed thresholds — tomography colorbars often have subtle gradients
    return (right_ab > 10 and right_l > 8) or (bottom_ab > 10 and bottom_l > 8)


def _panel_gap_score(panel_rgb: np.ndarray) -> float:
    """Detect vertical/horizontal gaps suggesting multiple panels.

    Crops edge margins to avoid colorbar/axis-label false positives,
    then uses median filtering to suppress thin grid lines.
    """
    h, w = panel_rgb.shape[:2]
    gray = rgb2gray(panel_rgb)

    # Crop margins to exclude colorbars and axis labels
    margin_y = max(1, int(h * 0.08))
    margin_x = max(1, int(w * 0.08))
    cropped = gray[margin_y : h - margin_y, margin_x : w - margin_x]
    ch, cw = cropped.shape
    if ch < 20 or cw < 20:
        return 0.0

    # Median filter to suppress thin lines (grid lines, ticks)
    gray_med = ndimage.median_filter(cropped, size=5)

    # Vertical projection: look for white gaps
    vproj = gray_med.mean(axis=0)
    is_gap_v = vproj > 0.94
    gap_runs_v = []
    in_gap = False
    start = 0
    for i in range(cw):
        if is_gap_v[i] and not in_gap:
            start = i
            in_gap = True
        elif not is_gap_v[i] and in_gap:
            gap_runs_v.append(i - start)
            in_gap = False
    if in_gap:
        gap_runs_v.append(cw - start)

    # Horizontal projection
    hproj = gray_med.mean(axis=1)
    is_gap_h = hproj > 0.94
    gap_runs_h = []
    in_gap = False
    start = 0
    for i in range(ch):
        if is_gap_h[i] and not in_gap:
            start = i
            in_gap = True
        elif not is_gap_h[i] and in_gap:
            gap_runs_h.append(i - start)
            in_gap = False
    if in_gap:
        gap_runs_h.append(ch - start)

    # Require wider gaps
    wide_gaps = sum(1 for g in gap_runs_v if g > cw * 0.10) + sum(
        1 for g in gap_runs_h if g > ch * 0.10
    )
    return min(1.0, wide_gaps / 3.0)


def _text_density_heuristic(panel_rgb: np.ndarray) -> float:
    """Estimate text density via high-frequency horizontal structure."""
    gray = rgb2gray(panel_rgb)
    h, w = gray.shape
    if h < 20:
        return 0.0

    # Horizontal derivative: text has lots of small horizontal transitions
    h_diff = np.abs(np.diff(gray, axis=1)).mean()
    # Normalize
    return float(np.clip(h_diff / 0.15, 0.0, 1.0))


def _horizontal_striation_score(gray: np.ndarray) -> float:
    """Detect dense horizontal stripes (e.g., seismic reflection events).

    High score = many parallel horizontal edges (observational, e.g., seismic).
    Low score = no dominant horizontal structure (conceptual).
    """
    gy, gx = np.gradient(gray.astype(np.float64))
    # Horizontal stripes = strong vertical gradient (gy), weak horizontal gradient (gx)
    horizontal_mask = (np.abs(gy) > 0.05) & (np.abs(gx) < 0.03)
    return float(np.clip(horizontal_mask.mean() * 3.0, 0.0, 1.0))


def classify(panel_rgb: np.ndarray) -> dict:
    """Classify a figure image into conceptual / observational / multi-panel / other.

    Args:
        panel_rgb: RGB uint8 array.

    Returns:
        dict with keys:
            figure_type: "conceptual_model" | "observational_data" | "multi_panel" | "other"
            confidence: float 0-1
            features: dict of raw feature values
            reason: str explaining the classification
    """
    h, w = panel_rgb.shape[:2]
    if h < 50 or w < 50:
        return {
            "figure_type": "other",
            "confidence": 0.95,
            "features": {},
            "reason": "Image too small",
        }

    gray = rgb2gray(panel_rgb)

    # Compute features
    edge_dens = _edge_density(gray)
    quant_score = _color_quantization_score(panel_rgb)
    grad_uniformity = _gradient_uniformity(gray)
    colorbar_present = _has_colorbar_region(panel_rgb)
    gap_score = _panel_gap_score(panel_rgb)
    text_dens = _text_density_heuristic(panel_rgb)
    striation = _horizontal_striation_score(gray)

    features = {
        "edge_density": round(edge_dens, 4),
        "color_quantization": round(quant_score, 4),
        "gradient_uniformity": round(grad_uniformity, 4),
        "has_colorbar_region": colorbar_present,
        "panel_gap_score": round(gap_score, 4),
        "text_density": round(text_dens, 4),
        "striation": round(striation, 4),
    }

    # Decision logic
    # Observational barrier: seismic waveforms have both gaps AND striations
    if gap_score >= 0.5 and striation >= 0.15:
        return {
            "figure_type": "observational_data",
            "confidence": min(0.9, 0.6 + gap_score * 0.2),
            "features": features,
            "reason": f"Multi-gap figure with horizontal striations (gap={gap_score:.2f}, striation={striation:.2f}) suggests waveform data",
        }

    # Multi-panel: strong gap evidence WITHOUT striations (conceptual multi-panel)
    if gap_score >= 0.5:
        return {
            "figure_type": "multi_panel",
            "confidence": min(0.95, 0.6 + gap_score * 0.3),
            "features": features,
            "reason": f"Detected {gap_score:.2f} wide panel gaps",
        }

    # Conceptual model: high edge density + high quantization + high text density
    # Penalize horizontal striation (seismic reflections are NOT conceptual)
    # Penalize colorbar presence (data maps have colorbars)
    conceptual_score = (
        edge_dens * 0.15
        + quant_score * 0.20
        + (1.0 - grad_uniformity) * 0.15
        + text_dens * 0.20
        - striation * 0.50
        - (0.25 if colorbar_present else 0.0)
    )

    # Observational: smooth gradients + colorbar + lower edge density + LOW quantization
    # Reward horizontal striation (seismic reflections)
    observational_score = (
        grad_uniformity * 0.25
        + (0.3 if colorbar_present else 0.0)
        + (1.0 - edge_dens) * 0.15
        + (1.0 - quant_score) * 0.15
        + striation * 0.35
    )

    if conceptual_score > observational_score + 0.15:
        return {
            "figure_type": "conceptual_model",
            "confidence": min(0.9, 0.5 + conceptual_score * 0.4),
            "features": features,
            "reason": f"High edge density ({edge_dens:.2f}) and color quantization ({quant_score:.2f}) suggest schematic/conceptual diagram",
        }

    if observational_score > conceptual_score + 0.15:
        return {
            "figure_type": "observational_data",
            "confidence": min(0.9, 0.5 + observational_score * 0.4),
            "features": features,
            "reason": f"Smooth gradients (uniformity={grad_uniformity:.2f}) and colorbar={colorbar_present} suggest data-driven figure",
        }

    # Ambiguous
    return {
        "figure_type": "other",
        "confidence": 0.6,
        "features": features,
        "reason": f"Ambiguous: conceptual_score={conceptual_score:.2f}, observational_score={observational_score:.2f}",
    }
