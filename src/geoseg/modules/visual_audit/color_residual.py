"""Color residual diagnostics for segmentation audit.

Computes per-pixel color-vector deviation of each label from its representative
color. Results are diagnostic signals for agent visual judgment only; they are
not automatic retry/accept gates.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy import ndimage
from skimage.color import deltaE_cie76, rgb2lab
from skimage.measure import label, regionprops


def _validate_shapes(labels: np.ndarray, image_rgb: np.ndarray) -> None:
    """Ensure labels and image share the same spatial shape."""
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape {labels.shape}")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must be (H, W, 3), got shape {image_rgb.shape}")
    if labels.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"Shape mismatch: labels {labels.shape} vs image {image_rgb.shape[:2]}"
        )


def compute_palette_match_residuals(
    image_rgb: np.ndarray,
    palette_rgb: np.ndarray,
    *,
    chunk_size: int = 200_000,
) -> dict[str, np.ndarray]:
    """Match pixels to exact palette colors and retain RGB/LAB diagnostics.

    ``margin_delta_e`` is the distance from the second-nearest palette color
    minus the nearest distance. Small values identify ambiguous color matches.
    Chunking bounds the temporary pixel-by-palette distance matrix for large
    publication figures.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    palette = np.asarray(palette_rgb, dtype=np.uint8)
    if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) < 2:
        raise ValueError("palette_rgb must have shape (K, 3) with K >= 2")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    h, w = image_rgb.shape[:2]
    flat_rgb = image_rgb.reshape(-1, 3).astype(np.float32)
    flat_lab = rgb2lab(image_rgb.astype(np.float64) / 255.0).reshape(-1, 3)
    palette_lab = rgb2lab(palette[np.newaxis, ...].astype(np.float64) / 255.0)[0]

    labels = np.empty(flat_rgb.shape[0], dtype=np.int32)
    rgb_residual = np.empty(flat_rgb.shape[0], dtype=np.float32)
    delta_e = np.empty(flat_rgb.shape[0], dtype=np.float32)
    margin = np.empty(flat_rgb.shape[0], dtype=np.float32)

    for start in range(0, flat_rgb.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_rgb.shape[0])
        lab_d2 = (
            (flat_lab[start:stop, None, :] - palette_lab[None, :, :]) ** 2
        ).sum(axis=2)
        nearest_two = np.partition(lab_d2, kth=1, axis=1)[:, :2]
        nearest_two.sort(axis=1)
        chunk_labels = lab_d2.argmin(axis=1).astype(np.int32)
        matched_rgb = palette[chunk_labels].astype(np.float32)

        labels[start:stop] = chunk_labels
        rgb_residual[start:stop] = np.linalg.norm(
            flat_rgb[start:stop] - matched_rgb, axis=1
        )
        delta_e[start:stop] = np.sqrt(nearest_two[:, 0])
        margin[start:stop] = np.sqrt(nearest_two[:, 1]) - delta_e[start:stop]

    return {
        "labels": labels.reshape(h, w),
        "rgb_residual": rgb_residual.reshape(h, w),
        "delta_e": delta_e.reshape(h, w),
        "margin_delta_e": margin.reshape(h, w),
    }


def estimate_text_mask(image_rgb: np.ndarray, dilation_iterations: int = 2) -> np.ndarray:
    """Lightweight text/annotation mask estimation.

    Combines adaptive thresholding and Laplacian edge response to find likely
    text or annotation pixels. Returns a boolean mask.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must be (H, W, 3), got shape {image_rgb.shape}")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=25, C=-5,
    )
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    laplacian = (laplacian > np.percentile(laplacian, 80)).astype(np.uint8) * 255
    text_mask = ((adaptive > 0) & (laplacian > 0))
    return ndimage.binary_dilation(text_mask, iterations=dilation_iterations)


def compute_label_representative_colors(
    labels: np.ndarray,
    image_rgb: np.ndarray,
) -> dict[int, dict[str, np.ndarray]]:
    """Compute median RGB and LAB representative colors per label.

    Uses median instead of mean to be robust against text/annotation outliers.
    """
    _validate_shapes(labels, image_rgb)

    representatives: dict[int, dict[str, np.ndarray]] = {}
    lab_image = rgb2lab(image_rgb.astype(np.float64) / 255.0)

    for lbl in sorted(set(labels.flatten()) - {0}):
        mask = labels == lbl
        rgb_pixels = image_rgb[mask]
        lab_pixels = lab_image[mask]

        median_rgb = np.median(rgb_pixels, axis=0).astype(np.uint8)
        median_lab = np.median(lab_pixels, axis=0)

        representatives[int(lbl)] = {
            "median_rgb": median_rgb,
            "median_lab": median_lab,
        }

    return representatives


def compute_color_residual_map(
    labels: np.ndarray,
    image_rgb: np.ndarray,
    representatives: dict[int, dict[str, np.ndarray]] | None = None,
) -> np.ndarray:
    """Per-pixel LAB delta-E residual to the label's representative color.

    Returns a 2D float array where high values indicate pixels that deviate
    strongly from their assigned label's representative color. Background
    pixels (label == 0) receive residual 0.
    """
    _validate_shapes(labels, image_rgb)

    if representatives is None:
        representatives = compute_label_representative_colors(labels, image_rgb)

    lab_image = rgb2lab(image_rgb.astype(np.float64) / 255.0)
    residual = np.zeros(labels.shape, dtype=np.float64)

    for lbl, reps in representatives.items():
        mask = labels == lbl
        if not mask.any():
            continue
        median_lab = reps["median_lab"].reshape(1, 1, 3)
        diff = deltaE_cie76(lab_image, median_lab)
        residual[mask] = diff[mask]

    return residual


def compute_label_residual_stats(
    labels: np.ndarray,
    image_rgb: np.ndarray,
    residual_map: np.ndarray | None = None,
) -> dict[int, dict[str, float]]:
    """Per-label residual statistics for ranking suspicious regions."""
    _validate_shapes(labels, image_rgb)

    if residual_map is None:
        residual_map = compute_color_residual_map(labels, image_rgb)
    elif residual_map.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: residual_map {residual_map.shape} vs labels {labels.shape}"
        )

    stats: dict[int, dict[str, float]] = {}
    for lbl in sorted(set(labels.flatten()) - {0}):
        vals = residual_map[labels == lbl]
        if vals.size == 0:
            continue
        stats[int(lbl)] = {
            "mean": round(float(vals.mean()), 2),
            "std": round(float(vals.std()), 2),
            "p90": round(float(np.percentile(vals, 90)), 2),
            "max": round(float(vals.max()), 2),
            "area": int(vals.size),
        }
    return stats


def find_high_deviation_regions(
    labels: np.ndarray,
    residual_map: np.ndarray,
    *,
    min_area_frac: float = 0.005,
    deviation_percentile: float = 95.0,
) -> list[dict[str, Any]]:
    """Find connected high-deviation sub-regions within labels.

    For each label, pixels above the label's own percentile residual are
    considered candidate deviants. Connected components of the union across
    labels are returned as spatial candidates. This is a diagnostic proposal,
    not a verdict.
    """
    if labels.shape != residual_map.shape:
        raise ValueError(
            f"Shape mismatch: labels {labels.shape} vs residual_map {residual_map.shape}"
        )

    h, w = labels.shape
    total_area = h * w
    min_area = max(30, int(total_area * min_area_frac))

    high_deviation_mask = np.zeros(labels.shape, dtype=bool)
    for lbl in sorted(set(labels.flatten()) - {0}):
        label_mask = labels == lbl
        if not label_mask.any():
            continue
        label_residuals = residual_map[label_mask]
        if label_residuals.size == 0:
            continue
        threshold = np.percentile(label_residuals, deviation_percentile)
        high_deviation_mask |= label_mask & (residual_map >= threshold)

    cc = label(high_deviation_mask, connectivity=2)
    candidates: list[dict[str, Any]] = []

    for region in regionprops(cc):
        if region.area < min_area:
            continue
        coords = region.coords
        ys = coords[:, 0]
        xs = coords[:, 1]
        candidates.append(
            {
                "bbox": [int(region.bbox[1]), int(region.bbox[0]),
                         int(region.bbox[3]), int(region.bbox[2])],
                "area": int(region.area),
                "centroid": [round(float(region.centroid[1]), 1),
                             round(float(region.centroid[0]), 1)],
                "mean_delta_e": round(float(residual_map[ys, xs].mean()), 2),
                "max_delta_e": round(float(residual_map[ys, xs].max()), 2),
            }
        )

    candidates.sort(key=lambda c: c["area"], reverse=True)
    return candidates


def _residual_to_heatmap(residual_map: np.ndarray) -> np.ndarray:
    """Convert residual values to a blue→green→yellow→red heatmap."""
    max_val = residual_map.max()
    if max_val <= 0:
        return np.zeros((*residual_map.shape, 3), dtype=np.uint8)

    norm = np.clip(residual_map / max_val, 0.0, 1.0)
    h, w = residual_map.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    red = np.clip(norm * 3.0 - 1.5, 0.0, 1.0)
    green = np.clip(1.0 - np.abs(norm * 2.0 - 1.0), 0.0, 1.0)
    blue = np.clip(1.5 - norm * 3.0, 0.0, 1.0)

    heatmap[..., 0] = (red * 255).astype(np.uint8)
    heatmap[..., 1] = (green * 255).astype(np.uint8)
    heatmap[..., 2] = (blue * 255).astype(np.uint8)
    return heatmap


def create_color_residual_overlay(
    residual_map: np.ndarray,
    image_rgb: np.ndarray,
    labels: np.ndarray,
    candidates: list[dict[str, Any]] | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay residual heatmap on the original image.

    Optionally draws red outlines around candidate high-deviation regions.
    """
    if residual_map.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"Shape mismatch: residual_map {residual_map.shape} vs image {image_rgb.shape[:2]}"
        )
    if labels.shape != residual_map.shape:
        raise ValueError(
            f"Shape mismatch: labels {labels.shape} vs residual_map {residual_map.shape}"
        )

    heatmap = _residual_to_heatmap(residual_map)
    overlay = (image_rgb * (1.0 - alpha) + heatmap * alpha).astype(np.uint8)

    if candidates:
        overlay = overlay.copy()
        for cand in candidates:
            x0, y0, x1, y1 = cand["bbox"]
            overlay[y0:y1, x0] = [255, 0, 0]
            overlay[y0:y1, x1 - 1] = [255, 0, 0]
            overlay[y0, x0:x1] = [255, 0, 0]
            overlay[y1 - 1, x0:x1] = [255, 0, 0]

    return overlay


def compute_color_residual_audit(
    labels: np.ndarray,
    image_rgb: np.ndarray,
) -> dict[str, Any]:
    """Aggregate color residual diagnostics for visual audit."""
    _validate_shapes(labels, image_rgb)

    representatives = compute_label_representative_colors(labels, image_rgb)
    residual_map = compute_color_residual_map(labels, image_rgb, representatives)
    stats = compute_label_residual_stats(labels, image_rgb, residual_map)
    candidates = find_high_deviation_regions(labels, residual_map)

    non_zero = residual_map[residual_map > 0]
    global_stats = {
        "mean_delta_e": round(float(non_zero.mean()), 2) if non_zero.size > 0 else 0.0,
        "max_delta_e": round(float(non_zero.max()), 2) if non_zero.size > 0 else 0.0,
        "high_deviation_region_count": len(candidates),
    }

    serializable_reps = {
        str(lbl): {
            "median_rgb": reps["median_rgb"].tolist(),
            "median_lab": reps["median_lab"].tolist(),
        }
        for lbl, reps in representatives.items()
    }

    return {
        "representative_colors": serializable_reps,
        "global_residual": global_stats,
        "per_label_residual": stats,
        "high_deviation_regions": candidates,
    }
