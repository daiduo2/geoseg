"""v4_kmeans: dual-path color segmentation for geophysics panels.

- jet_vivid path: VLM reps -> nearest-median in LAB + shape filter
- colorbar_guided path: colorbar seeds -> K-means RGB + hole fill + small-merge
- pastel_faded fallback: CV seeds or K-means++ + shape filter

Unified interface: segment() returns {"labels", "seeds", "overlay", "meta"}.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy import ndimage
from skimage.color import lab2rgb, rgb2lab

from geoseg.modules.segment_engines._shared import (
    _create_overlay,
    _shape_filter,
    _label_by_nearest,
    _estimate_background_color,
    _is_background_v2,
    _cv_seeds,
    _refine_vlm_seeds,
    _auto_k,
    saturation_ratio,
    _find_pixel_for_color,
    _scan_for_missing_colors,
    _parse_count_from_tag,
)


JET_VIVID_RATIO = 0.05


def _nearest_median(
    panel_lab: np.ndarray,
    seeds_lab: np.ndarray,
    median_size: int = 5,
) -> np.ndarray:
    """Per-pixel nearest seed in LAB, followed by median-filter smoothing."""
    labels = _label_by_nearest(panel_lab, seeds_lab)
    if median_size > 1:
        labels = ndimage.median_filter(labels, size=median_size)
    return labels


def _sample_colorbar_seeds(colorbar_rgb: np.ndarray, k: int) -> tuple[np.ndarray, list[str]]:
    """Sample k evenly-spaced RGBs along a colorbar strip."""
    h, w, _ = colorbar_rgb.shape
    if h >= w:
        ys = np.linspace(int(0.05 * h), int(0.95 * h) - 1, k).astype(int)
        cx = w // 2
        seeds = np.array([colorbar_rgb[y, cx] for y in ys])
    else:
        xs = np.linspace(int(0.05 * w), int(0.95 * w) - 1, k).astype(int)
        cy = h // 2
        seeds = np.array([colorbar_rgb[cy, x] for x in xs])
    names = _name_palette(seeds, k)
    return seeds.astype(np.uint8), names


def _name_palette(seeds_rgb: np.ndarray, k: int) -> list[str]:
    """Label k seed colors with conventional names."""
    standard = ["red", "orange", "yellow", "green", "blue", "purple"]
    if k > len(standard):
        standard = standard + [f"c{i}" for i in range(len(standard), k)]
    pool = standard[:k]
    rgb = seeds_rgb.astype(np.float32) / 255.0
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    diff = mx - mn + 1e-9
    h = np.zeros(k)
    for i in range(k):
        r, g, b = rgb[i]
        if mx[i] == r:
            h[i] = (60 * ((g - b) / diff[i]) + 360) % 360
        elif mx[i] == g:
            h[i] = 60 * ((b - r) / diff[i]) + 120
        else:
            h[i] = 60 * ((r - g) / diff[i]) + 240
    order = np.argsort(h)
    names = [""] * k
    for rank, original_idx in enumerate(order):
        names[original_idx] = pool[rank]
    return names


def _reorder_labels_by_median_y(labels: np.ndarray) -> np.ndarray:
    """Reorder labels so that top=lowest index, bottom=highest index."""
    h, w = labels.shape
    unique = np.unique(labels[labels >= 0])
    if len(unique) == 0:
        return labels.copy()

    median_y = {}
    for lbl in unique:
        ys = np.where(labels == lbl)[0]
        median_y[lbl] = np.median(ys) if len(ys) > 0 else h

    sorted_by_y = sorted(median_y.items(), key=lambda x: x[1])
    old_to_new = {old: new for new, (old, _) in enumerate(sorted_by_y)}

    out = np.full_like(labels, -1)
    for old, new in old_to_new.items():
        out[labels == old] = new
    return out


def _fill_holes(labels: np.ndarray) -> np.ndarray:
    """Fill holes inside each labeled region."""
    out = labels.copy()
    for lbl in range(int(labels.max()) + 1):
        mask = labels == lbl
        if not mask.any():
            continue
        filled = ndimage.binary_fill_holes(mask)
        out[filled & (labels != lbl)] = lbl
    return out


def _remove_small_components(labels: np.ndarray, min_area_frac: float = 0.001) -> np.ndarray:
    """Merge tiny connected components (< min_area_frac of panel area) into neighbors."""
    h, w = labels.shape
    out = labels.copy()
    min_area = max(50, int(h * w * min_area_frac))

    for lbl in range(int(labels.max()) + 1):
        mask = out == lbl
        if not mask.any():
            continue
        labeled, num = ndimage.label(mask)
        if num <= 1:
            continue
        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        for comp_id in range(1, num + 1):
            if sizes[comp_id - 1] < min_area:
                comp_mask = labeled == comp_id
                dilated = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool))
                neighbors = out[dilated & ~comp_mask & (out >= 0)]
                if len(neighbors) > 0:
                    new_lbl = int(np.bincount(neighbors).argmax())
                    out[comp_mask] = new_lbl
    return out


def _enhance_close_boundaries(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    palette_rgb: np.ndarray,
    color_dist_threshold: float = 55.0,
) -> np.ndarray:
    """Re-classify boundary pixels between adjacent layers with similar seed colors."""
    out = labels.copy()
    k = len(palette_rgb)
    if k < 2:
        return out

    for i in range(k - 1):
        d = float(np.linalg.norm(palette_rgb[i].astype(np.float32) - palette_rgb[i + 1].astype(np.float32)))
        if d >= color_dist_threshold:
            continue

        mask1 = out == i
        mask2 = out == (i + 1)
        if not mask1.any() or not mask2.any():
            continue

        dilated1 = ndimage.binary_dilation(mask1, structure=np.ones((3, 3), dtype=bool))
        dilated2 = ndimage.binary_dilation(mask2, structure=np.ones((3, 3), dtype=bool))
        boundary = dilated1 & dilated2
        if not boundary.any():
            continue

        coords = np.where(boundary)
        boundary_pixels = panel_rgb[coords].astype(np.float32)
        d1 = np.linalg.norm(boundary_pixels - palette_rgb[i].astype(np.float32), axis=1)
        d2 = np.linalg.norm(boundary_pixels - palette_rgb[i + 1].astype(np.float32), axis=1)
        reclass = d2 < d1
        out[coords[0][reclass], coords[1][reclass]] = i + 1
        out[coords[0][~reclass], coords[1][~reclass]] = i

    return out


def segment_jet_vivid(
    panel_rgb: np.ndarray,
    reps: list[dict] | None = None,
    n_layers: int = 5,
    max_auto_k: int = 0,
) -> dict:
    """Nearest-median segmentation for vivid jet-colormap panels.

    Returns {"labels", "seeds", "overlay", "meta"}.
    """
    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)
    bg_rgb = _estimate_background_color(panel_rgb)
    min_auto_count = max(50, h * w // 2000)

    if reps:
        cv_seeds_rgb, cv_tags = _cv_seeds(panel_rgb, k=len(reps))
        used_cv_indices: set[int] = set()

        refined_seeds, refined_reps = _refine_vlm_seeds(
            panel_rgb, reps, bg_rgb, cv_seeds_rgb, cv_tags, used_cv_indices
        )
        color_names = [r.get("color_name", f"layer_{i + 1}") for i, r in enumerate(reps)]
    else:
        cv_seeds_rgb, cv_tags = _cv_seeds(panel_rgb, k=n_layers)
        used_cv_indices: set[int] = set()
        refined_seeds = []
        refined_reps = []
        color_names = [f"layer_{i + 1}" for i in range(n_layers)]

    refined_seeds, refined_reps = _auto_k(
        panel_rgb, panel_lab, bg_rgb,
        refined_seeds, refined_reps,
        cv_seeds_rgb, cv_tags, used_cv_indices,
        max_auto_k, min_auto_count,
    )
    if len(refined_reps) > len(color_names):
        color_names = color_names + [r["name"] for r in refined_reps[len(color_names):]]

    if not refined_seeds:
        refined_seeds = [cv_seeds_rgb[i] for i in range(min(n_layers, len(cv_seeds_rgb)))]

    refined_seeds_arr = np.array(refined_seeds, dtype=np.uint8)
    seeds_lab = rgb2lab(refined_seeds_arr[np.newaxis, ...])[0]

    labels = _nearest_median(panel_lab, seeds_lab, median_size=5)
    labels = _shape_filter(labels)

    overlay = _create_overlay(panel_rgb, labels, refined_seeds_arr)

    return {
        "labels": labels,
        "seeds": refined_seeds_arr.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "v4_kmeans",
            "path": "jet_vivid",
            "reps_refined": refined_reps,
            "cv_seeds": cv_seeds_rgb.tolist() if len(cv_seeds_rgb) else [],
            "bg_rgb": bg_rgb.tolist(),
            "auto_k_added": len(refined_reps) - (len(reps) if reps else 0),
            "saturation_ratio": round(saturation_ratio(panel_rgb), 4),
        },
    }


def segment_colorbar_guided(
    panel_rgb: np.ndarray,
    colorbar_rgb: np.ndarray,
    n_layers: int = 5,
    color_dist_threshold: float = 55.0,
    explicit_seeds: list[dict] | None = None,
    n_color_zones: int = 0,
) -> dict:
    """Colorbar-guided K-means segmentation.

    Returns {"labels", "seeds", "overlay", "meta"}.
    """
    h, w, _ = panel_rgb.shape
    pixels = panel_rgb.reshape(-1, 3).astype(np.float64)

    # When VLM sees many color zones, increase k and tighten boundary merging
    effective_n_layers = n_layers
    effective_threshold = color_dist_threshold
    if n_color_zones >= 3:
        effective_n_layers = max(n_layers, n_color_zones + 1)
        effective_threshold = 35.0

    if explicit_seeds is not None and len(explicit_seeds) > 0:
        seeds_rgb = np.array([s["rgb"] for s in explicit_seeds], dtype=np.uint8)
        names = [s.get("name", f"layer_{i+1}") for i, s in enumerate(explicit_seeds)]
        k = len(explicit_seeds)
    else:
        seeds_rgb, names = _sample_colorbar_seeds(colorbar_rgb, effective_n_layers)
        k = effective_n_layers
    seeds_arr = seeds_rgb.astype(np.float64)

    centroids, labels_flat = kmeans2(pixels, seeds_arr, minit="matrix")
    labels = labels_flat.reshape(h, w).astype(np.int32)

    labels = _reorder_labels_by_median_y(labels)
    labels = _fill_holes(labels)
    labels = _remove_small_components(labels, min_area_frac=0.001)

    ordered_palette = np.zeros((k, 3), dtype=np.uint8)
    for lbl in range(k):
        mask = labels == lbl
        if mask.any():
            ordered_palette[lbl] = panel_rgb[mask].mean(axis=0).astype(np.uint8)
        else:
            ordered_palette[lbl] = (lab2rgb(centroids[lbl][np.newaxis, ...])[0] * 255).clip(0, 255).astype(np.uint8)

    labels = _enhance_close_boundaries(panel_rgb, labels, ordered_palette, effective_threshold)

    final_palette = np.zeros((k, 3), dtype=np.uint8)
    for lbl in range(k):
        mask = labels == lbl
        if mask.any():
            final_palette[lbl] = panel_rgb[mask].mean(axis=0).astype(np.uint8)
        else:
            final_palette[lbl] = ordered_palette[lbl]

    overlay = _create_overlay(panel_rgb, labels, final_palette)

    return {
        "labels": labels,
        "seeds": final_palette.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "v4_kmeans",
            "path": "colorbar_guided",
            "seed_origin": "explicit_seeds" if explicit_seeds is not None else "colorbar",
            "n_layers": k,
            "saturation_ratio": round(saturation_ratio(panel_rgb), 4),
        },
    }


def segment_pastel_faded(
    panel_rgb: np.ndarray,
    colorbar_rgb: np.ndarray | None = None,
    n_layers: int = 5,
    n_color_zones: int = 0,
) -> dict:
    """K-means in LAB space, optionally seeded from the panel's colorbar.

    Returns {"labels", "seeds", "overlay", "meta"}.
    """
    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb).reshape(-1, 3)

    if colorbar_rgb is not None and colorbar_rgb.size > 0:
        seeds_rgb, names = _sample_colorbar_seeds(colorbar_rgb, n_layers)
        seeds_lab = rgb2lab(seeds_rgb[np.newaxis, ...])[0]
        centroids, labels_flat = kmeans2(panel_lab, seeds_lab, minit="matrix")
        seed_origin = "colorbar"
    else:
        cv_seeds_rgb, cv_tags = _cv_seeds(panel_rgb, k=n_layers)
        if len(cv_seeds_rgb) >= n_layers:
            seeds_rgb = cv_seeds_rgb[:n_layers]
            seeds_lab = rgb2lab(seeds_rgb[np.newaxis, ...])[0]
            centroids, labels_flat = kmeans2(panel_lab, seeds_lab, minit="matrix")
            seed_origin = "cv_multi_source"
            names = [f"cv_{i}" for i in range(n_layers)]
        else:
            centroids, labels_flat = kmeans2(panel_lab, n_layers, minit="++", seed=42)
            approx = (lab2rgb(centroids[np.newaxis, ...])[0] * 255).clip(0, 255).astype(np.uint8)
            seeds_rgb = approx
            names = _name_palette(seeds_rgb, n_layers)
            seed_origin = "kmeans++_random"

    labels = labels_flat.reshape(h, w).astype(np.int32)
    labels = _shape_filter(labels)

    overlay = _create_overlay(panel_rgb, labels, seeds_rgb)

    return {
        "labels": labels,
        "seeds": seeds_rgb.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "v4_kmeans",
            "path": "pastel_faded",
            "seed_origin": seed_origin,
            "n_layers": n_layers,
            "saturation_ratio": round(saturation_ratio(panel_rgb), 4),
        },
    }


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
