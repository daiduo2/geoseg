"""e018: e014 + Morphology denoising.

Pipeline:
    1. Run e014 edge-guided to get initial labels
    2. Region analysis via skimage.measure.regionprops
    3. Small region merge (< min_area) into neighbor with longest shared boundary
    4. Thin region merge (perimeter^2 / area > 80) into neighbor
    5. Boundary smoothing: binary_closing then binary_opening (disk r=1)
    6. Hole filling inside large regions
    7. Compute final palette from original image using cleaned labels
    8. Return SegmentResult
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, closing, opening

from lib.segment import SegmentResult, saturation_ratio
from lib.segment_edge_guided import segment_jet_vivid_edge_guided


def _compute_shared_boundary(labels: np.ndarray, target_label: int, neighbor_label: int) -> int:
    """Count 4-connected boundary pixels between two label regions."""
    target_mask = labels == target_label
    neighbor_mask = labels == neighbor_label
    # Dilate target and intersect with neighbor
    dilated = ndimage.binary_dilation(target_mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool))
    return int(np.sum(dilated & neighbor_mask))


def _find_best_neighbor(labels: np.ndarray, comp_mask: np.ndarray, exclude_label: int) -> int:
    """Find neighbor label with longest shared boundary."""
    dilated = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool))
    neigh_labels = labels[dilated & ~comp_mask]
    if neigh_labels.size == 0:
        return -1
    vals, counts = np.unique(neigh_labels, return_counts=True)
    # Exclude the label itself if present
    mask = vals != exclude_label
    vals = vals[mask]
    counts = counts[mask]
    if len(vals) == 0:
        return -1
    return int(vals[counts.argmax()])


def _merge_small_regions(labels: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    """Merge connected components smaller than min_area into their best neighbor."""
    out = labels.copy()
    merged = 0
    for lbl in np.unique(out):
        if lbl < 0:
            continue
        mask = out == lbl
        cc = label(mask, connectivity=2)
        regions = regionprops(cc)
        for r in regions:
            if r.area >= min_area:
                continue
            comp_mask = cc == r.label
            best_neighbor = _find_best_neighbor(out, comp_mask, lbl)
            if best_neighbor >= 0:
                out[comp_mask] = best_neighbor
                merged += 1
    return out, merged


def _merge_thin_regions(labels: np.ndarray, thin_threshold: float = 80.0) -> tuple[np.ndarray, int]:
    """Merge connected components with perimeter^2/area > threshold into neighbor."""
    out = labels.copy()
    merged = 0
    for lbl in np.unique(out):
        if lbl < 0:
            continue
        mask = out == lbl
        cc = label(mask, connectivity=2)
        regions = regionprops(cc)
        for r in regions:
            area = max(r.area, 1e-9)
            perim = r.perimeter
            ratio = float("inf") if perim == 0 else (perim ** 2) / area
            if ratio <= thin_threshold:
                continue
            comp_mask = cc == r.label
            best_neighbor = _find_best_neighbor(out, comp_mask, lbl)
            if best_neighbor >= 0:
                out[comp_mask] = best_neighbor
                merged += 1
    return out, merged


def _smooth_boundaries(labels: np.ndarray) -> np.ndarray:
    """Apply closing then opening (disk r=1) per label mask, largest-first to avoid over-swallowing.

    Uses a signed-distance approach to resolve overlaps: after smoothing each label,
    pixels claimed by multiple labels are assigned to the one whose *original* mask
    had the closest boundary.
    """
    h, w = labels.shape
    unique_labels = np.unique(labels)
    label_areas = [(lbl, np.sum(labels == lbl)) for lbl in unique_labels if lbl >= 0]
    label_areas.sort(key=lambda x: x[1], reverse=True)

    # Precompute distance-to-boundary for every original label mask
    dist_to_boundary: dict[int, np.ndarray] = {}
    for lbl, _ in label_areas:
        mask = labels == lbl
        # Distance transform: positive inside, negative outside
        dist_in = ndimage.distance_transform_edt(mask)
        dist_out = ndimage.distance_transform_edt(~mask)
        dist_to_boundary[lbl] = dist_in - dist_out

    # Track which labels claim each pixel using numpy arrays for speed
    claim_best_label = np.full((h, w), -1, dtype=np.int32)
    claim_best_dist = np.full((h, w), -np.inf, dtype=np.float32)

    for lbl, _ in label_areas:
        mask = labels == lbl
        mask = closing(mask, footprint=disk(1))
        mask = opening(mask, footprint=disk(1))
        # Update best label for pixels where this label has higher dist_to_boundary
        d = dist_to_boundary[lbl]
        better = mask & (d > claim_best_dist)
        claim_best_label[better] = lbl
        claim_best_dist[better] = d[better]

    # Unambiguous pixels (only one claim) -> that label
    # Ambiguous pixels (multiple claims) -> label with highest dist_to_boundary
    out = claim_best_label.copy()

    # Fill any remaining unassigned pixels with original labels
    unassigned = out == -1
    if unassigned.any():
        out[unassigned] = labels[unassigned]
    return out


def _fill_holes(labels: np.ndarray, min_region_area: int = 100) -> np.ndarray:
    """Fill holes inside large regions (background pixels completely surrounded by one label)."""
    out = labels.copy()
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        if lbl < 0:
            continue
        mask = labels == lbl
        if np.sum(mask) < min_region_area:
            continue
        # Fill holes in this mask
        filled = ndimage.binary_fill_holes(mask)
        hole_mask = filled & ~mask
        if hole_mask.any():
            out[hole_mask] = lbl
    return out


def _remap_labels_sequential(labels: np.ndarray) -> np.ndarray:
    """Remap labels to 0..k-1."""
    unique = np.unique(labels)
    out = labels.copy()
    for new_id, old_id in enumerate(unique):
        out[labels == old_id] = new_id
    return out


def _compute_palette(panel_rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute mean RGB for each label."""
    unique = np.unique(labels)
    palette = []
    for lbl in unique:
        if lbl < 0:
            continue
        mask = labels == lbl
        mean_rgb = panel_rgb[mask].mean(axis=0).astype(np.uint8)
        palette.append(mean_rgb)
    return np.array(palette, dtype=np.uint8)


def segment_jet_vivid_morph_clean(
    panel_rgb: np.ndarray,
    reps: list[dict],
    max_auto_k: int = 3,
    edge_weight: float = 0.3,
    edge_percentile: float = 90.0,
    sigma: float = 4.0,
) -> SegmentResult:
    """e018: e014 edge-guided + morphology-based denoising.

    Parameters
    ----------
    panel_rgb : np.ndarray
        (H, W, 3) uint8 cropped panel.
    reps : list[dict]
        VLM representative points.
    max_auto_k : int
        Maximum extra seeds to auto-detect (passed to e014).
    edge_weight : float
        Spatial penalty strength for e014.
    edge_percentile : float
        Percentile threshold for edge detection.
    sigma : float
        Gaussian fall-off width for the edge penalty.

    Returns
    -------
    SegmentResult
    """
    h, w, _ = panel_rgb.shape
    min_area = max(50, h * w * 0.01)

    # --- Step 1: Run e014 edge-guided ---
    result_e014 = segment_jet_vivid_edge_guided(
        panel_rgb,
        reps,
        max_auto_k=max_auto_k,
        edge_weight=edge_weight,
        edge_percentile=edge_percentile,
        sigma=sigma,
    )
    labels = result_e014.labels.copy()

    # Count initial connected components
    cc_initial = label(labels >= 0, connectivity=2)
    n_cc_initial = len(regionprops(cc_initial))

    # --- Step 2: Small region merge ---
    labels, n_small_merged = _merge_small_regions(labels, min_area)

    # --- Step 3: Thin region merge ---
    labels, n_thin_merged = _merge_thin_regions(labels, thin_threshold=80.0)

    # --- Step 4: Boundary smoothing ---
    labels = _smooth_boundaries(labels)

    # --- Step 5: Hole filling ---
    labels = _fill_holes(labels, min_region_area=min_area)

    # --- Step 6: Remap labels sequentially ---
    labels = _remap_labels_sequential(labels)

    # --- Step 7: Compute final palette ---
    palette = _compute_palette(panel_rgb, labels)

    # Count final connected components
    cc_final = label(labels >= 0, connectivity=2)
    n_cc_final = len(regionprops(cc_final))

    color_names = [f"region_{i}" for i in range(len(palette))]

    return SegmentResult(
        labels=labels,
        palette=palette,
        color_names=color_names,
        path="jet_vivid_morph_clean",
        saturation_ratio=saturation_ratio(panel_rgb),
        notes={
            "e014_notes": result_e014.notes,
            "n_cc_initial": n_cc_initial,
            "n_cc_final": n_cc_final,
            "n_small_merged": n_small_merged,
            "n_thin_merged": n_thin_merged,
            "min_area": min_area,
        },
    )
