"""Visual audit view generation for segmentation quality review.

These views are intentionally designed to EXPOSE problems:
- No saturation/brightness boost that beautifies defects.
- No single alpha-blended overlay as the only view.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import segmentation
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.measure import label, regionprops

from geoseg.modules.segment_engines._shared import (
    _create_overlay,
    _distinct_colors,
)
from geoseg.modules.visual_audit.color_residual import (
    compute_color_residual_map,
    create_color_residual_overlay,
)


RED = np.array([255, 0, 0], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)
BLACK = np.array([0, 0, 0], dtype=np.uint8)


def _boundary_mask(labels: np.ndarray, mode: str = "thick") -> np.ndarray:
    """Return boolean mask of label boundaries."""
    return segmentation.find_boundaries(labels, mode=mode)


def create_boundary_on_image(
    labels: np.ndarray,
    image_rgb: np.ndarray,
    boundary_color: tuple[int, int, int] = (255, 0, 0),
    boundary_width: str = "thick",
) -> np.ndarray:
    """Draw segmentation boundaries on top of an existing image.

    Use with `no_text_rgb` to verify boundaries align with real geology.
    """
    if labels.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"Shape mismatch: labels {labels.shape} vs image {image_rgb.shape[:2]}"
        )

    result = image_rgb.copy()
    boundaries = _boundary_mask(labels, mode=boundary_width)
    result[boundaries] = boundary_color
    return result


def create_pure_mask(
    labels: np.ndarray,
    overlay_colors: np.ndarray | None = None,
) -> np.ndarray:
    """Create a pure segmentation mask with no original image blending.

    Every tiny label becomes visible because there is no alpha blending.
    """
    h, w = labels.shape
    n = int(labels.max()) + 1

    if overlay_colors is None:
        colors = _distinct_colors(n)
    else:
        colors = overlay_colors.astype(np.uint8)
        if len(colors) < n:
            padded = np.zeros((n, 3), dtype=np.uint8)
            padded[: len(colors)] = colors
            colors = padded

    mask = np.zeros((h, w, 3), dtype=np.uint8)
    unique = sorted(np.unique(labels))
    for lbl in unique:
        if lbl == 0:
            continue
        mask[labels == lbl] = colors[int(lbl) % len(colors)]

    boundaries = _boundary_mask(labels, mode="inner")
    mask[boundaries] = WHITE
    return mask


def create_fragment_highlight(
    labels: np.ndarray,
    image_rgb: np.ndarray,
    min_area_frac: float = 0.001,
) -> np.ndarray:
    """Highlight tiny isolated islands in red on top of the original image.

    Args:
        labels: int32 label array.
        image_rgb: original RGB image.
        min_area_frac: area fraction below which a connected component is
            considered a tiny island.

    Returns:
        RGB image with tiny islands highlighted in red.
    """
    if labels.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"Shape mismatch: labels {labels.shape} vs image {image_rgb.shape[:2]}"
        )

    h, w = labels.shape
    total_area = h * w
    min_area = max(30, int(total_area * min_area_frac))

    result = image_rgb.copy()
    unique = sorted(set(labels.flatten()) - {0})

    for lbl in unique:
        mask = labels == lbl
        cc = label(mask, connectivity=2)
        for region in regionprops(cc):
            if region.area < min_area:
                coords = region.coords
                result[coords[:, 0], coords[:, 1]] = RED

    return result


def create_text_residual_map(
    labels: np.ndarray,
    image_rgb: np.ndarray,
    text_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Highlight text regions and label boundaries to detect text-to-label leaks.

    If text_mask is None, attempts a lightweight adaptive threshold + Laplacian
    detection as a fallback. The caller should prefer providing a real text mask.
    """
    if labels.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"Shape mismatch: labels {labels.shape} vs image {image_rgb.shape[:2]}"
        )

    if text_mask is None:
        text_mask = _estimate_text_mask(image_rgb)

    result = image_rgb.copy()
    boundaries = _boundary_mask(labels, mode="thick")

    # Text regions in blue tint
    result[text_mask] = (result[text_mask] * 0.6 + np.array([0, 0, 255]) * 0.4).astype(np.uint8)
    # Label boundaries in red
    result[boundaries & ~text_mask] = RED
    return result


def _estimate_text_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Lightweight text region estimation fallback."""
    import cv2

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=25, C=-5,
    )
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    laplacian = (laplacian > np.percentile(laplacian, 80)).astype(np.uint8) * 255
    text_mask = ((adaptive > 0) & (laplacian > 0))
    return ndimage.binary_dilation(text_mask, iterations=2)


def create_topology_map(labels: np.ndarray) -> np.ndarray:
    """Pseudo-color labels ordered by median y-coordinate.

    Helps verify that layer ordering matches geological expectations
    (older/deeper layers at bottom, younger/surface layers at top).
    """
    from geoseg.modules.segment_engines.v4_kmeans import _reorder_labels_by_median_y

    ordered = _reorder_labels_by_median_y(labels)
    return create_pure_mask(ordered)


def create_difference_heatmap(
    labels: np.ndarray,
    image_rgb: np.ndarray,
) -> np.ndarray:
    """Visualize where segmentation boundaries deviate from image color edges.

    Green = boundary aligns with color edge.
    Red = boundary exists without corresponding color edge (suspect over-segmentation).
    """
    if labels.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"Shape mismatch: labels {labels.shape} vs image {image_rgb.shape[:2]}"
        )

    gray = rgb2gray(image_rgb)
    edges = sobel(gray)
    edge_mask = np.abs(edges) > 0.05
    seg_boundaries = _boundary_mask(labels, mode="thick")

    result = image_rgb.copy()
    aligned = seg_boundaries & edge_mask
    misaligned = seg_boundaries & ~edge_mask
    result[aligned] = np.array([0, 200, 0], dtype=np.uint8)
    result[misaligned] = RED
    return result


def create_color_residual_heatmap(
    labels: np.ndarray,
    image_rgb: np.ndarray,
) -> np.ndarray:
    """Visualize per-label color inconsistency as a heatmap overlay.

    Blue/green = pixel matches its label's representative color.
    Yellow/red = pixel strongly deviates (candidate for re-segmentation).
    """
    if labels.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"Shape mismatch: labels {labels.shape} vs image {image_rgb.shape[:2]}"
        )

    residual_map = compute_color_residual_map(labels, image_rgb)
    return create_color_residual_overlay(residual_map, image_rgb, labels, alpha=0.5)


def create_side_by_side(
    labels: np.ndarray,
    image_rgb: np.ndarray,
) -> np.ndarray:
    """Original image side-by-side with pure label mask for direct comparison."""
    if labels.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"Shape mismatch: labels {labels.shape} vs image {image_rgb.shape[:2]}"
        )

    mask = create_pure_mask(labels)
    h, w = image_rgb.shape[:2]
    gap = 10
    canvas = np.full((h, w * 2 + gap, 3), 32, dtype=np.uint8)
    canvas[:, :w] = image_rgb
    canvas[:, w + gap :] = mask
    return canvas


def create_plume_comparison(
    labels: np.ndarray,
    image_rgb: np.ndarray,
    gt_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compare predicted plume label(s) with optional manual GT mask.

    Left: original. Middle: predicted plume highlight. Right (if provided): GT.
    """
    if labels.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"Shape mismatch: labels {labels.shape} vs image {image_rgb.shape[:2]}"
        )

    from geoseg.modules.visual_audit.semantic import compute_plume_fidelity

    plume = compute_plume_fidelity(labels, image_rgb, gt_mask)
    plume_label = plume.get("plume_label")

    h, w = image_rgb.shape[:2]
    gap = 10
    n_panels = 3 if gt_mask is not None else 2
    canvas = np.full((h, w * n_panels + gap * (n_panels - 1), 3), 32, dtype=np.uint8)

    # Original
    canvas[:, :w] = image_rgb

    # Predicted plume highlight
    pred_view = image_rgb.copy()
    if plume_label is not None:
        pred_mask = labels == plume_label
        pred_view[pred_mask] = np.array([0, 255, 0], dtype=np.uint8)
    canvas[:, w + gap : 2 * w + gap] = pred_view

    # GT mask
    if gt_mask is not None:
        gt_view = image_rgb.copy()
        gt_view[gt_mask] = np.array([0, 255, 0], dtype=np.uint8)
        canvas[:, 2 * (w + gap) :] = gt_view

    return canvas


def create_audit_views(
    labels: np.ndarray,
    panel_rgb: np.ndarray,
    no_text_rgb: np.ndarray | None = None,
    text_mask: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Generate the full set of audit views.

    Returns:
        dict with keys:
        - boundary_on_original: boundaries drawn on panel_rgb
        - boundary_on_no_text: boundaries drawn on no_text_rgb (if provided)
        - pure_mask: discrete color mask, no blending
        - fragment_highlight: tiny islands in red
        - text_residual_map: text mask + boundaries
        - topology_map: labels reordered by median y
        - difference_heatmap: boundary vs color edge alignment
        - side_by_side: original vs pure mask
        - plume_comparison: original / predicted plume / GT (if provided)
    """
    views = {
        "boundary_on_original": create_boundary_on_image(labels, panel_rgb),
        "pure_mask": create_pure_mask(labels),
        "fragment_highlight": create_fragment_highlight(labels, panel_rgb),
        "text_residual_map": create_text_residual_map(labels, panel_rgb, text_mask),
        "topology_map": create_topology_map(labels),
        "difference_heatmap": create_difference_heatmap(labels, panel_rgb),
        "color_residual": create_color_residual_heatmap(labels, panel_rgb),
        "side_by_side": create_side_by_side(labels, panel_rgb),
        "plume_comparison": create_plume_comparison(labels, panel_rgb, gt_mask),
    }

    if no_text_rgb is not None:
        views["boundary_on_no_text"] = create_boundary_on_image(
            labels, no_text_rgb, boundary_color=(255, 0, 0)
        )

    return views


def save_views(views: dict[str, np.ndarray], output_dir: str) -> dict[str, str]:
    """Save all views to disk and return path mapping."""
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, arr in views.items():
        path = out / f"{name}.jpg"
        Image.fromarray(arr).save(path, quality=90)
        paths[name] = str(path)
    return paths
