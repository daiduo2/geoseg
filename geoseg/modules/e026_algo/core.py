"""E026 core segmentation algorithm (cleaned from process_content_zone_vivid_mask.py).

Removes all hardcoded paths and panel coordinates. Pure functions only.
"""

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv


def auto_extract_seeds(panel: np.ndarray, n_layers: int = 7) -> list:
    """Extract layer seed colors via KMeans on non-white pixels, sorted by vertical position.

    Args:
        panel: RGB image array (H, W, 3).
        n_layers: Number of layers to extract (default 7 for ph01).

    Returns:
        List of RGB seed colors, sorted top-to-bottom by median y-position.
    """
    h, w = panel.shape[:2]
    pixels = panel.reshape(-1, 3)
    non_white = np.all(pixels < 240, axis=1)
    content_pixels = pixels[non_white]
    if len(content_pixels) < n_layers * 100:
        content_pixels = pixels
    kmeans = KMeans(n_clusters=n_layers, random_state=42, n_init=10).fit(content_pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.predict(pixels)
    label_img = labels.reshape(h, w)
    median_y = {}
    for lbl in range(n_layers):
        ys = np.where(label_img == lbl)[0]
        median_y[lbl] = np.median(ys) if len(ys) > 0 else h
    sorted_by_y = sorted(median_y.items(), key=lambda x: x[1])
    return [centers[old].tolist() for old, _ in sorted_by_y]


def segment_fixed_nn(img: np.ndarray, seeds: list) -> np.ndarray:
    """Segment image by nearest-neighbor to seeds, with post-processing.

    Post-processing:
    1. Reorder labels top-to-bottom by median y.
    2. Fill holes in each layer.
    3. Remove small components (< 0.1% area), reclassify to largest neighbor.
    4. Boundary reclassification for adjacent similar layers.

    Args:
        img: RGB image array (H, W, 3).
        seeds: List of RGB seed colors.

    Returns:
        Label array (H, W), labels 1..n_layers. Background = 0.
    """
    h, w = img.shape[:2]
    n_layers = len(seeds)
    pixels = img.reshape(-1, 3)
    seeds_arr = np.array(seeds, dtype=float)
    d2 = ((pixels[:, None, :] - seeds_arr[None, :, :]) ** 2).sum(axis=2)
    labels = d2.argmin(axis=1).reshape(h, w)

    # Reorder labels top-to-bottom
    median_y = {}
    for lbl in range(n_layers):
        ys = np.where(labels == lbl)[0]
        median_y[lbl] = np.median(ys) if len(ys) > 0 else h
    sorted_by_y = sorted(median_y.items(), key=lambda x: x[1])
    old_to_new = {old: new for new, (old, _) in enumerate(sorted_by_y, start=1)}
    new_labels = np.zeros_like(labels)
    for old, new in old_to_new.items():
        new_labels[labels == old] = new
    labels = new_labels

    # Fill holes
    for lbl in range(1, n_layers + 1):
        mask = labels == lbl
        if mask.any():
            filled = ndimage.binary_fill_holes(mask)
            labels[filled & (labels == 0)] = lbl

    # Remove small components
    min_area = max(50, int(h * w * 0.001))
    for lbl in range(1, n_layers + 1):
        mask = labels == lbl
        if not mask.any():
            continue
        labeled, num = ndimage.label(mask)
        if num <= 1:
            continue
        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        for comp_id in range(1, num + 1):
            if sizes[comp_id - 1] < min_area:
                comp_mask = labeled == comp_id
                dilated = ndimage.binary_dilation(comp_mask)
                neighbors = labels[dilated & ~comp_mask & (labels > 0)]
                if len(neighbors) > 0:
                    labels[comp_mask] = int(np.bincount(neighbors).argmax())

    # Boundary reclassification for similar adjacent layers
    seeds_ordered = np.array([seeds[i - 1] for i in range(1, n_layers + 1)])
    for i in range(1, n_layers):
        d = np.linalg.norm(seeds_ordered[i - 1] - seeds_ordered[i])
        if d < 60:
            l1, l2 = i, i + 1
            mask1 = labels == l1
            mask2 = labels == l2
            dilated1 = ndimage.binary_dilation(mask1)
            dilated2 = ndimage.binary_dilation(mask2)
            boundary = dilated1 & dilated2
            if not boundary.any():
                continue
            coords = np.where(boundary)
            boundary_pixels = img[coords].astype(float)
            d1 = np.linalg.norm(boundary_pixels - seeds_ordered[l1 - 1], axis=1)
            d2 = np.linalg.norm(boundary_pixels - seeds_ordered[l2 - 1], axis=1)
            reclass = d2 < d1
            labels[coords[0][reclass], coords[1][reclass]] = l2
            labels[coords[0][~reclass], coords[1][~reclass]] = l1

    return labels


def vivid_color(rgb: np.ndarray, sat_boost: float = 0.35, val_boost: float = 0.1) -> np.ndarray:
    """Boost saturation (+ slight brightness) in HSV space.

    Args:
        rgb: RGB array shape (3,).
        sat_boost: Saturation increment (default 0.35).
        val_boost: Value increment (default 0.1).

    Returns:
        Enhanced RGB array shape (3,), uint8.
    """
    rgb_norm = rgb.astype(float) / 255.0
    hsv = rgb_to_hsv(rgb_norm.reshape(1, 1, 3)).reshape(3)
    hsv[1] = min(1.0, hsv[1] + sat_boost)
    hsv[2] = min(1.0, hsv[2] + val_boost)
    vivid_rgb = hsv_to_rgb(hsv.reshape(1, 1, 3)).reshape(3)
    return (vivid_rgb * 255).astype(np.uint8)


def create_vivid_overlay(original: np.ndarray, labels: np.ndarray, alpha: float = 0.5) -> tuple:
    """Create a vivid color overlay on top of the original image.

    Args:
        original: RGB image array (H, W, 3).
        labels: Label array (H, W), labels 1..n_layers.
        alpha: Overlay blending factor (default 0.5).

    Returns:
        (overlay_image, vivid_colors_list)
        overlay_image: RGB array (H, W, 3) with vivid colors + white boundaries.
        vivid_colors_list: List of vivid RGB arrays, one per layer.
    """
    h, w = labels.shape
    overlay = original.copy()
    n_layers = int(labels.max())

    vivid_colors = []
    for lbl in range(1, n_layers + 1):
        mask = labels == lbl
        if mask.any():
            mean_color = original[mask].mean(axis=0)
            vivid = vivid_color(mean_color, sat_boost=0.35, val_boost=0.1)
            vivid_colors.append(vivid)
        else:
            vivid_colors.append(np.array([200, 200, 200], dtype=np.uint8))

    colored = np.zeros_like(overlay)
    for lbl in range(1, n_layers + 1):
        mask = labels == lbl
        if mask.any():
            colored[mask] = vivid_colors[lbl - 1]

    blended = (overlay.astype(float) * (1 - alpha) + colored.astype(float) * alpha).astype(np.uint8)

    boundaries = np.zeros((h, w), dtype=bool)
    for lbl in range(1, n_layers + 1):
        mask = labels == lbl
        if mask.any():
            eroded = ndimage.binary_erosion(mask)
            boundaries |= (mask & ~eroded)
    boundaries = ndimage.binary_dilation(boundaries, iterations=1)
    blended[boundaries] = [255, 255, 255]

    return blended, vivid_colors
