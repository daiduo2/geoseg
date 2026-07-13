"""Component extraction from segmented labels (cleaned from compose_filtered_components.py).

Removes grid visualization and hardcoded layer names. Pure functions only.
"""

import numpy as np
from scipy import ndimage


def extract_components(
    labels: np.ndarray,
    original: np.ndarray,
    min_area: int = 200,
) -> list:
    """Extract connected components per layer, filtering by minimum area.

    Args:
        labels: Label array (H, W), labels 1..n_layers.
        original: RGB image array (H, W, 3), used for computing mean colors.
        min_area: Minimum pixel area to keep a component (default 200).

    Returns:
        List of component dicts, each with:
            {
                "id": int,
                "layer_id": int,
                "bbox": [x, y, w, h],
                "area": int,
                "centroid": [cx, cy],
                "mean_color": [r, g, b],
            }
    """
    h, w = labels.shape
    n_layers = int(labels.max())
    components = []
    comp_id = 0

    for lbl in range(1, n_layers + 1):
        mask = labels == lbl
        if not mask.any():
            continue

        labeled, num = ndimage.label(mask)
        for cid in range(1, num + 1):
            comp_mask = (labeled == cid)
            area = int(comp_mask.sum())
            if area < min_area:
                continue

            ys, xs = np.where(comp_mask)
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            cx = float(xs.mean())
            cy = float(ys.mean())
            mean_color = original[comp_mask].mean(axis=0).astype(int).tolist()

            components.append({
                "id": comp_id,
                "layer_id": lbl,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": area,
                "centroid": [cx, cy],
                "mean_color": mean_color,
            })
            comp_id += 1

    return components


def build_segmentation_result(
    labels: np.ndarray,
    original: np.ndarray,
    min_area: int = 200,
) -> dict:
    """Build the full segmentation_result dict from labels and original image.

    Args:
        labels: Label array (H, W).
        original: RGB image array (H, W, 3).
        min_area: Minimum component area (default 200).

    Returns:
        {
            "components": [...],
            "layers": [{"id": int, "color": [r,g,b]}],
        }
    """
    n_layers = int(labels.max())
    components = extract_components(labels, original, min_area=min_area)

    # Compute layer mean colors from original image
    layers = []
    for lbl in range(1, n_layers + 1):
        mask = labels == lbl
        if mask.any():
            color = original[mask].mean(axis=0).astype(int).tolist()
        else:
            color = [200, 200, 200]
        layers.append({"id": lbl, "color": color})

    return {
        "components": components,
        "layers": layers,
    }
