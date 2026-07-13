"""Automatic crop generation for visual audit.

Forces multi-scale inspection by extracting small, high-resolution regions
around known failure points (text remnants, plume, layer boundaries).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops


CropSpec = dict[str, object]


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _crop(image: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    h, w = image.shape[:2]
    y0 = _clamp(y0, 0, h)
    y1 = _clamp(y1, 0, h)
    x0 = _clamp(x0, 0, w)
    x1 = _clamp(x1, 0, w)
    return image[y0:y1, x0:x1]


def crop_top_surface(panel_rgb: np.ndarray) -> np.ndarray:
    """Top 20% of the panel: surface, weak zone, crust."""
    h = panel_rgb.shape[0]
    return _crop(panel_rgb, 0, h // 5, 0, panel_rgb.shape[1])


def crop_bottom_boundary(panel_rgb: np.ndarray) -> np.ndarray:
    """Bottom 15%: deepest layer boundaries."""
    h = panel_rgb.shape[0]
    return _crop(panel_rgb, int(h * 0.85), h, 0, panel_rgb.shape[1])


def _find_green_region_center(panel_rgb: np.ndarray) -> tuple[int, int] | None:
    """Find centroid of the largest greenish region in the upper half."""
    h, w = panel_rgb.shape[:2]
    upper = panel_rgb[: h // 2, :, :]

    # Simple green dominance mask
    green_dominant = (
        (upper[:, :, 1].astype(int) > upper[:, :, 0].astype(int))
        & (upper[:, :, 1].astype(int) > upper[:, :, 2].astype(int))
        & (upper[:, :, 1] > 80)
    )

    if not green_dominant.any():
        return None

    cc = label(green_dominant, connectivity=2)
    regions = regionprops(cc)
    if not regions:
        return None

    largest = max(regions, key=lambda r: r.area)
    cy, cx = largest.centroid
    return int(cy), int(cx)


def crop_green_fragments(panel_rgb: np.ndarray, size: int = 400) -> np.ndarray:
    """Crop around the largest green fragment cluster."""
    center = _find_green_region_center(panel_rgb)
    if center is None:
        # Fallback: upper-middle
        h, w = panel_rgb.shape[:2]
        center = (h // 4, w // 2)

    cy, cx = center
    half = size // 2
    return _crop(panel_rgb, cy - half, cy + half, cx - half, cx + half)


def _find_central_plume_center(panel_rgb: np.ndarray) -> tuple[int, int] | None:
    """Find centroid of the largest light-colored region in the central column."""
    h, w = panel_rgb.shape[:2]
    x0, x1 = int(w * 0.35), int(w * 0.65)
    central = panel_rgb[:, x0:x1, :]

    # Light region: high value and warm-ish
    lightness = central.mean(axis=2)
    threshold = np.percentile(lightness, 70)
    light_mask = lightness > threshold

    if not light_mask.any():
        return None

    cc = label(light_mask, connectivity=2)
    regions = regionprops(cc)
    if not regions:
        return None

    largest = max(regions, key=lambda r: r.area)
    cy, cx = largest.centroid
    return int(cy), int(cx + x0)


def crop_plume_body(panel_rgb: np.ndarray, size: int = 500) -> np.ndarray:
    """Crop around the central plume / uplift structure."""
    center = _find_central_plume_center(panel_rgb)
    if center is None:
        h, w = panel_rgb.shape[:2]
        center = (h // 2, w // 2)

    cy, cx = center
    half_h = size // 2
    half_w = size * 3 // 8
    return _crop(panel_rgb, cy - half_h, cy + half_h, cx - half_w, cx + half_w)


def _estimate_text_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Lightweight fallback text mask for crop placement."""
    import cv2
    from scipy import ndimage

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=25, C=-5,
    )
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    laplacian = (laplacian > np.percentile(laplacian, 80)).astype(np.uint8) * 255
    text_mask = ((adaptive > 0) & (laplacian > 0))
    return ndimage.binary_dilation(text_mask, iterations=2)


def _find_text_crop_centers(panel_rgb: np.ndarray, n: int = 2) -> list[tuple[int, int]]:
    """Find centroids of the largest text-like connected components."""
    text_mask = _estimate_text_mask(panel_rgb)
    if not text_mask.any():
        return []

    cc = label(text_mask, connectivity=2)
    regions = regionprops(cc)
    regions.sort(key=lambda r: r.area, reverse=True)

    centers = []
    for region in regions[:n]:
        cy, cx = region.centroid
        centers.append((int(cy), int(cx)))
    return centers


def crop_text_regions(panel_rgb: np.ndarray, size: int = 300) -> list[np.ndarray]:
    """Generate crops around the largest text-like regions."""
    centers = _find_text_crop_centers(panel_rgb)
    crops = []
    for cy, cx in centers:
        half = size // 2
        crops.append(_crop(panel_rgb, cy - half, cy + half, cx - half, cx + half))
    return crops


def create_audit_crops(
    panel_rgb: np.ndarray,
    include_text: bool = True,
) -> dict[str, np.ndarray | list[np.ndarray]]:
    """Generate a standard set of audit crops.

    Returns:
        dict with keys:
        - top_surface: upper region
        - bottom_boundary: lower region
        - green_fragments: green textured zone (Panel 3)
        - plume_body: central uplift/plume (Panel 3)
        - text_regions: list of crops around text-like regions
    """
    crops: dict[str, np.ndarray | list[np.ndarray]] = {
        "top_surface": crop_top_surface(panel_rgb),
        "bottom_boundary": crop_bottom_boundary(panel_rgb),
        "green_fragments": crop_green_fragments(panel_rgb),
        "plume_body": crop_plume_body(panel_rgb),
    }

    if include_text:
        crops["text_regions"] = crop_text_regions(panel_rgb)

    return crops


def save_crops(
    crops: dict[str, np.ndarray | list[np.ndarray]],
    output_dir: str,
) -> dict[str, str | list[str]]:
    """Save crops to disk and return path mapping."""
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str | list[str]] = {}

    for name, value in crops.items():
        if isinstance(value, list):
            path_list = []
            for i, arr in enumerate(value):
                path = out / f"{name}_{i}.jpg"
                Image.fromarray(arr).save(path, quality=90)
                path_list.append(str(path))
            paths[name] = path_list
        else:
            path = out / f"{name}.jpg"
            Image.fromarray(value).save(path, quality=90)
            paths[name] = str(path)

    return paths
