"""Edge-aware segmentation for 3D schematic panels.

Key insight: real boundaries have high color gradient even when colors are similar.
Gradients have low gradient. Merge decisions combine color similarity + edge strength.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.cluster import KMeans


def smooth_text(image: np.ndarray, kernel_size: int = 5, blur_sigma: float = 1.0) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    blurred = cv2.GaussianBlur(closed, (0, 0), blur_sigma)
    return cv2.addWeighted(closed, 0.7, blurred, 0.3, 0)


def compute_color_gradient(image: np.ndarray) -> np.ndarray:
    """Compute per-pixel color gradient magnitude (L2 norm of RGB gradients)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(sobel_x**2 + sobel_y**2)


def edge_aware_merge(
    label_img: np.ndarray,
    image: np.ndarray,
    color_threshold: float = 40.0,
    edge_threshold: float = 8.0,
) -> np.ndarray:
    """Merge labels that are color-similar AND have weak boundary edges.

    A boundary is kept if:
    - Colors are very different (dist >= color_threshold), OR
    - Boundary has strong edges (mean_grad >= edge_threshold)

    A boundary is merged if:
    - Colors are similar (dist < color_threshold), AND
    - Boundary has weak edges (mean_grad < edge_threshold)
    """
    h, w = label_img.shape
    gradient = compute_color_gradient(image)

    # Compute mean color per label
    unique = sorted(np.unique(label_img))
    n_labels = len(unique)
    label_to_idx = {lbl: i for i, lbl in enumerate(unique)}

    mean_colors = np.zeros((n_labels, 3), dtype=np.float32)
    for lbl in unique:
        idx = label_to_idx[lbl]
        mask = label_img == lbl
        mean_colors[idx] = image[mask].mean(axis=0)

    # Find boundary pixels between each label pair
    boundary_grads = {}
    boundary_pixels = {}

    for y in range(h):
        for x in range(w - 1):
            a = label_img[y, x]
            b = label_img[y, x + 1]
            if a != b:
                pair = (min(a, b), max(a, b))
                if pair not in boundary_pixels:
                    boundary_pixels[pair] = []
                boundary_pixels[pair].append((y, x))

    for y in range(h - 1):
        for x in range(w):
            a = label_img[y, x]
            b = label_img[y + 1, x]
            if a != b:
                pair = (min(a, b), max(a, b))
                if pair not in boundary_pixels:
                    boundary_pixels[pair] = []
                boundary_pixels[pair].append((y, x))

    for pair, pixels in boundary_pixels.items():
        grads = [gradient[y, x] for y, x in pixels]
        boundary_grads[pair] = np.mean(grads)

    # Union-find for merging
    parent = {lbl: lbl for lbl in unique}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i, li in enumerate(unique):
        for j, lj in enumerate(unique):
            if i >= j:
                continue
            pair = (min(li, lj), max(li, lj))
            if pair not in boundary_grads:
                continue

            color_dist = np.linalg.norm(mean_colors[i] - mean_colors[j])
            mean_grad = boundary_grads[pair]

            # Merge if color-similar AND weak boundary
            if color_dist < color_threshold and mean_grad < edge_threshold:
                union(li, lj)

    # Remap
    remap = {}
    next_lbl = 0
    result = np.zeros_like(label_img)
    for lbl in unique:
        root = find(lbl)
        if root not in remap:
            remap[root] = next_lbl
            next_lbl += 1
        result[label_img == lbl] = remap[root]

    return result


def segment_two_pass(image: np.ndarray, n_clusters: int = 6) -> np.ndarray:
    """Two-pass: k-means -> blur -> k-means, then edge-aware merge."""
    h, w = image.shape[:2]

    # Pass 1
    pixels = image.reshape(-1, 3).astype(np.float32)
    k1 = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels1 = k1.fit_predict(pixels).reshape(h, w)
    centers1 = k1.cluster_centers_.astype(np.float32)

    # Smooth labels spatially
    labels_smooth = cv2.GaussianBlur(labels1.astype(np.float32), (21, 21), 5)
    labels_smooth = np.round(labels_smooth).astype(np.int32)

    # Pass 2: k-means on blurred image
    blurred_img = cv2.GaussianBlur(image, (21, 21), 5)
    pixels2 = blurred_img.reshape(-1, 3).astype(np.float32)
    k2 = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels2 = k2.fit_predict(pixels2).reshape(h, w)

    # Edge-aware merge
    merged = edge_aware_merge(labels2, image, color_threshold=40.0, edge_threshold=8.0)

    return merged


def draw_boundaries(image: np.ndarray, labels: np.ndarray, color: tuple = (0, 0, 0), thickness: int = 2) -> np.ndarray:
    """Draw black boundaries between different labels."""
    result = image.copy()
    h, w = labels.shape

    # Horizontal boundaries
    for y in range(h - 1):
        for x in range(w):
            if labels[y, x] != labels[y + 1, x]:
                cv2.line(result, (x, y), (x, y + 1), color, thickness)

    # Vertical boundaries
    for y in range(h):
        for x in range(w - 1):
            if labels[y, x] != labels[y, x + 1]:
                cv2.line(result, (x, y), (x + 1, y), color, thickness)

    return result


def render_label_fill(labels: np.ndarray, n_labels: int | None = None) -> np.ndarray:
    """Render labels with uniform colors."""
    unique = sorted(np.unique(labels))
    if n_labels is None:
        n_labels = len(unique)

    import colorsys
    colors = []
    for i in range(n_labels):
        hue = (i * 0.618033988749895) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.92)
        colors.append([int(c * 255) for c in rgb])

    h, w = labels.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        color = np.array(colors[i % len(colors)], dtype=np.uint8)
        result[mask] = color

    return result


def detect_fragments(image: np.ndarray, labels: np.ndarray, min_area: int = 20, max_area: int = 1500) -> np.ndarray:
    """Detect small fragments not captured by main labels."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # LoG
    log = cv2.Laplacian(cv2.GaussianBlur(gray, (5, 5), 1.5), cv2.CV_64F)
    log_norm = cv2.normalize(np.abs(log), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, log_mask = cv2.threshold(log_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Blue fragments
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_blue = np.array([90, 20, 20])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Exclude areas already covered by main labels (except background)
    label_coverage = np.zeros_like(gray)
    unique = np.unique(labels)
    for lbl in unique:
        if lbl == 0:
            continue
        mask = labels == lbl
        label_coverage[mask] = 255

    combined = cv2.bitwise_or(log_mask, blue_mask)
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(label_coverage))

    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)

    frag_mask = np.zeros_like(gray)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            frag_mask[cc_labels == i] = 255

    return frag_mask


def overlay_fragments(image: np.ndarray, frag_mask: np.ndarray) -> np.ndarray:
    """Overlay fragments in blue."""
    result = image.copy()
    result[frag_mask > 0] = [0, 100, 255]
    return result


def process_panel(image_path: Path, n_clusters: int = 6) -> dict:
    img = np.array(Image.open(image_path).convert("RGB"))
    cleaned = smooth_text(img)
    labels = segment_two_pass(cleaned, n_clusters=n_clusters)
    frag_mask = detect_fragments(img, labels)

    fill = render_label_fill(labels)
    boundaries = draw_boundaries(fill, labels)
    frag_overlay = overlay_fragments(boundaries, frag_mask)

    return {
        "original": img,
        "cleaned": cleaned,
        "labels": labels,
        "fill": fill,
        "boundaries": boundaries,
        "fragments": frag_overlay,
    }


def create_figure(results: list[dict]) -> np.ndarray:
    n = len(results)
    h, w = results[0]["original"].shape[:2]
    cols = ["Original", "Cleaned", "Label Fill", "Boundaries+Frags"]

    header_h = 35
    label_w = 100
    cell_h, cell_w = h, w

    canvas = np.ones((n * cell_h + header_h, label_w + len(cols) * cell_w, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    for c, title in enumerate(cols):
        x = label_w + c * cell_w + cell_w // 2 - len(title) * 5
        cv2.putText(canvas, title, (x, 25), font, 0.6, (0, 0, 0), 2)

    row_titles = ["Panel 1", "Panel 2", "Panel 3"]
    keys = ["original", "cleaned", "fill", "fragments"]

    for r, res in enumerate(results):
        y = header_h + r * cell_h
        cv2.putText(canvas, row_titles[r], (10, y + 30), font, 0.6, (0, 0, 0), 2)

        for c, key in enumerate(keys):
            x = label_w + c * cell_w
            img = res[key]
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            canvas[y:y + cell_h, x:x + cell_w] = img

    return canvas


def main():
    base = Path(__file__).parent.parent.parent
    panel_paths = [base / f"panel_{i}_front.png" for i in range(1, 4)]

    results = []
    for p in panel_paths:
        print(f"Processing {p.name}...")
        res = process_panel(p, n_clusters=6)
        n_final = len(np.unique(res["labels"]))
        print(f"  -> {n_final} layers")
        results.append(res)

    fig = create_figure(results)
    out = base / "result_edge_aware.png"
    Image.fromarray(fig).save(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
