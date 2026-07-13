"""Process extracted front face panels: smoothing, layer segmentation, fragment detection.

Pipeline per panel:
1. Text smoothing (morphological close + gaussian blur)
2. Layer segmentation (k-means + region merging)
3. Fragment detection (LoG + connected components)
4. Compose 3x4 comparison figure
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def smooth_text(image: np.ndarray, kernel_size: int = 5, blur_sigma: float = 1.0) -> np.ndarray:
    """Morphological close to fill text, then slight gaussian blur."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    blurred = cv2.GaussianBlur(closed, (0, 0), blur_sigma)
    return cv2.addWeighted(closed, 0.7, blurred, 0.3, 0)


def segment_layers(image: np.ndarray, n_clusters: int = 6) -> np.ndarray:
    """K-means coarse segmentation followed by region merging."""
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(np.float32)

    label_img = labels.reshape(h, w).astype(np.int32)

    # Region merging: merge adjacent regions with similar colors
    merged = merge_similar_regions(label_img, centers, threshold=30.0)

    # Build colored segmentation map
    seg = np.zeros_like(image)
    unique_labels = np.unique(merged)
    for lbl in unique_labels:
        mask = merged == lbl
        seg[mask] = centers[lbl].astype(np.uint8)

    return seg


def merge_similar_regions(label_img: np.ndarray, centers: np.ndarray, threshold: float = 30.0) -> np.ndarray:
    """Merge adjacent regions with color distance below threshold."""
    h, w = label_img.shape
    merged = label_img.copy()
    n_centers = len(centers)

    # Build adjacency graph
    adj = np.zeros((n_centers, n_centers), dtype=bool)
    for y in range(h - 1):
        for x in range(w - 1):
            a = merged[y, x]
            b_right = merged[y, x + 1]
            b_down = merged[y + 1, x]
            if a != b_right:
                adj[a, b_right] = True
                adj[b_right, a] = True
            if a != b_down:
                adj[a, b_down] = True
                adj[b_down, a] = True

    # Union-find for merging
    parent = list(range(n_centers))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Merge adjacent similar colors
    for i in range(n_centers):
        for j in range(i + 1, n_centers):
            if adj[i, j]:
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < threshold:
                    union(i, j)

    # Remap labels
    remap = {}
    next_label = 0
    for i in range(n_centers):
        root = find(i)
        if root not in remap:
            remap[root] = next_label
            next_label += 1
        merged[label_img == i] = remap[root]

    return merged


def detect_fragments(image: np.ndarray, min_area: int = 30, max_area: int = 2000) -> np.ndarray:
    """Detect scattered fragments using LoG + connected components."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # LoG to highlight blob-like structures
    log = cv2.Laplacian(cv2.GaussianBlur(gray, (5, 5), 1.5), cv2.CV_64F)
    log_norm = cv2.normalize(np.abs(log), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold LoG response
    _, log_mask = cv2.threshold(log_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Also use simple color-based segmentation for fragments
    # Fragments are typically darker/blue blobs against orange/yellow background
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Blue-ish fragments in HSV
    lower_blue = np.array([90, 20, 20])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combine masks
    combined = cv2.bitwise_or(log_mask, blue_mask)

    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)

    frag_mask = np.zeros_like(gray)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            frag_mask[labels == i] = 255

    return frag_mask


def process_panel(image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process a single panel through all stages."""
    img = np.array(Image.open(image_path).convert("RGB"))

    smoothed = smooth_text(img)
    segmented = segment_layers(smoothed)
    fragments = detect_fragments(img)

    return img, smoothed, segmented, fragments


def create_comparison_figure(results: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Create 3x4 comparison figure."""
    n_panels = len(results)
    rows = n_panels
    cols = 4

    # Determine cell size from first image
    h, w = results[0][0].shape[:2]
    cell_h, cell_w = h, w

    # Labels
    row_labels = ["Panel 1", "Panel 2", "Panel 3"]
    col_labels = ["Original", "Smoothed", "Segmented", "Fragments"]

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_h = 30
    header_h = 30

    canvas = np.ones((rows * (cell_h + label_h) + header_h, cols * cell_w, 3), dtype=np.uint8) * 255

    # Column headers
    for c, label in enumerate(col_labels):
        x = c * cell_w + cell_w // 2 - len(label) * 6
        cv2.putText(canvas, label, (x, 25), font, 0.7, (0, 0, 0), 2)

    for r, (orig, smooth, seg, frag) in enumerate(results):
        y_base = header_h + r * (cell_h + label_h)

        # Row label
        cv2.putText(canvas, row_labels[r], (10, y_base + 20), font, 0.6, (0, 0, 0), 2)

        # Original
        canvas[y_base + label_h:y_base + label_h + cell_h, 0:cell_w] = orig
        # Smoothed
        canvas[y_base + label_h:y_base + label_h + cell_h, cell_w:2 * cell_w] = smooth
        # Segmented
        canvas[y_base + label_h:y_base + label_h + cell_h, 2 * cell_w:3 * cell_w] = seg
        # Fragments (grayscale to RGB)
        frag_rgb = cv2.cvtColor(frag, cv2.COLOR_GRAY2RGB)
        canvas[y_base + label_h:y_base + label_h + cell_h, 3 * cell_w:4 * cell_w] = frag_rgb

    return canvas


def main():
    base = Path(__file__).parent.parent.parent
    panel_paths = [base / f"panel_{i}_front.png" for i in range(1, 4)]

    results = []
    for p in panel_paths:
        print(f"Processing {p.name}...")
        results.append(process_panel(p))

    comparison = create_comparison_figure(results)
    out_path = base / "result_C.png"
    Image.fromarray(comparison).save(out_path)
    print(f"Saved comparison to {out_path} ({comparison.shape[1]}x{comparison.shape[0]})")


if __name__ == "__main__":
    main()
