"""
Color-clustering based text removal for geological schematic images.
Strategy: KMeans clustering to separate text from background, then inpaint.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cluster_colors(image: np.ndarray, k: int = 5) -> tuple:
    """
    Run KMeans clustering on all pixels.
    Returns (labels, centers, pixel_counts).
    """
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Count pixels per cluster
    counts = np.bincount(labels, minlength=k)

    return labels, centers, counts, h, w


def identify_text_cluster(centers: np.ndarray, counts: np.ndarray,
                          total_pixels: int) -> int:
    """
    Identify which cluster corresponds to text.

    Heuristics:
    1. Text is typically bright (high L2 norm of RGB)
    2. Text is often desaturated (low std of RGB channels)
    3. Text has small total area (not dominant)
    4. For white text on light bg, may need to merge top 2 bright clusters
    """
    k = len(centers)

    # Compute brightness (L2 norm)
    brightness = np.linalg.norm(centers, axis=1)

    # Compute saturation (std of RGB channels)
    saturation = np.std(centers, axis=1)

    # Relative area
    areas = counts / total_pixels

    print(f"  Cluster analysis:")
    for i in range(k):
        print(f"    Cluster {i}: center={centers[i]}, brightness={brightness[i]:.1f}, "
              f"saturation={saturation[i]:.1f}, area={areas[i]*100:.2f}%")

    # Strategy: look for bright clusters with small area
    # Score: higher brightness + lower saturation + smaller area = more text-like
    # Normalize metrics
    brightness_norm = brightness / np.max(brightness)
    saturation_norm = saturation / (np.max(saturation) + 1e-6)
    area_norm = 1.0 - (areas / (np.max(areas) + 1e-6))  # smaller is better

    # Composite score: bright, desaturated, small area
    scores = (0.5 * brightness_norm +
              0.3 * (1.0 - saturation_norm) +
              0.2 * area_norm)

    print(f"  Text-likeness scores: {scores}")

    # Check if top 2 brightest clusters are very similar (for white text on light bg)
    sorted_by_brightness = np.argsort(brightness)[::-1]
    top1, top2 = sorted_by_brightness[0], sorted_by_brightness[1]

    color_distance = np.linalg.norm(centers[top1] - centers[top2])
    brightness_ratio = brightness[top2] / (brightness[top1] + 1e-6)

    print(f"  Top 2 bright clusters: {top1} (bright={brightness[top1]:.1f}) and "
          f"{top2} (bright={brightness[top2]:.1f})")
    print(f"  Color distance between top 2: {color_distance:.1f}, "
          f"brightness ratio: {brightness_ratio:.3f}")

    # If top 2 are very similar and both bright, merge them
    if color_distance < 40 and brightness_ratio > 0.85:
        print(f"  -> Merging clusters {top1} and {top2} (similar bright clusters)")
        return (top1, top2)  # Return tuple to indicate merge

    text_cluster = int(np.argmax(scores))
    print(f"  -> Selected text cluster: {text_cluster}")
    return text_cluster


def create_text_mask(labels: np.ndarray, text_clusters, h: int, w: int) -> np.ndarray:
    """Create binary mask for text cluster pixels."""
    mask = np.zeros_like(labels, dtype=np.uint8)

    if isinstance(text_clusters, tuple):
        for tc in text_clusters:
            mask[labels == tc] = 255
    else:
        mask[labels == text_clusters] = 255

    return mask.reshape(h, w)


def inpaint_text(image: np.ndarray, mask: np.ndarray,
                 dilate_iter: int = 1, radius: int = 5) -> np.ndarray:
    """Dilate mask and inpaint text regions."""
    # Dilate mask slightly to cover text edges
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=dilate_iter)

    # Inpaint
    result = cv2.inpaint(image, mask_dilated, radius, cv2.INPAINT_TELEA)
    return result


def process_panel(image_path: str, k: int = 5) -> dict:
    """Process a single panel through the full pipeline."""
    print(f"\n{'='*60}")
    print(f"Processing: {Path(image_path).name}")
    print(f"{'='*60}")

    image = load_image(image_path)
    h, w = image.shape[:2]
    total_pixels = h * w

    print(f"Image size: {w}x{h}, total pixels: {total_pixels}")

    # Step 2: KMeans clustering
    labels, centers, counts, h, w = cluster_colors(image, k=k)

    # Step 3: Identify text cluster
    text_clusters = identify_text_cluster(centers, counts, total_pixels)

    # Step 4: Create mask
    mask = create_text_mask(labels, text_clusters, h, w)

    # Step 5: Inpaint
    result = inpaint_text(image, mask)

    return {
        'original': image,
        'mask': mask,
        'result': result,
        'text_clusters': text_clusters,
        'centers': centers,
        'counts': counts
    }


def create_comparison_figure(results: list, output_path: str):
    """Create 3x2 comparison figure."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    fig.suptitle('Color-Clustering Text Removal Results', fontsize=16)

    panel_names = ['Panel 1', 'Panel 2', 'Panel 3']

    for i, (result, name) in enumerate(zip(results, panel_names)):
        # Original
        axes[i, 0].imshow(result['original'])
        axes[i, 0].set_title(f'{name} - Original')
        axes[i, 0].axis('off')

        # Result
        axes[i, 1].imshow(result['result'])
        tc = result['text_clusters']
        tc_str = f"clusters {tc}" if isinstance(tc, tuple) else f"cluster {tc}"
        axes[i, 1].set_title(f'{name} - Text Removed ({tc_str})')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison figure saved to: {output_path}")


def main():
    input_dir = Path('/Users/daiduo2/geoseg/src/3d_schematic')
    output_dir = input_dir / 'agent_results'
    output_dir.mkdir(exist_ok=True)

    panels = [
        input_dir / 'panel_1_front.png',
        input_dir / 'panel_2_front.png',
        input_dir / 'panel_3_front.png',
    ]

    results = []
    for panel_path in panels:
        result = process_panel(str(panel_path), k=5)
        results.append(result)

    # Save individual results
    for i, result in enumerate(results):
        out_path = output_dir / f'panel_{i+1}_no_text.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(result['result'], cv2.COLOR_RGB2BGR))
        print(f"Saved: {out_path}")

    # Create comparison figure
    comparison_path = output_dir / 'cluster_comparison.png'
    create_comparison_figure(results, str(comparison_path))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for i, result in enumerate(results):
        tc = result['text_clusters']
        tc_str = f"clusters {tc}" if isinstance(tc, tuple) else f"cluster {tc}"
        print(f"Panel {i+1}: Text identified as {tc_str}")


if __name__ == '__main__':
    main()
