import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_swt_bidirectional(image_gray, canny_low=30, canny_high=100, max_stroke_width=60):
    """
    Simplified Stroke Width Transform with bidirectional ray tracing.
    """
    edges = cv2.Canny(image_gray, canny_low, canny_high)

    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    theta = np.arctan2(sobely, sobelx)

    h, w = edges.shape
    swt = np.full((h, w), np.inf, dtype=np.float32)

    edge_pixels = np.argwhere(edges > 0)

    for y, x in edge_pixels:
        angle = theta[y, x]
        for sign in [1, -1]:
            dx = np.cos(angle) * sign
            dy = np.sin(angle) * sign

            for step in range(1, max_stroke_width + 1):
                nx = int(round(x + dx * step))
                ny = int(round(y + dy * step))

                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    break

                if edges[ny, nx] > 0:
                    dist = np.hypot(nx - x, ny - y)
                    swt[y, x] = min(swt[y, x], dist)
                    break

    swt[np.isinf(swt)] = max_stroke_width
    return swt, edges


def compute_consistency_mask(swt, window_size=5, mad_threshold=4.0, max_sw=30):
    """
    Text pixels have consistent stroke widths in a neighborhood.
    """
    h, w = swt.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    half = window_size // 2
    padded = np.pad(swt, half, mode='edge')

    for y in range(h):
        for x in range(w):
            window = padded[y:y + window_size, x:x + window_size]
            median = np.median(window)

            if median <= 0 or median > max_sw:
                continue

            mad = np.median(np.abs(window - median))
            if mad < mad_threshold:
                mask[y, x] = 255

    return mask


def compute_laplacian_backup(image_gray, threshold=12, brightness_mask=None):
    """
    High-frequency Laplacian regions with geometric filtering.
    Optionally constrained to bright regions (text is light-colored).
    """
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F, ksize=3)
    laplacian = np.abs(laplacian)

    mx = laplacian.max()
    if mx > 0:
        laplacian = (laplacian / mx * 255).astype(np.uint8)
    else:
        laplacian = np.zeros_like(image_gray)

    _, high_freq = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)

    # Apply brightness constraint if provided
    if brightness_mask is not None:
        high_freq = cv2.bitwise_and(high_freq, brightness_mask)

    # Morphological opening to remove tiny noise
    kernel_small = np.ones((2, 2), np.uint8)
    high_freq = cv2.morphologyEx(high_freq, cv2.MORPH_OPEN, kernel_small)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(high_freq, connectivity=8)

    backup_mask = np.zeros_like(high_freq)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 5 or area > 800:
            continue

        if w == 0 or h == 0:
            continue

        aspect = max(w, h) / max(min(w, h), 1)
        compactness = (w * h) / max(area, 1)

        if aspect < 15 and compactness < 12:
            backup_mask[labels == i] = 255

    return backup_mask


def compute_mser_text_candidates(image_gray, brightness_mask=None):
    """
    MSER detects stable extremal regions.
    Restrict to smaller regions; optionally brightness-filtered.
    """
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(image_gray)

    mask = np.zeros_like(image_gray)
    for region in regions:
        if 10 < len(region) < 800:
            for pt in region:
                mask[pt[1], pt[0]] = 255

    if brightness_mask is not None:
        mask = cv2.bitwise_and(mask, brightness_mask)

    # Dilate slightly to connect character parts
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def filter_geological_lines(mask, max_aspect=10, max_compactness=15):
    """
    Remove long thin lines (geological boundaries) while keeping text blobs.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 5:
            continue

        if w == 0 or h == 0:
            continue

        aspect = max(w, h) / max(min(w, h), 1)
        compactness = (w * h) / max(area, 1)

        if aspect < max_aspect and compactness < max_compactness:
            filtered[labels == i] = 255

    return filtered


def remove_text(image_path, canny_low=20, canny_high=60, mad_threshold=4.0,
                laplacian_threshold=12, dilate_iter=2, inpaint_radius=5):
    """
    Full pipeline: SWT + Laplacian + MSER -> filter -> mask -> inpaint.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Brightness mask: text is light-colored in these schematics
    _, brightness_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    # Also include moderately bright regions
    _, brightness_mask2 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    brightness_mask = cv2.bitwise_or(brightness_mask, brightness_mask2)
    # Dilate to catch text edges
    kernel = np.ones((5, 5), np.uint8)
    brightness_mask = cv2.dilate(brightness_mask, kernel, iterations=1)

    # SWT
    swt, edges = compute_swt_bidirectional(gray, canny_low, canny_high)
    swt_mask = compute_consistency_mask(swt, mad_threshold=mad_threshold)

    # Laplacian backup (brightness-constrained)
    lap_mask = compute_laplacian_backup(gray, threshold=laplacian_threshold,
                                        brightness_mask=brightness_mask)

    # MSER candidates (brightness-constrained)
    mser_mask = compute_mser_text_candidates(gray, brightness_mask=brightness_mask)

    # Combine all three
    combined_mask = cv2.bitwise_or(swt_mask, lap_mask)
    combined_mask = cv2.bitwise_or(combined_mask, mser_mask)

    # Filter out geological lines
    filtered_mask = filter_geological_lines(combined_mask, max_aspect=10, max_compactness=15)

    # Dilate
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(filtered_mask, kernel, iterations=dilate_iter)

    # Inpaint
    inpainted = cv2.inpaint(img_rgb, dilated_mask, inpaintRadius=inpaint_radius,
                            flags=cv2.INPAINT_TELEA)

    return {
        'original': img_rgb,
        'swt': swt,
        'edges': edges,
        'swt_mask': swt_mask,
        'lap_mask': lap_mask,
        'mser_mask': mser_mask,
        'brightness_mask': brightness_mask,
        'combined_mask': combined_mask,
        'filtered_mask': filtered_mask,
        'dilated_mask': dilated_mask,
        'inpainted': inpainted
    }


def main():
    base = Path('/Users/daiduo2/geoseg/src/3d_schematic')
    out_dir = base / 'agent_results'
    out_dir.mkdir(exist_ok=True)

    panels = [
        base / 'panel_1_front.png',
        base / 'panel_2_front.png',
        base / 'panel_3_front.png',
    ]

    results = []
    for p in panels:
        print(f"Processing {p.name}...")
        res = remove_text(p, canny_low=20, canny_high=60,
                          mad_threshold=4.0, laplacian_threshold=12,
                          dilate_iter=2, inpaint_radius=5)
        results.append((p.name, res))

    # Build 3x2 comparison figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))

    for i, (name, res) in enumerate(results):
        axes[i, 0].imshow(res['original'])
        axes[i, 0].set_title(f"{name} — Original")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(res['inpainted'])
        axes[i, 1].set_title(f"{name} — Inpainted")
        axes[i, 1].axis('off')

    plt.tight_layout()
    out_path = out_dir / 'stroke_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {out_path}")

    # Per-panel diagnostic figures
    for name, res in results:
        fig2, ax2 = plt.subplots(2, 4, figsize=(18, 10))
        ax2[0, 0].imshow(res['original'])
        ax2[0, 0].set_title('Original')
        ax2[0, 0].axis('off')

        ax2[0, 1].imshow(res['edges'], cmap='gray')
        ax2[0, 1].set_title('Canny Edges')
        ax2[0, 1].axis('off')

        swt_vis = np.clip(res['swt'], 0, 50)
        ax2[0, 2].imshow(swt_vis, cmap='viridis')
        ax2[0, 2].set_title('Stroke Width (clipped 0-50)')
        ax2[0, 2].axis('off')

        ax2[0, 3].imshow(res['swt_mask'], cmap='gray')
        ax2[0, 3].set_title('SWT Consistency Mask')
        ax2[0, 3].axis('off')

        ax2[1, 0].imshow(res['lap_mask'], cmap='gray')
        ax2[1, 0].set_title('Laplacian Backup Mask')
        ax2[1, 0].axis('off')

        ax2[1, 1].imshow(res['mser_mask'], cmap='gray')
        ax2[1, 1].set_title('MSER Mask')
        ax2[1, 1].axis('off')

        ax2[1, 2].imshow(res['filtered_mask'], cmap='gray')
        ax2[1, 2].set_title('Filtered Mask')
        ax2[1, 2].axis('off')

        ax2[1, 3].imshow(res['dilated_mask'], cmap='gray')
        ax2[1, 3].set_title('Dilated Mask')
        ax2[1, 3].axis('off')

        plt.tight_layout()
        diag_path = out_dir / f'{Path(name).stem}_diagnostics.png'
        plt.savefig(diag_path, dpi=150, bbox_inches='tight')
        print(f"Saved diagnostics to {diag_path}")
        plt.close(fig2)

    print("Done.")


if __name__ == '__main__':
    main()
