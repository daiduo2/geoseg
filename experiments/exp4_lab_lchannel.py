"""Experiment 4: Segment plume ROI using LAB L-channel (brightness) instead of RGB color."""
from __future__ import annotations

import colorsys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def remove_text_v2(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive threshold + Laplacian + inpaint + median fill + Gaussian blend."""
    import scipy.ndimage as ndimage
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=25, C=-5
    )
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    laplacian = (laplacian > np.percentile(laplacian, 80)).astype(np.uint8) * 255
    text_mask = ((adaptive > 0) & (laplacian > 0))
    labeled, num = ndimage.label(text_mask)
    text_mask_clean = np.zeros_like(text_mask)
    for i in range(1, num + 1):
        comp = labeled == i
        if 8 < comp.sum() < 1200:
            text_mask_clean[comp] = True
    kernel = np.ones((5, 5), np.uint8)
    text_dilated = cv2.dilate(text_mask_clean.astype(np.uint8), kernel, iterations=2)
    inpainted = cv2.inpaint(image, text_dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    cleaned = inpainted.copy()
    mask_bool = text_dilated.astype(bool)
    for ch in range(3):
        channel = inpainted[:, :, ch].astype(np.float32)
        ys, xs = np.where(mask_bool)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 3), min(image.shape[0], y + 4)
            x0, x1 = max(0, x - 3), min(image.shape[1], x + 4)
            patch = channel[y0:y1, x0:x1]
            patch_mask = mask_bool[y0:y1, x0:x1]
            valid = patch[~patch_mask]
            if len(valid) > 0:
                cleaned[y, x, ch] = int(np.median(valid))
            else:
                cleaned[y, x, ch] = int(channel[y, x])
    blurred = cv2.GaussianBlur(cleaned, (7, 7), sigmaX=1.5)
    mask_f = text_dilated.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (5, 5), sigmaX=2)
    mask_3ch = np.stack([mask_f] * 3, axis=-1)
    cleaned = (blurred * mask_3ch + cleaned * (1 - mask_3ch)).astype(np.uint8)
    return cleaned, text_dilated


def enhance_v(image: np.ndarray) -> np.ndarray:
    """CLAHE enhancement on V channel."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def generate_overlay(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Generate colored overlay from labels."""
    overlay = image.copy()
    unique_labels = sorted(set(labels.flatten()) - {0})
    np.random.seed(42)
    colors = {
        lbl: tuple(int(c * 255) for c in colorsys.hsv_to_rgb(
            (i * 0.618033988749895) % 1.0, 0.7, 0.9
        ))
        for i, lbl in enumerate(unique_labels)
    }
    colors[0] = (0, 0, 0)
    for lbl in unique_labels:
        mask = labels == lbl
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.4 +
                         np.array(colors[lbl], dtype=np.float32) * 0.6).astype(np.uint8)
    return overlay


def main():
    panel_path = Path("src/3d_schematic/panel_3_front.png")
    primary_labels_path = Path("runs/3d_schematic_correct_e2e/panel_3_front/labels_primary.npz")
    out_dir = Path("runs/tubular_panel3")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 4: LAB L-channel k-means on plume ROI")
    print("=" * 60)

    # 1. Load panel and preprocess
    print("\n[1] Load panel and preprocess")
    panel_rgb = np.array(Image.open(panel_path).convert("RGB"))
    print(f"  Original shape: {panel_rgb.shape}")

    cleaned, text_mask = remove_text_v2(panel_rgb)
    enhanced = enhance_v(cleaned)
    print(f"  Text mask pixels: {text_mask.sum()}")

    # 2. Load primary labels, identify plume ROI via HSV warm-color overlap
    print("\n[2] Load primary labels, identify plume ROI via HSV warm-color overlap")
    primary = np.load(primary_labels_path)
    labels_primary = primary["labels"]
    print(f"  Primary labels shape: {labels_primary.shape}")
    print(f"  Primary unique labels: {sorted(set(labels_primary.flatten()) - {0})}")

    # HSV warm-color mask: red/orange/yellow hues with decent saturation/value
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # Warm hues: 0-30 (red-orange-yellow) and 170-180 (red wrap-around)
    warm_mask = (
        (((hue >= 0) & (hue <= 30)) | ((hue >= 170) & (hue <= 180)))
        & (sat > 30)
        & (val > 50)
    )
    print(f"  Warm-color mask pixels: {warm_mask.sum()}")

    # Overlap with primary labels (non-background)
    fg_mask = labels_primary != 0
    plume_roi = warm_mask & fg_mask
    print(f"  Plume ROI pixels (warm + foreground): {plume_roi.sum()}")

    if plume_roi.sum() < 100:
        print("  WARNING: Plume ROI too small, falling back to all warm pixels")
        plume_roi = warm_mask
        print(f"  Fallback ROI pixels: {plume_roi.sum()}")

    # 3. Convert plume ROI to LAB, extract L-channel
    print("\n[3] Convert plume ROI to LAB, extract L-channel")
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    print(f"  L-channel range: [{l_channel.min():.1f}, {l_channel.max():.1f}]")

    # 4. Run k-means on L-channel within plume ROI (1D data) with k=2 or 3
    print("\n[4] Run k-means on L-channel within plume ROI")
    roi_pixels = l_channel[plume_roi].reshape(-1, 1)
    print(f"  ROI pixels for clustering: {len(roi_pixels)}")

    best_k = None
    best_labels = None
    best_centroids = None
    best_inertia = float("inf")

    for k in [2, 3]:
        if len(roi_pixels) < k * 10:
            print(f"  Skipping k={k}: not enough pixels")
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        attempts = 10
        compactness, labels_k, centers = cv2.kmeans(
            roi_pixels.astype(np.float32), k, None, criteria, attempts, flags
        )
        print(f"  k={k}: inertia={compactness:.2f}, centroids={centers.flatten().tolist()}")
        if compactness < best_inertia:
            best_inertia = compactness
            best_k = k
            best_labels = labels_k.flatten()
            best_centroids = centers.flatten()

    print(f"\n  Best k={best_k} with inertia={best_inertia:.2f}")
    print(f"  Centroids (L-channel): {best_centroids.tolist()}")

    # 5. Fuse result back into full image labels
    print("\n[5] Fuse result back into full image labels")
    labels_exp4 = labels_primary.copy()

    # Assign new label IDs for the segmented plume regions
    # Use high IDs to avoid collision with primary labels
    base_label = 100
    roi_coords = np.where(plume_roi)
    for i, lbl in enumerate(best_labels):
        y, x = roi_coords[0][i], roi_coords[1][i]
        labels_exp4[y, x] = base_label + lbl

    unique_exp4 = sorted(set(labels_exp4.flatten()) - {0})
    print(f"  Experiment labels: {unique_exp4}")

    # 6. Save overlay and labels
    print("\n[6] Save overlay and labels")
    overlay = generate_overlay(enhanced, labels_exp4)
    overlay_path = out_dir / "exp4_lab_lchannel.jpg"
    Image.fromarray(overlay).save(overlay_path, quality=90)
    print(f"  Overlay saved: {overlay_path}")

    labels_path = out_dir / "exp4_lab_lchannel.npz"
    np.savez_compressed(labels_path, labels=labels_exp4)
    print(f"  Labels saved: {labels_path}")

    # 7. Visual assessment
    print("\n[7] Visual assessment")
    total_pixels = labels_exp4.size
    for lbl in unique_exp4:
        mask = labels_exp4 == lbl
        count = mask.sum()
        pct = count / total_pixels * 100
        if lbl >= 100:
            centroid_idx = lbl - base_label
            centroid_val = best_centroids[centroid_idx] if centroid_idx < len(best_centroids) else None
            print(f"  Plume segment {lbl}: {count} px ({pct:.2f}%), L-centroid={centroid_val:.1f}" if centroid_val is not None else f"  Plume segment {lbl}: {count} px ({pct:.2f}%)")
        else:
            print(f"  Original label {lbl}: {count} px ({pct:.2f}%)")

    print("\n" + "=" * 60)
    print("Experiment 4 complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
