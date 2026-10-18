"""Panel 3 plume tube segmentation with edge_guided as secondary engine."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.cluster.vq import kmeans2

from geoseg.modules.segment_engines.edge_guided import segment as eg_segment
from geoseg.modules.segment_engines.regional_fusion import (
    fuse_with_freeze,
    generate_overlay_with_legend,
)
from geoseg.modules.segment_engines.v4_kmeans import _reorder_labels_by_median_y


# ---- Reuse proven preprocessing ----


def remove_text_v2(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive threshold + Laplacian + inpaint + median fill + Gaussian blend."""
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


def enhance_v(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE enhancement on V channel."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ---- Plume detection (warm color overlap) ----


def find_plume_label(labels: np.ndarray, img_rgb: np.ndarray) -> tuple[int, int]:
    """Identify plume label by warm color overlap."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    warm_mask = (
        ((hsv[:, :, 0] < 30) | (hsv[:, :, 0] > 150))
        & (hsv[:, :, 1] > 40)
        & (hsv[:, :, 2] > 80)
    )
    best_label, best_score = 0, 0
    for lbl in sorted(set(labels.flatten()) - {0}):
        score = np.logical_and(labels == lbl, warm_mask).sum()
        if score > best_score:
            best_score = score
            best_label = lbl
    return best_label, best_score


def get_plume_seeds(img_rgb: np.ndarray, plume_mask: np.ndarray, k: int = 4) -> list[dict]:
    """Extract color seeds from plume ROI for edge_guided reps."""
    pixels = img_rgb[plume_mask].astype(np.float64)
    if len(pixels) < k:
        k = max(1, len(pixels))
    centroids, _ = kmeans2(pixels, k, minit="points", iter=50)
    centroids = np.clip(centroids, 0, 255).astype(np.uint8)
    reps = []
    for i, c in enumerate(centroids):
        # Find a pixel in plume closest to this centroid
        dists = np.linalg.norm(pixels - c, axis=1)
        idx = int(np.argmin(dists))
        ys, xs = np.where(plume_mask)
        y, x = ys[idx], xs[idx]
        reps.append({
            "y": int(y),
            "x": int(x),
            "color": c.tolist(),
            "name": f"plume_layer_{i+1}",
        })
    return reps


# ---- Main experiment ----


def main():
    panel_path = Path("src/3d_schematic/panel_3_front.png")
    output_dir = Path("runs/3d_schematic_edge_guided")
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_rgb = np.array(Image.open(panel_path).convert("RGB"))
    print(f"Original shape: {panel_rgb.shape}")

    # Stage 1: Preprocessing
    cleaned, text_mask = remove_text_v2(panel_rgb)
    enhanced = enhance_v(cleaned, clip_limit=2.0)
    Image.fromarray(enhanced).save(output_dir / "00_enhanced.jpg", quality=90)

    # Stage 2: Load primary labels
    primary_npz = Path("runs/3d_schematic_correct_e2e/panel_3_front/labels_primary.npz")
    labels_a = np.load(primary_npz)["labels"]
    labels_a = _reorder_labels_by_median_y(labels_a)
    unique = sorted(set(labels_a.flatten()) - {0})
    print(f"Primary labels: {unique}")

    # Stage 3: Identify plume
    plume_label, plume_score = find_plume_label(labels_a, enhanced)
    print(f"Plume label: {plume_label} (score={plume_score})")

    freeze_mask = labels_a != plume_label
    retry_mask = labels_a == plume_label
    print(f"Freeze pixels: {freeze_mask.sum()} ({freeze_mask.mean()*100:.1f}%)")
    print(f"Retry pixels:  {retry_mask.sum()} ({retry_mask.mean()*100:.1f}%)")

    # Stage 4: Masked image for secondary engine
    masked_rgb = enhanced.copy()
    mask_u8 = freeze_mask.astype(np.uint8) * 255
    masked_rgb = cv2.inpaint(masked_rgb, mask_u8, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # Extra contrast enhancement on plume region specifically
    plume_enhanced = enhance_v(masked_rgb, clip_limit=4.0)
    # Blend: only inside plume ROI use high contrast, outside keep inpainted
    plume_mask_u8 = retry_mask.astype(np.uint8) * 255
    plume_mask_f = cv2.GaussianBlur(plume_mask_u8.astype(np.float32) / 255.0, (5, 5), 0)
    plume_mask_3ch = np.stack([plume_mask_f] * 3, axis=-1)
    masked_rgb = (plume_enhanced * plume_mask_3ch + masked_rgb * (1 - plume_mask_3ch)).astype(np.uint8)

    Image.fromarray(masked_rgb).save(output_dir / "01_masked_input.jpg", quality=90)

    # Stage 5: Generate reps from plume seeds
    reps = get_plume_seeds(enhanced, retry_mask, k=4)
    print(f"Plume reps: {[r['name'] + '=' + str(r['color']) for r in reps]}")

    # Stage 6: Run edge_guided on masked image with varied params
    configs = [
        {"edge_weight": 0.5, "sigma": 3.0, "n_layers": 4, "reps": None, "name": "eg_default"},
        {"edge_weight": 0.8, "sigma": 2.0, "n_layers": 4, "reps": reps, "name": "eg_high_edge_reps"},
        {"edge_weight": 0.9, "sigma": 1.5, "n_layers": 5, "reps": reps, "name": "eg_max_edge_reps"},
    ]

    results = []
    for cfg in configs:
        name = cfg.pop("name")
        reps_cfg = cfg.pop("reps")
        print(f"\n[{name}] edge_weight={cfg['edge_weight']}, sigma={cfg['sigma']}, n_layers={cfg['n_layers']}, reps={reps_cfg is not None}")

        try:
            result_b = eg_segment(masked_rgb, reps=reps_cfg, **cfg)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        labels_b = _reorder_labels_by_median_y(result_b["labels"])
        overlay_b = result_b["overlay"]

        Image.fromarray(overlay_b).save(output_dir / f"02_{name}_secondary.jpg", quality=90)
        sec_labels = sorted(set(labels_b.flatten()) - {0})
        print(f"  Secondary labels: {sec_labels}")

        # Stage 7: Fuse
        fused = fuse_with_freeze(labels_a, labels_b, freeze_mask, seam_width=3)
        overlay_fused = generate_overlay_with_legend(enhanced, fused)
        Image.fromarray(overlay_fused).save(output_dir / f"03_{name}_fused.jpg", quality=90)

        # Metrics
        preserved = np.sum((labels_a[freeze_mask] == fused[freeze_mask]).astype(int))
        changed = np.sum((labels_a[retry_mask] != fused[retry_mask]).astype(int))
        print(f"  Freeze preserved: {preserved}/{freeze_mask.sum()} ({preserved/freeze_mask.sum()*100:.1f}%)")
        print(f"  Retry changed:    {changed}/{retry_mask.sum()} ({changed/retry_mask.sum()*100:.1f}%)")

        results.append({
            "name": name,
            "config": cfg,
            "preserved_pct": preserved / freeze_mask.sum() * 100,
            "changed_pct": changed / retry_mask.sum() * 100,
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"{r['name']}: freeze={r['preserved_pct']:.1f}%, retry_changed={r['changed_pct']:.1f}%")

    (output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
