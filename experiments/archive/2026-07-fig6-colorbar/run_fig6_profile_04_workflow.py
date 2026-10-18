#!/usr/bin/env python3
"""Text-aware geoseg workflow for fig6_profile_04 — v2 with n_layers=5."""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geoseg.modules.segment_engines.mask_aware import segment_with_text_mask
from geoseg.modules.segment_engines._shared import _create_overlay


def create_overlay(panel_rgb, labels, fill_mode="blend"):
    return _create_overlay(
        panel_rgb,
        labels,
        np.zeros((1, 3), dtype=np.uint8),
        alpha=0.65,
        boundary_mode="thin",
        skip_background=True,
        min_area_frac=0.0005,
        fill_mode=fill_mode,
    )


def remove_text_v2(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive threshold + Laplacian + inpaint + median fill + Gaussian blend.

    Based on test_3d_schematic_edge_guided.py approach, tuned for fig6 panels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=25, C=-5
    )
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    laplacian = (laplacian > np.percentile(laplacian, 80)).astype(np.uint8) * 255
    text_mask = ((adaptive > 0) & (laplacian > 0))
    # Simpler approach: just use connectedComponents on the boolean mask
    num, labeled_mask = cv2.connectedComponents(text_mask.astype(np.uint8), connectivity=8)
    text_mask_clean = np.zeros_like(text_mask)
    for i in range(1, num):
        comp = labeled_mask == i
        comp_area = comp.sum()
        if 8 < comp_area < 1200:
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


def main():
    base_dir = Path("/Users/daiduo2/geoseg")
    panel_path = base_dir / "runs/feng_fig6_final_v4/crop_tests/fig6_profile_04_cropped.jpg"
    out_dir = base_dir / "runs/feng_fig6_workflow_v8/fig6_profile_04"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load original panel
    img_rgb = np.array(Image.open(panel_path).convert("RGB"))
    print(f"Original panel shape: {img_rgb.shape}")

    # 2. Run text removal (v2 approach)
    print("\n--- Stage 2: Text removal v2 ---")
    cleaned, text_mask = remove_text_v2(img_rgb)
    cleaned_path = out_dir / "cleaned.jpg"
    mask_path = out_dir / "text_mask.jpg"
    Image.fromarray(cleaned).save(cleaned_path)
    Image.fromarray(text_mask).save(mask_path)
    print(f"Saved cleaned panel to {cleaned_path}")
    print(f"Saved text mask to {mask_path}")
    print(f"Text mask coverage: {text_mask.sum() / text_mask.size * 100:.2f}%")

    # 3. Run mask-aware segmentation with multiple engines
    engines = ["v4_kmeans", "edge_guided", "kmeans_full"]
    n_layers = 5  # Request 5 to get 4 actual layers
    results = {}

    for engine in engines:
        print(f"\n--- Stage 3: Running {engine} ---")
        try:
            result = segment_with_text_mask(
                engine_name=engine,
                image_rgb=img_rgb,
                text_mask=text_mask.astype(bool),
                n_layers=n_layers,
            )
            labels = result["labels"]
            overlay = result["overlay"]

            unique = sorted(set(labels.flatten()) - {0})
            print(f"  Engine: {engine}, labels: {unique}, n_layers: {len(unique)}")

            # Save per-engine outputs
            labels_path = out_dir / f"labels_{engine}.npz"
            overlay_path = out_dir / f"overlay_{engine}.jpg"
            np.savez(labels_path, labels=labels)
            Image.fromarray(overlay).save(overlay_path)
            print(f"  Saved labels to {labels_path}")
            print(f"  Saved overlay to {overlay_path}")

            results[engine] = {
                "labels": labels,
                "overlay": overlay,
                "unique_labels": unique,
                "labels_path": labels_path,
                "overlay_path": overlay_path,
            }
        except Exception as e:
            print(f"  ERROR with {engine}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Print summary for visual evaluation
    print("\n" + "=" * 60)
    print("VISUAL EVALUATION SUMMARY")
    print("=" * 60)
    for engine, r in results.items():
        print(f"\n{engine}:")
        print(f"  Labels: {r['unique_labels']}")
        print(f"  Overlay: {r['overlay_path']}")

    print("\n--- All engine outputs saved. Ready for visual evaluation. ---")
    return results


if __name__ == "__main__":
    main()
