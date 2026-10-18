#!/usr/bin/env python3
"""Fix light-region under-segmentation for fig6_profile_07.

Runs 3 variants, generates mask overlays, saves best to output dir.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2lab, lab2rgb
from scipy.cluster.vq import kmeans2

# Add src to path
sys.path.insert(0, "/Users/daiduo2/geoseg/src")

from geoseg.modules.segment_engines._shared import (
    _create_overlay,
    _label_by_nearest,
    _merge_small_regions,
    row_median_filter,
)
from geoseg.modules.segment_engines.v4_kmeans import (
    segment_colorbar_guided,
    _sample_colorbar_seeds,
    _reorder_labels_by_median_y,
    _fill_holes,
    _remove_small_components,
    _enhance_close_boundaries,
)

# Paths
PANEL_PATH = "/Users/daiduo2/geoseg/runs/feng_fig6_panels_v4/fig6_profile_07.png"
CURRENT_DIR = "/Users/daiduo2/geoseg/runs/feng_fig6_final/fig6_profile_07"
OUTPUT_DIR = "/Users/daiduo2/geoseg/runs/feng_fig6_final_v2/fig6_profile_07"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_panel_and_colorbar(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract panel region and colorbar from composite image."""
    h, w, _ = img_rgb.shape
    # The colorbar is at the bottom, panel is above it
    # Look for the colorbar strip (typically a thin horizontal strip at bottom)
    # Find the boundary by looking for the colorbar pattern

    # Colorbar is usually at the very bottom, ~10-20 pixels tall
    # Try bottom 40 pixels for colorbar
    colorbar_height = min(40, h // 5)
    colorbar = img_rgb[-colorbar_height:, :]

    # Panel is everything above the colorbar
    panel = img_rgb[:-colorbar_height, :]

    # Check if the bottom strip actually looks like a colorbar (has many colors)
    # If not, the colorbar might be even thinner or non-existent
    bottom_strip = img_rgb[-colorbar_height:, :]
    # A colorbar should have high color variance along its long axis
    strip_lab = rgb2lab(bottom_strip)
    if bottom_strip.shape[1] > bottom_strip.shape[0]:
        # Horizontal colorbar: variance along x
        var = np.var(strip_lab.reshape(-1, 3), axis=0).mean()
    else:
        var = np.var(strip_lab.reshape(-1, 3), axis=0).mean()

    # If variance is too low, colorbar might be thinner
    if var < 5.0:
        colorbar_height = min(20, h // 10)
        colorbar = img_rgb[-colorbar_height:, :]
        panel = img_rgb[:-colorbar_height, :]

    return panel, colorbar


def variant_a_colorbar_lab_nearest_median(panel_rgb, colorbar_rgb, n_layers=7):
    """Variant A: LAB nearest-median with MORE colorbar seeds."""
    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)

    # Sample n_layers evenly from colorbar
    seeds_rgb, names = _sample_colorbar_seeds(colorbar_rgb, n_layers)
    seeds_lab = rgb2lab(seeds_rgb[np.newaxis, ...])[0]

    # Nearest seed in LAB
    labels = _label_by_nearest(panel_lab, seeds_lab)

    # Median filter for smoothing
    labels = ndimage.median_filter(labels, size=5)

    # Reorder by depth
    labels = _reorder_labels_by_median_y(labels)

    # Remove very small components
    labels = _remove_small_components(labels, min_area_frac=0.0005)

    # Merge small regions
    labels = _merge_small_regions(labels, min_area_frac=0.002)

    overlay = _create_overlay(panel_rgb, labels, seeds_rgb, fill_mode="mask")

    return {
        "labels": labels,
        "seeds": seeds_rgb,
        "overlay": overlay,
        "meta": {
            "engine": "colorbar_lab_nearest_median",
            "n_layers": n_layers,
            "seeds_rgb": seeds_rgb.tolist(),
            "names": names,
        }
    }


def variant_b_colorbar_guided_with_median(panel_rgb, colorbar_rgb, n_layers=7):
    """Variant B: v4_kmeans colorbar_guided with n_layers=7 and median post-processing."""
    result = segment_colorbar_guided(
        panel_rgb,
        colorbar_rgb,
        n_layers=n_layers,
        color_dist_threshold=35.0,  # Tighter boundary merging for more layers
    )

    # Additional median post-processing
    labels = ndimage.median_filter(result["labels"], size=5)
    labels = _reorder_labels_by_median_y(labels)

    # Get final palette from labels
    k = int(labels.max()) + 1
    final_palette = np.zeros((k, 3), dtype=np.uint8)
    for lbl in range(k):
        mask = labels == lbl
        if mask.any():
            final_palette[lbl] = panel_rgb[mask].mean(axis=0).astype(np.uint8)
        else:
            final_palette[lbl] = result["seeds"][lbl] if lbl < len(result["seeds"]) else np.array([128, 128, 128])

    overlay = _create_overlay(panel_rgb, labels, final_palette, fill_mode="mask")

    return {
        "labels": labels,
        "seeds": final_palette,
        "overlay": overlay,
        "meta": {
            "engine": "v4_kmeans_colorbar_guided_median",
            "n_layers": k,
            "original_meta": result["meta"],
        }
    }


def variant_c_oversegment_then_merge(panel_rgb, colorbar_rgb, n_layers=7):
    """Variant C: Oversegment with more seeds, then merge adjacent similar labels."""
    h, w, _ = panel_rgb.shape

    # Start with MORE seeds than needed (oversegment)
    initial_k = 10
    seeds_rgb, _ = _sample_colorbar_seeds(colorbar_rgb, initial_k)

    # K-means with these seeds
    pixels = panel_rgb.reshape(-1, 3).astype(np.float64)
    seeds_arr = seeds_rgb.astype(np.float64)
    centroids, labels_flat = kmeans2(pixels, seeds_arr, minit="matrix")
    labels = labels_flat.reshape(h, w).astype(np.int32)

    # Reorder by depth
    labels = _reorder_labels_by_median_y(labels)

    # Fill holes
    labels = _fill_holes(labels)

    # Compute mean color of each label region
    k = initial_k
    region_colors = np.zeros((k, 3), dtype=np.float64)
    region_sizes = np.zeros(k, dtype=np.int64)
    for lbl in range(k):
        mask = labels == lbl
        if mask.any():
            region_colors[lbl] = panel_rgb[mask].mean(axis=0)
            region_sizes[lbl] = mask.sum()

    # Merge adjacent labels that are too similar in color
    # Compute pairwise LAB distances
    region_lab = rgb2lab(region_colors[np.newaxis, ...])[0]

    merge_threshold = 15.0  # LAB distance threshold for merging
    max_merge_iterations = 3

    for _ in range(max_merge_iterations):
        merged = False
        # Check adjacent labels (in depth order)
        for i in range(k - 1):
            if region_sizes[i] == 0 or region_sizes[i + 1] == 0:
                continue
            dist = float(np.linalg.norm(region_lab[i] - region_lab[i + 1]))
            if dist < merge_threshold:
                # Merge i+1 into i
                labels[labels == (i + 1)] = i
                # Also shift all labels > i+1 down by 1
                labels[labels > (i + 1)] -= 1
                # Update region colors and sizes
                mask = labels == i
                if mask.any():
                    region_colors[i] = panel_rgb[mask].mean(axis=0)
                    region_sizes[i] = mask.sum()
                region_sizes[i + 1] = 0
                region_lab = rgb2lab(region_colors[np.newaxis, ...])[0]
                k -= 1
                merged = True
                break
        if not merged:
            break

    # Renumber labels to be contiguous
    unique_labels = sorted(np.unique(labels[labels >= 0]))
    old_to_new = {old: new for new, old in enumerate(unique_labels)}
    new_labels = np.full_like(labels, -1)
    for old, new in old_to_new.items():
        new_labels[labels == old] = new
    labels = new_labels

    # Final median filter
    labels = ndimage.median_filter(labels, size=5)

    # Recompute final palette
    k = int(labels.max()) + 1
    final_palette = np.zeros((k, 3), dtype=np.uint8)
    for lbl in range(k):
        mask = labels == lbl
        if mask.any():
            final_palette[lbl] = panel_rgb[mask].mean(axis=0).astype(np.uint8)

    # Remove tiny components
    labels = _remove_small_components(labels, min_area_frac=0.001)
    labels = _merge_small_regions(labels, min_area_frac=0.002)

    overlay = _create_overlay(panel_rgb, labels, final_palette, fill_mode="mask")

    return {
        "labels": labels,
        "seeds": final_palette,
        "overlay": overlay,
        "meta": {
            "engine": "oversegment_then_merge",
            "n_layers": k,
            "initial_k": initial_k,
            "merge_threshold": merge_threshold,
        }
    }


def save_variant(result, variant_name, output_dir):
    """Save variant results to disk."""
    vdir = os.path.join(output_dir, f"variant_{variant_name}")
    os.makedirs(vdir, exist_ok=True)

    # Save labels
    np.savez(os.path.join(vdir, "labels.npz"), labels=result["labels"])

    # Save overlay
    overlay_img = Image.fromarray(result["overlay"])
    overlay_img.save(os.path.join(vdir, "overlay_mask.jpg"))

    # Save meta
    meta = result["meta"].copy()
    # Convert numpy arrays to lists for JSON
    if "seeds_rgb" in meta:
        meta["seeds_rgb"] = [[int(c) for c in s] for s in meta["seeds_rgb"]]
    with open(os.path.join(vdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return vdir


def create_comparison(original_rgb, mask_overlay, output_path):
    """Create side-by-side original vs mask comparison."""
    h, w, _ = original_rgb.shape
    # Create a combined image: original on left, mask on right
    combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined[:, :w, :] = original_rgb
    combined[:, w:, :] = mask_overlay
    Image.fromarray(combined).save(output_path)


def compute_label_stats(labels):
    """Compute label statistics for audit."""
    unique = np.unique(labels[labels >= 0])
    stats = {}
    for lbl in unique:
        count = int((labels == lbl).sum())
        ys, xs = np.where(labels == lbl)
        if len(ys) > 0:
            stats[int(lbl)] = {
                "pixels": count,
                "median_y": float(np.median(ys)),
                "median_x": float(np.median(xs)),
            }
    return stats


def main():
    # Load source image
    img = Image.open(PANEL_PATH).convert("RGB")
    img_rgb = np.array(img)

    # Extract panel and colorbar
    panel_rgb, colorbar_rgb = extract_panel_and_colorbar(img_rgb)
    print(f"Panel shape: {panel_rgb.shape}, Colorbar shape: {colorbar_rgb.shape}")

    # Save extracted panel for reference
    Image.fromarray(panel_rgb).save(os.path.join(OUTPUT_DIR, "panel_extracted.png"))
    Image.fromarray(colorbar_rgb).save(os.path.join(OUTPUT_DIR, "colorbar_extracted.png"))

    # Run variants
    print("\n=== Variant A: colorbar_lab_nearest_median (n_layers=7) ===")
    result_a = variant_a_colorbar_lab_nearest_median(panel_rgb, colorbar_rgb, n_layers=7)
    vdir_a = save_variant(result_a, "A", OUTPUT_DIR)
    print(f"  Saved to {vdir_a}")
    print(f"  Labels: {np.unique(result_a['labels'])}")
    stats_a = compute_label_stats(result_a["labels"])
    for lbl, s in sorted(stats_a.items()):
        print(f"    Label {lbl}: {s['pixels']} pixels, median_y={s['median_y']:.1f}")

    print("\n=== Variant B: v4_kmeans colorbar_guided + median (n_layers=7) ===")
    result_b = variant_b_colorbar_guided_with_median(panel_rgb, colorbar_rgb, n_layers=7)
    vdir_b = save_variant(result_b, "B", OUTPUT_DIR)
    print(f"  Saved to {vdir_b}")
    print(f"  Labels: {np.unique(result_b['labels'])}")
    stats_b = compute_label_stats(result_b["labels"])
    for lbl, s in sorted(stats_b.items()):
        print(f"    Label {lbl}: {s['pixels']} pixels, median_y={s['median_y']:.1f}")

    print("\n=== Variant C: oversegment_then_merge (initial_k=10, merge_threshold=15) ===")
    result_c = variant_c_oversegment_then_merge(panel_rgb, colorbar_rgb, n_layers=7)
    vdir_c = save_variant(result_c, "C", OUTPUT_DIR)
    print(f"  Saved to {vdir_c}")
    print(f"  Labels: {np.unique(result_c['labels'])}")
    stats_c = compute_label_stats(result_c["labels"])
    for lbl, s in sorted(stats_c.items()):
        print(f"    Label {lbl}: {s['pixels']} pixels, median_y={s['median_y']:.1f}")

    # Save comparison images for each variant
    for name, result in [("A", result_a), ("B", result_b), ("C", result_c)]:
        comp_path = os.path.join(OUTPUT_DIR, f"variant_{name}", "comparison.jpg")
        create_comparison(panel_rgb, result["overlay"], comp_path)
        print(f"  Comparison saved: {comp_path}")

    # Print summary for decision
    print("\n" + "=" * 60)
    print("SUMMARY FOR DECISION")
    print("=" * 60)
    for name, result, stats in [("A", result_a, stats_a), ("B", result_b, stats_b), ("C", result_c, stats_c)]:
        n_labels = len([l for l in np.unique(result["labels"]) if l >= 0])
        print(f"\nVariant {name}: {n_labels} labels")
        for lbl, s in sorted(stats.items()):
            print(f"  Label {lbl}: {s['pixels']} px, depth_y={s['median_y']:.0f}")

    # Return data for external decision
    return {
        "A": {"result": result_a, "stats": stats_a, "dir": vdir_a},
        "B": {"result": result_b, "stats": stats_b, "dir": vdir_b},
        "C": {"result": result_c, "stats": stats_c, "dir": vdir_c},
    }


if __name__ == "__main__":
    results = main()
