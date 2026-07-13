#!/usr/bin/env python3
"""Fix light-region under-segmentation for fig6_profile_07 - Variant D.

Variant D: LAB L-channel guided oversegmentation with adaptive merging.
This specifically targets the light-region (yellow/orange) merging issue.
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

sys.path.insert(0, "/Users/daiduo2/geoseg/src")

from geoseg.modules.segment_engines._shared import (
    _create_overlay,
    _label_by_nearest,
    _merge_small_regions,
    row_median_filter,
)
from geoseg.modules.segment_engines.v4_kmeans import (
    _sample_colorbar_seeds,
    _reorder_labels_by_median_y,
    _fill_holes,
    _remove_small_components,
)

PANEL_PATH = "/Users/daiduo2/geoseg/runs/feng_fig6_panels_v4/fig6_profile_07.png"
OUTPUT_DIR = "/Users/daiduo2/geoseg/runs/feng_fig6_final_v2/fig6_profile_07"


def extract_panel_and_colorbar(img_rgb):
    h, w, _ = img_rgb.shape
    colorbar_height = min(40, h // 5)
    colorbar = img_rgb[-colorbar_height:, :]
    panel = img_rgb[:-colorbar_height, :]
    return panel, colorbar


def variant_d_lab_l_guided(panel_rgb, colorbar_rgb, n_layers=7):
    """Variant D: Use LAB L-channel + colorbar seeds for light-region splitting.

    Strategy:
    1. Preprocess with row_median to suppress text
    2. Sample 8 seeds from colorbar (oversegment)
    3. Classify in LAB space with nearest seed
    4. Use L-channel gradient to detect boundaries between light regions
    5. Merge only when LAB distance is very small AND no strong L-gradient
    """
    h, w, _ = panel_rgb.shape

    # Step 1: Preprocess
    panel_proc = row_median_filter(panel_rgb, size=5)

    # Step 2: Sample seeds from colorbar (oversegment)
    seeds_rgb, names = _sample_colorbar_seeds(colorbar_rgb, 8)
    seeds_lab = rgb2lab(seeds_rgb[np.newaxis, ...])[0]

    # Step 3: Classify in LAB
    panel_lab = rgb2lab(panel_proc)
    labels = _label_by_nearest(panel_lab, seeds_lab)

    # Step 4: Use L-channel to detect light-region boundaries
    L = panel_lab[:, :, 0]  # Lightness channel

    # Compute vertical gradient of L (strong gradients = likely boundaries)
    L_grad_y = np.abs(np.gradient(L, axis=0))
    L_grad_y = ndimage.gaussian_filter(L_grad_y, sigma=1.0)

    # Step 5: Merge adjacent labels only if they're similar in LAB AND no strong L-gradient between them
    k = 8
    region_lab = np.zeros((k, 3))
    region_sizes = np.zeros(k)
    for lbl in range(k):
        mask = labels == lbl
        if mask.any():
            region_lab[lbl] = panel_lab[mask].mean(axis=0)
            region_sizes[lbl] = mask.sum()

    # Merge iteration: check adjacent labels in depth order
    labels = _reorder_labels_by_median_y(labels)

    # After reordering, recompute
    for lbl in range(k):
        mask = labels == lbl
        if mask.any():
            region_lab[lbl] = panel_lab[mask].mean(axis=0)
            region_sizes[lbl] = mask.sum()

    merge_threshold = 12.0  # Stricter than variant C
    max_merges = 3

    for _ in range(max_merges):
        merged = False
        # Check adjacent pairs (in depth order, i.e., adjacent label numbers)
        for i in range(k - 1):
            if region_sizes[i] == 0 or region_sizes[i + 1] == 0:
                continue

            # LAB distance between adjacent regions
            lab_dist = float(np.linalg.norm(region_lab[i] - region_lab[i + 1]))

            # Check L-gradient at the boundary between these two labels
            mask_i = labels == i
            mask_j = labels == (i + 1)
            dilated_i = ndimage.binary_dilation(mask_i, structure=np.ones((3, 3), dtype=bool))
            dilated_j = ndimage.binary_dilation(mask_j, structure=np.ones((3, 3), dtype=bool))
            boundary = dilated_i & dilated_j

            if boundary.any():
                avg_grad = float(L_grad_y[boundary].mean())
            else:
                avg_grad = 0.0

            # Merge only if very similar AND weak boundary
            if lab_dist < merge_threshold and avg_grad < 3.0:
                # Merge i+1 into i
                labels[labels == (i + 1)] = i
                labels[labels > (i + 1)] -= 1

                # Recompute
                mask = labels == i
                if mask.any():
                    region_lab[i] = panel_lab[mask].mean(axis=0)
                    region_sizes[i] = mask.sum()
                region_sizes[i + 1] = 0
                k -= 1
                merged = True
                break

        if not merged:
            break

    # Renumber labels
    unique_labels = sorted(np.unique(labels[labels >= 0]))
    old_to_new = {old: new for new, old in enumerate(unique_labels)}
    new_labels = np.full_like(labels, -1)
    for old, new in old_to_new.items():
        new_labels[labels == old] = new
    labels = new_labels

    # Final cleanup
    labels = ndimage.median_filter(labels, size=5)
    labels = _fill_holes(labels)
    labels = _remove_small_components(labels, min_area_frac=0.001)
    labels = _merge_small_regions(labels, min_area_frac=0.002)

    # Final palette
    k = int(labels.max()) + 1
    final_palette = np.zeros((k, 3), dtype=np.uint8)
    for lbl in range(k):
        mask = labels == lbl
        if mask.any():
            final_palette[lbl] = panel_rgb[mask].mean(axis=0).astype(np.uint8)

    overlay = _create_overlay(panel_rgb, labels, final_palette, fill_mode="mask")

    return {
        "labels": labels,
        "seeds": final_palette,
        "overlay": overlay,
        "meta": {
            "engine": "lab_l_guided",
            "n_layers": k,
            "initial_k": 8,
            "merge_threshold": merge_threshold,
            "preprocessing": "row_median",
        }
    }


def variant_e_explicit_seeds(panel_rgb, colorbar_rgb, n_layers=7):
    """Variant E: Explicitly sample more seeds in the light (yellow/orange) region.

    The colorbar goes: red -> orange -> yellow -> green -> cyan -> blue.
    For this panel, we need to capture:
    - Red/orange top (sediment)
    - Yellow upper-middle (LV-N)
    - Greenish-teal middle
    - Cyan lower
    - Blue bottom

    We sample 7 seeds with extra density in the yellow-green transition.
    """
    h, w, _ = panel_rgb.shape

    # Sample 7 seeds from colorbar with emphasis on light region
    # Use non-uniform sampling: more seeds in the yellow-green-cyan region
    cb_h, cb_w, _ = colorbar_rgb.shape

    if cb_h >= cb_w:
        # Vertical colorbar
        ys = np.linspace(int(0.05 * cb_h), int(0.95 * cb_h) - 1, 7).astype(int)
        cx = cb_w // 2
        seeds = np.array([colorbar_rgb[y, cx] for y in ys])
    else:
        # Horizontal colorbar
        # Sample with more density in the middle (yellow-green-cyan)
        xs = np.array([
            int(0.05 * cb_w),   # red
            int(0.20 * cb_w),   # orange
            int(0.35 * cb_w),   # yellow
            int(0.50 * cb_w),   # green
            int(0.65 * cb_w),   # cyan
            int(0.80 * cb_w),   # blue
            int(0.92 * cb_w),   # dark blue
        ])
        cy = cb_h // 2
        seeds = np.array([colorbar_rgb[cy, x] for x in xs])

    names = ["red", "orange", "yellow", "green", "cyan", "blue", "dark_blue"]

    # K-means with these seeds
    pixels = panel_rgb.reshape(-1, 3).astype(np.float64)
    seeds_arr = seeds.astype(np.float64)
    centroids, labels_flat = kmeans2(pixels, seeds_arr, minit="matrix")
    labels = labels_flat.reshape(h, w).astype(np.int32)

    # Reorder by depth
    labels = _reorder_labels_by_median_y(labels)

    # Fill holes
    labels = _fill_holes(labels)

    # Remove small components
    labels = _remove_small_components(labels, min_area_frac=0.001)

    # Median filter
    labels = ndimage.median_filter(labels, size=5)

    # Compute final palette
    k = int(labels.max()) + 1
    final_palette = np.zeros((k, 3), dtype=np.uint8)
    for lbl in range(k):
        mask = labels == lbl
        if mask.any():
            final_palette[lbl] = panel_rgb[mask].mean(axis=0).astype(np.uint8)

    # Merge small regions
    labels = _merge_small_regions(labels, min_area_frac=0.002)

    overlay = _create_overlay(panel_rgb, labels, final_palette, fill_mode="mask")

    return {
        "labels": labels,
        "seeds": final_palette,
        "overlay": overlay,
        "meta": {
            "engine": "explicit_seeds_dense",
            "n_layers": k,
            "seed_positions": xs.tolist() if cb_h < cb_w else ys.tolist(),
        }
    }


def save_variant(result, variant_name, output_dir):
    vdir = os.path.join(output_dir, f"variant_{variant_name}")
    os.makedirs(vdir, exist_ok=True)
    np.savez(os.path.join(vdir, "labels.npz"), labels=result["labels"])
    Image.fromarray(result["overlay"]).save(os.path.join(vdir, "overlay_mask.jpg"))
    meta = result["meta"].copy()
    with open(os.path.join(vdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return vdir


def create_comparison(original_rgb, mask_overlay, output_path):
    h, w, _ = original_rgb.shape
    combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined[:, :w, :] = original_rgb
    combined[:, w:, :] = mask_overlay
    Image.fromarray(combined).save(output_path)


def compute_label_stats(labels):
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
    img = Image.open(PANEL_PATH).convert("RGB")
    img_rgb = np.array(img)
    panel_rgb, colorbar_rgb = extract_panel_and_colorbar(img_rgb)

    print(f"Panel shape: {panel_rgb.shape}, Colorbar shape: {colorbar_rgb.shape}")

    print("\n=== Variant D: LAB L-guided (initial_k=8) ===")
    result_d = variant_d_lab_l_guided(panel_rgb, colorbar_rgb, n_layers=7)
    vdir_d = save_variant(result_d, "D", OUTPUT_DIR)
    print(f"  Labels: {np.unique(result_d['labels'])}")
    stats_d = compute_label_stats(result_d["labels"])
    for lbl, s in sorted(stats_d.items()):
        print(f"    Label {lbl}: {s['pixels']} pixels, median_y={s['median_y']:.1f}")

    print("\n=== Variant E: Explicit dense seeds (n=7) ===")
    result_e = variant_e_explicit_seeds(panel_rgb, colorbar_rgb, n_layers=7)
    vdir_e = save_variant(result_e, "E", OUTPUT_DIR)
    print(f"  Labels: {np.unique(result_e['labels'])}")
    stats_e = compute_label_stats(result_e["labels"])
    for lbl, s in sorted(stats_e.items()):
        print(f"    Label {lbl}: {s['pixels']} pixels, median_y={s['median_y']:.1f}")

    for name, result in [("D", result_d), ("E", result_e)]:
        comp_path = os.path.join(OUTPUT_DIR, f"variant_{name}", "comparison.jpg")
        create_comparison(panel_rgb, result["overlay"], comp_path)

    print("\n" + "=" * 60)
    print("ADDITIONAL VARIANTS SUMMARY")
    print("=" * 60)
    for name, result, stats in [("D", result_d, stats_d), ("E", result_e, stats_e)]:
        n_labels = len([l for l in np.unique(result["labels"]) if l >= 0])
        print(f"\nVariant {name}: {n_labels} labels")
        for lbl, s in sorted(stats.items()):
            print(f"  Label {lbl}: {s['pixels']} px, depth_y={s['median_y']:.0f}")

    return {"D": {"result": result_d, "stats": stats_d, "dir": vdir_d},
            "E": {"result": result_e, "stats": stats_e, "dir": vdir_e}}


if __name__ == "__main__":
    results = main()
