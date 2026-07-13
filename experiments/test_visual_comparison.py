"""Visual comparison: baseline vs improved segmentation on text-heavy panels.

Tests three improvements identified by experiment agents:
1. row_median_filter (size=5) as preprocessing instead of adaptive_blur
2. median_filter (size=5) postprocessing on colorbar_guided + pastel_faded paths
3. SLIC superpixel + kmeans as alternative engine
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2lab
from skimage.segmentation import slic
from scipy.cluster.vq import kmeans2

from geoseg.modules.segment_engines._shared import (
    _create_overlay,
    adaptive_blur,
)
from geoseg.modules.segment_engines.v4_kmeans import segment
from geoseg.modules.segment_engines.metrics import compute_all


def row_median_filter(panel_rgb: np.ndarray, size: int = 5) -> np.ndarray:
    """1D median filter along rows — suppresses horizontal text/noise,
    preserves vertical layer boundaries."""
    return ndimage.median_filter(panel_rgb, size=(1, size, 1))


def segment_slic_kmeans(panel_rgb: np.ndarray, n_layers: int = 5) -> dict:
    """SLIC superpixel + superpixel-level kmeans."""
    h, w = panel_rgb.shape[:2]
    segments = slic(
        panel_rgb,
        n_segments=500,
        compactness=10.0,
        channel_axis=2,
        start_label=0,
    )
    n_sp = int(segments.max()) + 1

    # Compute mean RGB per superpixel
    sp_means = np.zeros((n_sp, 3), dtype=np.float64)
    for sp_id in range(n_sp):
        mask = segments == sp_id
        if mask.any():
            sp_means[sp_id] = panel_rgb[mask].mean(axis=0)

    # K-means on superpixel means
    centroids, sp_labels = kmeans2(sp_means, n_layers, minit="++", seed=42)

    # Map superpixel labels back to pixel labels
    labels = sp_labels[segments].astype(np.int32)

    # Reorder by median y (top to bottom)
    unique = np.unique(labels)
    median_y = {lbl: np.median(np.where(labels == lbl)[0]) for lbl in unique}
    sorted_by_y = sorted(median_y.items(), key=lambda x: x[1])
    old_to_new = {old: new for new, (old, _) in enumerate(sorted_by_y)}
    out = np.full_like(labels, -1)
    for old, new in old_to_new.items():
        out[labels == old] = new

    # Palette from mean color of each final label
    palette = np.zeros((n_layers, 3), dtype=np.uint8)
    for lbl in range(n_layers):
        mask = out == lbl
        if mask.any():
            palette[lbl] = panel_rgb[mask].mean(axis=0).astype(np.uint8)

    overlay = _create_overlay(panel_rgb, out, palette)

    return {
        "labels": out,
        "overlay": overlay,
        "meta": {"engine": "slic_kmeans", "n_layers": n_layers},
    }


def run_comparison(image_path: str, n_layers: int = 5) -> dict:
    """Run baseline + all improvements, save side-by-side overlays."""
    img = Image.open(image_path).convert("RGB")
    panel_rgb = np.array(img)

    results = {}

    # --- Baseline: adaptive_blur + v4_kmeans ---
    blurred = adaptive_blur(panel_rgb)
    baseline = segment(blurred, n_layers=n_layers)
    results["baseline"] = {
        "labels": baseline["labels"],
        "overlay": baseline["overlay"],
        "meta": baseline["meta"],
    }

    # --- Improvement 1: row_median + v4_kmeans ---
    filtered = row_median_filter(panel_rgb, size=5)
    row_med = segment(filtered, n_layers=n_layers)
    results["row_median"] = {
        "labels": row_med["labels"],
        "overlay": row_med["overlay"],
        "meta": row_med["meta"],
    }

    # --- Improvement 2: row_median + v4_kmeans + median post on all paths ---
    # We manually apply median postprocessing to cover colorbar_guided/pastel_faded
    filtered2 = row_median_filter(panel_rgb, size=5)
    row_med_post = segment(filtered2, n_layers=n_layers)
    labels_post = ndimage.median_filter(row_med_post["labels"], size=5)
    # Rebuild overlay with postprocessed labels
    # Use seeds from row_med_post and remap colors
    from geoseg.modules.segment_engines._shared import _distinct_colors
    n = int(labels_post.max()) + 1
    palette = _distinct_colors(n)
    overlay_post = _create_overlay(panel_rgb, labels_post, palette)
    results["row_median_post"] = {
        "labels": labels_post,
        "overlay": overlay_post,
        "meta": {**row_med_post["meta"], "postprocess": "median_5"},
    }

    # --- Improvement 3: SLIC + kmeans ---
    slic_result = segment_slic_kmeans(panel_rgb, n_layers=n_layers)
    results["slic_kmeans"] = slic_result

    # --- Save comparison image ---
    out_dir = Path("runs/visual_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    # Side-by-side: original | baseline | row_median | row_median_post | slic
    h, w = panel_rgb.shape[:2]
    # Resize overlays to same height for side-by-side
    combined = np.concatenate([
        panel_rgb,
        baseline["overlay"],
        row_med["overlay"],
        overlay_post,
        slic_result["overlay"],
    ], axis=1)

    Image.fromarray(combined).save(out_dir / f"{stem}_comparison.png")

    # Also save individual overlays
    for name, res in results.items():
        Image.fromarray(res["overlay"]).save(out_dir / f"{stem}_{name}.png")

    # Compute metrics
    metrics = {}
    for name, res in results.items():
        metrics[name] = compute_all(res["labels"], panel_rgb)

    return metrics


if __name__ == "__main__":
    test_images = [
        "runs/test_panel_fix/page_002_img_0_panels.jpg",
        "runs/test_panel_fix/page_011_img_0_panels.jpg",
        "runs/test_panel_fix/page_003_img_0_panels.jpg",
        "runs/test_panel_fix/page_010_img_0_panels.jpg",
        "runs/test_panel_fix/page_004_img_0_panels.jpg",
        "runs/test_panel_fix/page_013_img_0_panels.jpg",
    ]

    all_metrics = {}
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"Skip missing: {img_path}")
            continue
        print(f"\n=== {img_path} ===")
        metrics = run_comparison(img_path, n_layers=5)
        all_metrics[Path(img_path).stem] = metrics
        for name, m in metrics.items():
            frag = m.get("total_fragment_area_fraction", 0)
            ba = m.get("boundary_alignment", 0)
            n = m.get("n_layers", 0)
            print(f"  {name:20s}: layers={n}, BA={ba:.3f}, frag_area={frag:.4f}")

    print(f"\nOverlays saved to: runs/visual_comparison/")
