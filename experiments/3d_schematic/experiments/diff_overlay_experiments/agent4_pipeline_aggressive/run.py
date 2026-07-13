"""
Agent-4: Aggressive vs Conservative Diff-Overlay Pipeline Comparison

Tests robustness boundaries of the diff-overlay segmentation scheme by running
two extreme parameter sets against all 3 panels and generating comparison visuals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.segmentation import felzenszwalb

# Add parent to path for importing design_diff_overlay
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from design_diff_overlay import diff_overlay_pipeline, render_label_fill


PANEL_PATHS = [
    Path("/Users/daiduo2/geoseg/src/3d_schematic/panel_1_front.png"),
    Path("/Users/daiduo2/geoseg/src/3d_schematic/panel_2_front.png"),
    Path("/Users/daiduo2/geoseg/src/3d_schematic/panel_3_front.png"),
]

OUT_DIR = Path("/Users/daiduo2/geoseg/src/3d_schematic/diff_overlay_experiments/agent4_pipeline_aggressive")

AGGRESSIVE_PARAMS = {
    "blur_ksize": 21,
    "blur_sigma": 5.0,
    "diff_thresh": 10,
    "expand_radius": 25,
    "felz_scale": 300,
    "felz_sigma": 0.5,
    "overlay_label": -1,
}

CONSERVATIVE_PARAMS = {
    "blur_ksize": 11,
    "blur_sigma": 2.0,
    "diff_thresh": 30,
    "expand_radius": 10,
    "felz_scale": 300,
    "felz_sigma": 0.5,
    "overlay_label": -1,
}


def run_pipeline(image_path: Path, params: dict, name: str) -> dict:
    """Run diff_overlay_pipeline and save intermediates."""
    img = np.array(Image.open(image_path).convert("RGB"))
    result = diff_overlay_pipeline(img, **params)

    suffix = (
        f"_{name}_k{params['blur_ksize']}"
        f"_s{params['blur_sigma']}"
        f"_t{params['diff_thresh']}"
        f"_e{params['expand_radius']}"
    )

    # Save detail map
    detail_norm = (result["detail"] / (result["detail"].max() + 1e-8) * 255).astype(np.uint8)
    Image.fromarray(detail_norm).save(OUT_DIR / f"{image_path.stem}_detail{suffix}.png")

    # Save overlay mask
    overlay_uint8 = result["overlay_mask"].astype(np.uint8) * 255
    Image.fromarray(overlay_uint8).save(OUT_DIR / f"{image_path.stem}_overlay{suffix}.png")

    # Save label fill
    fill = render_label_fill(result["final_labels"], params["overlay_label"])
    Image.fromarray(fill).save(OUT_DIR / f"{image_path.stem}_fill{suffix}.png")

    # Save overlay-only visualization
    Image.fromarray(result["overlay_only"]).save(OUT_DIR / f"{image_path.stem}_overlay_vis{suffix}.png")

    return result


def build_comparison_figure(
    panel_paths: list[Path],
    aggressive_results: list[dict],
    conservative_results: list[dict],
) -> plt.Figure:
    """Build 3x4 comparison figure."""
    n_panels = len(panel_paths)
    fig, axes = plt.subplots(n_panels, 4, figsize=(20, 5 * n_panels))
    if n_panels == 1:
        axes = axes.reshape(1, -1)

    for row, (path, agg, cons) in enumerate(zip(panel_paths, aggressive_results, conservative_results)):
        img = np.array(Image.open(path).convert("RGB"))
        agg_fill = render_label_fill(agg["final_labels"], AGGRESSIVE_PARAMS["overlay_label"])
        cons_fill = render_label_fill(cons["final_labels"], CONSERVATIVE_PARAMS["overlay_label"])

        # Col 1: Original
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{path.stem} - Original", fontsize=12)
        axes[row, 0].axis("off")

        # Col 2: Aggressive label fill
        axes[row, 1].imshow(agg_fill)
        axes[row, 1].set_title(
            f"Aggressive\nk={AGGRESSIVE_PARAMS['blur_ksize']} "
            f"s={AGGRESSIVE_PARAMS['blur_sigma']} "
            f"t={AGGRESSIVE_PARAMS['diff_thresh']} "
            f"e={AGGRESSIVE_PARAMS['expand_radius']}",
            fontsize=10,
        )
        axes[row, 1].axis("off")

        # Col 3: Conservative label fill
        axes[row, 2].imshow(cons_fill)
        axes[row, 2].set_title(
            f"Conservative\nk={CONSERVATIVE_PARAMS['blur_ksize']} "
            f"s={CONSERVATIVE_PARAMS['blur_sigma']} "
            f"t={CONSERVATIVE_PARAMS['diff_thresh']} "
            f"e={CONSERVATIVE_PARAMS['expand_radius']}",
            fontsize=10,
        )
        axes[row, 2].axis("off")

        # Col 4: Mask comparison
        # aggressive = red, conservative = green, both = yellow
        agg_mask = agg["overlay_mask"].astype(np.uint8)
        cons_mask = cons["overlay_mask"].astype(np.uint8)
        comparison = np.zeros((*agg_mask.shape, 3), dtype=np.uint8)
        comparison[..., 0] = agg_mask * 255   # Red channel = aggressive
        comparison[..., 1] = cons_mask * 255  # Green channel = conservative
        # Yellow where both
        both = (agg_mask & cons_mask) * 255
        comparison[..., 2] = both * 0  # No blue

        axes[row, 3].imshow(comparison)
        axes[row, 3].set_title("Overlay Mask Comparison\nRed=Aggressive  Green=Conservative  Yellow=Both", fontsize=10)
        axes[row, 3].axis("off")

    plt.tight_layout()
    return fig


def compute_overlay_stats(result: dict) -> dict:
    """Compute statistics about the overlay mask."""
    mask = result["overlay_mask"]
    total_pixels = mask.size
    overlay_pixels = int(mask.sum())
    overlay_ratio = overlay_pixels / total_pixels

    # Count unique geo labels (excluding overlay)
    geo_labels = result["geo_labels"]
    unique_geo = len(np.unique(geo_labels[~mask]))

    # Average detail intensity inside vs outside overlay
    detail = result["detail"]
    avg_detail_inside = float(detail[mask].mean()) if overlay_pixels > 0 else 0.0
    avg_detail_outside = float(detail[~mask].mean()) if overlay_pixels < total_pixels else 0.0

    return {
        "overlay_pixels": overlay_pixels,
        "overlay_ratio": overlay_ratio,
        "unique_geo_labels": unique_geo,
        "avg_detail_inside": avg_detail_inside,
        "avg_detail_outside": avg_detail_outside,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    aggressive_results = []
    conservative_results = []

    print("=" * 60)
    print("Agent-4: Aggressive vs Conservative Pipeline Comparison")
    print("=" * 60)

    for path in PANEL_PATHS:
        print(f"\nProcessing {path.name}...")

        agg = run_pipeline(path, AGGRESSIVE_PARAMS, "aggressive")
        cons = run_pipeline(path, CONSERVATIVE_PARAMS, "conservative")

        aggressive_results.append(agg)
        conservative_results.append(cons)

        # Print stats
        agg_stats = compute_overlay_stats(agg)
        cons_stats = compute_overlay_stats(cons)

        print(f"  Aggressive overlay: {agg_stats['overlay_ratio']*100:.1f}% of image, {agg_stats['unique_geo_labels']} geo labels")
        print(f"  Conservative overlay: {cons_stats['overlay_ratio']*100:.1f}% of image, {cons_stats['unique_geo_labels']} geo labels")

    # Build and save comparison figure
    print("\nBuilding comparison figure...")
    fig = build_comparison_figure(PANEL_PATHS, aggressive_results, conservative_results)
    fig_path = OUT_DIR / "comparison_3x4.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure to {fig_path}")

    # Save detailed per-panel figures (larger, for closer inspection)
    for i, (path, agg, cons) in enumerate(zip(PANEL_PATHS, aggressive_results, conservative_results)):
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        img = np.array(Image.open(path).convert("RGB"))
        agg_fill = render_label_fill(agg["final_labels"], AGGRESSIVE_PARAMS["overlay_label"])
        cons_fill = render_label_fill(cons["final_labels"], CONSERVATIVE_PARAMS["overlay_label"])

        axes[0].imshow(img)
        axes[0].set_title("Original", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(agg_fill)
        axes[1].set_title(
            f"Aggressive: k={AGGRESSIVE_PARAMS['blur_ksize']} s={AGGRESSIVE_PARAMS['blur_sigma']} "
            f"t={AGGRESSIVE_PARAMS['diff_thresh']} e={AGGRESSIVE_PARAMS['expand_radius']}",
            fontsize=12,
        )
        axes[1].axis("off")

        axes[2].imshow(cons_fill)
        axes[2].set_title(
            f"Conservative: k={CONSERVATIVE_PARAMS['blur_ksize']} s={CONSERVATIVE_PARAMS['blur_sigma']} "
            f"t={CONSERVATIVE_PARAMS['diff_thresh']} e={CONSERVATIVE_PARAMS['expand_radius']}",
            fontsize=12,
        )
        axes[2].axis("off")

        agg_mask = agg["overlay_mask"].astype(np.uint8)
        cons_mask = cons["overlay_mask"].astype(np.uint8)
        comparison = np.zeros((*agg_mask.shape, 3), dtype=np.uint8)
        comparison[..., 0] = agg_mask * 255
        comparison[..., 1] = cons_mask * 255
        axes[3].imshow(comparison)
        axes[3].set_title("Mask: Red=Aggressive  Green=Conservative  Yellow=Both", fontsize=12)
        axes[3].axis("off")

        plt.tight_layout()
        panel_fig_path = OUT_DIR / f"{path.stem}_comparison.png"
        fig.savefig(panel_fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {panel_fig_path}")

    # Save summary report
    report_path = OUT_DIR / "report.txt"
    with open(report_path, "w") as f:
        f.write("# Agent-4: Aggressive vs Conservative Diff-Overlay Pipeline Report\n\n")
        f.write("## Parameters\n\n")
        f.write("### Aggressive\n")
        for k, v in AGGRESSIVE_PARAMS.items():
            f.write(f"  {k}={v}\n")
        f.write("\n### Conservative\n")
        for k, v in CONSERVATIVE_PARAMS.items():
            f.write(f"  {k}={v}\n")
        f.write("\n## Per-Panel Statistics\n\n")
        for path, agg, cons in zip(PANEL_PATHS, aggressive_results, conservative_results):
            agg_s = compute_overlay_stats(agg)
            cons_s = compute_overlay_stats(cons)
            f.write(f"### {path.name}\n")
            f.write(f"  Aggressive overlay:  {agg_s['overlay_ratio']*100:.1f}% | geo labels: {agg_s['unique_geo_labels']}\n")
            f.write(f"  Conservative overlay: {cons_s['overlay_ratio']*100:.1f}% | geo labels: {cons_s['unique_geo_labels']}\n")
            f.write(f"  Detail inside/outside (agg): {agg_s['avg_detail_inside']:.1f} / {agg_s['avg_detail_outside']:.1f}\n")
            f.write(f"  Detail inside/outside (cons): {cons_s['avg_detail_inside']:.1f} / {cons_s['avg_detail_outside']:.1f}\n\n")

    print(f"\nSaved report to {report_path}")
    print("\nDone! Inspect the generated images to evaluate robustness.")


if __name__ == "__main__":
    main()
