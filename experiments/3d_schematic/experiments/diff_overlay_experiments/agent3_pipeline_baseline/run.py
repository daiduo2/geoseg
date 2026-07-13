"""
Agent-3: 差分叠层完整 pipeline 基线实验
对比 diff_overlay_pipeline vs v3 pipeline
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.segmentation import felzenszwalb

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from design_diff_overlay import diff_overlay_pipeline, render_label_fill as render_diff_fill
from process_final_v3 import process_panel, render_label_fill as render_v3_fill, draw_boundaries, post_merge


def run_v3_pipeline(image_path: Path) -> dict:
    """Run v3 pipeline on a single panel."""
    return process_panel(image_path)


def run_diff_overlay_pipeline(image_path: Path, params: dict) -> dict:
    """Run diff-overlay pipeline on a single panel, with post-merge on geo labels."""
    img = np.array(Image.open(image_path).convert("RGB"))
    result = diff_overlay_pipeline(img, **params)

    overlay_mask = result["overlay_mask"]
    overlay_label = params.get("overlay_label", -1)

    # Apply post_merge to geo_labels (non-overlay region) before merging
    geo_labels = post_merge(result["geo_labels"], img)
    print(f"  geo labels after post_merge: {len(np.unique(geo_labels))}")

    # Re-merge: overlay_mask gets overlay_label, rest gets merged geo_labels
    final_labels = geo_labels.copy()
    final_labels[overlay_mask] = overlay_label

    # Render label fill with overlay in gray
    fill = render_diff_fill(final_labels, overlay_label)

    # Draw boundaries
    boundaries = draw_boundaries(img, final_labels)

    return {
        "original": img,
        "labels": final_labels,
        "fill": fill,
        "boundaries": boundaries,
        "overlay_mask": overlay_mask,
        "overlay_only": result["overlay_only"],
    }


def create_comparison_figure(v3_results: list[dict], diff_results: list[dict]) -> np.ndarray:
    """Create side-by-side comparison figure.

    Layout per panel (row):
        Col 1: Original
        Col 2: v3 Label Fill
        Col 3: Diff-Overlay Label Fill (overlay = gray)
        Col 4: v3 Boundaries
        Col 5: Diff-Overlay Boundaries
        Col 6: Diff-Overlay Overlay Visualization (magenta)
    """
    n = len(v3_results)
    assert n == len(diff_results)
    h, w = v3_results[0]["original"].shape[:2]

    cols = [
        "Original",
        "v3 Label Fill",
        "Diff Label Fill",
        "v3 Boundaries",
        "Diff Boundaries",
        "Diff Overlay",
    ]
    n_cols = len(cols)
    header_h = 40
    cell_h, cell_w = h, w

    canvas = np.ones((n * cell_h + header_h, n_cols * cell_w, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Headers
    for c, title in enumerate(cols):
        x = c * cell_w + cell_w // 2 - len(title) * 5
        cv2.putText(canvas, title, (x, 28), font, 0.55, (0, 0, 0), 2)

    row_titles = ["Panel 1", "Panel 2", "Panel 3"]
    for r in range(n):
        y = header_h + r * cell_h
        cv2.putText(canvas, row_titles[r], (10, y + 25), font, 0.6, (0, 0, 0), 2)

        v3 = v3_results[r]
        diff = diff_results[r]

        # Col 1: Original
        canvas[y:y + cell_h, 0 * cell_w:1 * cell_w] = v3["original"]
        # Col 2: v3 Label Fill
        canvas[y:y + cell_h, 1 * cell_w:2 * cell_w] = v3["fill"]
        # Col 3: Diff Label Fill
        canvas[y:y + cell_h, 2 * cell_w:3 * cell_w] = diff["fill"]
        # Col 4: v3 Boundaries
        canvas[y:y + cell_h, 3 * cell_w:4 * cell_w] = v3["boundaries"]
        # Col 5: Diff Boundaries
        canvas[y:y + cell_h, 4 * cell_w:5 * cell_w] = diff["boundaries"]
        # Col 6: Diff Overlay
        canvas[y:y + cell_h, 5 * cell_w:6 * cell_w] = diff["overlay_only"]

    return canvas


def create_zoom_comparison(v3_result: dict, diff_result: dict, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    """Create zoomed-in comparison for a specific region.

    crop_box: (x1, y1, x2, y2)
    Layout: 2 rows x 4 cols
        Row 1: v3 Original, v3 Fill, v3 Boundaries, v3 Labels
        Row 2: Diff Original, Diff Fill, Diff Boundaries, Diff Overlay
    """
    x1, y1, x2, y2 = crop_box
    h_crop = y2 - y1
    w_crop = x2 - x1

    v3_orig = v3_result["original"][y1:y2, x1:x2]
    v3_fill = v3_result["fill"][y1:y2, x1:x2]
    v3_bound = v3_result["boundaries"][y1:y2, x1:x2]

    diff_orig = diff_result["original"][y1:y2, x1:x2]
    diff_fill = diff_result["fill"][y1:y2, x1:x2]
    diff_bound = diff_result["boundaries"][y1:y2, x1:x2]
    diff_overlay = diff_result["overlay_only"][y1:y2, x1:x2]

    # Add label numbers to v3 labels for clarity
    v3_labels_crop = v3_result["labels"][y1:y2, x1:x2]
    diff_labels_crop = diff_result["labels"][y1:y2, x1:x2]

    # Create label number overlays
    v3_label_vis = v3_fill.copy()
    diff_label_vis = diff_fill.copy()

    # Build composite
    n_rows, n_cols = 2, 4
    header_h = 35
    cell_h, cell_w = h_crop, w_crop

    canvas = np.ones((n_rows * cell_h + header_h, n_cols * cell_w, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    row_titles = ["v3 Pipeline", "Diff-Overlay Pipeline"]
    col_titles = ["Original", "Label Fill", "Boundaries", "Overlay/Labels"]

    for c, title in enumerate(col_titles):
        x = c * cell_w + cell_w // 2 - len(title) * 4
        cv2.putText(canvas, title, (x, 25), font, 0.5, (0, 0, 0), 1)

    for r, title in enumerate(row_titles):
        y = header_h + r * cell_h
        cv2.putText(canvas, title, (10, y + 20), font, 0.5, (0, 0, 0), 1)

    # Row 0: v3
    y0 = header_h
    canvas[y0:y0 + cell_h, 0 * cell_w:1 * cell_w] = v3_orig
    canvas[y0:y0 + cell_h, 1 * cell_w:2 * cell_w] = v3_fill
    canvas[y0:y0 + cell_h, 2 * cell_w:3 * cell_w] = v3_bound
    canvas[y0:y0 + cell_h, 3 * cell_w:4 * cell_w] = v3_label_vis

    # Row 1: diff
    y1_off = header_h + cell_h
    canvas[y1_off:y1_off + cell_h, 0 * cell_w:1 * cell_w] = diff_orig
    canvas[y1_off:y1_off + cell_h, 1 * cell_w:2 * cell_w] = diff_fill
    canvas[y1_off:y1_off + cell_h, 2 * cell_w:3 * cell_w] = diff_bound
    canvas[y1_off:y1_off + cell_h, 3 * cell_w:4 * cell_w] = diff_overlay

    return canvas


def analyze_text_regions(v3_result: dict, diff_result: dict) -> dict:
    """Analyze text region behavior in both pipelines."""
    v3_labels = v3_result["labels"]
    diff_labels = diff_result["labels"]
    overlay_mask = diff_result["overlay_mask"]

    v3_unique = len(np.unique(v3_labels))
    diff_unique = len(np.unique(diff_labels))

    # Count small regions in v3 (potential text artifacts)
    v3_counts = np.bincount(v3_labels.flatten())
    total_pixels = v3_labels.size
    small_regions = np.sum((v3_counts > 0) & (v3_counts < total_pixels * 0.01))

    overlay_pixels = int(overlay_mask.sum())
    overlay_ratio = overlay_pixels / total_pixels * 100

    return {
        "v3_label_count": v3_unique,
        "diff_label_count": diff_unique,
        "v3_small_regions": int(small_regions),
        "overlay_pixels": overlay_pixels,
        "overlay_ratio_pct": overlay_ratio,
    }


def main():
    base = Path("/Users/daiduo2/geoseg/src/3d_schematic")
    out_dir = base / "diff_overlay_experiments" / "agent3_pipeline_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_paths = [base / f"panel_{i}_front.png" for i in range(1, 4)]

    # Default recommended parameters
    diff_params = {
        "blur_ksize": 15,
        "blur_sigma": 3.0,
        "diff_thresh": 20.0,
        "expand_radius": 15,
        "felz_scale": 300.0,
        "felz_sigma": 0.5,
        "overlay_label": -1,
    }

    print("=" * 60)
    print("Agent-3: Diff-Overlay Pipeline Baseline Experiment")
    print("=" * 60)

    v3_results = []
    diff_results = []
    analyses = []

    for i, p in enumerate(panel_paths):
        print(f"\n--- Panel {i + 1}: {p.name} ---")

        # v3 pipeline
        print("  Running v3 pipeline...")
        v3_res = run_v3_pipeline(p)
        v3_results.append(v3_res)

        # diff-overlay pipeline
        print("  Running diff-overlay pipeline...")
        diff_res = run_diff_overlay_pipeline(p, diff_params)
        diff_results.append(diff_res)

        # Analysis
        analysis = analyze_text_regions(v3_res, diff_res)
        analyses.append(analysis)
        print(f"  v3 labels: {analysis['v3_label_count']}, diff labels: {analysis['diff_label_count']}")
        print(f"  v3 small regions (<1%): {analysis['v3_small_regions']}")
        print(f"  overlay coverage: {analysis['overlay_ratio_pct']:.1f}%")

    # Save individual results
    for i, (v3, diff) in enumerate(zip(v3_results, diff_results)):
        panel_name = f"panel_{i + 1}"
        Image.fromarray(v3["fill"]).save(out_dir / f"{panel_name}_v3_fill.png")
        Image.fromarray(v3["boundaries"]).save(out_dir / f"{panel_name}_v3_boundaries.png")
        Image.fromarray(diff["fill"]).save(out_dir / f"{panel_name}_diff_fill.png")
        Image.fromarray(diff["boundaries"]).save(out_dir / f"{panel_name}_diff_boundaries.png")
        Image.fromarray(diff["overlay_only"]).save(out_dir / f"{panel_name}_diff_overlay.png")

    # Create full comparison figure
    print("\nGenerating full comparison figure...")
    comparison = create_comparison_figure(v3_results, diff_results)
    comparison_path = out_dir / "comparison_full.png"
    Image.fromarray(comparison).save(comparison_path)
    print(f"  Saved: {comparison_path}")

    # Create zoom comparison for Panel 3 (most text)
    print("\nGenerating Panel 3 zoom comparison...")
    h, w = v3_results[2]["original"].shape[:2]
    # Focus on left-middle text region: "refractory, Mg-rich peridotite residues"
    crop = (w // 12, h // 5, 7 * w // 12, 4 * h // 5)
    zoom = create_zoom_comparison(v3_results[2], diff_results[2], crop)
    zoom_path = out_dir / "comparison_zoom_panel3.png"
    Image.fromarray(zoom).save(zoom_path)
    print(f"  Saved: {zoom_path}")

    # Save analysis report
    report_path = out_dir / "analysis_report.txt"
    with open(report_path, "w") as f:
        f.write("Diff-Overlay Pipeline Baseline Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write("Parameters:\n")
        for k, v in diff_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        for i, a in enumerate(analyses):
            f.write(f"Panel {i + 1}:\n")
            f.write(f"  v3 label count: {a['v3_label_count']}\n")
            f.write(f"  diff label count: {a['diff_label_count']}\n")
            f.write(f"  v3 small regions (<1% area): {a['v3_small_regions']}\n")
            f.write(f"  overlay coverage: {a['overlay_ratio_pct']:.1f}%\n")
            f.write("\n")
    print(f"  Saved: {report_path}")

    print("\n" + "=" * 60)
    print("Experiment complete. Results saved to:")
    print(f"  {out_dir}")
    print("=" * 60)

    return v3_results, diff_results, analyses


if __name__ == "__main__":
    main()
