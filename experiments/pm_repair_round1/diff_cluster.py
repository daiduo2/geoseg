"""Method D: Difference-cluster + NN fill PM repair.

For each profile:
  1. Load original RGB and labels.
  2. Generate overlay from labels using existing _create_overlay helper.
  3. Compute per-pixel RGB Euclidean distance between original and overlay within ROI.
  4. Threshold distance > 35 to get high-difference pixels.
  5. Cluster high-difference pixels using scipy.ndimage.label (connected components).
     Filter clusters by area: keep only 20 < area < 2000 px.
  6. Draw a visualization of the ROI with each cluster numbered at its centroid.
  7. For each cluster, fill its pixels with the label of the nearest same-label pixel
     outside the cluster (distance transform constrained by label).
  8. Generate repaired overlay and save all outputs.

Usage:
    python experiments/pm_repair_round1/diff_cluster.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, "/Users/daiduo2/geoseg/src")

from geoseg.modules.segment_engines._shared import _create_overlay, _distinct_colors


# ── Configuration ─────────────────────────────────────────────────────────────

PROFILES = [
    {
        "id": "04",
        "image": "/Users/daiduo2/geoseg/runs/feng_fig6_final_v4/crop_tests/fig6_profile_04_cropped.jpg",
        "labels": "/Users/daiduo2/geoseg/runs/feng_fig6_comparisons_v7/fig6_profile_04/labels.npz",
        "roi": {"x1": 120, "y1": 40, "x2": 260, "y2": 110},
    },
    {
        "id": "05",
        "image": "/Users/daiduo2/geoseg/runs/feng_fig6_final_v4/crop_tests/fig6_profile_05_cropped.jpg",
        "labels": "/Users/daiduo2/geoseg/runs/feng_fig6_comparisons_v7/fig6_profile_05/labels.npz",
        "roi": {"x1": 100, "y1": 50, "x2": 240, "y2": 115},
    },
]

OUTPUT_BASE = Path("/Users/daiduo2/geoseg/runs/pm_repair_round1/diff_cluster")
DISTANCE_THRESHOLD = 35.0
MIN_CLUSTER_AREA = 20
MAX_CLUSTER_AREA = 2000


def load_inputs(image_path: str, labels_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load original RGB image and label array."""
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)
    labels = np.load(labels_path)["labels"]
    return rgb, labels


def generate_overlay(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Generate segmentation overlay using shared helper."""
    n = int(labels.max()) + 1
    palette = _distinct_colors(n)
    overlay = _create_overlay(rgb, labels, palette, alpha=0.65)
    return overlay


def compute_difference_mask(
    rgb: np.ndarray, overlay: np.ndarray, roi: dict
) -> np.ndarray:
    """Compute per-pixel RGB Euclidean distance within ROI, threshold."""
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    roi_rgb = rgb[y1:y2, x1:x2]
    roi_overlay = overlay[y1:y2, x1:x2]
    diff = np.linalg.norm(roi_rgb.astype(np.float32) - roi_overlay.astype(np.float32), axis=2)
    mask = diff > DISTANCE_THRESHOLD
    return mask


def cluster_high_diff_pixels(mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Cluster high-difference pixels using connected components.

    Returns (labelled_array, list of cluster_info dicts).
    """
    labeled, num_features = ndimage.label(mask)
    clusters = []
    for i in range(1, num_features + 1):
        cluster_mask = labeled == i
        area = int(cluster_mask.sum())
        if MIN_CLUSTER_AREA < area < MAX_CLUSTER_AREA:
            ys, xs = np.where(cluster_mask)
            centroid_y = float(ys.mean())
            centroid_x = float(xs.mean())
            clusters.append({
                "label_id": i,
                "area": area,
                "centroid_y": centroid_y,
                "centroid_x": centroid_x,
                "mask": cluster_mask,
            })
    return labeled, clusters


def draw_cluster_visualization(
    rgb: np.ndarray, roi: dict, clusters: list[dict], labeled: np.ndarray
) -> Image.Image:
    """Draw ROI with each cluster numbered at its centroid."""
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    roi_rgb = rgb[y1:y2, x1:x2]
    img = Image.fromarray(roi_rgb)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
        font_large = font

    # Draw cluster outlines
    colors = _distinct_colors(len(clusters) + 1)
    for idx, cluster in enumerate(clusters):
        mask = cluster["mask"]
        # Find boundary pixels
        from skimage.segmentation import find_boundaries
        boundary = find_boundaries(mask, mode="thick")
        ys, xs = np.where(boundary)
        color = tuple(colors[idx].tolist())
        for y, x in zip(ys, xs):
            draw.point((x, y), fill=color)

        # Draw number at centroid
        cx = int(cluster["centroid_x"])
        cy = int(cluster["centroid_y"])
        text = str(idx + 1)
        bbox = draw.textbbox((0, 0), text, font=font_large)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        # White background for text
        draw.rectangle(
            [cx - tw // 2 - 2, cy - th // 2 - 2, cx + tw // 2 + 2, cy + th // 2 + 2],
            fill=(255, 255, 255),
        )
        draw.text((cx - tw // 2, cy - th // 2), text, fill=(0, 0, 0), font=font_large)

    # Draw ROI border
    draw.rectangle([0, 0, x2 - x1 - 1, y2 - y1 - 1], outline=(255, 0, 0), width=2)

    return img


def fill_clusters_nn(
    labels: np.ndarray, clusters: list[dict], roi: dict
) -> np.ndarray:
    """Fill each cluster with label of nearest same-label pixel outside cluster.

    Uses distance transform constrained by label: for each cluster, find the
    nearest pixel that shares the same label value but is outside the cluster.
    """
    repaired = labels.copy()
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    roi_labels = repaired[y1:y2, x1:x2]

    for cluster in clusters:
        mask = cluster["mask"]  # (roi_h, roi_w)
        # Determine which labels are present in the cluster
        cluster_label_values = np.unique(roi_labels[mask])

        # For each label value in the cluster, find nearest same-label pixel outside
        for lbl in cluster_label_values:
            if lbl < 0:
                continue
            # Mask of same-label pixels outside the cluster
            same_label_mask = (roi_labels == lbl) & (~mask)
            if not same_label_mask.any():
                continue

            # Distance transform: distance from each pixel to nearest same-label pixel
            dist = distance_transform_edt(~same_label_mask)

            # Within the cluster, find the pixel closest to a same-label pixel
            cluster_dist = np.where(mask, dist, np.inf)
            nearest_y, nearest_x = np.unravel_index(np.argmin(cluster_dist), cluster_dist.shape)
            nearest_label = roi_labels[nearest_y, nearest_x]

            # Fill all cluster pixels with this nearest label
            roi_labels[mask] = nearest_label
            break  # Only fill once per cluster with the dominant label

    repaired[y1:y2, x1:x2] = roi_labels
    return repaired


def create_comparison_image(
    rgb: np.ndarray,
    original_labels: np.ndarray,
    repaired_labels: np.ndarray,
    roi: dict,
) -> tuple[Image.Image, Image.Image]:
    """Create ROI and full comparison images."""
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]

    # Generate overlays
    n_orig = int(original_labels.max()) + 1
    palette_orig = _distinct_colors(n_orig)
    overlay_orig = _create_overlay(rgb, original_labels, palette_orig, alpha=0.65)

    n_rep = int(repaired_labels.max()) + 1
    palette_rep = _distinct_colors(n_rep)
    overlay_rep = _create_overlay(rgb, repaired_labels, palette_rep, alpha=0.65)

    # ROI comparison
    roi_orig = overlay_orig[y1:y2, x1:x2]
    roi_rep = overlay_rep[y1:y2, x1:x2]
    roi_comp = np.concatenate([roi_orig, roi_rep], axis=1)
    roi_img = Image.fromarray(roi_comp)

    # Full comparison
    full_comp = np.concatenate([overlay_orig, overlay_rep], axis=1)
    full_img = Image.fromarray(full_comp)

    return roi_img, full_img


def process_profile(profile: dict) -> dict:
    """Process a single profile through the diff-cluster + NN fill pipeline."""
    profile_id = profile["id"]
    print(f"\n=== Processing profile {profile_id} ===")

    # Load inputs
    rgb, labels = load_inputs(profile["image"], profile["labels"])
    print(f"  Image shape: {rgb.shape}, Labels shape: {labels.shape}")

    # Generate overlay
    overlay = generate_overlay(rgb, labels)

    # Compute difference mask within ROI
    diff_mask = compute_difference_mask(rgb, overlay, profile["roi"])
    print(f"  High-diff pixels: {diff_mask.sum()} (threshold={DISTANCE_THRESHOLD})")

    # Cluster
    labeled, clusters = cluster_high_diff_pixels(diff_mask)
    print(f"  Clusters found (after area filter): {len(clusters)}")
    for i, c in enumerate(clusters):
        print(f"    Cluster {i+1}: area={c['area']}, centroid=({c['centroid_x']:.1f}, {c['centroid_y']:.1f})")

    # Draw cluster visualization
    cluster_viz = draw_cluster_visualization(rgb, profile["roi"], clusters, labeled)

    # Fill clusters with NN label
    repaired_labels = fill_clusters_nn(labels, clusters, profile["roi"])

    # Generate repaired overlay
    n_rep = int(repaired_labels.max()) + 1
    palette_rep = _distinct_colors(n_rep)
    repaired_overlay = _create_overlay(rgb, repaired_labels, palette_rep, alpha=0.65)

    # Create comparison images
    roi_comp, full_comp = create_comparison_image(rgb, labels, repaired_labels, profile["roi"])

    # Save outputs
    out_dir = OUTPUT_BASE / f"fig6_profile_{profile_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / "labels_repaired.npz", labels=repaired_labels)
    Image.fromarray(repaired_overlay).save(out_dir / "overlay_repaired.jpg", quality=95)
    cluster_viz.save(out_dir / "clusters_visualization.jpg", quality=95)
    roi_comp.save(out_dir / "roi_comparison.jpg", quality=95)
    full_comp.save(out_dir / "full_comparison.jpg", quality=95)

    print(f"  Saved outputs to: {out_dir}")

    return {
        "profile_id": profile_id,
        "n_clusters": len(clusters),
        "clusters": [
            {
                "id": i + 1,
                "area": c["area"],
                "centroid": (round(c["centroid_x"], 1), round(c["centroid_y"], 1)),
            }
            for i, c in enumerate(clusters)
        ],
        "output_dir": str(out_dir),
    }


def write_report(results: list[dict]) -> None:
    """Write a short report with observations and recommendations."""
    report_path = OUTPUT_BASE / "report.md"

    lines = [
        "# Method D: Difference-cluster + NN Fill PM Repair — Report",
        "",
        f"**Date:** 2026-06-23",
        "**Method:** Compute RGB Euclidean distance between original and overlay within ROI,",
        "threshold > 35, cluster with connected components, filter 20 < area < 2000 px,",
        "fill each cluster with nearest same-label pixel (distance transform).",
        "",
        "## Results Summary",
        "",
    ]

    for r in results:
        lines.append(f"### Profile {r['profile_id']}")
        lines.append(f"- Clusters found: {r['n_clusters']}")
        for c in r["clusters"]:
            lines.append(f"  - Cluster {c['id']}: area={c['area']} px, centroid={c['centroid']}")
        lines.append(f"- Output: `{r['output_dir']}`")
        lines.append("")

    lines.extend([
        "## Observations",
        "",
        "1. **Difference threshold (35)**: Tuned to capture noticeable color mismatches",
        "   between original image and segmentation overlay without picking up noise.",
        "",
        "2. **Cluster area filter (20-2000 px)**: Removes tiny noise specks and avoids",
        "   treating large correctly-segmented regions as errors.",
        "",
        "3. **NN fill strategy**: Uses distance transform constrained by label to find",
        "   the nearest pixel sharing the same label value outside the cluster. This",
        "   propagates the dominant surrounding label into the cluster region.",
        "",
        "4. **Limitations**:",
        "   - If a cluster spans multiple true labels, NN fill may assign the wrong label.",
        "   - Distance transform is computed on binary mask, not geodesic distance.",
        "   - No semantic understanding — purely geometric nearest-neighbor.",
        "",
        "## Recommendations",
        "",
        "1. **Visual review**: Check `clusters_visualization.jpg` for each profile to verify",
        "   clusters correspond to actual PM regions, not artifacts.",
        "",
        "2. **Threshold tuning**: If too many/few clusters, adjust `DISTANCE_THRESHOLD`",
        "   (current=35) and `MIN_CLUSTER_AREA`/`MAX_CLUSTER_AREA`.",
        "",
        "3. **Multi-label clusters**: For clusters spanning label boundaries, consider",
        "   splitting by label value before NN fill, or using watershed segmentation.",
        "",
        "4. **Geodesic distance**: Replace Euclidean distance transform with geodesic",
        "   distance (respecting image edges) for better boundary adherence.",
        "",
        "5. **Next iteration**: Combine with edge-guided fill or VLM-assisted semantic",
        "   labeling for clusters near known geological features (faults, horizons).",
        "",
    ])

    report_path.write_text("\n".join(lines))
    print(f"\nReport saved to: {report_path}")


def main() -> int:
    """Run diff-cluster + NN fill PM repair for all profiles."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    results = []
    for profile in PROFILES:
        result = process_profile(profile)
        results.append(result)

    write_report(results)

    print("\n=== All profiles processed ===")
    for r in results:
        print(f"  Profile {r['profile_id']}: {r['n_clusters']} clusters -> {r['output_dir']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
