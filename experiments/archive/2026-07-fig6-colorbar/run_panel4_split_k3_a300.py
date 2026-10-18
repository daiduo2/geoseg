"""Run split_label_by_color_components on panel_4 with LAB k=3, min_area=300."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from geoseg.modules.post_process.split import split_label_by_color_components
from geoseg.modules.segment_engines._shared import _distinct_colors
from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend


BASE = Path("/Users/daiduo2/geoseg/runs/preprocess_newimage_merged/panels/panel_4")
LABELS_PATH = BASE / "labels.npz"
PANEL_PATH = BASE / "visual_audit" / "panel.png"
OUT_DIR = BASE / "visual_audit"
OUT_STEM = "split_k3_a300"
TARGET_LABEL = 3


def _mean_l2(rgb: np.ndarray, center: np.ndarray) -> float:
    """Mean RGB L2 distance from each pixel to a center color."""
    return float(np.linalg.norm(rgb.astype(np.float32) - center.astype(np.float32), axis=1).mean())


def main() -> None:
    panel_rgb = np.array(Image.open(PANEL_PATH).convert("RGB"))
    labels = np.load(LABELS_PATH)["labels"]

    split_labels = split_label_by_color_components(
        labels,
        panel_rgb,
        target_label=TARGET_LABEL,
        color_space="LAB",
        k=3,
        min_component_area=300,
        seed=42,
    )

    # ---- Metrics ----
    target_mask = labels == TARGET_LABEL
    target_pixels = panel_rgb[target_mask]
    baseline_median = np.median(target_pixels, axis=0)
    baseline_l2 = _mean_l2(target_pixels, baseline_median)

    new_label_ids = sorted({int(v) for v in np.unique(split_labels[target_mask])})
    split_per_component_l2 = []
    for lbl in new_label_ids:
        comp_mask = split_labels == lbl
        comp_pixels = panel_rgb[comp_mask]
        comp_median = np.median(comp_pixels, axis=0)
        split_per_component_l2.append(_mean_l2(comp_pixels, comp_median))
    # Weighted mean by component area so result reflects all target pixels.
    component_areas = [(split_labels == lbl).sum() for lbl in new_label_ids]
    total_area = sum(component_areas)
    split_l2 = float(
        sum(l2 * area for l2, area in zip(split_per_component_l2, component_areas)) / total_area
        if total_area else 0.0
    )

    n_final_labels = int(len({int(v) for v in np.unique(split_labels) if v != 0}))
    n_split_components = len(new_label_ids)

    metrics = {
        "target_region_baseline_l2": round(baseline_l2, 4),
        "target_region_split_l2": round(split_l2, 4),
        "n_final_labels": n_final_labels,
        "n_split_components": n_split_components,
    }

    # ---- Save label array ----
    np.save(OUT_DIR / f"{OUT_STEM}_labels.npy", split_labels)

    # ---- Overlay with legend ----
    overlay = generate_overlay_with_legend(panel_rgb, split_labels, alpha=0.65)
    Image.fromarray(overlay).save(OUT_DIR / f"{OUT_STEM}_overlay.jpg")

    # ---- Side-by-side comparison ----
    baseline_mask = np.zeros_like(panel_rgb)
    split_mask = np.full_like(panel_rgb, 32)

    colors = _distinct_colors(max(n_split_components, 1))
    for i, lbl in enumerate(new_label_ids):
        color = colors[i % len(colors)]
        baseline_mask[target_mask] = baseline_median.astype(np.uint8)
        split_mask[split_labels == lbl] = color

    # Normalize baseline mask to full color.
    baseline_view = panel_rgb.copy()
    baseline_view[target_mask] = (
        baseline_view[target_mask] * 0.35 + baseline_median.astype(np.uint8) * 0.65
    ).astype(np.uint8)

    compare = np.concatenate([panel_rgb, baseline_view, split_mask], axis=1)
    Image.fromarray(compare).save(OUT_DIR / f"{OUT_STEM}_compare.jpg")

    # ---- Save metrics JSON ----
    with open(OUT_DIR / f"{OUT_STEM}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved: {OUT_DIR / f'{OUT_STEM}_labels.npy'}")
    print(f"Saved: {OUT_DIR / f'{OUT_STEM}_overlay.jpg'}")
    print(f"Saved: {OUT_DIR / f'{OUT_STEM}_compare.jpg'}")
    print(f"Saved: {OUT_DIR / f'{OUT_STEM}_metrics.json'}")


if __name__ == "__main__":
    main()
