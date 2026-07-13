"""Run split_label_by_color_components on panel_3 label 1 and generate fig6 residual outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from geoseg.modules.post_process.split import split_label_by_color_components


def build_median_color_mask(labels: np.ndarray, img_rgb: np.ndarray) -> np.ndarray:
    """Color each label by its median RGB in the original image."""
    mask = np.zeros_like(img_rgb)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        region = labels == label_id
        if not region.any():
            continue
        median_color = np.median(img_rgb[region], axis=0).astype(np.uint8)
        mask[region] = median_color
    return mask


def mean_rgb_l2(mask: np.ndarray, img_rgb: np.ndarray) -> float:
    """Mean RGB L2 distance over the whole panel."""
    diff = mask.astype(np.float32) - img_rgb.astype(np.float32)
    l2 = np.linalg.norm(diff, axis=2)
    return float(np.mean(l2))


def save_diff_image(diff: np.ndarray, path: Path) -> None:
    """Save absolute RGB difference scaled to uint8."""
    scaled = np.clip(diff * 2, 0, 255).astype(np.uint8)
    Image.fromarray(scaled).save(path)


def build_comparison_grid(
    img_rgb: np.ndarray,
    baseline_mask: np.ndarray,
    split_mask: np.ndarray,
    baseline_diff: np.ndarray,
    split_diff: np.ndarray,
) -> np.ndarray:
    """Build horizontal grid: original | baseline mask | split mask | baseline diff | split diff."""
    row_height = img_rgb.shape[0]
    baseline_diff_vis = np.clip(baseline_diff * 2, 0, 255).astype(np.uint8)
    split_diff_vis = np.clip(split_diff * 2, 0, 255).astype(np.uint8)
    return np.concatenate(
        [img_rgb, baseline_mask, split_mask, baseline_diff_vis, split_diff_vis],
        axis=1,
    )


def main() -> None:
    panel_dir = ROOT / "runs/preprocess_newimage_merged/panels/panel_3"
    out_dir = panel_dir / "visual_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_path = panel_dir / "labels.npz"
    img_path = out_dir / "panel.png"

    labels = np.load(labels_path)["labels"]
    img_rgb = np.array(Image.open(img_path).convert("RGB"))

    target_label = 1
    color_space = "LAB"
    k = 3
    min_component_area = 300

    split_labels = split_label_by_color_components(
        labels,
        img_rgb,
        target_label,
        color_space=color_space,
        k=k,
        min_component_area=min_component_area,
        seed=42,
    )

    baseline_mask = build_median_color_mask(labels, img_rgb)
    split_mask = build_median_color_mask(split_labels, img_rgb)

    baseline_diff = np.abs(baseline_mask.astype(np.float32) - img_rgb.astype(np.float32))
    split_diff = np.abs(split_mask.astype(np.float32) - img_rgb.astype(np.float32))

    np.save(out_dir / "split_k3_a300_labels.npy", split_labels)
    Image.fromarray(baseline_mask).save(out_dir / "split_k3_a300_mask_baseline.jpg")
    Image.fromarray(split_mask).save(out_dir / "split_k3_a300_mask_split.jpg")
    save_diff_image(baseline_diff, out_dir / "split_k3_a300_diff_baseline.jpg")
    save_diff_image(split_diff, out_dir / "split_k3_a300_diff_split.jpg")

    grid = build_comparison_grid(img_rgb, baseline_mask, split_mask, baseline_diff, split_diff)
    Image.fromarray(grid).save(out_dir / "split_k3_a300_grid.jpg")

    n_final_labels = int(len(np.unique(split_labels)) - (1 if 0 in np.unique(split_labels) else 0))
    n_split_components = int(len(np.unique(split_labels[labels == target_label])))

    metrics = {
        "baseline_l2": mean_rgb_l2(baseline_mask, img_rgb),
        "split_l2": mean_rgb_l2(split_mask, img_rgb),
        "l2_reduction": mean_rgb_l2(baseline_mask, img_rgb) - mean_rgb_l2(split_mask, img_rgb),
        "n_final_labels": n_final_labels,
        "n_split_components": n_split_components,
    }

    with open(out_dir / "split_k3_a300_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
