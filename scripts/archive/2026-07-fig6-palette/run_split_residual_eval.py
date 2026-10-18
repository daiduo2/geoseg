"""Run split_label_by_color_components with mask-based residual evaluation.

Follows the fig6 mask-based residual route:
1. Build baseline mask where each label is colored by median RGB.
2. Build split mask where each new component is colored by median RGB.
3. Compute abs RGB diff against original panel for both masks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from geoseg.modules.post_process.split import split_label_by_color_components


def _median_color_mask(labels: np.ndarray, img_rgb: np.ndarray) -> np.ndarray:
    """Return RGB mask where each label is filled with its median RGB color."""
    mask = np.zeros_like(img_rgb)
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        if lab == 0:
            continue
        region = labels == lab
        if not region.any():
            continue
        median_rgb = np.median(img_rgb[region], axis=0).astype(np.uint8)
        mask[region] = median_rgb
    return mask


def _mean_rgb_l2(mask: np.ndarray, original: np.ndarray) -> float:
    """Mean RGB L2 norm over the whole panel."""
    diff = mask.astype(np.float32) - original.astype(np.float32)
    l2 = np.sqrt(np.sum(diff ** 2, axis=2))
    return float(np.mean(l2))


def _save_image(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=95)


def _build_grid(
    original: np.ndarray,
    baseline_mask: np.ndarray,
    split_mask: np.ndarray,
    baseline_diff: np.ndarray,
    split_diff: np.ndarray,
) -> np.ndarray:
    """Build 1x5 comparison grid."""
    import cv2

    # Convert diffs to uint8 heatmaps for visualization.
    def _to_u8(x: np.ndarray) -> np.ndarray:
        if x.dtype == np.uint8:
            return x
        xmax = x.max()
        if xmax > 0:
            return np.clip(x / xmax * 255, 0, 255).astype(np.uint8)
        return np.zeros_like(x, dtype=np.uint8)

    def _gray3(x: np.ndarray) -> np.ndarray:
        g = _to_u8(x)
        return np.stack([g, g, g], axis=2)

    tiles = [
        original,
        baseline_mask,
        split_mask,
        _gray3(baseline_diff.mean(axis=2)),
        _gray3(split_diff.mean(axis=2)),
    ]

    h, w = original.shape[:2]
    grid = np.zeros((h, w * 5, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        grid[:, i * w : (i + 1) * w] = tile

    # Add titles.
    titles = ["original", "baseline mask", "split mask", "baseline diff", "split diff"]
    for i, title in enumerate(titles):
        cv2.putText(
            grid,
            title,
            (i * w + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    return grid


def main() -> None:
    base = Path("/Users/daiduo2/geoseg/runs/preprocess_newimage_merged/panels/panel_0")
    labels_path = base / "labels.npz"
    img_path = base / "visual_audit" / "panel.png"
    out_dir = base / "visual_audit"

    labels = np.load(labels_path)["labels"]
    img_rgb = np.array(Image.open(img_path).convert("RGB"))

    target_label = 2
    color_space = "LAB"
    k = 3
    min_component_area = 300

    # Baseline median-color mask (before split).
    baseline_mask = _median_color_mask(labels, img_rgb)
    baseline_diff = np.abs(baseline_mask.astype(np.float32) - img_rgb.astype(np.float32))
    baseline_l2 = _mean_rgb_l2(baseline_mask, img_rgb)

    # Run split utility.
    split_labels = split_label_by_color_components(
        labels,
        img_rgb,
        target_label=target_label,
        color_space=color_space,
        k=k,
        min_component_area=min_component_area,
        seed=42,
    )

    # Split median-color mask.
    split_mask = _median_color_mask(split_labels, img_rgb)
    split_diff = np.abs(split_mask.astype(np.float32) - img_rgb.astype(np.float32))
    split_l2 = _mean_rgb_l2(split_mask, img_rgb)

    # Metrics.
    n_final_labels = len(set(np.unique(split_labels)) - {0})
    n_split_components = len(set(np.unique(split_labels)) - set(np.unique(labels)) - {0})

    metrics = {
        "baseline_l2": baseline_l2,
        "split_l2": split_l2,
        "l2_reduction": baseline_l2 - split_l2,
        "n_final_labels": n_final_labels,
        "n_split_components": n_split_components,
    }

    # Save outputs.
    np.save(out_dir / "split_k3_a300_labels.npy", split_labels)
    _save_image(baseline_mask, out_dir / "split_k3_a300_mask_baseline.jpg")
    _save_image(split_mask, out_dir / "split_k3_a300_mask_split.jpg")
    _save_image(baseline_diff.astype(np.uint8), out_dir / "split_k3_a300_diff_baseline.jpg")
    _save_image(split_diff.astype(np.uint8), out_dir / "split_k3_a300_diff_split.jpg")

    grid = _build_grid(img_rgb, baseline_mask, split_mask, baseline_diff, split_diff)
    _save_image(grid, out_dir / "split_k3_a300_grid.jpg")

    with open(out_dir / "split_k3_a300_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
