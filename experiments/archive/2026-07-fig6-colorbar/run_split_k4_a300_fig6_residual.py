"""Fig6 mask-based residual evaluation for split_label_by_color_components.

Builds median-color masks over the WHOLE PANEL and computes abs RGB diff
against the original panel.  Does NOT use overlay-with-legend.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from geoseg.modules.post_process.split import split_label_by_color_components


BASE = Path("/Users/daiduo2/geoseg/runs/preprocess_newimage_merged/panels/panel_2")
LABELS_PATH = BASE / "labels.npz"
IMAGE_PATH = BASE / "visual_audit" / "panel.png"
OUT_DIR = BASE / "visual_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LABEL = 1
COLOR_SPACE = "LAB"
K = 4
MIN_COMPONENT_AREA = 300
STEM = "split_k4_a300"


def load_labels(p: Path) -> np.ndarray:
    data = np.load(p)
    keys = list(data.keys())
    if "labels" in keys:
        return data["labels"]
    return data[keys[0]]


def build_median_color_mask(labels: np.ndarray, img_rgb: np.ndarray) -> np.ndarray:
    """Color every label by its median RGB in the original image."""
    mask = np.zeros_like(img_rgb)
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        region = labels == lbl
        median_color = np.median(img_rgb[region], axis=0).astype(np.uint8)
        mask[region] = median_color
    return mask


def abs_rgb_diff(mask_rgb: np.ndarray, img_rgb: np.ndarray) -> np.ndarray:
    """Absolute per-channel RGB difference, scaled to uint8."""
    diff = np.abs(mask_rgb.astype(np.float32) - img_rgb.astype(np.float32))
    return diff.astype(np.uint8)


def mean_rgb_l2_whole(mask_rgb: np.ndarray, img_rgb: np.ndarray) -> float:
    """Mean RGB L2 distance over the whole panel."""
    diff = mask_rgb.astype(np.float32) - img_rgb.astype(np.float32)
    return float(np.mean(np.linalg.norm(diff, axis=2)))


def build_comparison_grid(
    img_rgb: np.ndarray,
    baseline_mask: np.ndarray,
    split_mask: np.ndarray,
    baseline_diff: np.ndarray,
    split_diff: np.ndarray,
) -> np.ndarray:
    """Horizontal grid: original | baseline mask | split mask | baseline diff | split diff."""
    images = [img_rgb, baseline_mask, split_mask, baseline_diff, split_diff]
    h = img_rgb.shape[0]
    # Ensure all images have the same height and 3 channels.
    normalized = []
    for im in images:
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=2)
        normalized.append(im[:h])
    return np.concatenate(normalized, axis=1)


def main() -> None:
    labels = load_labels(LABELS_PATH)
    img_rgb = np.array(Image.open(IMAGE_PATH).convert("RGB"))

    if labels.shape[:2] != img_rgb.shape[:2]:
        raise ValueError(f"Shape mismatch: labels {labels.shape}, image {img_rgb.shape}")

    split_labels = split_label_by_color_components(
        labels,
        img_rgb,
        TARGET_LABEL,
        color_space=COLOR_SPACE,
        k=K,
        min_component_area=MIN_COMPONENT_AREA,
        seed=42,
    )

    # Median-color masks over the whole panel.
    baseline_mask = build_median_color_mask(labels, img_rgb)
    split_mask = build_median_color_mask(split_labels, img_rgb)

    # Absolute RGB diff images.
    baseline_diff = abs_rgb_diff(baseline_mask, img_rgb)
    split_diff = abs_rgb_diff(split_mask, img_rgb)

    # Metrics over the whole panel.
    baseline_l2 = mean_rgb_l2_whole(baseline_mask, img_rgb)
    split_l2 = mean_rgb_l2_whole(split_mask, img_rgb)
    l2_reduction = baseline_l2 - split_l2

    n_final_labels = int(len({int(v) for v in np.unique(split_labels) if v != 0}))
    n_split_components = int(len({int(v) for v in np.unique(split_labels[labels == TARGET_LABEL]) if v != 0}))

    # Save artifacts.
    np.save(OUT_DIR / f"{STEM}_labels.npy", split_labels)
    Image.fromarray(baseline_mask).save(OUT_DIR / f"{STEM}_mask_baseline.jpg", quality=95)
    Image.fromarray(split_mask).save(OUT_DIR / f"{STEM}_mask_split.jpg", quality=95)
    Image.fromarray(baseline_diff).save(OUT_DIR / f"{STEM}_diff_baseline.jpg", quality=95)
    Image.fromarray(split_diff).save(OUT_DIR / f"{STEM}_diff_split.jpg", quality=95)

    grid = build_comparison_grid(img_rgb, baseline_mask, split_mask, baseline_diff, split_diff)
    Image.fromarray(grid).save(OUT_DIR / f"{STEM}_grid.jpg", quality=95)

    metrics = {
        "panel_id": 2,
        "k": K,
        "min_area": MIN_COMPONENT_AREA,
        "baseline_l2": round(baseline_l2, 4),
        "split_l2": round(split_l2, 4),
        "l2_reduction": round(l2_reduction, 4),
        "n_final_labels": n_final_labels,
        "n_split_components": n_split_components,
        "output_mask_baseline": str(OUT_DIR / f"{STEM}_mask_baseline.jpg"),
        "output_mask_split": str(OUT_DIR / f"{STEM}_mask_split.jpg"),
        "output_diff_baseline": str(OUT_DIR / f"{STEM}_diff_baseline.jpg"),
        "output_diff_split": str(OUT_DIR / f"{STEM}_diff_split.jpg"),
        "output_grid": str(OUT_DIR / f"{STEM}_grid.jpg"),
    }
    (OUT_DIR / f"{STEM}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
