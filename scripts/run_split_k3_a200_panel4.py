"""Run split_label_by_color_components for panel_4 label 3 and save artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geoseg.modules.post_process.split import split_label_by_color_components


BASE = Path("/Users/daiduo2/geoseg/runs/preprocess_newimage_merged/panels/panel_4")
LABELS_PATH = BASE / "labels.npz"
IMAGE_PATH = BASE / "visual_audit" / "panel.png"
OUT_DIR = BASE / "visual_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LABEL = 3
COLOR_SPACE = "LAB"
K = 3
MIN_COMPONENT_AREA = 200


def load_labels(p: Path) -> np.ndarray:
    data = np.load(p)
    keys = list(data.keys())
    if "labels" in keys:
        return data["labels"]
    return data[keys[0]]


def random_color_palette(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]
    return palette


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    n = int(labels.max()) + 1
    palette = random_color_palette(max(n, 2))
    return palette[labels.astype(np.int32)]


def overlay_labels(img_rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    colored = colorize_labels(labels)
    blended = cv2.addWeighted(img_rgb, 1.0 - alpha, colored, alpha, 0)
    return blended


def mean_rgb_l2(mask: np.ndarray, img_rgb: np.ndarray, ref_color: np.ndarray) -> float:
    pixels = img_rgb[mask]
    if pixels.size == 0:
        return 0.0
    diff = np.linalg.norm(pixels.astype(np.float32) - ref_color.astype(np.float32), axis=1)
    return float(np.mean(diff))


def median_rgb(mask: np.ndarray, img_rgb: np.ndarray) -> np.ndarray:
    pixels = img_rgb[mask]
    if pixels.size == 0:
        return np.zeros(3, dtype=np.uint8)
    return np.median(pixels, axis=0).astype(np.uint8)


def add_legend(img_rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()

    n = int(labels.max())
    palette = random_color_palette(n + 1)
    x, y = 10, 10
    box_w, box_h = 18, 18
    spacing = 4
    for label_id in range(1, n + 1):
        color = tuple(int(c) for c in palette[label_id])
        draw.rectangle([x, y, x + box_w, y + box_h], fill=color, outline=(255, 255, 255))
        draw.text(
            (x + box_w + 6, y),
            f"Label {label_id}",
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        y += box_h + spacing

    return np.array(pil)


def build_side_by_side(
    img_rgb: np.ndarray,
    original_labels: np.ndarray,
    split_labels: np.ndarray,
    target_label: int,
) -> np.ndarray:
    baseline = np.where(original_labels == target_label, target_label, 0).astype(np.int32)
    img_h, img_w = img_rgb.shape[:2]
    margin = 10
    total_w = img_w * 3 + margin * 2
    canvas = np.zeros((img_h, total_w, 3), dtype=np.uint8)

    canvas[:, :img_w] = img_rgb
    canvas[:, img_w + margin : 2 * img_w + margin] = overlay_labels(img_rgb, baseline)
    canvas[:, 2 * img_w + 2 * margin :] = overlay_labels(img_rgb, split_labels)

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        font = ImageFont.load_default()
    titles = ["Original", f"Baseline mask (label {target_label})", "Split mask"]
    for i, title in enumerate(titles):
        x = i * (img_w + margin) + 10
        draw.text((x, 10), title, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))

    return np.array(pil)


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
    )

    # Compute metrics
    baseline_mask = labels == TARGET_LABEL
    baseline_median = median_rgb(baseline_mask, img_rgb)
    baseline_l2 = mean_rgb_l2(baseline_mask, img_rgb, baseline_median)

    new_component_ids = sorted(set(np.unique(split_labels[baseline_mask])) - {0})
    split_l2_sum = 0.0
    split_pixel_count = 0
    for cid in new_component_ids:
        comp_mask = (split_labels == cid) & baseline_mask
        if not comp_mask.any():
            continue
        comp_median = median_rgb(comp_mask, img_rgb)
        comp_pixels = img_rgb[comp_mask]
        diff = np.linalg.norm(comp_pixels.astype(np.float32) - comp_median.astype(np.float32), axis=1)
        split_l2_sum += float(np.sum(diff))
        split_pixel_count += int(comp_pixels.shape[0])

    split_l2 = split_l2_sum / split_pixel_count if split_pixel_count else 0.0
    n_final_labels = int(np.max(split_labels))
    n_split_components = len(new_component_ids)

    # Save outputs
    np.save(OUT_DIR / "split_k3_a200_labels.npy", split_labels)
    overlay = overlay_labels(img_rgb, split_labels)
    overlay_legend = add_legend(overlay, split_labels)
    Image.fromarray(overlay_legend).save(OUT_DIR / "split_k3_a200_overlay.jpg", quality=95)

    compare = build_side_by_side(img_rgb, labels, split_labels, TARGET_LABEL)
    Image.fromarray(compare).save(OUT_DIR / "split_k3_a200_compare.jpg", quality=95)

    metrics = {
        "target_region_baseline_l2": round(baseline_l2, 4),
        "target_region_split_l2": round(split_l2, 4),
        "n_final_labels": n_final_labels,
        "n_split_components": n_split_components,
        "panel_id": 4,
        "k": K,
        "min_area": MIN_COMPONENT_AREA,
        "l2_reduction": round((baseline_l2 - split_l2) / baseline_l2 if baseline_l2 else 0.0, 4),
        "output_overlay": str(OUT_DIR / "split_k3_a200_overlay.jpg"),
        "output_compare": str(OUT_DIR / "split_k3_a200_compare.jpg"),
    }
    (OUT_DIR / "split_k3_a200_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
