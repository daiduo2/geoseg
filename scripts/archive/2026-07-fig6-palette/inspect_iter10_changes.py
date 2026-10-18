#!/usr/bin/env python3
"""Generate per-label detail crops for iteration 10 palette adjustments."""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def delta_e(lab1, lab2):
    return np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))


def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def render_mask(labels, palette):
    return palette[labels].astype(np.uint8)


def sample_label_rgb(mask_image, labels, label):
    pixels = mask_image[labels == label]
    if len(pixels) == 0:
        return None
    return np.median(pixels, axis=0).astype(np.uint8).tolist()


def extract_palette_from_mask(mask_image, labels):
    palette = np.zeros((16, 3), dtype=np.uint8)
    for lab in range(16):
        rgb = sample_label_rgb(mask_image, labels, lab)
        if rgb is not None:
            palette[lab] = rgb
    return palette


def crop_around_label(labels, label, pad=25):
    coords = np.argwhere(labels == label)
    if len(coords) == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(labels.shape[0], y_max + pad)
    x_max = min(labels.shape[1], x_max + pad)
    return (y_min, x_min, y_max, x_max)


def mean_rgb_l2(original, mask, valid):
    d = np.sqrt(np.sum((original.astype(float) - mask.astype(float)) ** 2, axis=2))
    return float(d[valid].mean())


def main():
    ref_root = Path(
        "/Users/daiduo2/geoseg/runs/fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5_global_palette_refinement"
    )
    src_root = Path(
        "/Users/daiduo2/geoseg/runs/fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5"
    )
    state_path = ref_root / "workflow_color_correction" / "state.json"
    before_dir = ref_root / "workflow_color_correction" / "iter_9_verified"
    after_dir = ref_root / "workflow_color_correction" / "iter_10"
    out_dir = ref_root / "workflow_color_correction" / "iter_10_inspect"
    detail_dir = out_dir / "per_label_details"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)

    with open(state_path) as f:
        state = json.load(f)

    profiles = state["profiles"]
    iter10_changes = state["adjustments"][9]["changes"]

    per_label_decisions = []

    for profile in profiles:
        labels = np.load(src_root / profile / "labels.npz")["labels"]
        text_mask = np.load(src_root / profile / "text_mask.npz")["mask"].astype(bool)
        original = load_image(src_root / profile / "cleaned.jpg")
        h, w = labels.shape
        original = original[:h, :w]
        text_mask = text_mask[:h, :w]
        valid = ~text_mask

        before_mask = load_image(before_dir / profile / "mask_corrected.jpg")[:h, :w]
        after_mask = load_image(after_dir / profile / "mask_corrected.jpg")[:h, :w]

        before_palette = extract_palette_from_mask(before_mask, labels)
        after_palette = extract_palette_from_mask(after_mask, labels)
        original_lab = rgb2lab(original)

        profile_detail_dir = detail_dir / profile
        profile_detail_dir.mkdir(parents=True, exist_ok=True)

        for change in iter10_changes:
            if change["profile"] != profile:
                continue
            lab = int(change["label"])
            suggested = np.array(change["suggested_rgb"])
            before_rgb = before_palette[lab]
            after_rgb = after_palette[lab]

            before_lab = rgb2lab(before_rgb.reshape(1, 1, 3).astype(float) / 255.0)[0, 0]
            after_lab = rgb2lab(after_rgb.reshape(1, 1, 3).astype(float) / 255.0)[0, 0]
            mask_pixels = (labels == lab) & valid
            if mask_pixels.sum() == 0:
                de_before = de_after = None
            else:
                de_before = float(delta_e(original_lab[mask_pixels], before_lab).mean())
                de_after = float(delta_e(original_lab[mask_pixels], after_lab).mean())

            per_label_decisions.append(
                {
                    "profile": profile,
                    "label": lab,
                    "suggested_rgb": suggested.tolist(),
                    "before_rgb": before_rgb.tolist(),
                    "after_rgb": after_rgb.tolist(),
                    "delta_e_before": de_before,
                    "delta_e_after": de_after,
                }
            )

            bbox = crop_around_label(labels, lab, pad=25)
            if bbox is None:
                continue
            y_min, x_min, y_max, x_max = bbox
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            titles = ["Original", "Iter 9 verified", "Iter 10"]
            images = [
                original[y_min:y_max, x_min:x_max],
                before_mask[y_min:y_max, x_min:x_max],
                after_mask[y_min:y_max, x_min:x_max],
            ]
            fig.suptitle(
                f"{profile} label {lab}\n"
                f"before {before_rgb.tolist()} dE={de_before if de_before is None else de_before:.1f} -> "
                f"after {after_rgb.tolist()} dE={de_after if de_after is None else de_after:.1f}",
                fontsize=9,
            )
            for ax, im, title in zip(axes, images, titles):
                ax.imshow(im)
                ax.set_title(title, fontsize=8)
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(profile_detail_dir / f"label_{lab:02d}_detail.jpg", dpi=150)
            plt.close()

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(per_label_decisions, f, indent=2)

    print(f"Detail images saved to: {detail_dir}")
    print(f"Summary saved to: {summary_path}")
    for d in per_label_decisions:
        print(
            f"{d['profile']} label {d['label']:2d}: "
            f"before {d['before_rgb']} dE={d['delta_e_before']:.2f} -> "
            f"after {d['after_rgb']} dE={d['delta_e_after']:.2f}"
        )


if __name__ == "__main__":
    main()
