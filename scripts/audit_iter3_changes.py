#!/usr/bin/env python3
"""Audit iteration 3 palette changes: compute per-label Delta E and generate comparison crops."""

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


def render_diff(original, mask, text_mask=None):
    diff = np.abs(original.astype(float) - mask.astype(float)).mean(axis=2)
    if text_mask is not None:
        diff = np.where(text_mask, 0, diff)
    norm = np.clip(diff / 255.0 * 2.0, 0, 1)
    norm_u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    if text_mask is not None:
        colored = np.where(text_mask[..., None], 0, colored)
    return colored


def build_iter2_verified_palette(initial, iter1_kept, iter2_kept):
    palette = initial.copy()
    for change in iter1_kept:
        lab = int(change["label"])
        palette[lab] = np.array(change["suggested_rgb"])
    for change in iter2_kept:
        lab = int(change["label"])
        palette[lab] = np.array(change["suggested_rgb"])
    return palette


def bbox_around(mask, pad=20):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0] - 1)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1] - 1)
    return (y0, y1 + 1, x0, x1 + 1)


def main():
    ref_root = Path(
        "/Users/daiduo2/geoseg/runs/fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5_global_palette_refinement"
    )
    src_root = Path(
        "/Users/daiduo2/geoseg/runs/fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5"
    )
    state_path = ref_root / "workflow_color_correction" / "state.json"
    out_dir = ref_root / "workflow_color_correction" / "iter_3_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(state_path) as f:
        state = json.load(f)

    profiles = state["profiles"]
    iter1_kept = state["verification"]["kept"]
    iter2_kept = state["verification_iter2"]["kept"]
    iter3_changes = state["adjustments"][2]["changes"]

    decisions = []

    for profile in profiles:
        labels = np.load(src_root / profile / "labels.npz")["labels"]
        text_mask = np.load(src_root / profile / "text_mask.npz")["mask"].astype(bool)
        original = load_image(src_root / profile / "cleaned.jpg")
        h, w = labels.shape
        original = original[:h, :w]
        text_mask = text_mask[:h, :w]
        valid = ~text_mask

        initial = np.load(ref_root / profile / "palette_initial.npz")["palette"].copy()
        iter2_palette = build_iter2_verified_palette(initial, iter1_kept, iter2_kept)
        current = np.array(state["palettes"][profile]).copy()

        original_lab = rgb2lab(original)

        for change in iter3_changes:
            if change["profile"] != profile:
                continue
            lab = int(change["label"])
            suggested = np.array(change["suggested_rgb"])
            before_rgb = iter2_palette[lab]

            before_lab = rgb2lab(before_rgb.reshape(1, 1, 3).astype(float) / 255.0)[0, 0]
            after_lab = rgb2lab(suggested.reshape(1, 1, 3).astype(float) / 255.0)[0, 0]
            mask_pixels = (labels == lab) & valid

            if mask_pixels.sum() == 0:
                de_before = de_after = None
                improved = False
            else:
                de_before = float(delta_e(original_lab[mask_pixels], before_lab).mean())
                de_after = float(delta_e(original_lab[mask_pixels], after_lab).mean())
                improved = de_after < de_before

            decisions.append(
                {
                    "profile": profile,
                    "label": lab,
                    "suggested_rgb": suggested.tolist(),
                    "before_rgb": before_rgb.tolist(),
                    "improved": improved,
                    "delta_e_before": de_before,
                    "delta_e_after": de_after,
                }
            )

            # Build crop around changed region
            bbox = bbox_around(mask_pixels, pad=30)
            if bbox is None:
                continue
            y0, y1, x0, x1 = bbox

            before_palette = iter2_palette.copy()
            after_palette = current.copy()

            before_mask_full = render_mask(labels, before_palette)
            after_mask_full = render_mask(labels, after_palette)
            before_diff_full = render_diff(original, before_mask_full, text_mask)
            after_diff_full = render_diff(original, after_mask_full, text_mask)

            crop_original = original[y0:y1, x0:x1]
            crop_before = before_mask_full[y0:y1, x0:x1]
            crop_after = after_mask_full[y0:y1, x0:x1]
            crop_before_diff = before_diff_full[y0:y1, x0:x1]
            crop_after_diff = after_diff_full[y0:y1, x0:x1]

            fig, axes = plt.subplots(1, 5, figsize=(14, 3))
            fig.suptitle(
                f"{profile} label {lab} | before dE={de_before:.1f} -> after dE={de_after:.1f} "
                f"({'KEEP' if improved else 'REVERT'} by metric)",
                fontsize=10,
            )
            titles = ["Original", "Before", "After", "Before diff", "After diff"]
            images = [crop_original, crop_before, crop_after, crop_before_diff, crop_after_diff]
            for ax, im, title in zip(axes, images, titles):
                ax.imshow(im)
                ax.set_title(title, fontsize=8)
                ax.axis("off")
            plt.tight_layout()
            crop_path = out_dir / f"{profile}_label{lab:02d}.jpg"
            plt.savefig(crop_path, dpi=150)
            plt.close()

    # Print summary
    print(f"Generated {len(list(out_dir.glob('*.jpg')))} crop images in {out_dir}")
    print("\nPer-change metric summary:")
    for d in decisions:
        status = "KEEP" if d["improved"] else "REVERT"
        print(
            f"  {status} {d['profile']} label {d['label']:2d} -> {d['suggested_rgb']} "
            f"(dE before={d['delta_e_before']:.2f}, after={d['delta_e_after']:.2f})"
        )

    summary_path = out_dir / "metric_summary.json"
    with open(summary_path, "w") as f:
        json.dump(decisions, f, indent=2)
    print(f"\nMetric summary written to {summary_path}")


if __name__ == "__main__":
    main()
