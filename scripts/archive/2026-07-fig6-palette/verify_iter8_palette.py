#!/usr/bin/env python3
"""Verify iteration 8 palette adjustments and revert entries that worsened color fidelity."""

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


def mean_rgb_l2(original, mask, valid):
    d = np.sqrt(np.sum((original.astype(float) - mask.astype(float)) ** 2, axis=2))
    return float(d[valid].mean())


def sample_label_rgb(mask_image, labels, label):
    pixels = mask_image[labels == label]
    if len(pixels) == 0:
        return None
    return np.median(pixels, axis=0).astype(np.uint8).tolist()


def iter7_verified_palette(labels, before_mask):
    """Sample the palette that produced the existing iter_7_verified mask."""
    palette = np.zeros((16, 3), dtype=np.uint8)
    for lab in range(16):
        rgb = sample_label_rgb(before_mask, labels, lab)
        if rgb is not None:
            palette[lab] = rgb
    return palette


def crop_around_label(mask, labels, label, pad=20):
    coords = np.argwhere(labels == label)
    if len(coords) == 0:
        return None, None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(mask.shape[0], y_max + pad)
    x_max = min(mask.shape[1], x_max + pad)
    return (y_min, x_min, y_max, x_max), mask[y_min:y_max, x_min:x_max]


def main():
    ref_root = Path(
        "/Users/daiduo2/geoseg/runs/fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5_global_palette_refinement"
    )
    src_root = Path(
        "/Users/daiduo2/geoseg/runs/fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5"
    )
    state_path = ref_root / "workflow_color_correction" / "state.json"
    before_dir = ref_root / "workflow_color_correction" / "iter_7_verified"
    out_dir = ref_root / "workflow_color_correction" / "iter_8_verified"
    detail_dir = out_dir / "per_label_details"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)

    with open(state_path) as f:
        state = json.load(f)

    profiles = state["profiles"]
    iter8_changes = state["adjustments"][7]["changes"]

    final_palettes = {}
    iter7_l2 = {}
    verified_l2 = {}
    per_label_decisions = []

    for profile in profiles:
        labels = np.load(src_root / profile / "labels.npz")["labels"]
        text_mask = np.load(src_root / profile / "text_mask.npz")["mask"].astype(bool)
        original = load_image(src_root / profile / "cleaned.jpg")
        h, w = labels.shape
        original = original[:h, :w]
        text_mask = text_mask[:h, :w]
        valid = ~text_mask

        before_mask = load_image(before_dir / profile / "mask_corrected.jpg")
        before_mask = before_mask[:h, :w]

        current = iter7_verified_palette(labels, before_mask).copy()
        original_lab = rgb2lab(original)

        profile_detail_dir = detail_dir / profile
        profile_detail_dir.mkdir(parents=True, exist_ok=True)

        for change in iter8_changes:
            if change["profile"] != profile:
                continue
            lab = int(change["label"])
            suggested = np.array(change["suggested_rgb"])
            before_rgb = np.array(sample_label_rgb(before_mask, labels, lab))
            if before_rgb is None:
                continue

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

            keep_visual = improved

            per_label_decisions.append(
                {
                    "profile": profile,
                    "label": lab,
                    "suggested_rgb": suggested.tolist(),
                    "before_rgb": before_rgb.tolist(),
                    "improved_metric": improved,
                    "kept": keep_visual,
                    "delta_e_before": de_before,
                    "delta_e_after": de_after,
                }
            )
            if not keep_visual:
                current[lab] = before_rgb

            bbox, _ = crop_around_label(labels, labels, lab, pad=25)
            if bbox is not None:
                y_min, x_min, y_max, x_max = bbox
                before_render = before_mask[y_min:y_max, x_min:x_max]
                after_render = render_mask(labels, current)[y_min:y_max, x_min:x_max]
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                titles = ["Original", "Iter 7 verified", "Iter 8 verified"]
                images = [
                    original[y_min:y_max, x_min:x_max],
                    before_render,
                    after_render,
                ]
                status = "KEPT" if keep_visual else "REVERTED"
                fig.suptitle(
                    f"{profile} label {lab} ({status})\n"
                    f"before {before_rgb.tolist()} dE={de_before:.1f} -> "
                    f"after {suggested.tolist()} dE={de_after:.1f}",
                    fontsize=9,
                )
                for ax, im, title in zip(axes, images, titles):
                    ax.imshow(im)
                    ax.set_title(title, fontsize=8)
                    ax.axis("off")
                plt.tight_layout()
                plt.savefig(profile_detail_dir / f"label_{lab:02d}_detail.jpg", dpi=150)
                plt.close()

        final_palettes[profile] = current.tolist()

        iter7_l2[profile] = mean_rgb_l2(original, before_mask, valid)

        profile_dir = out_dir / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        verified_mask = render_mask(labels, current)
        Image.fromarray(verified_mask).save(profile_dir / "mask_corrected.jpg")
        saved_verified_mask = load_image(profile_dir / "mask_corrected.jpg")[:h, :w]
        verified_l2[profile] = mean_rgb_l2(original, saved_verified_mask, valid)

        Image.fromarray(render_diff(original, saved_verified_mask, text_mask)).save(
            profile_dir / "diff_corrected.jpg"
        )

    state["palettes"] = final_palettes
    state["iteration"] = "8_verified"
    state["mask_paths"] = {p: str(out_dir / p / "mask_corrected.jpg") for p in profiles}
    state["diff_paths"] = {p: str(out_dir / p / "diff_corrected.jpg") for p in profiles}
    state["verification_iter8"] = {
        "kept": [d for d in per_label_decisions if d["kept"]],
        "reverted": [d for d in per_label_decisions if not d["kept"]],
        "iter7_l2": iter7_l2,
        "verified_l2": verified_l2,
    }

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    rows = len(profiles)
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.5), squeeze=False)
    fig.suptitle(
        "Iteration 8 verification: iter 7 verified vs. iter 8 verified (worse entries reverted)",
        fontsize=12,
    )

    for r, profile in enumerate(profiles):
        labels = np.load(src_root / profile / "labels.npz")["labels"]
        text_mask = np.load(src_root / profile / "text_mask.npz")["mask"].astype(bool)
        original = load_image(src_root / profile / "cleaned.jpg")
        h, w = labels.shape
        original = original[:h, :w]
        text_mask = text_mask[:h, :w]

        before_mask = load_image(before_dir / profile / "mask_corrected.jpg")[:h, :w]
        verified_mask = load_image(out_dir / profile / "mask_corrected.jpg")[:h, :w]
        before_diff = render_diff(original, before_mask, text_mask)
        verified_diff = render_diff(original, verified_mask, text_mask)

        images = [original, before_mask, verified_mask, before_diff, verified_diff]
        titles = [
            f"{profile}\nOriginal",
            f"Iter 7 verified\nL2={iter7_l2[profile]:.1f}",
            f"Iter 8 verified\nL2={verified_l2[profile]:.1f}",
            "Iter 7 diff",
            "Iter 8 diff",
        ]
        for c, (im, title) in enumerate(zip(images, titles)):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    grid_path = out_dir / "comparison_grid.jpg"
    plt.tight_layout()
    plt.savefig(grid_path, dpi=200)
    plt.close()

    print(f"Final comparison grid: {grid_path}")
    kept = sum(1 for d in per_label_decisions if d["kept"])
    reverted = len(per_label_decisions) - kept
    print(f"Adjustments kept: {kept}, reverted: {reverted}")
    for d in per_label_decisions:
        status = "KEPT" if d["kept"] else "REVERTED"
        print(
            f"  {status} {d['profile']} label {d['label']} -> {d['suggested_rgb']} "
            f"(dE before={d['delta_e_before']:.2f}, after={d['delta_e_after']:.2f})"
        )


if __name__ == "__main__":
    main()
