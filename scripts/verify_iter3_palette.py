#!/usr/bin/env python3
"""Verify iteration 3 palette adjustments and revert entries that worsened color fidelity."""

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


def build_iter2_verified_palette(initial, iter1_kept, iter2_kept):
    palette = initial.copy()
    for change in iter1_kept:
        palette[int(change["label"])] = np.array(change["suggested_rgb"])
    for change in iter2_kept:
        palette[int(change["label"])] = np.array(change["suggested_rgb"])
    return palette


def main():
    ref_root = Path(
        "/Users/daiduo2/geoseg/runs/fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5_global_palette_refinement"
    )
    src_root = Path(
        "/Users/daiduo2/geoseg/runs/fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5"
    )
    state_path = ref_root / "workflow_color_correction" / "state.json"
    out_dir = ref_root / "workflow_color_correction" / "iter_3_verified"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(state_path) as f:
        state = json.load(f)

    profiles = state["profiles"]
    iter1_kept = state["verification"]["kept"]
    iter2_kept = state["verification_iter2"]["kept"]
    iter3_changes = state["adjustments"][2]["changes"]

    final_palettes = {}
    iter2_l2 = {}
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

        initial = np.load(ref_root / profile / "palette_initial.npz")["palette"].copy()
        iter2_verified = build_iter2_verified_palette(initial, iter1_kept, iter2_kept)
        current = np.array(state["palettes"][profile]).copy()

        original_lab = rgb2lab(original)

        for change in iter3_changes:
            if change["profile"] != profile:
                continue
            lab = int(change["label"])
            suggested = np.array(change["suggested_rgb"])
            before_rgb = iter2_verified[lab]
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
            per_label_decisions.append(
                {
                    "profile": profile,
                    "label": lab,
                    "suggested_rgb": suggested.tolist(),
                    "before_rgb": before_rgb.tolist(),
                    "improved": improved,
                    "kept": improved,
                    "delta_e_before": de_before,
                    "delta_e_after": de_after,
                }
            )
            if not improved:
                current[lab] = before_rgb

        final_palettes[profile] = current.tolist()

        iter2_mask = render_mask(labels, iter2_verified)
        verified_mask = render_mask(labels, current)
        iter2_l2[profile] = mean_rgb_l2(original, iter2_mask, valid)
        verified_l2[profile] = mean_rgb_l2(original, verified_mask, valid)

        profile_dir = out_dir / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(verified_mask).save(profile_dir / "mask_corrected.jpg")
        Image.fromarray(render_diff(original, verified_mask, text_mask)).save(
            profile_dir / "diff_corrected.jpg"
        )
        Image.fromarray(iter2_mask).save(profile_dir / "mask_iter2_verified.jpg")
        Image.fromarray(render_diff(original, iter2_mask, text_mask)).save(
            profile_dir / "diff_iter2_verified.jpg"
        )

    # Update state
    state["palettes"] = final_palettes
    state["iteration"] = "3_verified"
    state["mask_paths"] = {
        p: str(out_dir / p / "mask_corrected.jpg") for p in profiles
    }
    state["diff_paths"] = {
        p: str(out_dir / p / "diff_corrected.jpg") for p in profiles
    }
    state["verification_iter3"] = {
        "kept": [d for d in per_label_decisions if d["kept"]],
        "reverted": [d for d in per_label_decisions if not d["kept"]],
        "iter2_l2": iter2_l2,
        "verified_l2": verified_l2,
    }

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    # Build comparison grid
    rows = len(profiles)
    cols = 5
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 2.5), squeeze=False
    )
    fig.suptitle(
        "Iteration 3 verification: iter 2 verified vs. iter 3 verified (worse entries reverted)",
        fontsize=12,
    )

    for r, profile in enumerate(profiles):
        labels = np.load(src_root / profile / "labels.npz")["labels"]
        text_mask = np.load(src_root / profile / "text_mask.npz")["mask"].astype(bool)
        original = load_image(src_root / profile / "cleaned.jpg")
        h, w = labels.shape
        original = original[:h, :w]
        text_mask = text_mask[:h, :w]
        valid = ~text_mask

        initial = np.load(ref_root / profile / "palette_initial.npz")["palette"]
        iter2_verified = build_iter2_verified_palette(initial, iter1_kept, iter2_kept)
        current = np.array(final_palettes[profile])

        iter2_mask = render_mask(labels, iter2_verified)
        verified_mask = render_mask(labels, current)
        iter2_diff = render_diff(original, iter2_mask, text_mask)
        verified_diff = render_diff(original, verified_mask, text_mask)

        images = [original, iter2_mask, verified_mask, iter2_diff, verified_diff]
        titles = [
            f"{profile}\nOriginal",
            f"Iter 2 verified\nL2={iter2_l2[profile]:.1f}",
            f"Iter 3 verified\nL2={verified_l2[profile]:.1f}",
            "Iter 2 diff",
            "Iter 3 diff",
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
