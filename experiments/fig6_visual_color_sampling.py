#!/usr/bin/env python3
"""Visual color sampling: color each label by the original image median.

No residual optimization, no text-label splitting. For every label we simply
sample the median RGB of the original pixels inside that label (ignoring very
dark/bright annotation pixels so the median is not pulled by text). The result
is a mask where each region is colored like the original geology.

This is a diagnostic/visual tool: we look at the output and decide by eye which
labels need their color adjusted away from the current brute-force palette.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path("/Users/daiduo2/geoseg")
GLOBAL_DIR = ROOT / "runs" / "fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5_global_palette_refinement"
PANEL_DIR = ROOT / "runs" / "feng_fig6_final_v4" / "crop_tests"
LABELS_DIR = ROOT / "runs" / "fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5"
OUT_DIR = GLOBAL_DIR / "visual_color_sampling"

PROFILES = [f"fig6_profile_{i:02d}" for i in range(3, 8)]
PROFILE_LABELS = {
    "fig6_profile_03": "Profile 03",
    "fig6_profile_04": "Profile 04",
    "fig6_profile_05": "Profile 05",
    "fig6_profile_06": "Profile 06",
    "fig6_profile_07": "Profile 07",
}

DARK_THRESHOLD = 55
BRIGHT_THRESHOLD = 210


def _load_font(size: int):
    for p in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def load_global_refined_palette() -> np.ndarray:
    summary = json.loads((GLOBAL_DIR / "summary.json").read_text(encoding="utf-8"))
    return np.array(summary["refined_palette"], dtype=np.uint8)


def load_brute_force_palette(panel_id: str) -> np.ndarray:
    summary = json.loads((GLOBAL_DIR / "brute_force_correction" / "summary.json").read_text(encoding="utf-8"))
    palette = np.array(summary[panel_id]["final_palette"], dtype=np.uint8)
    return palette


def render_mask(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl in np.unique(labels):
        if lbl < len(palette):
            mask[labels == lbl] = palette[lbl]
    return mask


def sample_median_colors(original: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return a palette where each label is the median color of its original pixels.

    Very dark or very bright pixels are ignored so text/annotation does not bias
    the sampled geological color.
    """
    max_label = int(labels.max())
    palette = np.zeros((max_label + 1, 3), dtype=np.uint8)
    gray = original.mean(axis=2)
    text_mask = (gray < DARK_THRESHOLD) | (gray > BRIGHT_THRESHOLD)

    for lbl in np.unique(labels):
        label_mask = labels == lbl
        valid = label_mask & ~text_mask
        if valid.sum() > 0:
            palette[lbl] = np.median(original[valid], axis=0).astype(np.uint8)
        elif label_mask.sum() > 0:
            palette[lbl] = np.median(original[label_mask], axis=0).astype(np.uint8)
    return palette


def assemble_grid(
    rows: list[tuple],
    title: str,
    profile_labels: list[str],
) -> Image.Image:
    header_labels = ["Original", "Global mask", "Sampled-color mask"]
    n_rows = len(rows)
    h, w = rows[0][0].shape[:2]
    header_h = 50
    label_w = 200
    col_gap = 4
    row_gap = 6
    canvas_h = header_h + n_rows * (h + row_gap)
    canvas_w = label_w + len(header_labels) * (w + col_gap)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(16)
    big_font = _load_font(22)

    bbox = draw.textbbox((0, 0), title, font=big_font)
    title_w = bbox[2] - bbox[0]
    draw.text(((canvas_w - title_w) // 2, 12), title, fill=(220, 220, 220), font=big_font)

    for i, hdr in enumerate(header_labels):
        x = label_w + i * (w + col_gap) + w // 2
        draw.text((x, header_h - 26), hdr, fill=(200, 200, 200), font=font, anchor="mm")

    for r, (row, plabel) in enumerate(zip(rows, profile_labels)):
        y = header_h + r * (h + row_gap)
        draw.text((label_w // 2, y + h // 2), plabel, fill=(200, 200, 200), font=font, anchor="mm")
        for c, img in enumerate(row):
            x = label_w + c * (w + col_gap)
            canvas.paste(Image.fromarray(img), (x, y))

    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_palette = load_global_refined_palette()

    rows: list[tuple] = []
    profile_labels: list[str] = []

    for panel_id in PROFILES:
        labels = np.load(LABELS_DIR / panel_id / "labels.npz")["labels"]
        original = np.array(Image.open(PANEL_DIR / f"{panel_id}_cropped.jpg").convert("RGB"))

        # Global refined palette mask (baseline).
        global_mask = render_mask(labels, base_palette)

        # Sampled-color mask.
        sampled_palette = sample_median_colors(original, labels)
        sampled_mask = render_mask(labels, sampled_palette)

        profile_out = OUT_DIR / panel_id
        profile_out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(global_mask).save(profile_out / "mask_global.jpg", quality=95)
        Image.fromarray(sampled_mask).save(profile_out / "mask_sampled.jpg", quality=95)
        np.savez_compressed(profile_out / "sampled_palette.npz", palette=sampled_palette)

        rows.append((original, global_mask, sampled_mask))
        profile_labels.append(PROFILE_LABELS[panel_id])
        print(f"{panel_id}: sampled palette saved")

    grid = assemble_grid(
        rows,
        "Visual color sampling: original vs global vs sampled median",
        profile_labels,
    )
    grid_path = OUT_DIR / "visual_color_sampling_v16.jpg"
    grid.save(grid_path, quality=95)
    print(f"\nSaved grid: {grid_path}")


if __name__ == "__main__":
    main()
