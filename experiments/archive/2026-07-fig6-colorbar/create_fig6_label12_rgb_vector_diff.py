#!/usr/bin/env python3
"""Rebuild label-12 residual grid using true RGB vector difference.

For each profile we show:
  Original | Seed mask | RGB vector diff (seed) | Adjusted mask | RGB vector diff (adjusted)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path("/Users/daiduo2/geoseg")
SRC_DIR = ROOT / "runs" / "fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5_label12_recolor"
PANEL_DIR = ROOT / "runs" / "feng_fig6_final_v4" / "crop_tests"
OUT_DIR = SRC_DIR

PROFILES = [f"fig6_profile_{i:02d}" for i in range(3, 8)]
PROFILE_LABELS = {
    "fig6_profile_03": "Profile 03",
    "fig6_profile_04": "Profile 04 (PM artifact)",
    "fig6_profile_05": "Profile 05 (PM artifact)",
    "fig6_profile_06": "Profile 06",
    "fig6_profile_07": "Profile 07",
}


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


def rgb_vector_difference_signed(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Signed RGB vector difference shifted to [0, 255] with 128 = zero."""
    diff = original.astype(np.float32) - mask.astype(np.float32)
    return np.clip(diff + 128.0, 0.0, 255.0).astype(np.uint8)


def rgb_vector_difference_abs(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Absolute per-channel RGB difference (no shift, dark = no diff)."""
    diff = np.abs(original.astype(np.float32) - mask.astype(np.float32))
    return np.clip(diff, 0.0, 255.0).astype(np.uint8)


def rgb_vector_difference_gray(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Absolute per-channel RGB difference converted to luminance."""
    diff = np.abs(original.astype(np.float32) - mask.astype(np.float32))
    lum = 0.299 * diff[..., 0] + 0.587 * diff[..., 1] + 0.114 * diff[..., 2]
    gray = np.clip(lum, 0.0, 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=2)


def mean_l2(original: np.ndarray, mask: np.ndarray) -> float:
    return float(np.linalg.norm(original.astype(np.float32) - mask.astype(np.float32), axis=2).mean())


def assemble_grid(
    rows: list[tuple[np.ndarray, ...]],
    header_labels: list[str],
    title: str,
    profile_labels: list[str],
    row_metrics: list[tuple[float, float]],
) -> Image.Image:
    n_rows = len(rows)
    h, w = rows[0][0].shape[:2]
    header_h = 50
    label_w = 160
    col_gap = 4
    row_gap = 6
    canvas_h = header_h + n_rows * (h + row_gap)
    canvas_w = label_w + len(header_labels) * (w + col_gap)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(18)
    big_font = _load_font(24)

    bbox = draw.textbbox((0, 0), title, font=big_font)
    title_w = bbox[2] - bbox[0]
    draw.text(((canvas_w - title_w) // 2, 12), title, fill=(220, 220, 220), font=big_font)

    for i, hdr in enumerate(header_labels):
        x = label_w + i * (w + col_gap) + w // 2
        draw.text((x, header_h - 28), hdr, fill=(200, 200, 200), font=font, anchor="mm")

    for r, (row, plabel, (seed_l2, adj_l2)) in enumerate(zip(rows, profile_labels, row_metrics)):
        y = header_h + r * (h + row_gap)
        draw.text((label_w // 2, y + h // 2), plabel, fill=(200, 200, 200), font=font, anchor="mm")
        for c, img in enumerate(row):
            x = label_w + c * (w + col_gap)
            canvas.paste(Image.fromarray(img), (x, y))
            if c in (2, 4):
                metric = seed_l2 if c == 2 else adj_l2
                metric_text = f"L2≈{metric:.1f}"
                draw.text((x + 4, y + 4), metric_text, fill=(255, 255, 255), font=font)

    return canvas


def main() -> None:
    diff_modes = {
        "signed": (rgb_vector_difference_signed, "RGB Vector Diff (signed, 128 = zero)"),
        "abs": (rgb_vector_difference_abs, "RGB Vector Diff (absolute)"),
        "gray": (rgb_vector_difference_gray, "RGB Vector Diff (luminance)"),
    }

    for mode, (diff_fn, title_suffix) in diff_modes.items():
        rows: list[tuple[np.ndarray, ...]] = []
        row_metrics: list[tuple[float, float]] = []

        for panel_id in PROFILES:
            original = np.array(Image.open(PANEL_DIR / f"{panel_id}_cropped.jpg").convert("RGB"))
            seed_mask = np.array(Image.open(SRC_DIR / panel_id / "seed_mask.jpg").convert("RGB"))
            adj_mask = np.array(Image.open(SRC_DIR / panel_id / "adjusted_mask.jpg").convert("RGB"))

            seed_diff = diff_fn(original, seed_mask)
            adj_diff = diff_fn(original, adj_mask)
            rows.append((original, seed_mask, seed_diff, adj_mask, adj_diff))
            row_metrics.append((mean_l2(original, seed_mask), mean_l2(original, adj_mask)))

        header = ["Original", "Seed mask", f"Seed {title_suffix}", "Adjusted mask", f"Adj {title_suffix}"]
        grid = assemble_grid(
            rows,
            header,
            f"Label-12 recolor — {title_suffix}",
            [PROFILE_LABELS[p] for p in PROFILES],
            row_metrics,
        )
        out_path = OUT_DIR / f"rgb_vector_diff_label12_{mode}_v7.jpg"
        grid.save(out_path, quality=95)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
