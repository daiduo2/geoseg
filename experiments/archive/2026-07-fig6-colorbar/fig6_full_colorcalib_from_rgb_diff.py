#!/usr/bin/env python3
"""Analyze per-label color bias from absolute RGB vector difference and recalibrate.

Workflow:
1. Load PM-smoothed labels and original panel.
2. Compute per-label median color in the original image.
3. Compare to the current palette; rank labels by L2 RGB bias and per-channel bias.
4. Recalibrate selected labels by replacing their palette color with the median
   original color (optionally blended with the original palette to keep some
   colorbar alignment).
5. Render new masks and regenerate the absolute RGB vector difference grid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path("/Users/daiduo2/geoseg")
LABELS_DIR = ROOT / "runs" / "fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5"
PANEL_DIR = ROOT / "runs" / "feng_fig6_final_v4" / "crop_tests"
SRC_RECOLOR_DIR = ROOT / "runs" / "fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5_label12_recolor"
OUT_DIR = ROOT / "runs" / "fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5_full_colorcalib"

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


def load_seed_palette() -> np.ndarray:
    summary = json.loads((SRC_RECOLOR_DIR / "summary.json").read_text(encoding="utf-8"))
    return np.array(summary["base_palette"], dtype=np.uint8)


def compute_label_median_colors(labels: np.ndarray, original: np.ndarray) -> dict[int, np.ndarray]:
    medians: dict[int, np.ndarray] = {}
    for lbl in sorted(np.unique(labels)):
        if lbl == 0:
            continue
        mask = labels == lbl
        if mask.any():
            medians[int(lbl)] = np.median(original[mask], axis=0).astype(np.uint8)
    return medians


def analyze_color_bias(palette: np.ndarray, medians: dict[int, np.ndarray]) -> list[dict]:
    rows = []
    for lbl, med in sorted(medians.items()):
        pal = palette[lbl]
        diff = med.astype(np.float32) - pal.astype(np.float32)
        l2 = float(np.linalg.norm(diff))
        rows.append({
            "label": lbl,
            "palette_rgb": pal.tolist(),
            "median_rgb": med.tolist(),
            "diff_rgb": diff.tolist(),
            "l2_bias": round(l2, 2),
            "abs_r": int(abs(diff[0])),
            "abs_g": int(abs(diff[1])),
            "abs_b": int(abs(diff[2])),
        })
    rows.sort(key=lambda x: x["l2_bias"], reverse=True)
    return rows


def recalibrate_palette(
    palette: np.ndarray,
    medians: dict[int, np.ndarray],
    labels_to_adjust: set[int] | None = None,
    blend: float = 1.0,
) -> np.ndarray:
    """Return palette with selected labels moved toward their median original color."""
    new_palette = palette.copy().astype(np.float32)
    if labels_to_adjust is None:
        labels_to_adjust = set(medians.keys())
    for lbl in labels_to_adjust:
        if lbl not in medians:
            continue
        target = medians[lbl].astype(np.float32)
        new_palette[lbl] = (1.0 - blend) * new_palette[lbl] + blend * target
    return np.clip(new_palette, 0, 255).astype(np.uint8)


def render_mask(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        mask[labels == lbl] = palette[lbl]
    return mask


def abs_rgb_diff(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    diff = np.abs(original.astype(np.float32) - mask.astype(np.float32))
    return np.clip(diff, 0, 255).astype(np.uint8)


def mean_l2(original: np.ndarray, mask: np.ndarray) -> float:
    return float(np.linalg.norm(original.astype(np.float32) - mask.astype(np.float32), axis=2).mean())


def assemble_grid(
    rows: list[tuple],
    title: str,
    profile_labels: list[str],
    row_metrics: list[tuple[float, float]],
) -> Image.Image:
    header_labels = ["Original", "Initial mask", "Initial abs RGB diff", "Calibrated mask", "Calibrated abs RGB diff"]
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

    for r, (row, plabel, (init_l2, cal_l2)) in enumerate(zip(rows, profile_labels, row_metrics)):
        y = header_h + r * (h + row_gap)
        draw.text((label_w // 2, y + h // 2), plabel, fill=(200, 200, 200), font=font, anchor="mm")
        for c, img in enumerate(row):
            x = label_w + c * (w + col_gap)
            canvas.paste(Image.fromarray(img), (x, y))
            if c in (2, 4):
                metric = init_l2 if c == 2 else cal_l2
                metric_text = f"L2≈{metric:.1f}"
                draw.text((x + 4, y + 4), metric_text, fill=(255, 255, 255), font=font)

    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_palette = load_seed_palette()
    analysis: dict[str, list[dict]] = {}
    all_rows: list[tuple] = []
    profile_labels: list[str] = []
    row_metrics: list[tuple[float, float]] = []

    # Strategy: recalibrate labels whose L2 bias is in the top 3 OR whose R-channel
    # absolute bias is particularly strong (catches the red areas the user noticed).
    for panel_id in PROFILES:
        labels = np.load(LABELS_DIR / panel_id / "labels.npz")["labels"]
        original = np.array(Image.open(PANEL_DIR / f"{panel_id}_cropped.jpg").convert("RGB"))

        medians = compute_label_median_colors(labels, original)
        bias = analyze_color_bias(base_palette, medians)
        analysis[panel_id] = bias

        # Select labels: strong color bias. We use a dual threshold:
        # - L2 RGB bias > 12 (clearly visible color mismatch)
        # - or abs(R) > 15 (catches the red areas the user pointed out).
        # This typically covers label 10 (R too high), label 12/14 (large L2),
        # and a few smaller-bias labels.
        adjust_labels = {
            row["label"]
            for row in bias
            if row["l2_bias"] > 12.0 or row["abs_r"] > 15
        }

        # Load previously label-12-adjusted palette if available.
        prev_palette_path = SRC_RECOLOR_DIR / panel_id / "palette_adjusted.npz"
        if prev_palette_path.exists():
            prev = np.load(prev_palette_path)
            current_palette = prev["palette"].copy()
        else:
            current_palette = base_palette.copy()

        initial_mask = render_mask(labels, current_palette)
        initial_diff = abs_rgb_diff(original, initial_mask)
        init_l2 = mean_l2(original, initial_mask)

        calibrated_palette = recalibrate_palette(
            current_palette,
            medians,
            labels_to_adjust=adjust_labels,
            blend=1.0,
        )
        calibrated_mask = render_mask(labels, calibrated_palette)
        calibrated_diff = abs_rgb_diff(original, calibrated_mask)
        cal_l2 = mean_l2(original, calibrated_mask)

        profile_out = OUT_DIR / panel_id
        profile_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(profile_out / "palette_calibrated.npz", palette=calibrated_palette)
        np.savez_compressed(profile_out / "labels.npz", labels=labels)
        Image.fromarray(initial_mask).save(profile_out / "mask_initial.jpg", quality=95)
        Image.fromarray(calibrated_mask).save(profile_out / "mask_calibrated.jpg", quality=95)
        Image.fromarray(initial_diff).save(profile_out / "diff_initial.jpg", quality=95)
        Image.fromarray(calibrated_diff).save(profile_out / "diff_calibrated.jpg", quality=95)

        all_rows.append((original, initial_mask, initial_diff, calibrated_mask, calibrated_diff))
        profile_labels.append(PROFILE_LABELS[panel_id])
        row_metrics.append((init_l2, cal_l2))

    # Save analysis artifacts.
    (OUT_DIR / "analysis.json").write_text(
        json.dumps({"base_palette": base_palette.tolist(), "profiles": analysis}, indent=2),
        encoding="utf-8",
    )
    save_summary_table(analysis, OUT_DIR / "analysis.md")

    grid = assemble_grid(
        all_rows,
        "Full color calibration from abs RGB vector difference",
        profile_labels,
        row_metrics,
    )
    grid_path = OUT_DIR / "rgb_vector_diff_full_calibrated_v8.jpg"
    grid.save(grid_path, quality=95)
    print(f"Saved grid: {grid_path}")
    print(f"Saved analysis: {OUT_DIR / 'analysis.json'}")
    print(f"Saved summary table: {OUT_DIR / 'analysis.md'}")


def save_summary_table(
    analysis: dict[str, list[dict]],
    out_path: Path,
) -> None:
    """Write a markdown table of the most biased labels per profile."""
    lines = ["# Per-label color bias analysis\n"]
    lines.append("Palette color vs. median original color inside each label.\n")
    for panel_id in PROFILES:
        lines.append(f"\n## {PROFILE_LABELS[panel_id]}\n")
        lines.append("| label | palette RGB | median RGB | L2 bias | |R| | |G| | |B| |\n")
        lines.append("|------|-------------|------------|---------|---|---|---|---|\n")
        for row in analysis[panel_id][:8]:  # top 8
            lines.append(
                f"| {row['label']} | {tuple(row['palette_rgb'])} | "
                f"{tuple(row['median_rgb'])} | {row['l2_bias']} | "
                f"{row['abs_r']} | {row['abs_g']} | {row['abs_b']} |\n"
            )
    out_path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
