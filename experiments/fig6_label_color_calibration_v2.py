#!/usr/bin/env python3
"""Fig.6 global colorbar-aware palette refinement (design B).

Approach:
1. Load PM-smoothed labels and original panels.
2. Extract the shared 16-color seed palette from the colorbar.
3. For each label, collect per-profile median RGB inside the segmented region.
4. Aggregate to a robust global median-of-medians per label.
5. Refine via shrinkage toward the original seed color, applying correction only
   where the aggregate bias exceeds a tolerance (Delta E > 5 or L2 > 8).
6. Render masks WITHOUT skipping label 0.
7. Produce an absolute RGB vector difference grid and summary JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2lab, deltaE_cie76

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseg.modules.segment_engines.v4_kmeans import _sample_colorbar_seeds

ROOT = Path("/Users/daiduo2/geoseg")
FIGURE_PATH = ROOT / "fig6_detected_panels.jpg"
LABELS_DIR = ROOT / "runs" / "fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5"
PANEL_DIR = ROOT / "runs" / "feng_fig6_final_v4" / "crop_tests"
BASELINE_DIR = ROOT / "runs" / "fig6_colorbar_16zone_experiment_no_merge_clean_text"
OUT_DIR = ROOT / "runs" / "fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5_global_palette_refinement"

COLORBAR_ROI = (1346, 1376, 317, 10)
PROFILES = [f"fig6_profile_{i:02d}" for i in range(3, 8)]
PROFILE_LABELS = {
    "fig6_profile_03": "Profile 03",
    "fig6_profile_04": "Profile 04",
    "fig6_profile_05": "Profile 05",
    "fig6_profile_06": "Profile 06",
    "fig6_profile_07": "Profile 07",
}

# Design B hyperparameters.
ALPHA = 0.8
L2_TOLERANCE = 8.0
DELTA_E_TOLERANCE = 5.0
GRID_HEIGHT = 200


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
    """Sample 16 evenly-spaced seed colors from the shared colorbar."""
    colorbar_rgb = np.array(Image.open(FIGURE_PATH).convert("RGB"))
    x, y, w, h = COLORBAR_ROI
    strip = colorbar_rgb[y : y + h, x : x + w]
    seeds, _ = _sample_colorbar_seeds(strip, k=16)
    return seeds.astype(np.uint8)


def load_labels_and_text_mask(profile: str) -> tuple[np.ndarray, np.ndarray]:
    """Load label array and text mask for a profile."""
    profile_dir = LABELS_DIR / profile
    labels = np.load(profile_dir / "labels.npz")["labels"]
    text_mask = np.load(profile_dir / "text_mask.npz")["mask"]
    return labels, text_mask


def compute_label_median_colors(
    labels: np.ndarray,
    original: np.ndarray,
    text_mask: np.ndarray,
) -> dict[int, np.ndarray]:
    """Median RGB per label, excluding text-mask pixels."""
    medians: dict[int, np.ndarray] = {}
    valid = ~text_mask
    for lbl in sorted(np.unique(labels)):
        mask = (labels == lbl) & valid
        if mask.any():
            medians[int(lbl)] = np.median(original[mask], axis=0).astype(np.uint8)
    return medians


def aggregate_global_medians(
    all_medians: dict[str, dict[int, np.ndarray]],
) -> dict[int, np.ndarray]:
    """Median-of-medians per label across all profiles."""
    global_medians: dict[int, np.ndarray] = {}
    for lbl in range(16):
        colors = [
            medians[lbl].astype(np.float32)
            for medians in all_medians.values()
            if lbl in medians
        ]
        if colors:
            global_medians[lbl] = np.median(np.stack(colors), axis=0).astype(np.uint8)
    return global_medians


def refine_palette(
    seed_palette: np.ndarray,
    global_medians: dict[int, np.ndarray],
    alpha: float = ALPHA,
    l2_tol: float = L2_TOLERANCE,
    de_tol: float = DELTA_E_TOLERANCE,
) -> tuple[np.ndarray, dict[int, dict]]:
    """Shrink global medians toward seed colors when bias exceeds tolerance."""
    refined = seed_palette.copy().astype(np.float32)
    metadata: dict[int, dict] = {}
    seed_lab = rgb2lab(seed_palette.reshape(-1, 1, 3))

    for lbl, aggregate in sorted(global_medians.items()):
        seed = seed_palette[lbl].astype(np.float32)
        diff = aggregate.astype(np.float32) - seed
        l2 = float(np.linalg.norm(diff))

        agg_lab = rgb2lab(aggregate.reshape(1, 1, 3))
        delta_e = float(np.squeeze(deltaE_cie76(seed_lab[lbl], agg_lab[0, 0])))

        corrected = bool(l2 > l2_tol or delta_e > de_tol)
        if corrected:
            refined[lbl] = seed + alpha * diff

        metadata[lbl] = {
            "seed_rgb": seed.astype(np.uint8).tolist(),
            "aggregate_rgb": aggregate.tolist(),
            "diff_rgb": diff.tolist(),
            "l2_bias": round(l2, 2),
            "delta_e": round(delta_e, 2),
            "corrected": corrected,
            "refined_rgb": np.clip(refined[lbl], 0, 255).astype(np.uint8).tolist(),
        }

    return np.clip(refined, 0, 255).astype(np.uint8), metadata


def render_mask(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Render label map to RGB including label 0."""
    h, w = labels.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl in np.unique(labels):
        mask[labels == lbl] = palette[int(lbl)]
    return mask


def abs_rgb_diff(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pixel-wise absolute RGB difference."""
    diff = np.abs(original.astype(np.float32) - mask.astype(np.float32))
    return np.clip(diff, 0, 255).astype(np.uint8)


def mean_l2(original: np.ndarray, mask: np.ndarray) -> float:
    """Mean per-pixel L2 RGB distance."""
    return float(np.linalg.norm(original.astype(np.float32) - mask.astype(np.float32), axis=2).mean())


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    """Resize image to a target height keeping aspect ratio."""
    pil = Image.fromarray(img)
    aspect = pil.width / pil.height
    new_w = int(round(height * aspect))
    return np.array(pil.resize((new_w, height), Image.LANCZOS))


def assemble_grid(
    rows: list[tuple],
    title: str,
    profile_labels: list[str],
    row_metrics: list[tuple[float, float]],
) -> Image.Image:
    """Build a comparison grid with variable-width rows (panels differ in width)."""
    header_labels = ["Original", "Initial mask", "Initial abs RGB diff", "Calibrated mask", "Calibrated abs RGB diff"]
    n_rows = len(rows)
    header_h = 60
    label_w = 140
    col_gap = 6
    row_gap = 8

    # Resize each row image to a common height; widths stay proportional.
    scaled_rows: list[list[np.ndarray]] = []
    for row in rows:
        scaled = [_resize_to_height(img, GRID_HEIGHT) for img in row]
        scaled_rows.append(scaled)

    # Compute per-column width as the maximum scaled width across rows.
    n_cols = len(header_labels)
    col_widths = [max(scaled_rows[r][c].shape[1] for r in range(n_rows)) for c in range(n_cols)]

    canvas_w = label_w + sum(col_widths) + col_gap * (n_cols - 1)
    canvas_h = header_h + n_rows * (GRID_HEIGHT + row_gap)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(16)
    big_font = _load_font(22)

    bbox = draw.textbbox((0, 0), title, font=big_font)
    title_w = bbox[2] - bbox[0]
    draw.text(((canvas_w - title_w) // 2, 14), title, fill=(220, 220, 220), font=big_font)

    x_cursor = label_w
    for hdr, cw in zip(header_labels, col_widths):
        draw.text((x_cursor + cw // 2, header_h - 26), hdr, fill=(200, 200, 200), font=font, anchor="mm")
        x_cursor += cw + col_gap

    for r, (scaled, plabel, (init_l2, cal_l2)) in enumerate(zip(scaled_rows, profile_labels, row_metrics)):
        y = header_h + r * (GRID_HEIGHT + row_gap)
        draw.text((label_w // 2, y + GRID_HEIGHT // 2), plabel, fill=(200, 200, 200), font=font, anchor="mm")
        x_cursor = label_w
        for c, img in enumerate(scaled):
            cw = col_widths[c]
            paste_x = x_cursor + (cw - img.shape[1]) // 2
            canvas.paste(Image.fromarray(img), (paste_x, y))
            if c in (2, 4):
                metric = init_l2 if c == 2 else cal_l2
                draw.text((paste_x + 4, y + 4), f"L2≈{metric:.1f}", fill=(255, 255, 255), font=font)
            x_cursor += cw + col_gap

    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_palette = load_seed_palette()

    all_medians: dict[str, dict[int, np.ndarray]] = {}
    originals: dict[str, np.ndarray] = {}
    labels_by_profile: dict[str, np.ndarray] = {}

    for profile in PROFILES:
        labels, text_mask = load_labels_and_text_mask(profile)
        original = np.array(Image.open(PANEL_DIR / f"{profile}_cropped.jpg").convert("RGB"))
        labels_by_profile[profile] = labels
        originals[profile] = original
        all_medians[profile] = compute_label_median_colors(labels, original, text_mask)

    global_medians = aggregate_global_medians(all_medians)
    refined_palette, palette_metadata = refine_palette(seed_palette, global_medians)

    # Save the refined seed reference strip.
    strip = np.zeros((40, 16 * 40, 3), dtype=np.uint8)
    for i, color in enumerate(refined_palette):
        strip[:, i * 40 : (i + 1) * 40] = color
    Image.fromarray(strip).save(OUT_DIR / "16_seed_reference_refined.jpg", quality=95)

    grid_rows: list[tuple] = []
    profile_labels: list[str] = []
    row_metrics: list[tuple[float, float]] = []
    summary_profiles: dict[str, dict] = {}

    for profile in PROFILES:
        labels = labels_by_profile[profile]
        original = originals[profile]

        initial_mask = render_mask(labels, seed_palette)
        calibrated_mask = render_mask(labels, refined_palette)

        initial_diff = abs_rgb_diff(original, initial_mask)
        calibrated_diff = abs_rgb_diff(original, calibrated_mask)

        init_l2 = mean_l2(original, initial_mask)
        cal_l2 = mean_l2(original, calibrated_mask)

        profile_out = OUT_DIR / profile
        profile_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(profile_out / "labels.npz", labels=labels)
        np.savez_compressed(profile_out / "palette_initial.npz", palette=seed_palette)
        np.savez_compressed(profile_out / "palette_refined.npz", palette=refined_palette)
        Image.fromarray(initial_mask).save(profile_out / "mask_initial.jpg", quality=95)
        Image.fromarray(calibrated_mask).save(profile_out / "mask_calibrated.jpg", quality=95)
        Image.fromarray(initial_diff).save(profile_out / "abs_diff_initial.jpg", quality=95)
        Image.fromarray(calibrated_diff).save(profile_out / "abs_diff_calibrated.jpg", quality=95)

        grid_rows.append((original, initial_mask, initial_diff, calibrated_mask, calibrated_diff))
        profile_labels.append(PROFILE_LABELS[profile])
        row_metrics.append((init_l2, cal_l2))

        summary_profiles[profile] = {
            "mean_l2_initial": round(init_l2, 4),
            "mean_l2_calibrated": round(cal_l2, 4),
            "mean_l2_reduction": round(init_l2 - cal_l2, 4),
            "n_labels": int(len(np.unique(labels))),
        }

    grid = assemble_grid(
        grid_rows,
        "Global colorbar-aware palette refinement (B)",
        profile_labels,
        row_metrics,
    )
    grid_path = OUT_DIR / "rgb_vector_diff_global_palette_refinement.jpg"
    grid.save(grid_path, quality=95)

    summary = {
        "source_labels_dir": str(LABELS_DIR),
        "panel_dir": str(PANEL_DIR),
        "baseline_dir": str(BASELINE_DIR),
        "method": "global_palette_refinement",
        "alpha": ALPHA,
        "l2_tolerance": L2_TOLERANCE,
        "delta_e_tolerance": DELTA_E_TOLERANCE,
        "seed_palette": seed_palette.tolist(),
        "refined_palette": refined_palette.tolist(),
        "palette_metadata": palette_metadata,
        "profiles": summary_profiles,
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved refined seed strip: {OUT_DIR / '16_seed_reference_refined.jpg'}")
    print(f"Saved comparison grid:    {grid_path}")
    print(f"Saved summary JSON:       {summary_path}")
    for profile in PROFILES:
        s = summary_profiles[profile]
        print(
            f"{profile}: L2 {s['mean_l2_initial']:.2f} -> {s['mean_l2_calibrated']:.2f} "
            f"(reduction {s['mean_l2_reduction']:.2f})"
        )


if __name__ == "__main__":
    main()
