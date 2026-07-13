#!/usr/bin/env python3
"""Text-aware color correction that keeps geological labels unified.

Label 12 (and similar labels) mix real geological pixels with PM text/annotation.
Previous attempts split these into separate labels, which fragmented the visual
result and made the problem worse.

This script keeps every label unified. For each high-residual label we:
1. Detect text/annotation pixels inside the label by luminance (dark <55 or bright >210).
2. Compute the optimal color using ONLY the non-text geology pixels.
3. Apply that color to the entire label.

Text pixels are allowed to keep a high residual; they do not break the layer
into a separate label. The geological layer itself becomes a solid, correctly
colored region.
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
OUT_DIR = GLOBAL_DIR / "text_aware_unified_correction"

PROFILES = [f"fig6_profile_{i:02d}" for i in range(3, 8)]
PROFILE_LABELS = {
    "fig6_profile_03": "Profile 03",
    "fig6_profile_04": "Profile 04 (PM artifact)",
    "fig6_profile_05": "Profile 05 (PM artifact)",
    "fig6_profile_06": "Profile 06",
    "fig6_profile_07": "Profile 07",
}

RESIDUAL_THRESHOLD = 25.0
MAX_ITERATIONS = 10

SEARCH_RANGE = 30
SEARCH_STEP = 3
FINE_RANGE = 5
FINE_STEP = 1

# Pixels darker/brighter than these are treated as text/annotation when
# computing the optimal color for a label. They remain in the label visually.
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


def detect_text_mask_for_label(original: np.ndarray, label_mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask (relative to the label) of text/annotation pixels."""
    gray = original[label_mask].mean(axis=1)
    return (gray < DARK_THRESHOLD) | (gray > BRIGHT_THRESHOLD)


def render_mask(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl in np.unique(labels):
        if lbl < len(palette):
            mask[labels == lbl] = palette[lbl]
    return mask


def per_label_residuals(original: np.ndarray, mask: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    l2 = np.linalg.norm(original.astype(np.float32) - mask.astype(np.float32), axis=2)
    residuals: dict[int, float] = {}
    for lbl in np.unique(labels):
        m = labels == lbl
        if m.any():
            residuals[int(lbl)] = float(l2[m].mean())
    return residuals


def _search_best_color(
    pixels: np.ndarray,
    center: np.ndarray,
    range_size: int,
    step: int,
    batch_size: int = 1024,
) -> tuple[np.ndarray, float]:
    """Return (best_color, best_residual) for a batch of candidate colors."""
    center = center.astype(np.int32)
    channels = [
        np.arange(max(0, center[i] - range_size), min(256, center[i] + range_size + 1), step)
        for i in range(3)
    ]
    rr, gg, bb = np.meshgrid(channels[0], channels[1], channels[2], indexing="ij")
    candidates = np.stack([rr.ravel(), gg.ravel(), bb.ravel()], axis=1).astype(np.float32)

    best_color = center.astype(np.uint8)
    best_residual = float("inf")

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        diff = pixels[None, :, :] - batch[:, None, :]
        residuals = np.linalg.norm(diff, axis=2).mean(axis=1)
        idx = int(np.argmin(residuals))
        if residuals[idx] < best_residual:
            best_residual = float(residuals[idx])
            best_color = batch[idx].astype(np.uint8)

    return best_color, best_residual


def text_aware_optimal_color(
    original: np.ndarray,
    labels: np.ndarray,
    lbl: int,
) -> np.ndarray:
    """Find the color that minimizes mean L2 residual for the non-text pixels of a label."""
    label_mask = labels == lbl
    pixels = original[label_mask].astype(np.float32)

    text_mask = detect_text_mask_for_label(original, label_mask)
    geology_pixels = pixels[~text_mask]

    # If the label is almost entirely text, fall back to all pixels to avoid
    # degenerate optimization.
    if len(geology_pixels) < 100:
        geology_pixels = pixels

    median = np.median(geology_pixels, axis=0).astype(np.uint8)
    coarse_best, _ = _search_best_color(geology_pixels, median, SEARCH_RANGE, SEARCH_STEP)
    fine_best, _ = _search_best_color(geology_pixels, coarse_best, FINE_RANGE, FINE_STEP)
    return fine_best


def iterative_text_aware_correct(
    labels: np.ndarray,
    original: np.ndarray,
    base_palette: np.ndarray,
    threshold: float,
    max_iter: int,
) -> tuple[np.ndarray, list[dict[int, float]], list[int]]:
    palette = base_palette.copy()
    history: list[dict[int, float]] = []
    corrected_labels: set[int] = set()

    for _ in range(max_iter):
        mask = render_mask(labels, palette)
        residuals = per_label_residuals(original, mask, labels)
        history.append(residuals)

        high = {lbl for lbl, r in residuals.items() if r > threshold}
        if not high:
            break

        for lbl in high:
            palette[lbl] = text_aware_optimal_color(original, labels, lbl)
            corrected_labels.add(lbl)

        new_mask = render_mask(labels, palette)
        new_residuals = per_label_residuals(original, new_mask, labels)
        max_change = max(abs(new_residuals[lbl] - residuals[lbl]) for lbl in high)
        if max_change < 1e-3:
            history.append(new_residuals)
            break

    return palette, history, sorted(corrected_labels)


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
    row_corrections: list[list[int]],
) -> Image.Image:
    header_labels = ["Original", "Global mask", "Global abs RGB diff", "Text-aware unified mask", "Text-aware unified abs RGB diff"]
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

    for r, (row, plabel, (global_l2, corr_l2), corrected) in enumerate(zip(rows, profile_labels, row_metrics, row_corrections)):
        y = header_h + r * (h + row_gap)
        label_text = plabel + (f"\ncorrected: {corrected}" if corrected else "\nno correction")
        draw.text((label_w // 2, y + h // 2), label_text, fill=(200, 200, 200), font=font, anchor="mm")
        for c, img in enumerate(row):
            x = label_w + c * (w + col_gap)
            canvas.paste(Image.fromarray(img), (x, y))
            if c in (2, 4):
                metric = global_l2 if c == 2 else corr_l2
                metric_text = f"L2≈{metric:.1f}"
                draw.text((x + 4, y + 4), metric_text, fill=(255, 255, 255), font=font)

    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_palette = load_global_refined_palette()

    rows: list[tuple] = []
    profile_labels: list[str] = []
    row_metrics: list[tuple[float, float]] = []
    row_corrections: list[list[int]] = []
    summary: dict[str, dict] = {}

    for panel_id in PROFILES:
        labels = np.load(LABELS_DIR / panel_id / "labels.npz")["labels"]
        original = np.array(Image.open(PANEL_DIR / f"{panel_id}_cropped.jpg").convert("RGB"))

        global_mask = render_mask(labels, base_palette)
        global_diff = abs_rgb_diff(original, global_mask)
        global_l2 = mean_l2(original, global_mask)

        corrected_palette, history, corrected = iterative_text_aware_correct(
            labels, original, base_palette, RESIDUAL_THRESHOLD, MAX_ITERATIONS
        )
        corrected_mask = render_mask(labels, corrected_palette)
        corrected_diff = abs_rgb_diff(original, corrected_mask)
        corrected_l2 = mean_l2(original, corrected_mask)

        profile_out = OUT_DIR / panel_id
        profile_out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(global_mask).save(profile_out / "mask_global.jpg", quality=95)
        Image.fromarray(global_diff).save(profile_out / "diff_global.jpg", quality=95)
        Image.fromarray(corrected_mask).save(profile_out / "mask_corrected.jpg", quality=95)
        Image.fromarray(corrected_diff).save(profile_out / "diff_corrected.jpg", quality=95)
        (profile_out / "residual_history.json").write_text(
            json.dumps({f"iter_{i}": {str(k): round(v, 2) for k, v in hist.items()} for i, hist in enumerate(history)}, indent=2),
            encoding="utf-8",
        )

        rows.append((original, global_mask, global_diff, corrected_mask, corrected_diff))
        profile_labels.append(PROFILE_LABELS[panel_id])
        row_metrics.append((global_l2, corrected_l2))
        row_corrections.append(corrected)

        summary[panel_id] = {
            "global_l2": round(global_l2, 2),
            "corrected_l2": round(corrected_l2, 2),
            "reduction": round(global_l2 - corrected_l2, 2),
            "iterations": len(history) - 1 if len(history) > 1 else len(history),
            "final_per_label_residuals": {str(k): round(v, 2) for k, v in history[-1].items()},
            "corrected_labels": corrected,
        }
        print(f"{panel_id}: corrected {corrected}, L2 {global_l2:.1f} -> {corrected_l2:.1f}")

    grid = assemble_grid(
        rows,
        f"Text-aware unified correction (threshold={RESIDUAL_THRESHOLD})",
        profile_labels,
        row_metrics,
        row_corrections,
    )
    grid_path = OUT_DIR / "abs_rgb_diff_text_aware_unified_v15.jpg"
    grid.save(grid_path, quality=95)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved grid: {grid_path}")
    print(f"Saved summary: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
