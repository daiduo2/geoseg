#!/usr/bin/env python3
"""Iterative residual-threshold color correction.

Treat every high-residual region as color bias. For each profile:
1. Start from the globally-refined shared palette.
2. Repeatedly find labels whose mean L2 RGB residual exceeds a threshold.
3. Set each offending label's color to the L2-optimal color = per-channel mean
   of the original pixels belonging to that label.
4. Stop when all labels are below the threshold or no further improvement.

This is purely residual-driven: no manual ROI, no segmentation assumption.
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
OUT_DIR = GLOBAL_DIR / "iterative_threshold_correction"

PROFILES = [f"fig6_profile_{i:02d}" for i in range(3, 8)]
PROFILE_LABELS = {
    "fig6_profile_03": "Profile 03",
    "fig6_profile_04": "Profile 04 (PM artifact)",
    "fig6_profile_05": "Profile 05 (PM artifact)",
    "fig6_profile_06": "Profile 06",
    "fig6_profile_07": "Profile 07",
}

# Stop when every label's mean L2 residual is below this value.
RESIDUAL_THRESHOLD = 25.0
MAX_ITERATIONS = 10


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


def render_mask(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl in np.unique(labels):
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


def l1_optimal_color(original: np.ndarray, labels: np.ndarray, lbl: int) -> np.ndarray:
    """L1-optimal color = per-channel median of the original pixels."""
    pixels = original[labels == lbl].astype(np.float32)
    return np.median(pixels, axis=0).astype(np.uint8)


def iterative_correct(
    labels: np.ndarray,
    original: np.ndarray,
    base_palette: np.ndarray,
    threshold: float,
    max_iter: int,
) -> tuple[np.ndarray, list[dict[int, float]], list[int]]:
    """Return (corrected_palette, residual_history, corrected_labels)."""
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
            palette[lbl] = l1_optimal_color(original, labels, lbl)
            corrected_labels.add(lbl)

        # Check whether the correction actually moved residuals.
        new_mask = render_mask(labels, palette)
        new_residuals = per_label_residuals(original, new_mask, labels)
        max_change = max(abs(new_residuals[lbl] - residuals[lbl]) for lbl in high)
        if max_change < 1e-3:
            # Median is already optimal; further iterations won't help.
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
    row_metrics: list[tuple[float, float, float]],
    row_corrections: list[list[int]],
) -> Image.Image:
    header_labels = ["Original", "Global mask", "Global abs RGB diff", "Iter-corrected mask", "Iter-corrected abs RGB diff"]
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

        corrected_palette, history, corrected = iterative_correct(
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
        print(f"{panel_id}: corrected {corrected}, L2 {global_l2:.1f} -> {corrected_l2:.1f}, iters {summary[panel_id]['iterations']}")

    grid = assemble_grid(
        rows,
        f"Iterative residual-threshold correction (threshold={RESIDUAL_THRESHOLD}, max_iter={MAX_ITERATIONS})",
        profile_labels,
        row_metrics,
        row_corrections,
    )
    grid_path = OUT_DIR / "abs_rgb_diff_iterative_threshold_v11.jpg"
    grid.save(grid_path, quality=95)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved grid: {grid_path}")
    print(f"Saved summary: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
