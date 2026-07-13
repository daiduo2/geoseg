#!/usr/bin/env python3
"""Generate abs RGB vector difference comparison for global palette refinement,
with an optional local ROI override for profile 05 around the PM artifact.

For the local override:
- Inside the user-circled ROI, each label's displayed color is replaced by the
  median original color of that label within the ROI only.
- Outside the ROI the global refined palette is used.
- This demonstrates "take the circled region out and correct it separately by
  sampling from the original image at the same geometry".
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
OUT_DIR = GLOBAL_DIR / "local_roi_correction"

PROFILES = [f"fig6_profile_{i:02d}" for i in range(3, 8)]
PROFILE_LABELS = {
    "fig6_profile_03": "Profile 03",
    "fig6_profile_04": "Profile 04 (PM artifact)",
    "fig6_profile_05": "Profile 05 (PM artifact)",
    "fig6_profile_06": "Profile 06",
    "fig6_profile_07": "Profile 07",
}

# Local ROI for profile 05 PM artifact region (user-circled area).
# Coordinates are (x1, y1, x2, y2) in full-panel pixel space.
LOCAL_ROI_05 = (50, 25, 320, 90)


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


def load_global_refined_palette() -> np.ndarray:
    summary = json.loads((GLOBAL_DIR / "summary.json").read_text(encoding="utf-8"))
    return np.array(summary["refined_palette"], dtype=np.uint8)


def render_mask(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl in np.unique(labels):
        mask[labels == lbl] = palette[lbl]
    return mask


def apply_local_roi_correction(
    labels: np.ndarray,
    original: np.ndarray,
    global_palette: np.ndarray,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    """Build a spatially-varying mask: global palette outside ROI,
    per-label median original color inside ROI.
    """
    x1, y1, x2, y2 = roi
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(labels.shape[1], x2), min(labels.shape[0], y2)

    corrected_palette = global_palette.copy()
    roi_mask = np.zeros_like(labels, dtype=bool)
    roi_mask[y1:y2, x1:x2] = True
    for lbl in np.unique(labels[roi_mask]):
        label_in_roi = (labels == lbl) & roi_mask
        if label_in_roi.any():
            corrected_palette[lbl] = np.median(original[label_in_roi], axis=0).astype(np.uint8)

    return render_mask(labels, corrected_palette)


def abs_rgb_diff(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    diff = np.abs(original.astype(np.float32) - mask.astype(np.float32))
    return np.clip(diff, 0, 255).astype(np.uint8)


def mean_l2(original: np.ndarray, mask: np.ndarray) -> float:
    return float(np.linalg.norm(original.astype(np.float32) - mask.astype(np.float32), axis=2).mean())


def draw_roi_box(img: np.ndarray, roi: tuple[int, int, int, int], color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    x1, y1, x2, y2 = roi
    out = img.copy()
    out[y1:y2, x1] = color
    out[y1:y2, x2 - 1] = color
    out[y1, x1:x2] = color
    out[y2 - 1, x1:x2] = color
    return out


def assemble_grid(
    rows: list[tuple],
    title: str,
    profile_labels: list[str],
    row_metrics: list[tuple[float, float, float]],
) -> Image.Image:
    header_labels = ["Original", "Global mask", "Global abs RGB diff", "Local-corrected mask", "Local-corrected abs RGB diff"]
    n_rows = len(rows)
    h, w = rows[0][0].shape[:2]
    header_h = 50
    label_w = 170
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

    for r, (row, plabel, (global_l2, local_l2)) in enumerate(zip(rows, profile_labels, row_metrics)):
        y = header_h + r * (h + row_gap)
        draw.text((label_w // 2, y + h // 2), plabel, fill=(200, 200, 200), font=font, anchor="mm")
        for c, img in enumerate(row):
            x = label_w + c * (w + col_gap)
            canvas.paste(Image.fromarray(img), (x, y))
            if c in (2, 4):
                metric = global_l2 if c == 2 else local_l2
                metric_text = f"L2≈{metric:.1f}"
                draw.text((x + 4, y + 4), metric_text, fill=(255, 255, 255), font=font)

    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    palette = load_global_refined_palette()

    rows: list[tuple] = []
    profile_labels: list[str] = []
    row_metrics: list[tuple[float, float]] = []
    summary: dict[str, dict] = {}

    for panel_id in PROFILES:
        labels = np.load(LABELS_DIR / panel_id / "labels.npz")["labels"]
        original = np.array(Image.open(PANEL_DIR / f"{panel_id}_cropped.jpg").convert("RGB"))

        global_mask = render_mask(labels, palette)
        global_diff = abs_rgb_diff(original, global_mask)
        global_l2 = mean_l2(original, global_mask)

        if panel_id == "fig6_profile_05":
            local_mask = apply_local_roi_correction(labels, original, palette, LOCAL_ROI_05)
            local_mask_box = draw_roi_box(local_mask, LOCAL_ROI_05)
            local_diff = abs_rgb_diff(original, local_mask)
            local_diff_box = draw_roi_box(local_diff, LOCAL_ROI_05)
            roi_label = f" (ROI {LOCAL_ROI_05})"
        else:
            local_mask = global_mask
            local_mask_box = global_mask
            local_diff = global_diff
            local_diff_box = global_diff
            roi_label = ""

        local_l2 = mean_l2(original, local_mask)

        profile_out = OUT_DIR / panel_id
        profile_out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(global_mask).save(profile_out / "mask_global.jpg", quality=95)
        Image.fromarray(global_diff).save(profile_out / "diff_global.jpg", quality=95)
        if panel_id == "fig6_profile_05":
            Image.fromarray(local_mask).save(profile_out / "mask_local_corrected.jpg", quality=95)
            Image.fromarray(local_diff).save(profile_out / "diff_local_corrected.jpg", quality=95)

        rows.append((original, global_mask, global_diff, local_mask_box, local_diff_box))
        profile_labels.append(PROFILE_LABELS[panel_id] + roi_label)
        row_metrics.append((global_l2, local_l2))

        summary[panel_id] = {
            "global_l2": round(global_l2, 2),
            "local_l2": round(local_l2, 2),
            "reduction": round(global_l2 - local_l2, 2),
        }

    grid = assemble_grid(
        rows,
        "Global palette refinement + local ROI correction (profile 05 PM region)",
        profile_labels,
        row_metrics,
    )
    grid_path = OUT_DIR / "abs_rgb_diff_global_plus_local_v9.jpg"
    grid.save(grid_path, quality=95)

    # Focused single-row view for profile 05 so the circled correction is easier to inspect.
    idx_05 = PROFILES.index("fig6_profile_05")
    focused_grid = assemble_grid(
        [rows[idx_05]],
        "Profile 05: global palette refinement vs local ROI correction",
        [profile_labels[idx_05]],
        [row_metrics[idx_05]],
    )
    focused_path = OUT_DIR / "abs_rgb_diff_global_plus_local_profile05_focus_v9.jpg"
    focused_grid.save(focused_path, quality=95)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved grid: {grid_path}")
    print(f"Saved focused grid: {focused_path}")
    print(f"Saved summary: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
