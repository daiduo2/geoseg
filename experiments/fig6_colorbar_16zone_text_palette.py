#!/usr/bin/env python3
"""Fig.6 16-zone colorbar matching with text denoising + colorbar palette.

Changes from roi_preproc version:
1. Detect and remove top-label text (e.g. "sediment") in addition to PM ROIs.
2. Use the original sampled colorbar colors as the overlay/mask palette.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.color import rgb2lab

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseg.modules.segment_engines._shared import _create_overlay
from geoseg.modules.segment_engines.v4_kmeans import _sample_colorbar_seeds

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments" / "pm_repair_round3"
sys.path.insert(0, str(_EXPERIMENTS))
from pm_repair import assign_label_to_background, repair_pm_artifact  # type: ignore


ROOT = Path("/Users/daiduo2/geoseg")
PANEL_DIR = ROOT / "runs" / "feng_fig6_final_v4" / "crop_tests"
FIGURE_PATH = ROOT / "fig6_detected_panels.jpg"
OUT_DIR = ROOT / "runs" / "fig6_colorbar_16zone_text_palette"

COLORBAR_ROI = (1346, 1376, 317, 10)
PROFILES = [f"fig6_profile_{i:02d}" for i in range(3, 8)]

VISUAL_ROIS = {
    "fig6_profile_04": (124, 17, 162, 41),
    "fig6_profile_05": (95, 35, 165, 80),
}

OCR_ROIS = {
    "fig6_profile_05": (108, 39, 150, 66),
}

ZONE_TO_LAYER = {
    0: 0, 1: 0, 2: 0,
    3: 1, 4: 1, 5: 1,
    6: 2, 7: 2, 8: 2,
    9: 3, 10: 3, 11: 3,
    12: 4, 13: 4, 14: 4, 15: 4,
}


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def extract_colorbar_strip() -> np.ndarray:
    img = np.array(Image.open(FIGURE_PATH).convert("RGB"))
    x, y, w, h = COLORBAR_ROI
    return img[y : y + h, x : x + w]


def sample_16_seeds(colorbar_rgb: np.ndarray) -> np.ndarray:
    seeds, _ = _sample_colorbar_seeds(colorbar_rgb, k=16)
    return seeds


def build_layer_palette(seeds_rgb: np.ndarray) -> np.ndarray:
    """Build a 5-layer palette as the median of the seed colors in each layer."""
    palette = np.zeros((5, 3), dtype=np.uint8)
    for layer in range(5):
        zones = [z for z, l_zone in ZONE_TO_LAYER.items() if l_zone == layer]
        palette[layer] = np.median(seeds_rgb[zones], axis=0).astype(np.uint8)
    return palette


def _fill_text_nearest(image_rgb: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    if not text_mask.any():
        return image_rgb.copy()
    filled = image_rgb.copy()
    _, indices = ndimage.distance_transform_edt(~text_mask, return_indices=True)
    rr, cc = np.where(text_mask)
    filled[rr, cc] = image_rgb[indices[0][rr, cc], indices[1][rr, cc]]
    return filled


def _assign_text_to_nearest_label(
    labels: np.ndarray, text_mask: np.ndarray
) -> np.ndarray:
    if not text_mask.any():
        return labels.copy()
    cleaned = labels.copy()
    valid_mask = (~text_mask) & (labels != 0)
    if not valid_mask.any():
        return cleaned
    _, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
    rr, cc = np.where(text_mask)
    cleaned[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]
    return cleaned


def detect_text_mask(panel_rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Detect text/annotation pixels using local color outliers.

    Mirrors the approach in scripts/generate_fig6_v7_outputs.py.
    The top/background sediment layer (label 0) is excluded.
    """
    pf = panel_rgb.astype(np.float32)

    med = np.stack(
        [ndimage.median_filter(panel_rgb[:, :, c], size=7) for c in range(3)],
        axis=2,
    ).astype(np.float32)
    rgb_dist = np.linalg.norm(pf - med, axis=2)

    lab = rgb2lab(panel_rgb)
    L = lab[:, :, 0]
    L_med = ndimage.median_filter(L, size=7)

    dark_mask = (rgb_dist > 30) | ((np.abs(L - L_med) > 20) & (L < 120))

    sat = pf.max(axis=2) - pf.min(axis=2)
    sat_med = ndimage.median_filter(sat, size=7)
    bright = pf.max(axis=2)
    white_mask = (sat_med - sat > 40) & (bright > 150) & (sat < 60)

    text_mask = (dark_mask | white_mask) & (labels != 0)

    struct = np.ones((5, 5), dtype=bool)
    text_mask = ndimage.binary_dilation(text_mask, structure=struct, iterations=4)

    cc, num = ndimage.label(text_mask)
    for i in range(1, num + 1):
        comp = cc == i
        area = int(comp.sum())
        if area < 30 or area > max(15000, panel_rgb.shape[0] * panel_rgb.shape[1] // 4):
            text_mask[comp] = False

    return text_mask


def filter_label_text_mask(text_mask: np.ndarray) -> np.ndarray:
    """Keep only label-like text components (top region, not full-width boundaries)."""
    h, w = text_mask.shape
    cc, num = ndimage.label(text_mask)
    filtered = np.zeros_like(text_mask)
    for i in range(1, num + 1):
        comp = cc == i
        ys, xs = np.where(comp)
        if len(xs) == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        comp_w = x_max - x_min + 1
        comp_h = y_max - y_min + 1
        area = int(comp.sum())

        in_top_region = y_max < h * 0.45
        not_full_width = comp_w < w * 0.75
        not_thin_line = comp_h > 3
        not_tiny = area > 80

        if in_top_region and not_full_width and not_thin_line and not_tiny:
            filtered[comp] = True
    return filtered


def segment_with_16_zones(
    panel_rgb: np.ndarray,
    colorbar_rgb: np.ndarray,
    text_mask: np.ndarray,
    layer_palette: np.ndarray,
) -> dict:
    h, w = panel_rgb.shape[:2]
    panel_lab = rgb2lab(panel_rgb)

    seeds_rgb = sample_16_seeds(colorbar_rgb)
    seeds_lab = rgb2lab(seeds_rgb[np.newaxis, ...])[0]

    filled_rgb = _fill_text_nearest(panel_rgb, text_mask)
    filled_lab = rgb2lab(filled_rgb)

    flat_lab = filled_lab.reshape(-1, 3)
    d2 = ((flat_lab[:, None, :] - seeds_lab[None, :, :]) ** 2).sum(axis=2)
    zone_labels = d2.argmin(axis=1).reshape(h, w).astype(np.int32)
    layer_labels = np.vectorize(ZONE_TO_LAYER.get)(zone_labels).astype(np.int32)
    layer_labels = ndimage.median_filter(layer_labels, size=5)
    layer_labels = _assign_text_to_nearest_label(layer_labels, text_mask)

    overlay = _create_overlay(
        panel_rgb, layer_labels, layer_palette,
        alpha=0.65, boundary_mode="thin", skip_background=False, fill_mode="blend",
    )
    mask = _create_overlay(
        panel_rgb, layer_labels, layer_palette,
        alpha=1.0, boundary_mode="thin", skip_background=False, fill_mode="mask",
    )
    return {
        "labels": layer_labels,
        "overlay": overlay,
        "mask": mask,
        "seeds": seeds_rgb,
        "palette": layer_palette,
    }


def preprocess_roi(
    panel_rgb: np.ndarray,
    roi: tuple[int, int, int, int],
    inpaint_radius: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = roi
    h, w = panel_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    mask = np.zeros(panel_rgb.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    cleaned = cv2.inpaint(panel_rgb, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cleaned, mask.astype(bool)


def detect_top_label_mask(panel_rgb: np.ndarray) -> np.ndarray:
    """Detect label text in the upper portion (e.g. 'sediment')."""
    # Use a coarse initial segmentation to guide text detection away from boundaries.
    h, w = panel_rgb.shape[:2]
    lab = rgb2lab(panel_rgb)
    L = lab[:, :, 0]
    L_med = ndimage.median_filter(L, size=7)

    pf = panel_rgb.astype(np.float32)
    med = np.stack(
        [ndimage.median_filter(panel_rgb[:, :, c], size=7) for c in range(3)],
        axis=2,
    ).astype(np.float32)
    rgb_dist = np.linalg.norm(pf - med, axis=2)

    dark_mask = (rgb_dist > 30) | ((np.abs(L - L_med) > 20) & (L < 120))

    sat = pf.max(axis=2) - pf.min(axis=2)
    sat_med = ndimage.median_filter(sat, size=7)
    bright = pf.max(axis=2)
    white_mask = (sat_med - sat > 40) & (bright > 150) & (sat < 60)

    text_mask = dark_mask | white_mask
    struct = np.ones((5, 5), dtype=bool)
    text_mask = ndimage.binary_dilation(text_mask, structure=struct, iterations=4)

    cc, num = ndimage.label(text_mask)
    filtered = np.zeros_like(text_mask)
    for i in range(1, num + 1):
        comp = cc == i
        ys, xs = np.where(comp)
        if len(xs) == 0:
            continue
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        area = int(comp.sum())
        comp_w = x_max - x_min + 1

        in_top = y_max < h * 0.45
        not_full_width = comp_w < w * 0.75
        if in_top and not_full_width and area > 80:
            filtered[comp] = True
    return filtered


def run_one_variant(
    panel_id: str,
    panel_rgb: np.ndarray,
    colorbar_rgb: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    layer_palette: np.ndarray,
) -> dict:
    # Start from optional ROI preprocessing.
    if roi is not None:
        preproc, roi_mask = preprocess_roi(panel_rgb, roi)
    else:
        preproc = panel_rgb
        roi_mask = np.zeros(panel_rgb.shape[:2], dtype=bool)

    # Detect top-label text (e.g. sediment) and combine with ROI mask.
    label_mask = detect_top_label_mask(preproc)
    text_mask = roi_mask | label_mask

    result = segment_with_16_zones(preproc, colorbar_rgb, text_mask, layer_palette)
    labels = result["labels"]

    # Post-hoc PM repair (same as before) only for profiles with ROIs.
    if panel_id in VISUAL_ROIS:
        labels = assign_label_to_background(labels)
        repaired = repair_pm_artifact(panel_rgb, labels, roi=VISUAL_ROIS[panel_id])
        panel_rgb = repaired["cleaned_rgb"]
        labels = repaired["labels"]

    overlay = _create_overlay(
        panel_rgb, labels, layer_palette,
        alpha=0.65, boundary_mode="thin", skip_background=False,
        fill_mode="blend", overlay_colors=layer_palette,
    )
    mask = _create_overlay(
        panel_rgb, labels, layer_palette,
        alpha=1.0, boundary_mode="thin", skip_background=False,
        fill_mode="mask", overlay_colors=layer_palette,
    )

    # White boundaries -> black.
    white = np.all(overlay == [255, 255, 255], axis=2)
    overlay[white] = [0, 0, 0]
    white = np.all(mask == [255, 255, 255], axis=2)
    mask[white] = [0, 0, 0]

    return {
        "labels": labels,
        "overlay": overlay,
        "mask": mask,
        "text_mask": text_mask,
    }


def create_comparison_image(
    original_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    title: str,
) -> np.ndarray:
    h, w = original_rgb.shape[:2]
    label_h = 40
    canvas = np.full((h + label_h, w * 3, 3), 255, dtype=np.uint8)
    canvas[label_h:, :w] = original_rgb
    canvas[label_h:, w : 2 * w] = overlay_rgb
    canvas[label_h:, 2 * w :] = mask_rgb

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = _load_font(24)
    for i, text in enumerate(["Original", f"Overlay {title}", "Mask"]):
        x = i * w + w // 2
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x - text_w // 2, 8), text, fill=(0, 0, 0), font=font)

    return np.array(img)


def create_big_comparison() -> None:
    """Stack all variant comparisons into one big image."""
    big_path = OUT_DIR / "big_comparison.jpg"
    font = _load_font(22)
    big_font = _load_font(28)

    rows: list[Image.Image] = []
    profile_labels = {
        "fig6_profile_03": "Profile 03",
        "fig6_profile_04": "Profile 04",
        "fig6_profile_05": "Profile 05",
        "fig6_profile_06": "Profile 06",
        "fig6_profile_07": "Profile 07",
    }

    for panel_id in PROFILES:
        variants = ["baseline"]
        if panel_id in VISUAL_ROIS:
            variants.append("visual_roi")
        if panel_id in OCR_ROIS:
            variants.append("ocr_roi")
        for variant in variants:
            comp_path = OUT_DIR / panel_id / f"comparison_{variant}.jpg"
            img = Image.open(comp_path).convert("RGB")
            label = f"{profile_labels[panel_id]} — {variant}"
            # Add top label bar.
            w, h = img.size
            label_h = 36
            canvas = Image.new("RGB", (w, h + label_h), (255, 255, 255))
            canvas.paste(img, (0, label_h))
            draw = ImageDraw.Draw(canvas)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            draw.text(((w - text_w) // 2, 6), label, fill=(0, 0, 0), font=font)
            rows.append(canvas)

    max_w = max(r.width for r in rows)
    total_h = sum(r.height for r in rows)
    title_h = 60
    canvas = Image.new("RGB", (max_w, total_h + title_h), (255, 255, 255))

    draw = ImageDraw.Draw(canvas)
    title = "Fig.6 16-Zone — Text Denoising + Colorbar Palette"
    bbox = draw.textbbox((0, 0), title, font=big_font)
    title_w = bbox[2] - bbox[0]
    draw.text(((max_w - title_w) // 2, 16), title, fill=(0, 0, 0), font=big_font)

    y = title_h
    for row in rows:
        x = (max_w - row.width) // 2
        canvas.paste(row, (x, y))
        y += row.height

    canvas.save(big_path, quality=95)
    print(f"Saved big comparison: {big_path}")


def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    colorbar_rgb = extract_colorbar_strip()
    print(f"Colorbar strip shape: {colorbar_rgb.shape}")

    seeds_rgb = sample_16_seeds(colorbar_rgb)
    layer_palette = build_layer_palette(seeds_rgb)
    print(f"Layer palette (BGR):\n{layer_palette}")

    summary: dict = {
        "colorbar_roi": COLORBAR_ROI,
        "layer_palette": layer_palette.tolist(),
        "profiles": {},
    }

    for panel_id in PROFILES:
        print(f"\n=== {panel_id} ===")
        original = np.array(
            Image.open(PANEL_DIR / f"{panel_id}_cropped.jpg").convert("RGB")
        )
        print(f"  Panel shape: {original.shape}")

        variants: dict[str, tuple[np.ndarray, dict]] = {}

        result = run_one_variant(panel_id, original, colorbar_rgb, roi=None, layer_palette=layer_palette)
        variants["baseline"] = (original, result)

        if panel_id in VISUAL_ROIS:
            result = run_one_variant(
                panel_id, original, colorbar_rgb, roi=VISUAL_ROIS[panel_id], layer_palette=layer_palette
            )
            variants["visual_roi"] = (original, result)

        if panel_id in OCR_ROIS:
            result = run_one_variant(
                panel_id, original, colorbar_rgb, roi=OCR_ROIS[panel_id], layer_palette=layer_palette
            )
            variants["ocr_roi"] = (original, result)

        panel_out = OUT_DIR / panel_id
        panel_out.mkdir(parents=True, exist_ok=True)

        panel_summary = {}
        for variant_name, (orig_img, res) in variants.items():
            comp = create_comparison_image(orig_img, res["overlay"], res["mask"], variant_name)
            comp_path = panel_out / f"comparison_{variant_name}.jpg"
            Image.fromarray(comp).save(comp_path, quality=95)
            np.savez_compressed(panel_out / f"labels_{variant_name}.npz", labels=res["labels"])
            np.savez_compressed(panel_out / f"text_mask_{variant_name}.npz", mask=res["text_mask"])
            panel_summary[variant_name] = {
                "comparison_path": str(comp_path),
                "n_labels": int(len(np.unique(res["labels"]))),
                "text_mask_pixels": int(res["text_mask"].sum()),
            }
            print(f"  {variant_name}: saved {comp_path}")

        summary["profiles"][panel_id] = panel_summary

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    create_big_comparison()
    print(f"\nAll outputs: {OUT_DIR}")
    return summary


if __name__ == "__main__":
    main()
