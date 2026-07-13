#!/usr/bin/env python3
"""Fig.6 16-zone colorbar matching with ROI-based PM preprocessing.

For profiles 04/05, inpaint the known PM ROI *before* segmentation so the
PM pixels do not bias the 16-zone colorbar assignment. Compare three ROI
variants: no ROI, visual ROI, and (for 05) OCR ROI.
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
OUT_DIR = ROOT / "runs" / "fig6_colorbar_16zone_roi_preproc"

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


def segment_with_16_zones(
    panel_rgb: np.ndarray,
    colorbar_rgb: np.ndarray,
    text_mask: np.ndarray,
) -> dict:
    h, w = panel_rgb.shape[:2]
    panel_lab = rgb2lab(panel_rgb)

    seeds_rgb, _ = _sample_colorbar_seeds(colorbar_rgb, 16)
    seeds_lab = rgb2lab(seeds_rgb[np.newaxis, ...])[0]

    filled_rgb = _fill_text_nearest(panel_rgb, text_mask)
    filled_lab = rgb2lab(filled_rgb)

    flat_lab = filled_lab.reshape(-1, 3)
    d2 = ((flat_lab[:, None, :] - seeds_lab[None, :, :]) ** 2).sum(axis=2)
    zone_labels = d2.argmin(axis=1).reshape(h, w).astype(np.int32)
    layer_labels = np.vectorize(ZONE_TO_LAYER.get)(zone_labels).astype(np.int32)
    layer_labels = ndimage.median_filter(layer_labels, size=5)
    layer_labels = _assign_text_to_nearest_label(layer_labels, text_mask)

    palette = np.zeros((5, 3), dtype=np.uint8)
    for lbl in range(5):
        mask = layer_labels == lbl
        if mask.any():
            palette[lbl] = np.median(panel_rgb[mask], axis=0).astype(np.uint8)

    overlay = _create_overlay(
        panel_rgb, layer_labels, palette,
        alpha=0.65, boundary_mode="thin", skip_background=False, fill_mode="blend",
    )
    mask = _create_overlay(
        panel_rgb, layer_labels, palette,
        alpha=1.0, boundary_mode="thin", skip_background=False, fill_mode="mask",
    )
    return {"labels": layer_labels, "overlay": overlay, "mask": mask, "palette": palette}


def preprocess_roi(
    panel_rgb: np.ndarray,
    roi: tuple[int, int, int, int],
    inpaint_radius: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Inpaint a single ROI and return cleaned image + boolean mask."""
    x1, y1, x2, y2 = roi
    h, w = panel_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    mask = np.zeros(panel_rgb.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    cleaned = cv2.inpaint(panel_rgb, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cleaned, mask.astype(bool)


def run_one_variant(
    panel_id: str,
    panel_rgb: np.ndarray,
    colorbar_rgb: np.ndarray,
    roi: tuple[int, int, int, int] | None,
) -> dict:
    """Run 16-zone segmentation with optional ROI preprocessing."""
    if roi is not None:
        preproc, roi_mask = preprocess_roi(panel_rgb, roi)
        text_mask = roi_mask
    else:
        preproc = panel_rgb
        text_mask = np.zeros(panel_rgb.shape[:2], dtype=bool)

    result = segment_with_16_zones(preproc, colorbar_rgb, text_mask)
    labels = result["labels"]

    # Post-hoc PM repair (same as before) only for profiles with ROIs.
    if panel_id in VISUAL_ROIS:
        labels = assign_label_to_background(labels)
        repaired = repair_pm_artifact(panel_rgb, labels, roi=VISUAL_ROIS[panel_id])
        panel_rgb = repaired["cleaned_rgb"]
        labels = repaired["labels"]

    # Recolor overlay from final panel/labels.
    colors: dict[int, np.ndarray] = {}
    for lbl in np.unique(labels):
        mask = labels == lbl
        if mask.any():
            colors[int(lbl)] = np.median(panel_rgb[mask], axis=0).astype(np.uint8)

    overlay_colors = np.zeros((max(colors.keys()) + 1, 3), dtype=np.uint8)
    for lbl, color in colors.items():
        overlay_colors[lbl] = color

    overlay = _create_overlay(
        panel_rgb, labels, overlay_colors,
        alpha=0.65, boundary_mode="thin", skip_background=False,
        fill_mode="blend", overlay_colors=overlay_colors,
    )
    mask = _create_overlay(
        panel_rgb, labels, overlay_colors,
        alpha=1.0, boundary_mode="thin", skip_background=False,
        fill_mode="mask", overlay_colors=overlay_colors,
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


def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    colorbar_rgb = extract_colorbar_strip()
    print(f"Colorbar strip shape: {colorbar_rgb.shape}")

    summary: dict = {"colorbar_roi": COLORBAR_ROI, "profiles": {}}

    for panel_id in PROFILES:
        print(f"\n=== {panel_id} ===")
        original = np.array(
            Image.open(PANEL_DIR / f"{panel_id}_cropped.jpg").convert("RGB")
        )
        print(f"  Panel shape: {original.shape}")

        variants: dict[str, tuple[np.ndarray, dict]] = {}

        # Baseline: no ROI preprocessing.
        result = run_one_variant(panel_id, original, colorbar_rgb, roi=None)
        variants["baseline"] = (original, result)

        # Visual ROI preprocessing.
        if panel_id in VISUAL_ROIS:
            result = run_one_variant(
                panel_id, original, colorbar_rgb, roi=VISUAL_ROIS[panel_id]
            )
            variants["visual_roi"] = (original, result)

        # OCR ROI preprocessing.
        if panel_id in OCR_ROIS:
            result = run_one_variant(
                panel_id, original, colorbar_rgb, roi=OCR_ROIS[panel_id]
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
            panel_summary[variant_name] = {
                "comparison_path": str(comp_path),
                "n_labels": int(len(np.unique(res["labels"]))),
            }
            print(f"  {variant_name}: saved {comp_path}")

        summary["profiles"][panel_id] = panel_summary

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nAll outputs: {OUT_DIR}")
    return summary


if __name__ == "__main__":
    main()
