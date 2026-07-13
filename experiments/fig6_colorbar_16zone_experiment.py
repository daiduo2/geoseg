#!/usr/bin/env python3
"""Fig.6 colorbar 16-zone matching experiment.

Pipeline (per profile):
1. Load cropped panel.
2. Extract shared colorbar strip from fig6_detected_panels.jpg.
3. Sample 16 evenly-spaced seed colors from the colorbar.
4. Run mask-aware nearest-seed segmentation in LAB space.
5. Optionally merge 16 zones -> 5 geological layers by grouping adjacent seeds.
6. Detect & smooth general text annotations into nearest label.
7. Apply targeted PM artifact repair for profiles 04/05 (same ROIs as before).
8. Save overlay, mask, and side-by-side comparison.

Self-contained: only reads existing source modules; does not modify them.
"""
from __future__ import annotations

import argparse
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

# Also import the experimental PM repair helper.
_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments" / "pm_repair_round3"
sys.path.insert(0, str(_EXPERIMENTS))
from pm_repair import (
    repair_pm_artifact,
    repair_pm_artifact_no_merge,
    assign_label_to_background,
)  # type: ignore


ROOT = Path("/Users/daiduo2/geoseg")
PANEL_DIR = ROOT / "runs" / "feng_fig6_final_v4" / "crop_tests"
FIGURE_PATH = ROOT / "fig6_detected_panels.jpg"
OUT_DIR = ROOT / "runs" / "fig6_colorbar_16zone_experiment"

# Right-hand colorbar ROI in fig6_detected_panels.jpg (red-to-blue scale).
# Tight crop around the actual colored strip, excluding white/black margins.
COLORBAR_ROI = (1346, 1376, 317, 10)

PROFILES = [f"fig6_profile_{i:02d}" for i in range(3, 8)]

# PM repair ROIs inherited from pm_repair.py.
PM_REPAIR_ROIS = {
    "fig6_profile_04": (124, 17, 162, 41),
    "fig6_profile_05": (95, 35, 165, 80),
}

# In no-merge mode text inpainting pulls cyan/blue lower-seed labels into the
# PM annotation ROIs. These are the same ROIs plus the two PM instances on
# profile 06 that also need the same cleanup.
PM_REPAIR_ROIS_NO_MERGE = {
    "fig6_profile_04": [(124, 17, 162, 41)],
    "fig6_profile_05": [(95, 35, 165, 80)],
    "fig6_profile_06": [
        (315, 35, 375, 65),  # left PM
        (470, 20, 540, 70),  # right PM
    ],
}

# Tight ROIs for known text annotations. Each is processed locally so only the
# actual dark strokes inside the ROI are masked, not the whole rectangle.
TEXT_ROIS = {
    "fig6_profile_04": [(120, 15, 165, 45)],      # PM
    "fig6_profile_05": [(100, 38, 160, 72)],      # PM
    "fig6_profile_06": [
        (55, 22, 100, 50),   # left BM
        (135, 22, 180, 50),  # second BM
        (200, 25, 275, 75),  # LVS + star
        (315, 35, 375, 65),  # left PM
        (470, 20, 540, 70),  # right PM
        (540, 35, 610, 75),  # LV-N
    ],
    "fig6_profile_07": [(140, 38, 190, 68)],      # LV-N
}

# Merge 16 consecutive zones into 5 layer groups (top/red -> bottom/blue).
ZONE_TO_LAYER = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 1,
    6: 2,
    7: 2,
    8: 2,
    9: 3,
    10: 3,
    11: 3,
    12: 4,
    13: 4,
    14: 4,
    15: 4,
}


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def extract_colorbar_strip() -> np.ndarray:
    """Return the shared colorbar strip from the full figure."""
    img = np.array(Image.open(FIGURE_PATH).convert("RGB"))
    x, y, w, h = COLORBAR_ROI
    return img[y : y + h, x : x + w]


def sample_16_seeds(colorbar_rgb: np.ndarray) -> np.ndarray:
    """Sample 16 evenly-spaced RGB seeds along the colorbar."""
    seeds, _ = _sample_colorbar_seeds(colorbar_rgb, k=16)
    return seeds


def create_seed_reference_strip(seeds_rgb: np.ndarray, height: int = 40) -> np.ndarray:
    """Build a horizontal strip showing the 16 sampled colorbar seed colors."""
    n = len(seeds_rgb)
    strip = np.zeros((height, n * height, 3), dtype=np.uint8)
    for i, color in enumerate(seeds_rgb):
        strip[:, i * height : (i + 1) * height] = color
    return strip


def _fill_text_nearest(image_rgb: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    """Fill text pixels with the nearest non-text pixel color."""
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
    """Assign text pixels to the nearest valid (non-text, non-zero) label."""
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


def _mask_dark_text_in_roi(
    panel_rgb: np.ndarray, roi: tuple[int, int, int, int]
) -> np.ndarray:
    """Return a tight text mask inside a known annotation ROI.

    The local background inside a small ROI is the colored layer, so dark
    letters can be found by thresholding the difference to the local median
    lightness. A small closing fills gaps inside letters without growing the
    mask into surrounding geology.
    """
    x1, y1, x2, y2 = roi
    h, w = panel_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), dtype=bool)

    crop = panel_rgb[y1:y2, x1:x2]
    lab = rgb2lab(crop)
    L = lab[:, :, 0]
    L_med = ndimage.median_filter(L, size=5)
    local_dark = (L < 95) & (L_med - L > 8)
    local_dark = ndimage.binary_closing(local_dark, iterations=1)
    local_dark = ndimage.binary_dilation(local_dark, iterations=2)

    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = local_dark
    return mask


def build_text_mask(panel_rgb: np.ndarray, panel_id: str = "") -> np.ndarray:
    """Detect text/annotation pixels including 'PM' and top-right 'sediment'.

    Uses three cues:
    - A global dark-letter detector (local-median lightness difference) with a
      tight area filter for small annotations.
    - A top-right bright low-saturation detector for the white 'sediment' label.
    - Known text ROIs from ``TEXT_ROIS`` are processed locally to guarantee that
      e.g. 'PM' and 'LV-N' are fully covered without masking the whole rectangle.

    Large geological regions are protected by connected-component area filters.
    The global adaptive+Laplacian text estimator is intentionally not used
    because it catches color boundaries and creates thousands of false positives.
    """
    h, w = panel_rgb.shape[:2]
    lab = rgb2lab(panel_rgb)
    L = lab[:, :, 0]

    # 1. Global dark letters / symbols on lighter backgrounds.
    L_med = ndimage.median_filter(L, size=7)
    dark_text = (L < 90) & (L_med - L > 12)
    dark_text = ndimage.binary_opening(dark_text, iterations=1)
    dark_text = ndimage.binary_dilation(dark_text, iterations=2)

    cc, num = ndimage.label(dark_text)
    for i in range(1, num + 1):
        comp = cc == i
        area = int(comp.sum())
        if area < 25 or area > 800:
            dark_text[comp] = False

    # 2. Top-right white 'sediment' label.
    sat = panel_rgb.max(axis=2) - panel_rgb.min(axis=2)
    bright = panel_rgb.max(axis=2)
    top_right = np.zeros((h, w), dtype=bool)
    x0 = int(w * 0.72)
    y1 = int(h * 0.30)
    top_right[:y1, x0:] = True
    sediment_mask = top_right & (bright > 180) & (sat < 70)
    sediment_mask = ndimage.binary_dilation(sediment_mask, iterations=2)

    cc, num = ndimage.label(sediment_mask)
    for i in range(1, num + 1):
        comp = cc == i
        area = int(comp.sum())
        if area < 30 or area > 4000:
            sediment_mask[comp] = False

    # 3. Local text masks for known annotation ROIs.
    combined = dark_text | sediment_mask
    for roi in TEXT_ROIS.get(panel_id, []):
        combined |= _mask_dark_text_in_roi(panel_rgb, roi)

    combined = ndimage.binary_dilation(combined, iterations=1)
    return combined


def preprocess_remove_text(
    panel_rgb: np.ndarray,
    panel_id: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a text mask and produce two cleaned versions.

    - ``segment_rgb``: text pixels replaced by the nearest non-text pixel color
      before segmentation, so clustering is not biased by annotation colors.
    - ``visual_rgb``: OpenCV inpainted version for display.

    Returns ``(segment_rgb, text_mask, visual_rgb)``.
    """
    text_mask = build_text_mask(panel_rgb, panel_id)
    if not text_mask.any():
        return panel_rgb.copy(), text_mask, panel_rgb.copy()

    segment_rgb = _fill_text_nearest(panel_rgb, text_mask)
    visual_rgb = cv2.inpaint(
        panel_rgb,
        text_mask.astype(np.uint8) * 255,
        inpaintRadius=7,
        flags=cv2.INPAINT_TELEA,
    )
    return segment_rgb, text_mask, visual_rgb


def detect_text_mask(panel_rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Detect text/annotation pixels using local color outliers.

    Mirrors the approach in scripts/generate_fig6_v7_outputs.py:
    - RGB local median distance (catches dark text and sharp symbols).
    - LAB L-channel local median (catches dark strokes).
    - Saturation drop on bright pixels (catches white text on colored layers).

    The top/background sediment layer (label 0) is excluded.
    """
    pf = panel_rgb.astype(np.float32)
    h, w = panel_rgb.shape[:2]

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
        if area < 30 or area > max(15000, h * w // 4):
            text_mask[comp] = False

    return text_mask


def segment_with_16_zones(
    panel_rgb: np.ndarray,
    colorbar_rgb: np.ndarray,
    text_mask: np.ndarray,
    merge_zones: bool = True,
) -> dict:
    """Nearest-seed segmentation using 16 colorbar zones."""
    h, w = panel_rgb.shape[:2]
    panel_lab = rgb2lab(panel_rgb)

    seeds_rgb = sample_16_seeds(colorbar_rgb)
    seeds_lab = rgb2lab(seeds_rgb[np.newaxis, ...])[0]

    # Fill text pixels before clustering.
    filled_rgb = _fill_text_nearest(panel_rgb, text_mask)
    filled_lab = rgb2lab(filled_rgb)

    # Nearest seed in LAB.
    flat_lab = filled_lab.reshape(-1, 3)
    d2 = ((flat_lab[:, None, :] - seeds_lab[None, :, :]) ** 2).sum(axis=2)
    zone_labels = d2.argmin(axis=1).reshape(h, w).astype(np.int32)

    if merge_zones:
        # Merge 16 zones into 5 layers.
        layer_labels = np.vectorize(ZONE_TO_LAYER.get)(zone_labels).astype(np.int32)
        n_classes = 5
    else:
        # Keep all 16 zones as independent labels.
        layer_labels = zone_labels
        n_classes = 16

    # Basic cleanup.
    layer_labels = ndimage.median_filter(layer_labels, size=5)

    # Reassign text pixels after clustering.
    layer_labels = _assign_text_to_nearest_label(layer_labels, text_mask)

    # Use colorbar seed colors directly when not merging; otherwise median panel colors.
    if merge_zones:
        palette = np.zeros((n_classes, 3), dtype=np.uint8)
        for lbl in range(n_classes):
            mask = layer_labels == lbl
            if mask.any():
                palette[lbl] = np.median(panel_rgb[mask], axis=0).astype(np.uint8)
        overlay = _create_overlay(
            panel_rgb,
            layer_labels,
            palette,
            alpha=0.65,
            boundary_mode="thin",
            skip_background=False,
            fill_mode="blend",
        )
        mask = _create_overlay(
            panel_rgb,
            layer_labels,
            palette,
            alpha=1.0,
            boundary_mode="thin",
            skip_background=False,
            fill_mode="mask",
        )
    else:
        palette = seeds_rgb.astype(np.uint8)
        # For no-merge mode, show colorbar seed colors almost opaquely so the
        # overlay is a direct colorbar-color map rather than a blend.
        overlay = _create_overlay(
            panel_rgb,
            layer_labels,
            palette,
            alpha=0.90,
            boundary_mode="thin",
            skip_background=False,
            fill_mode="blend",
            overlay_colors=palette,
        )
        mask = _create_overlay(
            panel_rgb,
            layer_labels,
            palette,
            alpha=1.0,
            boundary_mode="thin",
            skip_background=False,
            fill_mode="mask",
            overlay_colors=palette,
        )
        # Pure colorbar-color map without boundary lines, for inspecting the
        # exact zone colors sampled from the colorbar.
        pure_overlay = _create_overlay(
            np.full_like(panel_rgb, 128),
            layer_labels,
            palette,
            alpha=1.0,
            boundary_mode="outer",
            skip_background=False,
            fill_mode="mask",
            overlay_colors=palette,
        )

    return {
        "labels": layer_labels,
        "overlay": overlay,
        "mask": mask,
        "seeds": seeds_rgb,
        "palette": palette,
        "pure_overlay": pure_overlay if not merge_zones else None,
    }


def smooth_text_into_labels(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reassign text pixels to nearest geological label and inpaint background."""
    from geoseg.modules.post_process.merge import remove_labels_by_ids

    text_mask = detect_text_mask(panel_rgb, labels)
    if not text_mask.any():
        return panel_rgb.copy(), labels.copy(), text_mask

    max_lbl = int(labels.max())
    text_label = max_lbl + 1
    labels_with_text = labels.copy()
    labels_with_text[text_mask] = text_label
    cleaned_labels = remove_labels_by_ids(labels_with_text, [text_label], fill="nearest")

    cleaned = cv2.inpaint(
        panel_rgb,
        text_mask.astype(np.uint8) * 255,
        inpaintRadius=11,
        flags=cv2.INPAINT_TELEA,
    )

    colors: dict[int, np.ndarray] = {}
    for lbl in np.unique(cleaned_labels):
        mask = cleaned_labels == lbl
        if mask.any():
            colors[int(lbl)] = np.median(cleaned[mask], axis=0).astype(np.uint8)

    for lbl, color in colors.items():
        m = text_mask & (cleaned_labels == lbl)
        if m.any():
            cleaned[m] = color

    return cleaned, cleaned_labels, text_mask


def create_comparison_image(
    original_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    mask_rgb: np.ndarray,
) -> np.ndarray:
    """Create Original | Overlay | Mask side-by-side comparison."""
    h, w = original_rgb.shape[:2]
    label_h = 40
    canvas = np.full((h + label_h, w * 3, 3), 255, dtype=np.uint8)
    canvas[label_h:, :w] = original_rgb
    canvas[label_h:, w : 2 * w] = overlay_rgb
    canvas[label_h:, 2 * w :] = mask_rgb

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = _load_font(24)
    for i, text in enumerate(["Original", "Overlay (16-zone)", "Mask"]):
        x = i * w + w // 2
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x - text_w // 2, 8), text, fill=(0, 0, 0), font=font)

    return np.array(img)


def create_comparison_with_cleaned(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    mask_rgb: np.ndarray,
) -> np.ndarray:
    """Create Original | Cleaned | Overlay | Mask side-by-side comparison."""
    h, w = original_rgb.shape[:2]
    label_h = 40
    canvas = np.full((h + label_h, w * 4, 3), 255, dtype=np.uint8)
    canvas[label_h:, :w] = original_rgb
    canvas[label_h:, w : 2 * w] = cleaned_rgb
    canvas[label_h:, 2 * w : 3 * w] = overlay_rgb
    canvas[label_h:, 3 * w :] = mask_rgb

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = _load_font(24)
    for i, text in enumerate(["Original", "Cleaned", "Overlay", "Mask"]):
        x = i * w + w // 2
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x - text_w // 2, 8), text, fill=(0, 0, 0), font=font)

    return np.array(img)


def main(merge_zones: bool = True, remove_text: bool = False, out_dir: Path | None = None) -> dict:
    output_root = out_dir or OUT_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    colorbar_rgb = extract_colorbar_strip()
    print(f"Colorbar strip shape: {colorbar_rgb.shape}")
    print(f"Mode: merge_zones={merge_zones}, remove_text={remove_text}")

    summary: dict = {
        "colorbar_roi": COLORBAR_ROI,
        "merge_zones": merge_zones,
        "remove_text": remove_text,
        "profiles": {},
    }

    for panel_id in PROFILES:
        print(f"\n=== {panel_id} ===")
        original = np.array(
            Image.open(PANEL_DIR / f"{panel_id}_cropped.jpg").convert("RGB")
        )
        print(f"  Panel shape: {original.shape}")

        # 1. Optional text-noise preprocessing.
        segment_input = original.copy()
        visual_cleaned = original.copy()
        text_mask = np.zeros(original.shape[:2], dtype=bool)
        if remove_text:
            segment_input, text_mask, visual_cleaned = preprocess_remove_text(original, panel_id)
            print(f"  text mask pixels: {int(text_mask.sum())}")

        # 2. 16-zone colorbar segmentation on the text-free input.
        result = segment_with_16_zones(
            segment_input,
            colorbar_rgb,
            text_mask=text_mask,
            merge_zones=merge_zones,
        )
        labels = result["labels"]
        print(f"  unique labels: {sorted(np.unique(labels))}")

        # 3. PM artifact repair only in 5-layer merge mode.
        if merge_zones and panel_id in PM_REPAIR_ROIS:
            roi = PM_REPAIR_ROIS[panel_id]
            labels = assign_label_to_background(labels)
            repaired = repair_pm_artifact(visual_cleaned, labels, roi=roi)
            visual_cleaned = repaired["cleaned_rgb"]
            labels = repaired["labels"]
            print(f"  PM repair ROI: {repaired['roi']}")

        # 3b. No-merge mode: remove cyan/blue artifact labels pulled in by text fill.
        # Use row-horizontal fill so the replacement respects horizontal layers
        # instead of leaking labels from vertically adjacent strata.
        if not merge_zones and panel_id in PM_REPAIR_ROIS_NO_MERGE:
            rois = PM_REPAIR_ROIS_NO_MERGE[panel_id]
            per_roi_artifacts = []
            for roi in rois:
                # The right PM on profile_06 also picks up label 11.
                if panel_id == "fig6_profile_06" and roi == (470, 20, 540, 70):
                    per_roi_artifacts.append([11, 12])
                else:
                    per_roi_artifacts.append([12])
            labels = repair_pm_artifact_no_merge(
                labels,
                rois,
                per_roi_artifact_labels=per_roi_artifacts,
                fill_mode="row_horizontal",
                row_margin=40,
            )
            print(f"  PM no-merge repair (row_horizontal): {rois}")

        # 3. Build overlay / mask from the visually cleaned image so the saved
        #    comparison shows text-removed panels.
        if merge_zones:
            # Use median panel colors for merged layers.
            colors: dict[int, np.ndarray] = {}
            for lbl in np.unique(labels):
                if lbl == 0:
                    continue
                mask = labels == lbl
                if mask.any():
                    colors[int(lbl)] = np.median(visual_cleaned[mask], axis=0).astype(np.uint8)

            overlay_colors = np.zeros((max(colors.keys()) + 1, 3), dtype=np.uint8)
            for lbl, color in colors.items():
                overlay_colors[lbl] = color

            overlay = _create_overlay(
                visual_cleaned,
                labels,
                overlay_colors,
                alpha=0.65,
                boundary_mode="thin",
                skip_background=False,
                fill_mode="blend",
                overlay_colors=overlay_colors,
            )
            mask = _create_overlay(
                visual_cleaned,
                labels,
                overlay_colors,
                alpha=1.0,
                boundary_mode="thin",
                skip_background=False,
                fill_mode="mask",
                overlay_colors=overlay_colors,
            )
        else:
            palette = result["palette"]
            overlay = _create_overlay(
                visual_cleaned,
                labels,
                palette,
                alpha=0.90,
                boundary_mode="thin",
                skip_background=False,
                fill_mode="blend",
                overlay_colors=palette,
            )
            mask = _create_overlay(
                visual_cleaned,
                labels,
                palette,
                alpha=1.0,
                boundary_mode="thin",
                skip_background=False,
                fill_mode="mask",
                overlay_colors=palette,
            )
            # Pure colorbar-color map without boundary lines, for inspecting the
            # exact zone colors sampled from the colorbar.
            pure_overlay = _create_overlay(
                np.full_like(visual_cleaned, 128),
                labels,
                palette,
                alpha=1.0,
                boundary_mode="outer",
                skip_background=False,
                fill_mode="mask",
                overlay_colors=palette,
            )
            # Boundary-free overlay/mask for downstream grids and visual audit.
            overlay_no_boundary = _create_overlay(
                visual_cleaned,
                labels,
                palette,
                alpha=0.90,
                boundary_mode="none",
                skip_background=False,
                fill_mode="blend",
                overlay_colors=palette,
            )
            mask_no_boundary = _create_overlay(
                np.full_like(visual_cleaned, 128),
                labels,
                palette,
                alpha=1.0,
                boundary_mode="none",
                skip_background=False,
                fill_mode="mask",
                overlay_colors=palette,
            )

        # Recolor white boundaries to black for consistency with v7 style.
        white = np.all(overlay == [255, 255, 255], axis=2)
        overlay[white] = [0, 0, 0]
        white = np.all(mask == [255, 255, 255], axis=2)
        mask[white] = [0, 0, 0]

        # 4. Save outputs.
        panel_out = output_root / panel_id
        panel_out.mkdir(parents=True, exist_ok=True)

        Image.fromarray(visual_cleaned).save(panel_out / "cleaned.jpg", quality=95)
        Image.fromarray(overlay).save(panel_out / "overlay.jpg", quality=95)
        Image.fromarray(mask).save(panel_out / "mask.jpg", quality=95)
        if not merge_zones:
            Image.fromarray(pure_overlay).save(
                panel_out / "pure_overlay.jpg", quality=95
            )
            Image.fromarray(overlay_no_boundary).save(
                panel_out / "overlay_no_boundary.jpg", quality=95
            )
            Image.fromarray(mask_no_boundary).save(
                panel_out / "mask_no_boundary.jpg", quality=95
            )
            comparison_no_boundary = create_comparison_image(
                original, overlay_no_boundary, mask_no_boundary
            )
            Image.fromarray(comparison_no_boundary).save(
                panel_out / "comparison_no_boundary.jpg", quality=95
            )
        np.savez_compressed(panel_out / "labels.npz", labels=labels)
        np.savez_compressed(panel_out / "text_mask.npz", mask=text_mask)

        if remove_text:
            comparison = create_comparison_with_cleaned(original, visual_cleaned, overlay, mask)
        else:
            comparison = create_comparison_image(original, overlay, mask)
        Image.fromarray(comparison).save(panel_out / "comparison.jpg", quality=95)
        Image.fromarray(comparison).save(output_root / f"{panel_id}_comparison.jpg", quality=95)

        summary["profiles"][panel_id] = {
            "panel_shape": list(original.shape),
            "text_mask_pixels": int(text_mask.sum()),
            "n_final_labels": int(len(np.unique(labels)) - (1 if 0 in np.unique(labels) else 0)),
            "comparison_path": str(panel_out / "comparison.jpg"),
        }
        print(f"  Saved: {panel_out / 'comparison.jpg'}")

    if not merge_zones:
        seed_strip = create_seed_reference_strip(sample_16_seeds(colorbar_rgb))
        Image.fromarray(seed_strip).save(output_root / "16_seed_reference.jpg", quality=95)
        print(f"  Seed reference: {output_root / '16_seed_reference.jpg'}")

    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nAll outputs saved to: {output_root}")
    return summary


def assemble_text_removal_comparison(
    baseline_dir: Path = ROOT / "runs" / "fig6_colorbar_16zone_experiment_no_merge",
    cleaned_dir: Path = ROOT / "runs" / "fig6_colorbar_16zone_experiment_no_merge_clean_text",
    output_path: Path | None = None,
    target_width: int = 2400,
    use_no_boundary: bool = False,
    include_baseline: bool = True,
) -> Path:
    """Build a grid comparing original panel and text-removed 16-zone overlays.

    By default shows three columns (original | baseline | text-removed). Pass
    ``include_baseline=False`` for a two-column grid (original | text-removed)
    and ``use_no_boundary=True`` to use the boundary-free overlays.
    """
    if output_path is None:
        suffix = "_no_boundary" if use_no_boundary else ""
        name = (
            f"text_removal_comparison_grid{suffix}.jpg"
            if include_baseline
            else f"text_removal_comparison_grid{suffix}_2col.jpg"
        )
        output_path = cleaned_dir / name

    overlay_name = "overlay_no_boundary.jpg" if use_no_boundary else "overlay.jpg"

    rows: list[Image.Image] = []
    for panel_id in PROFILES:
        original = Image.open(PANEL_DIR / f"{panel_id}_cropped.jpg").convert("RGB")
        cleaned_overlay = Image.open(
            cleaned_dir / panel_id / overlay_name
        ).convert("RGB")
        if cleaned_overlay.size != original.size:
            cleaned_overlay = cleaned_overlay.resize(
                original.size, Image.Resampling.LANCZOS
            )

        if include_baseline:
            baseline_overlay = Image.open(
                baseline_dir / panel_id / overlay_name
            ).convert("RGB")
            if baseline_overlay.size != original.size:
                baseline_overlay = baseline_overlay.resize(
                    original.size, Image.Resampling.LANCZOS
                )
            row = Image.new("RGB", (original.width * 3, original.height))
            row.paste(original, (0, 0))
            row.paste(baseline_overlay, (original.width, 0))
            row.paste(cleaned_overlay, (original.width * 2, 0))
            headers = ["original", "baseline (no text removal)", "text removed"]
        else:
            row = Image.new("RGB", (original.width * 2, original.height))
            row.paste(original, (0, 0))
            row.paste(cleaned_overlay, (original.width, 0))
            headers = ["original", "text removed"]

        aspect = row.height / row.width
        new_height = int(target_width * aspect)
        rows.append(
            row.resize((target_width, new_height), Image.Resampling.LANCZOS)
        )

    row_height = rows[0].height
    header_height = 60
    label_width = 120
    gap = 10
    total_height = header_height + len(rows) * (row_height + gap)
    canvas_width = label_width + target_width + gap

    canvas = Image.new("RGB", (canvas_width, total_height), color=(32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(20)
    small_font = _load_font(16)

    col_width = target_width // len(headers)
    for idx, label in enumerate(headers):
        x = label_width + gap + idx * col_width + col_width // 2
        draw.text(
            (x, header_height // 2 - 10),
            label,
            fill=(255, 255, 255),
            font=small_font,
            anchor="mm",
        )

    for row_idx, (panel_id, row_img) in enumerate(zip(PROFILES, rows)):
        y = header_height + row_idx * (row_height + gap)
        draw.text(
            (label_width // 2, y + row_height // 2),
            panel_id,
            fill=(255, 255, 255),
            font=font,
            anchor="mm",
        )
        canvas.paste(row_img, (label_width + gap, y))

    canvas.save(output_path, quality=92)
    print(f"Text-removal comparison grid saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fig.6 colorbar 16-zone matching experiment."
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Keep all 16 zones as separate labels and use colorbar seed colors.",
    )
    parser.add_argument(
        "--remove-text",
        action="store_true",
        help="Inpaint text/annotation regions (PM, sediment, etc.) before segmentation.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/fig6_colorbar_16zone_experiment).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Assemble a 5-row original-vs-text-removed no-boundary comparison grid (2 columns).",
    )
    args = parser.parse_args()

    if args.compare:
        assemble_text_removal_comparison(use_no_boundary=True, include_baseline=False)
    else:
        out_dir = args.out_dir
        if out_dir is None:
            suffix = ""
            if args.no_merge:
                suffix += "_no_merge"
            if args.remove_text:
                suffix += "_clean_text"
            if not suffix:
                suffix = "_default"
            out_dir = ROOT / "runs" / f"fig6_colorbar_16zone_experiment{suffix}"

        main(merge_zones=not args.no_merge, remove_text=args.remove_text, out_dir=out_dir)
