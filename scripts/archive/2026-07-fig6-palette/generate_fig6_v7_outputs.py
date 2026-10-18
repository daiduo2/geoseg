#!/usr/bin/env python3
"""Generate v7 outputs: hue-matched overlays + text/annotation smoothing.

Minimal standalone script. Reuses existing library functions only:
- remove_text-style pixel detection (local RGB/LAB outlier + saturation).
- remove_labels_by_ids for nearest-neighbor fill.
- create_overlay for overlay/mask generation.
- draw_overlay_legend for overlay legend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.color import rgb2lab

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseg.modules.post_process.merge import remove_labels_by_ids
from geoseg.core.image_ops import create_overlay
from geoseg.modules.visual_audit.rendering import draw_overlay_legend


PANELS = [f"fig6_profile_{i:02d}" for i in range(3, 8)]
LABELS_DIR = Path("runs/feng_fig6_final_v5")
PANEL_DIR = Path("runs/feng_fig6_final_v4/crop_tests")
OUT_DIR = Path("runs/feng_fig6_comparisons_v7")

# Component-level fixes for PM-induced over-segmentation.
PM_FIXES = {
    "fig6_profile_04": {"source": 3, "target": 2, "max_area": 400},
    "fig6_profile_05": {"source": 2, "target": 3, "max_area": 600},
}


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try common system fonts, fallback to default."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def merge_small_components(
    labels: np.ndarray,
    source_label: int,
    target_label: int,
    max_area: int,
) -> np.ndarray:
    """Merge small connected components of source_label into target_label."""
    result = labels.copy()
    mask = labels == source_label
    cc, num = ndimage.label(mask)
    for i in range(1, num + 1):
        comp = cc == i
        if int(comp.sum()) >= max_area:
            continue
        result[comp] = target_label
    return result


def compute_label_colors(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    exclude_mask: np.ndarray | None = None,
    quantile_range: tuple[float, float] = (0.1, 0.9),
) -> dict[int, np.ndarray]:
    """Compute per-label median RGB after clipping text/boundary outliers."""
    colors: dict[int, np.ndarray] = {}
    exclude = exclude_mask if exclude_mask is not None else np.zeros(labels.shape, dtype=bool)
    for lbl in sorted(np.unique(labels)):
        mask = (labels == lbl) & (~exclude)
        if not mask.any():
            continue
        pixels = panel_rgb[mask].astype(np.float32)
        if quantile_range is not None:
            low, high = quantile_range
            q_low = np.percentile(pixels, low * 100, axis=0)
            q_high = np.percentile(pixels, high * 100, axis=0)
            pixels = np.clip(pixels, q_low, q_high)
        colors[int(lbl)] = np.median(pixels, axis=0).astype(np.uint8)
    return colors


def colors_to_array(colors: dict[int, np.ndarray]) -> np.ndarray:
    """Convert label->color dict to an array indexed by label value."""
    max_lbl = max(colors.keys())
    arr = np.zeros((max_lbl + 1, 3), dtype=np.uint8)
    for lbl, c in colors.items():
        arr[lbl] = c
    return arr


def detect_text_mask(panel_rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Detect text/annotation pixels using local color outliers.

    Combines:
    - RGB local median distance (catches dark text and sharp symbols).
    - LAB L-channel local median (catches dark strokes).
    - Saturation drop on bright pixels (catches white text on colored layers).

    The top/background sediment layer (label 0) is excluded because it only
    contains large white annotation text that does not create segmentation
    artifacts; smoothing it would punch holes in the deep-red region.
    """
    pf = panel_rgb.astype(np.float32)
    h, w = panel_rgb.shape[:2]

    # RGB local median distance.
    med = np.stack(
        [ndimage.median_filter(panel_rgb[:, :, c], size=7) for c in range(3)],
        axis=2,
    ).astype(np.float32)
    rgb_dist = np.linalg.norm(pf - med, axis=2)

    # LAB L-channel outlier.
    lab = rgb2lab(panel_rgb)
    L = lab[:, :, 0]
    L_med = ndimage.median_filter(L, size=7)

    dark_mask = (rgb_dist > 30) | (
        (np.abs(L - L_med) > 20) & (L < 120)
    )

    # White/light-gray text on colored backgrounds.
    sat = pf.max(axis=2) - pf.min(axis=2)
    sat_med = ndimage.median_filter(sat, size=7)
    bright = pf.max(axis=2)
    white_mask = (sat_med - sat > 40) & (bright > 150) & (sat < 60)

    text_mask = (dark_mask | white_mask) & (labels != 0)

    # Dilate to cover full letter strokes.
    struct = np.ones((5, 5), dtype=bool)
    text_mask = ndimage.binary_dilation(text_mask, structure=struct, iterations=4)

    # Keep only text-like connected components.
    cc, num = ndimage.label(text_mask)
    for i in range(1, num + 1):
        comp = cc == i
        area = int(comp.sum())
        if area < 30 or area > max(15000, h * w // 4):
            text_mask[comp] = False

    return text_mask


def smooth_text_into_labels(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reassign text pixels to nearest geological label and inpaint background.

    Returns:
        cleaned_rgb: panel with text pixels inpainted/filled.
        cleaned_labels: labels with text pixels reassigned.
        text_mask: boolean mask of detected text pixels.
    """
    text_mask = detect_text_mask(panel_rgb, labels)
    if not text_mask.any():
        return panel_rgb.copy(), labels.copy(), text_mask

    max_lbl = int(labels.max())
    text_label = max_lbl + 1
    labels_with_text = labels.copy()
    labels_with_text[text_mask] = text_label
    cleaned_labels = remove_labels_by_ids(
        labels_with_text, [text_label], fill="nearest"
    )

    # Inpaint the panel background.
    cleaned = cv2.inpaint(
        panel_rgb,
        text_mask.astype(np.uint8) * 255,
        inpaintRadius=11,
        flags=cv2.INPAINT_TELEA,
    )

    # Fill remaining text pixels with the assigned label color so the overlay
    # does not show any text ghost.
    colors = compute_label_colors(cleaned, cleaned_labels, text_mask)
    for lbl in np.unique(cleaned_labels):
        m = text_mask & (cleaned_labels == lbl)
        if m.any():
            cleaned[m] = colors[int(lbl)]

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
    for i, text in enumerate(["Original", "Overlay", "Mask"]):
        x = i * w + w // 2
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x - text_w // 2, 8), text, fill=(0, 0, 0), font=font)

    return np.array(img)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for panel_id in PANELS:
        labels = np.load(LABELS_DIR / panel_id / "labels.npz")["labels"]
        original = np.array(
            Image.open(PANEL_DIR / f"{panel_id}_cropped.jpg").convert("RGB")
        )

        # Component-level PM fixes.
        fix = PM_FIXES.get(panel_id)
        if fix is not None:
            labels = merge_small_components(
                labels,
                source_label=fix["source"],
                target_label=fix["target"],
                max_area=fix["max_area"],
            )

        # Text smoothing.
        cleaned, labels, text_mask = smooth_text_into_labels(original, labels)

        # Hue-matched colors.
        colors = compute_label_colors(cleaned, labels, text_mask)
        overlay_colors = colors_to_array(colors)
        seeds = np.empty((0, 3), dtype=np.uint8)

        overlay = create_overlay(
            cleaned,
            labels,
            seeds,
            alpha=0.65,
            fill_mode="blend",
            overlay_colors=overlay_colors,
            skip_background=False,
        )
        mask = create_overlay(
            cleaned,
            labels,
            seeds,
            alpha=1.0,
            fill_mode="mask",
            overlay_colors=overlay_colors,
            skip_background=False,
        )
        # Recolor default white boundaries to black.
        white = np.all(overlay == [255, 255, 255], axis=2)
        overlay[white] = [0, 0, 0]
        white = np.all(mask == [255, 255, 255], axis=2)
        mask[white] = [0, 0, 0]

        overlay_legend = draw_overlay_legend(overlay, labels, label_colors=colors)
        comparison = create_comparison_image(original, overlay, mask)

        panel_out = OUT_DIR / panel_id
        panel_out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay_legend).save(
            panel_out / "overlay_legend.jpg", quality=90
        )
        Image.fromarray(comparison).save(panel_out / "comparison.jpg", quality=90)
        np.savez_compressed(panel_out / "labels.npz", labels=labels)
        np.savez_compressed(panel_out / "text_mask.npz", mask=text_mask)

        # Collect comparison at the root of OUT_DIR.
        Image.fromarray(comparison).save(
            OUT_DIR / f"{panel_id}_comparison.jpg", quality=90
        )
        print(f"{panel_id}: text px={int(text_mask.sum())}, done")


if __name__ == "__main__":
    main()
