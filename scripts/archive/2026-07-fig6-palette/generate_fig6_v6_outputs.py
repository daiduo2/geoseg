#!/usr/bin/env python3
"""Generate v6 hue-matched overlays, 3-up comparisons, and PM artifact fixes.

Minimal standalone script. Uses existing library functions only:
- create_overlay for overlay/mask generation with custom colors.
- draw_overlay_legend for overlay legend.
- merge_labels_by_ids is not needed here because we merge at the component level.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseg.core.image_ops import create_overlay
from geoseg.modules.visual_audit.rendering import draw_overlay_legend


PANELS = [f"fig6_profile_{i:02d}" for i in range(3, 8)]
LABELS_DIR = Path("runs/feng_fig6_final_v5")
PANEL_DIR = Path("runs/feng_fig6_final_v4/crop_tests")
OUT_DIR = Path("runs/feng_fig6_comparisons_v6")

# PM-induced over-segmentation fixes, identified by color/geometry from the overlays:
# - 04: small blue (label 3) protrusion on the yellow-green layer (label 2).
# - 05: small pink/purple (label 2) protrusion on the blue layer (label 3).
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


def compute_label_colors(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    quantile_range: tuple[float, float] = (0.1, 0.9),
) -> dict[int, np.ndarray]:
    """Compute per-label median RGB after clipping text/boundary outliers.

    Includes label 0 because in these cropped panels it corresponds to a
    visible geological layer (e.g. the deep-red sediment top layer), not the
    plot background.
    """
    colors: dict[int, np.ndarray] = {}
    for lbl in sorted(np.unique(labels)):
        pixels = panel_rgb[labels == lbl].astype(np.float32)
        if quantile_range is not None and pixels.shape[0] > 0:
            low, high = quantile_range
            q_low = np.percentile(pixels, low * 100, axis=0)
            q_high = np.percentile(pixels, high * 100, axis=0)
            pixels = np.clip(pixels, q_low, q_high)
        if pixels.shape[0] > 0:
            colors[int(lbl)] = np.median(pixels, axis=0).astype(np.uint8)
    return colors


def colors_to_array(colors: dict[int, np.ndarray]) -> np.ndarray:
    """Convert label->color dict to an array indexed by label value."""
    max_lbl = max(colors.keys())
    arr = np.zeros((max_lbl + 1, 3), dtype=np.uint8)
    for lbl, c in colors.items():
        arr[lbl] = c
    return arr


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


def create_comparison_image(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    colors: dict[int, np.ndarray],
    alpha: float = 0.65,
) -> np.ndarray:
    """Create Original | Overlay | Mask side-by-side comparison."""
    overlay_colors = colors_to_array(colors)
    seeds = np.empty((0, 3), dtype=np.uint8)

    overlay = create_overlay(
        panel_rgb,
        labels,
        seeds,
        alpha=alpha,
        fill_mode="blend",
        overlay_colors=overlay_colors,
        skip_background=False,
    )
    mask = create_overlay(
        panel_rgb,
        labels,
        seeds,
        alpha=1.0,
        fill_mode="mask",
        overlay_colors=overlay_colors,
        skip_background=False,
    )

    h, w = panel_rgb.shape[:2]
    label_h = 40
    canvas = np.full((h + label_h, w * 3, 3), 255, dtype=np.uint8)
    canvas[label_h:, :w] = panel_rgb
    canvas[label_h:, w : 2 * w] = overlay
    canvas[label_h:, 2 * w :] = mask

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
        panel = np.array(
            Image.open(PANEL_DIR / f"{panel_id}_cropped.jpg").convert("RGB")
        )

        # Apply PM artifact fix for 04 / 05.
        fix = PM_FIXES.get(panel_id)
        if fix is not None:
            labels = merge_small_components(
                labels,
                source_label=fix["source"],
                target_label=fix["target"],
                max_area=fix["max_area"],
            )

        colors = compute_label_colors(panel, labels)
        overlay_colors = colors_to_array(colors)
        seeds = np.empty((0, 3), dtype=np.uint8)

        overlay = create_overlay(
            panel,
            labels,
            seeds,
            alpha=0.65,
            fill_mode="blend",
            overlay_colors=overlay_colors,
            skip_background=False,
        )
        overlay_legend = draw_overlay_legend(overlay, labels, label_colors=colors)
        comparison = create_comparison_image(panel, labels, colors)

        panel_out = OUT_DIR / panel_id
        panel_out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay_legend).save(
            panel_out / "overlay_legend.jpg", quality=90
        )
        Image.fromarray(comparison).save(panel_out / "comparison.jpg", quality=90)
        np.savez_compressed(panel_out / "labels.npz", labels=labels)

        # Also collect comparison at the root of OUT_DIR.
        Image.fromarray(comparison).save(
            OUT_DIR / f"{panel_id}_comparison.jpg", quality=90
        )
        print(f"{panel_id}: done")


if __name__ == "__main__":
    main()
