#!/usr/bin/env python3
"""Regenerate ROI-preprocessing comparisons without black boundary lines.

Loads the labels saved by `fig6_colorbar_16zone_roi_preproc.py` and re-renders
the overlay/mask with boundary_mode='none', keeping the same per-label median
panel colors as the original experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geoseg.modules.segment_engines._shared import _create_overlay


ROOT = Path("/Users/daiduo2/geoseg")
SRC_DIR = ROOT / "runs" / "fig6_colorbar_16zone_roi_preproc"
OUT_DIR = SRC_DIR / "no_boundary"
PANEL_DIR = ROOT / "runs" / "feng_fig6_final_v4" / "crop_tests"

PROFILES = [f"fig6_profile_{i:02d}" for i in range(3, 8)]
PROFILE_LABELS = {
    "fig6_profile_03": "Profile 03",
    "fig6_profile_04": "Profile 04 (PM artifact)",
    "fig6_profile_05": "Profile 05 (PM artifact)",
    "fig6_profile_06": "Profile 06",
    "fig6_profile_07": "Profile 07",
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


def build_overlay_colors(panel_rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-label median panel colors, matching roi_preproc.py logic."""
    colors: dict[int, np.ndarray] = {}
    for lbl in np.unique(labels):
        mask = labels == lbl
        if mask.any():
            colors[int(lbl)] = np.median(panel_rgb[mask], axis=0).astype(np.uint8)
    overlay_colors = np.zeros((max(colors.keys()) + 1, 3), dtype=np.uint8)
    for lbl, color in colors.items():
        overlay_colors[lbl] = color
    return overlay_colors


def render_without_boundary(
    panel_rgb: np.ndarray, labels: np.ndarray, overlay_colors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    overlay = _create_overlay(
        panel_rgb, labels, overlay_colors,
        alpha=0.65, boundary_mode="none", skip_background=False,
        fill_mode="blend", overlay_colors=overlay_colors,
    )
    mask = _create_overlay(
        panel_rgb, labels, overlay_colors,
        alpha=1.0, boundary_mode="none", skip_background=False,
        fill_mode="mask", overlay_colors=overlay_colors,
    )
    return overlay, mask


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font = _load_font(22)
    big_font = _load_font(28)
    rows: list[Image.Image] = []

    for panel_id in PROFILES:
        # Enforce a consistent logical order.
        preferred_order = ["baseline", "visual_roi", "ocr_roi"]
        found = set(
            v.stem.replace("labels_", "")
            for v in (SRC_DIR / panel_id).glob("labels_*.npz")
        )
        variants = [v for v in preferred_order if v in found]

        for variant in variants:
            labels_path = SRC_DIR / panel_id / f"labels_{variant}.npz"
            labels = np.load(labels_path)["labels"]
            original = np.array(
                Image.open(PANEL_DIR / f"{panel_id}_cropped.jpg").convert("RGB")
            )
            overlay_colors = build_overlay_colors(original, labels)
            overlay, mask = render_without_boundary(original, labels, overlay_colors)

            comp = create_comparison_image(original, overlay, mask, variant)
            comp_path = OUT_DIR / panel_id / f"comparison_{variant}.jpg"
            comp_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(comp).save(comp_path, quality=95)

            # Save standalone overlay/mask as well.
            Image.fromarray(overlay).save(OUT_DIR / panel_id / f"overlay_{variant}.jpg", quality=95)
            Image.fromarray(mask).save(OUT_DIR / panel_id / f"mask_{variant}.jpg", quality=95)

            label = f"{PROFILE_LABELS[panel_id]} — {variant}"
            w, h = Image.open(comp_path).size
            label_h = 36
            canvas = Image.new("RGB", (w, h + label_h), (255, 255, 255))
            canvas.paste(Image.open(comp_path), (0, label_h))
            draw = ImageDraw.Draw(canvas)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            draw.text(((w - text_w) // 2, 6), label, fill=(0, 0, 0), font=font)
            rows.append(canvas)

            print(f"Saved {comp_path}")

    max_w = max(r.width for r in rows)
    total_h = sum(r.height for r in rows)
    title_h = 60
    canvas = Image.new("RGB", (max_w, total_h + title_h), (255, 255, 255))

    draw = ImageDraw.Draw(canvas)
    title = "Fig.6 16-Zone ROI Preprocessing — No Boundary Lines"
    bbox = draw.textbbox((0, 0), title, font=big_font)
    title_w = bbox[2] - bbox[0]
    draw.text(((max_w - title_w) // 2, 16), title, fill=(0, 0, 0), font=big_font)

    y = title_h
    for row in rows:
        x = (max_w - row.width) // 2
        canvas.paste(row, (x, y))
        y += row.height

    big_path = OUT_DIR / "big_comparison_no_boundary.jpg"
    canvas.save(big_path, quality=95)
    print(f"\nSaved big comparison: {big_path}")


if __name__ == "__main__":
    main()
