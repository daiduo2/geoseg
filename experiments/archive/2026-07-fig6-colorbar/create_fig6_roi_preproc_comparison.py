#!/usr/bin/env python3
"""Create a big comparison image for 16-zone ROI preprocessing experiment.

Layout (per profile, side-by-side Original | Overlay | Mask):
- Profile 03, 06, 07: baseline only (one row each)
- Profile 04: baseline | visual_roi (two rows)
- Profile 05: baseline | visual_roi | ocr_roi (three rows)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/daiduo2/geoseg")
BASE_DIR = ROOT / "runs" / "fig6_colorbar_16zone_roi_preproc"
OUT_PATH = BASE_DIR / "roi_preproc_big_comparison.jpg"

PROFILES = [f"fig6_profile_{i:02d}" for i in range(3, 8)]
PROFILE_LABELS = {
    "fig6_profile_03": "Profile 03",
    "fig6_profile_04": "Profile 04 (PM artifact)",
    "fig6_profile_05": "Profile 05 (PM artifact)",
    "fig6_profile_06": "Profile 06",
    "fig6_profile_07": "Profile 07",
}

VARIANTS = {
    "fig6_profile_03": ["baseline"],
    "fig6_profile_04": ["baseline", "visual_roi"],
    "fig6_profile_05": ["baseline", "visual_roi", "ocr_roi"],
    "fig6_profile_06": ["baseline"],
    "fig6_profile_07": ["baseline"],
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


def _add_label(img: Image.Image, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Add a top label bar to an image."""
    w, h = img.size
    label_h = 36
    canvas = Image.new("RGB", (w, h + label_h), (255, 255, 255))
    canvas.paste(img, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((w - text_w) // 2, 6), text, fill=(0, 0, 0), font=font)
    return canvas


def main() -> None:
    font = _load_font(22)
    big_font = _load_font(28)

    # Load all comparison images with their variant labels.
    rows: list[Image.Image] = []
    for profile_id in PROFILES:
        variants = VARIANTS[profile_id]
        profile_label = PROFILE_LABELS[profile_id]

        for variant in variants:
            comp_path = BASE_DIR / profile_id / f"comparison_{variant}.jpg"
            img = Image.open(comp_path).convert("RGB")
            label = f"{profile_label} — {variant}"
            rows.append(_add_label(img, label, font))

    # Determine grid size: fixed width = widest row; stack vertically.
    max_w = max(r.width for r in rows)
    total_h = sum(r.height for r in rows)
    # Add a top title bar.
    title_h = 60
    canvas = Image.new("RGB", (max_w, total_h + title_h), (255, 255, 255))

    draw = ImageDraw.Draw(canvas)
    title = "Fig.6 16-Zone Colorbar Segmentation — ROI Preprocessing Comparison"
    bbox = draw.textbbox((0, 0), title, font=big_font)
    title_w = bbox[2] - bbox[0]
    draw.text(((max_w - title_w) // 2, 16), title, fill=(0, 0, 0), font=big_font)

    y = title_h
    for row in rows:
        # Center horizontally.
        x = (max_w - row.width) // 2
        canvas.paste(row, (x, y))
        y += row.height

    canvas.save(OUT_PATH, quality=95)
    print(f"Saved big comparison: {OUT_PATH}")
    print(f"Image size: {canvas.size}")


if __name__ == "__main__":
    main()
