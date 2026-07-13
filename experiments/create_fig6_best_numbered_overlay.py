#!/usr/bin/env python3
"""Create a single numbered overlay summary for all 5 Fig.6 profiles.

Best results (visual evaluation):
- fig6_profile_03: feng_fig6_comparisons_v7
- fig6_profile_04: pm_repair_ocr_experiment
- fig6_profile_05: pm_repair_experiment
- fig6_profile_06: feng_fig6_comparisons_v7
- fig6_profile_07: feng_fig6_comparisons_v7

Output:
- runs/fig6_profile_all_best_summary/fig6_profile_XX_numbered.jpg
- runs/fig6_profile_all_best_summary/fig6_all_profiles_summary.jpg
- runs/fig6_profile_all_best_summary/fig6_all_profiles_side_by_side.jpg
- runs/fig6_profile_all_best_summary/summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage import segmentation

sys.path.insert(0, "/Users/daiduo2/geoseg/src")

from geoseg.modules.segment_engines._shared import (
    _create_overlay,
    _merge_small_regions,
)


ROOT = Path("/Users/daiduo2/geoseg")
OUT_DIR = ROOT / "runs" / "fig6_profile_all_best_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROFILES = [
    {
        "id": "fig6_profile_03",
        "image": ROOT / "runs/feng_fig6_final_v4/crop_tests/fig6_profile_03_cropped.jpg",
        "labels": ROOT / "runs/feng_fig6_comparisons_v7/fig6_profile_03/labels.npz",
        "source": "feng_fig6_comparisons_v7",
    },
    {
        "id": "fig6_profile_04",
        "image": ROOT / "runs/feng_fig6_final_v4/crop_tests/fig6_profile_04_cropped.jpg",
        "labels": ROOT / "runs/pm_repair_ocr_experiment/fig6_profile_04/labels_repaired.npz",
        "source": "pm_repair_ocr_experiment",
    },
    {
        "id": "fig6_profile_05",
        "image": ROOT / "runs/feng_fig6_final_v4/crop_tests/fig6_profile_05_cropped.jpg",
        "labels": ROOT / "runs/pm_repair_experiment/fig6_profile_05/labels_repaired.npz",
        "source": "pm_repair_experiment",
    },
    {
        "id": "fig6_profile_06",
        "image": ROOT / "runs/feng_fig6_final_v4/crop_tests/fig6_profile_06_cropped.jpg",
        "labels": ROOT / "runs/feng_fig6_comparisons_v7/fig6_profile_06/labels.npz",
        "source": "feng_fig6_comparisons_v7",
    },
    {
        "id": "fig6_profile_07",
        "image": ROOT / "runs/feng_fig6_final_v4/crop_tests/fig6_profile_07_cropped.jpg",
        "labels": ROOT / "runs/feng_fig6_comparisons_v7/fig6_profile_07/labels.npz",
        "source": "feng_fig6_comparisons_v7",
    },
]


def load_labels(path: Path) -> np.ndarray:
    data = np.load(path)
    key = "labels" if "labels" in data else data.files[0]
    return data[key].astype(np.int32)


def make_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _rank_labels_by_depth(cleaned: np.ndarray) -> tuple[np.ndarray, list[tuple[int, object]]]:
    """Re-label regions top-to-bottom so layer numbers are consistent across panels.

    Returns the re-ranked label image and the sorted region list.
    """
    from skimage.measure import regionprops

    regions = []
    for r in regionprops(cleaned + 1):
        original_label = int(r.label) - 1
        regions.append((original_label, r))
    regions.sort(key=lambda item: item[1].centroid[0])

    label_to_rank = {lbl: idx for idx, (lbl, _) in enumerate(regions, start=1)}
    ranked = np.vectorize(label_to_rank.get, otypes=[np.int32])(cleaned)
    return ranked, regions


def _layer_colors_from_image(image: np.ndarray, ranked: np.ndarray, n_labels: int) -> np.ndarray:
    """Return a (n_labels, 3) palette where each layer gets the median RGB of its region."""
    colors = np.zeros((n_labels, 3), dtype=np.uint8)
    for idx in range(1, n_labels + 1):
        pixels = image[ranked == idx]
        if len(pixels) > 0:
            colors[idx - 1] = np.median(pixels, axis=0).astype(np.uint8)
    return colors


def create_numbered_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    title: str,
    fill_mode: str = "blend",
    mask_bg: tuple[int, int, int] | None = None,
) -> tuple[Image.Image, list[dict]]:
    """Return an overlay with original-hue layers and a separate numbered colorbar.

    Args:
        fill_mode: "blend" (semi-transparent on original) or "mask" (pure mask).
        mask_bg: If fill_mode is "mask", replace the default dark mask background
            with this RGB color (e.g. white).
    """
    cleaned = _merge_small_regions(labels, min_area_frac=0.001)

    # Re-map labels so layer 1 is always the top layer, layer 2 below it, etc.
    ranked, regions = _rank_labels_by_depth(cleaned)
    n_labels = len(regions)
    overlay_colors = _layer_colors_from_image(image, ranked, n_labels)

    overlay_rgb = _create_overlay(
        image,
        ranked,
        np.zeros((1, 3), dtype=np.uint8),
        alpha=0.65,
        boundary_mode="thin",
        skip_background=False,
        min_area_frac=0.001,
        fill_mode=fill_mode,
        overlay_colors=overlay_colors,
    )

    if fill_mode == "mask" and mask_bg is not None:
        bg_mask = (overlay_rgb == 32).all(axis=2)
        overlay_rgb[bg_mask] = mask_bg

    # Draw region boundaries as black lines instead of the default white.
    boundaries = segmentation.find_boundaries(ranked, mode="thin")
    overlay_rgb[boundaries] = (0, 0, 0)

    color_map = {idx: overlay_colors[idx - 1] for idx in range(1, n_labels + 1)}

    panel_h, panel_w = overlay_rgb.shape[:2]

    # Separate rectangular colorbar on the right with a clear gap.
    cb_gap = 20
    cb_w = 30
    cb_band_h = 18
    cb_h = n_labels * cb_band_h
    cb_x = panel_w + cb_gap
    cb_y0 = max(0, (panel_h - cb_h) // 2)
    total_w = cb_x + cb_w

    canvas = np.full((panel_h, total_w, 3), 255, dtype=np.uint8)
    canvas[:panel_h, :panel_w] = overlay_rgb

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = make_font(10)
    font_title = make_font(15)

    # Panel title
    draw.text(
        (8, 6),
        title,
        fill=(255, 255, 255),
        font=font_title,
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )

    # Rectangular colorbar: stacked color bands with no gaps between them.
    for idx in range(1, n_labels + 1):
        color = tuple(int(c) for c in color_map[idx])
        y1 = cb_y0 + (idx - 1) * cb_band_h
        y2 = y1 + cb_band_h
        draw.rectangle(
            [cb_x, y1, cb_x + cb_w - 1, y2 - 1],
            fill=color,
            outline=(0, 0, 0),
            width=1,
        )

        # Layer number centered in the band.
        text = str(idx)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        cx = cb_x + cb_w // 2
        cy = (y1 + y2) // 2
        draw.text(
            (cx - tw // 2, cy - th // 2),
            text,
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    numbered_regions = []
    for idx, (original_label, r) in enumerate(regions, start=1):
        numbered_regions.append({
            "number": idx,
            "original_label_id": original_label,
            "centroid": [int(r.centroid[1]), int(r.centroid[0])],
            "area": int(r.area),
            "color_rgb": list(int(c) for c in color_map[idx]),
        })

    return img, numbered_regions


def build_side_by_side_comparison(
    profiles_data: list[dict],
) -> Image.Image:
    """Build a two-column comparison: originals (left) vs mask overlays (right)."""
    left_imgs = [Image.fromarray(d["image"]) for d in profiles_data]
    right_imgs = [d["mask_overlay"] for d in profiles_data]
    titles = [d["title"] for d in profiles_data]

    left_w = max(img.width for img in left_imgs)
    right_w = max(img.width for img in right_imgs)
    col_gap = 40
    row_gap = 20
    header_h = 40
    margin_left = 140
    margin_top = 10

    row_heights = [max(left.height, right.height) for left, right in zip(left_imgs, right_imgs)]
    total_h = header_h + sum(row_heights) + row_gap * (len(profiles_data) + 1) + margin_top
    total_w = margin_left + left_w + col_gap + right_w + row_gap

    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font_header = make_font(22)
    font_title = make_font(18)

    header_y = margin_top + 6
    left_header_x = margin_left + left_w // 2
    right_header_x = margin_left + left_w + col_gap + right_w // 2
    for text, cx in [("Original", left_header_x), ("Overlay (mask)", right_header_x)]:
        bbox = draw.textbbox((0, 0), text, font=font_header)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, header_y), text, fill=(0, 0, 0), font=font_header)

    y = header_h + margin_top
    for left, right, title in zip(left_imgs, right_imgs, titles):
        row_h = max(left.height, right.height)
        y_left = y + (row_h - left.height) // 2
        y_right = y + (row_h - right.height) // 2
        x_left = margin_left + (left_w - left.width) // 2
        x_right = margin_left + left_w + col_gap + (right_w - right.width) // 2
        img.paste(left, (x_left, y_left))
        img.paste(right, (x_right, y_right))

        bbox = draw.textbbox((0, 0), title, font=font_title)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (margin_left - tw - 12, y + row_h // 2 - th // 2),
            title,
            fill=(0, 0, 0),
            font=font_title,
        )

        y += row_h + row_gap

    return img


def main() -> None:
    summary = {"profiles": [], "output_dir": str(OUT_DIR)}
    individual_images: list[Image.Image] = []
    profiles_data: list[dict] = []

    for profile in PROFILES:
        pid = profile["id"]
        image = np.array(Image.open(profile["image"]).convert("RGB"))
        labels = load_labels(profile["labels"])
        print(f"\n=== {pid} ===")
        print(f"  Source: {profile['source']}")
        print(f"  Image shape: {image.shape}")
        print(f"  Labels unique: {np.unique(labels)}")

        # Individual overlay: blend on original.
        numbered_img, numbered_regions = create_numbered_overlay(
            image, labels, title=pid, fill_mode="blend"
        )
        out_path = OUT_DIR / f"{pid}_numbered.jpg"
        numbered_img.save(out_path, quality=95)
        print(f"  Saved: {out_path}")

        # Mask overlay for side-by-side comparison.
        mask_overlay, _ = create_numbered_overlay(
            image, labels, title=pid, fill_mode="mask", mask_bg=(255, 255, 255)
        )

        summary["profiles"].append({
            "panel_id": pid,
            "source": profile["source"],
            "labels_path": str(profile["labels"]),
            "image_path": str(profile["image"]),
            "numbered_overlay": str(out_path),
            "n_partitions": len(numbered_regions),
            "partitions": numbered_regions,
        })
        individual_images.append(numbered_img)
        profiles_data.append({
            "image": image,
            "overlay": numbered_img,
            "mask_overlay": mask_overlay,
            "title": pid,
        })

    # Build combined canvas: single row of panels
    gap = 20
    max_h = max(img.height for img in individual_images)
    total_w = sum(img.width for img in individual_images) + gap * (len(individual_images) + 1)
    canvas_h = gap + max_h + gap
    canvas = np.full((canvas_h, total_w, 3), 255, dtype=np.uint8)

    x = gap
    for img_pil in individual_images:
        arr = np.array(img_pil)
        canvas[gap : gap + arr.shape[0], x : x + arr.shape[1]] = arr
        x += img_pil.width + gap

    summary_path = OUT_DIR / "fig6_all_profiles_summary.jpg"
    Image.fromarray(canvas).save(summary_path, quality=95)
    print(f"\nSaved combined summary: {summary_path}")

    side_by_side = build_side_by_side_comparison(profiles_data)
    side_by_side_path = OUT_DIR / "fig6_all_profiles_side_by_side.jpg"
    side_by_side.save(side_by_side_path, quality=95)
    print(f"Saved side-by-side comparison: {side_by_side_path}")

    summary_json_path = OUT_DIR / "summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
