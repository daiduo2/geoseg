#!/usr/bin/env python3
"""Create side-by-side comparison images for best overlays and originals."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST_OVERLAYS_DIR = PROJECT_ROOT / "docs" / "best_overlays"
BEST_OVERLAYS_3D_DIR = PROJECT_ROOT / "docs" / "best_overlays_3d_schematic"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "comparison_overlays"

LABEL_HEIGHT = 40
FONT_SIZE = 24


def find_font() -> ImageFont.FreeTypeFont:
    """Return a TrueType font with CJK support, or fall back to default."""
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, FONT_SIZE)
    return ImageFont.load_default()


FONT = find_font()


def load_image(path: Path) -> Image.Image:
    """Load an image and convert to RGB."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def resize_to_height(img: Image.Image, target_height: int) -> Image.Image:
    """Resize image to a target height keeping aspect ratio."""
    width, height = img.size
    if height == target_height:
        return img
    ratio = target_height / height
    new_width = int(round(width * ratio))
    return img.resize((new_width, target_height), Image.Resampling.LANCZOS)


def create_comparison(
    left_path: Path,
    right_path: Path,
    left_label: str,
    right_label: str,
    output_path: Path,
    target_height: int = 800,
) -> None:
    """Create a labeled side-by-side comparison image."""
    left = resize_to_height(load_image(left_path), target_height)
    right = resize_to_height(load_image(right_path), target_height)

    total_width = left.width + right.width
    total_height = target_height + LABEL_HEIGHT

    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    canvas.paste(left, (0, LABEL_HEIGHT))
    canvas.paste(right, (left.width, LABEL_HEIGHT))

    draw.rectangle([0, 0, total_width, LABEL_HEIGHT], fill="#f0f0f0")
    draw.line([(0, LABEL_HEIGHT), (total_width, LABEL_HEIGHT)], fill="#cccccc", width=1)
    draw.line([(left.width, 0), (left.width, total_height)], fill="#cccccc", width=1)

    left_bbox = draw.textbbox((0, 0), left_label, font=FONT)
    right_bbox = draw.textbbox((0, 0), right_label, font=FONT)
    left_text_w = left_bbox[2] - left_bbox[0]
    right_text_w = right_bbox[2] - right_bbox[0]

    draw.text(
        ((left.width - left_text_w) // 2, (LABEL_HEIGHT - FONT_SIZE) // 2),
        left_label,
        fill="black",
        font=FONT,
    )
    draw.text(
        (left.width + (right.width - right_text_w) // 2, (LABEL_HEIGHT - FONT_SIZE) // 2),
        right_label,
        fill="black",
        font=FONT,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    print(f"Saved: {output_path}")


def process_3d_schematic() -> None:
    """Generate comparisons for the 3D schematic best overlays."""
    base = BEST_OVERLAYS_3D_DIR
    pairs = [
        ("01_panel_1_original.png", "07_panel_1_best_segmentation_v4_kmeans_n8.png", "Panel 1 原始", "Panel 1 分割"),
        ("02_panel_2_original.png", "08_panel_2_best_segmentation_slic_kmeans_n8.png", "Panel 2 原始", "Panel 2 分割"),
        ("03_panel_3_original.png", "09_panel_3_best_segmentation_v4_kmeans_n10.png", "Panel 3 原始", "Panel 3 分割"),
        ("01_panel_1_original.png", "04_panel_1_text_removed.png", "Panel 1 原始", "Panel 1 去文字"),
        ("02_panel_2_original.png", "05_panel_2_text_removed.png", "Panel 2 原始", "Panel 2 去文字"),
        ("03_panel_3_original.png", "06_panel_3_text_removed.png", "Panel 3 原始", "Panel 3 去文字"),
        ("04_panel_1_text_removed.png", "07_panel_1_best_segmentation_v4_kmeans_n8.png", "Panel 1 去文字", "Panel 1 分割"),
        ("05_panel_2_text_removed.png", "08_panel_2_best_segmentation_slic_kmeans_n8.png", "Panel 2 去文字", "Panel 2 分割"),
        ("06_panel_3_text_removed.png", "09_panel_3_best_segmentation_v4_kmeans_n10.png", "Panel 3 去文字", "Panel 3 分割"),
    ]

    for left_name, right_name, left_label, right_label in pairs:
        left_path = base / left_name
        right_path = base / right_name
        if not left_path.exists() or not right_path.exists():
            print(f"Skipping missing pair: {left_name} / {right_name}")
            continue
        out_name = f"3d_schematic_{Path(left_name).stem}_vs_{Path(right_name).stem}.png"
        create_comparison(
            left_path,
            right_path,
            left_label,
            right_label,
            OUTPUT_DIR / out_name,
        )


def get_ph01_original_crop() -> Path:
    """Crop Structural pattern 1 panel from the M0.5 page 7 extracted figure."""
    source = PROJECT_ROOT / "runs" / "M0.5" / "images" / "page_007_img_0.png"
    crop_path = OUTPUT_DIR / "ph01_panel1_original.png"
    if crop_path.exists():
        return crop_path
    img = Image.open(source).convert("RGB")
    # Panel 1 = top row, 2nd column (Structural pattern 1)
    # Derived from gradient-based boundary detection on page_007_img_0.png
    cropped = img.crop((279, 34, 531, 370))
    cropped.save(crop_path, quality=95)
    print(f"Saved ph01 panel crop: {crop_path}")
    return crop_path


def find_best_overlays_original(overlay_name: str) -> Path | None:
    """Attempt to locate the original source image for a best_overlay file."""
    mapping: dict[str, Path | None] = {
        "01_2d_velocity_silixa_page5_v4_pastel.png": (
            PROJECT_ROOT / "docs" / "all_overlays" / "agent_review_vlm" / "silixa2021" / "page5_img1" / "01_original.jpg"
        ),
        "02_2d_velocity_c11b8db_edge_guided.png": (
            PROJECT_ROOT / "docs" / "all_overlays" / "agent_review_vlm" / "gras2019" / "c11b8db80b521d6fb1d17cbd552d29dba2113ab3af362eae113faddfb8890309" / "01_original.jpg"
        ),
        "03_horizon_refine_c11b8db.png": (
            PROJECT_ROOT / "docs" / "all_overlays" / "agent_review_vlm" / "gras2019" / "c11b8db80b521d6fb1d17cbd552d29dba2113ab3af362eae113faddfb8890309" / "01_original.jpg"
        ),
        "04_regional_fusion.jpg": PROJECT_ROOT / "runs" / "test_panel_fix" / "page_011_img_0_panels.jpg",
        "05_3d_schematic_panel1_fused.jpg": BEST_OVERLAYS_3D_DIR / "01_panel_1_original.png",
        "06_tubular_plume_warm_merge.jpg": BEST_OVERLAYS_3D_DIR / "03_panel_3_original.png",
        "07_engine_compare_best_auto.jpg": BEST_OVERLAYS_3D_DIR / "03_panel_3_original.png",
        "08_engine_compare_manual_gt.jpg": BEST_OVERLAYS_3D_DIR / "03_panel_3_original.png",
        "09_text_robust_row_median_post.png": (
            PROJECT_ROOT / "docs" / "all_overlays" / "agent_review_vlm" / "ph01" / "page_004_img_0" / "01_original.jpg"
        ),
        "10_self_heal_silixa_final.jpg": (
            PROJECT_ROOT / "docs" / "all_overlays" / "agent_review_vlm" / "silixa2021" / "page5_img1" / "01_original.jpg"
        ),
        "12_literature_silixa_page4.jpg": (
            PROJECT_ROOT / "docs" / "all_overlays" / "agent_review_vlm" / "silixa2021" / "page4_img0" / "01_original.jpg"
        ),
    }
    if overlay_name == "ph01.jpg":
        return get_ph01_original_crop()
    original = mapping.get(overlay_name)
    if original and original.exists():
        return original
    return None


def process_best_overlays() -> None:
    """Generate comparisons for the general best overlays where originals exist."""
    for overlay_path in sorted(BEST_OVERLAYS_DIR.iterdir()):
        if overlay_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue

        overlay_name = overlay_path.name
        original_path = find_best_overlays_original(overlay_name)

        if original_path is None:
            if "side_by_side" in overlay_name.lower():
                out_name = f"best_overlay_{Path(overlay_name).stem}_comparison.png"
                out_path = OUTPUT_DIR / out_name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                load_image(overlay_path).save(out_path, quality=95)
                print(f"Copied pre-built comparison: {out_path}")
            else:
                print(f"No original found for {overlay_name}, skipping")
            continue

        stem = Path(overlay_name).stem
        out_name = f"best_overlay_{stem}_comparison.png"
        create_comparison(
            original_path,
            overlay_path,
            "原始图像",
            "Overlay",
            OUTPUT_DIR / out_name,
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    process_3d_schematic()
    process_best_overlays()
    print(f"\nAll comparisons written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
