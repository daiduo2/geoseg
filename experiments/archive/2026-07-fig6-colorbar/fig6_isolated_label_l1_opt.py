#!/usr/bin/env python3
"""Isolate the most biased label per profile and perform L1-optimal color fit.

For each profile:
1. Identify the label with the largest L1 RGB bias vs current palette.
2. Crop/mask the original image to only that label's region.
3. Compute the L1-optimal color = per-channel median of the isolated pixels.
4. Render: original | isolated label | current color patch | optimized color patch |
   current L1 residual (isolated) | optimized L1 residual (isolated).

This demonstrates the user's proposal: remove other regions' influence and optimize
a single label's color freely using L1 vector norm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path("/Users/daiduo2/geoseg")
LABELS_DIR = ROOT / "runs" / "fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5"
PANEL_DIR = ROOT / "runs" / "feng_fig6_final_v4" / "crop_tests"
SRC_RECOLOR_DIR = ROOT / "runs" / "fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5_label12_recolor"
OUT_DIR = ROOT / "runs" / "fig6_colorbar_16zone_experiment_no_merge_clean_text_pm_smooth_s5_isolated_l1_opt"

PROFILES = [f"fig6_profile_{i:02d}" for i in range(3, 8)]
PROFILE_LABELS = {
    "fig6_profile_03": "Profile 03",
    "fig6_profile_04": "Profile 04 (PM artifact)",
    "fig6_profile_05": "Profile 05 (PM artifact)",
    "fig6_profile_06": "Profile 06",
    "fig6_profile_07": "Profile 07",
}


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


def load_palette(panel_id: str) -> np.ndarray:
    # Use the original 16-zone colorbar palette as baseline to demonstrate L1 optimization.
    summary = json.loads((SRC_RECOLOR_DIR / "summary.json").read_text(encoding="utf-8"))
    return np.array(summary["base_palette"], dtype=np.uint8)


def l1_residual(original: np.ndarray, color: np.ndarray) -> float:
    """Mean L1 norm of RGB vector differences for all pixels."""
    diff = np.abs(original.astype(np.float32) - color.astype(np.float32))
    return float(diff.sum(axis=1).mean())


def isolate_label(original: np.ndarray, labels: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a masked RGB image and the pixel list for the target label."""
    mask = labels == target
    isolated = np.zeros_like(original)
    isolated[mask] = original[mask]
    pixels = original[mask].astype(np.float32)
    return isolated, pixels


def create_patch(color: np.ndarray, size: int = 80) -> np.ndarray:
    return np.full((size, size, 3), color, dtype=np.uint8)


def create_residual_image(original: np.ndarray, labels: np.ndarray, target: int, color: np.ndarray) -> np.ndarray:
    """Absolute RGB residual restricted to the target label region."""
    mask = labels == target
    residual = np.zeros_like(original)
    diff = np.abs(original.astype(np.float32) - color.astype(np.float32))
    residual[mask] = np.clip(diff[mask], 0, 255).astype(np.uint8)
    return residual


def assemble_row(
    original: np.ndarray,
    isolated: np.ndarray,
    current_patch: np.ndarray,
    optimized_patch: np.ndarray,
    current_residual: np.ndarray,
    optimized_residual: np.ndarray,
    label: int,
    current_l1: float,
    optimized_l1: float,
) -> Image.Image:
    h, w = original.shape[:2]
    patch_h = 80
    patch_w = 80
    header_h = 30
    n_cols = 6
    canvas = Image.new("RGB", (w * 2 + patch_w * 2 + w * 2 + (n_cols - 1) * 6, h + header_h), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(14)
    small_font = _load_font(12)

    headers = [
        "Original",
        f"Label {label} isolated",
        "Current color",
        "L1-optimal color",
        f"Current L1={current_l1:.1f}",
        f"Optimized L1={optimized_l1:.1f}",
    ]
    xs = [0, w + 6, 2 * w + 12, 2 * w + patch_w + 18, 2 * w + 2 * patch_w + 24, 3 * w + 2 * patch_w + 30]

    for x, hdr in zip(xs, headers):
        bbox = draw.textbbox((0, 0), hdr, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (w if "Original" in hdr or "isolated" in hdr or "L1=" in hdr else patch_w) // 2 - text_w // 2, 8), hdr, fill=(220, 220, 220), font=font)

    canvas.paste(Image.fromarray(original), (xs[0], header_h))
    canvas.paste(Image.fromarray(isolated), (xs[1], header_h))
    canvas.paste(Image.fromarray(current_patch), (xs[2], header_h))
    canvas.paste(Image.fromarray(optimized_patch), (xs[3], header_h))
    canvas.paste(Image.fromarray(current_residual), (xs[4], header_h))
    canvas.paste(Image.fromarray(optimized_residual), (xs[5], header_h))

    # Label info
    info = f"label {label}: current L1 {current_l1:.1f} → optimized L1 {optimized_l1:.1f} (Δ {current_l1 - optimized_l1:.1f})"
    draw.text((10, header_h + h + 4), info, fill=(200, 200, 200), font=small_font)

    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    row_images: list[Image.Image] = []

    for panel_id in PROFILES:
        labels = np.load(LABELS_DIR / panel_id / "labels.npz")["labels"]
        original = np.array(Image.open(PANEL_DIR / f"{panel_id}_cropped.jpg").convert("RGB"))
        palette = load_palette(panel_id)

        # Find label with largest L1 RGB bias.
        best_label = -1
        best_l1 = -1.0
        best_pixels = np.array([])
        for lbl in sorted(np.unique(labels)):
            if lbl == 0:
                continue
            mask = labels == lbl
            if not mask.any():
                continue
            pixels = original[mask].astype(np.float32)
            current_color = palette[lbl].astype(np.float32)
            l1 = float(np.abs(pixels - current_color).sum(axis=1).mean())
            if l1 > best_l1:
                best_l1 = l1
                best_label = int(lbl)
                best_pixels = pixels

        if best_label == -1 or best_pixels.size == 0:
            continue

        # L1-optimal color = per-channel median.
        optimized_color = np.median(best_pixels, axis=0).astype(np.uint8)
        current_color = palette[best_label]

        isolated, _ = isolate_label(original, labels, best_label)
        current_residual = create_residual_image(original, labels, best_label, current_color)
        optimized_residual = create_residual_image(original, labels, best_label, optimized_color)

        current_l1 = l1_residual(best_pixels, current_color)
        optimized_l1 = l1_residual(best_pixels, optimized_color)

        current_patch = create_patch(current_color)
        optimized_patch = create_patch(optimized_color)

        row_img = assemble_row(
            original,
            isolated,
            current_patch,
            optimized_patch,
            current_residual,
            optimized_residual,
            best_label,
            current_l1,
            optimized_l1,
        )
        row_path = OUT_DIR / f"{panel_id}_label{best_label}_l1_opt.jpg"
        row_img.save(row_path, quality=95)
        row_images.append(row_img)

        summary[panel_id] = {
            "label": best_label,
            "current_color": current_color.tolist(),
            "optimized_color": optimized_color.tolist(),
            "current_l1": round(current_l1, 2),
            "optimized_l1": round(optimized_l1, 2),
            "reduction": round(current_l1 - optimized_l1, 2),
            "pixel_count": int(best_pixels.shape[0]),
            "row_path": str(row_path),
        }
        print(f"{panel_id}: label {best_label} L1 {current_l1:.1f} → {optimized_l1:.1f}")

    # Big stacked figure.
    widths = [img.width for img in row_images]
    max_w = max(widths)
    total_h = sum(img.height for img in row_images)
    big = Image.new("RGB", (max_w, total_h), (32, 32, 32))
    y = 0
    for img in row_images:
        big.paste(img, (0, y))
        y += img.height
    big_path = OUT_DIR / "isolated_l1_opt_all_profiles.jpg"
    big.save(big_path, quality=95)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved big figure: {big_path}")
    print(f"Saved summary: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
