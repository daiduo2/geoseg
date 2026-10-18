#!/usr/bin/env python3
"""Run all existing text-removal algorithms on fig6_profile_03..07.

Produces side-by-side comparison grids for visual audit. Each panel gets:
- cleaned image
- overlay (where applicable)
- text mask (where applicable)
- a comparison grid of every method plus the original
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import segmentation
from skimage.color import rgb2lab
from skimage.measure import regionprops, label as sklabel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseg.modules.text_removal import remove_text
from geoseg.modules.post_process.merge import remove_labels_by_ids
from geoseg.modules.segment_engines._shared import (
    _create_overlay,
    _distinct_colors,
    adaptive_blur,
    row_median_filter,
)
from geoseg.modules.segment_engines.regional_fusion import (
    generate_overlay_with_legend,
)

PANELS = [f"fig6_profile_{i:02d}" for i in range(3, 8)]
ORIG_DIR = Path("runs/feng_fig6_final_v4/crop_tests")
LABELS_DIR = Path("runs/feng_fig6_final_v5")
OUT_DIR = Path("runs/feng_fig6_text_remove_audit")


def load_panel(panel_id: str) -> tuple[np.ndarray, np.ndarray]:
    img = np.array(
        Image.open(ORIG_DIR / f"{panel_id}_cropped.jpg").convert("RGB")
    )
    labels = np.load(LABELS_DIR / panel_id / "labels.npz")["labels"]
    return img, labels


def save_result(
    out_dir: Path,
    name: str,
    cleaned: np.ndarray,
    overlay: np.ndarray | None,
    mask: np.ndarray | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cleaned).save(out_dir / f"{name}_cleaned.jpg", quality=95)
    if overlay is not None:
        Image.fromarray(overlay).save(out_dir / f"{name}_overlay.jpg", quality=95)
    if mask is not None:
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            out_dir / f"{name}_mask.jpg", quality=95
        )


def make_overlay(panel_rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return generate_overlay_with_legend(panel_rgb, labels)


# --------------------------------------------------------------------------- #
# Algorithm A: library text_removal.remove_text (MSER + Laplacian + inpaint)
# --------------------------------------------------------------------------- #
def run_text_removal_py(img: np.ndarray, labels: np.ndarray) -> dict:
    cleaned, mask = remove_text(img)
    return {
        "cleaned": cleaned,
        "overlay": make_overlay(cleaned, labels),
        "mask": mask.astype(bool),
    }


# --------------------------------------------------------------------------- #
# Algorithm B: row median filter (anisotropic preprocessing)
# --------------------------------------------------------------------------- #
def make_row_median_runner(size: int) -> Callable:
    def runner(img: np.ndarray, labels: np.ndarray) -> dict:
        cleaned = row_median_filter(img, size=size)
        return {
            "cleaned": cleaned,
            "overlay": make_overlay(cleaned, labels),
            "mask": None,
        }

    runner.__name__ = f"row_median_{size}"
    return runner


# --------------------------------------------------------------------------- #
# Algorithm C: adaptive Gaussian blur
# --------------------------------------------------------------------------- #
def run_adaptive_blur(img: np.ndarray, labels: np.ndarray) -> dict:
    cleaned = adaptive_blur(img)
    return {
        "cleaned": cleaned,
        "overlay": make_overlay(cleaned, labels),
        "mask": None,
    }


# --------------------------------------------------------------------------- #
# Algorithm D: v7 outlier-based text mask + smooth_text_into_labels
# --------------------------------------------------------------------------- #
def detect_text_mask_v7(panel_rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
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


def compute_label_colors(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    exclude_mask: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    colors: dict[int, np.ndarray] = {}
    exclude = exclude_mask if exclude_mask is not None else np.zeros(labels.shape, dtype=bool)
    for lbl in sorted(np.unique(labels)):
        mask = (labels == lbl) & (~exclude)
        if not mask.any():
            continue
        pixels = panel_rgb[mask].astype(np.float32)
        q_low = np.percentile(pixels, 10, axis=0)
        q_high = np.percentile(pixels, 90, axis=0)
        pixels = np.clip(pixels, q_low, q_high)
        colors[int(lbl)] = np.median(pixels, axis=0).astype(np.uint8)
    return colors


def run_v7_outlier(img: np.ndarray, labels: np.ndarray) -> dict:
    text_mask = detect_text_mask_v7(img, labels)
    if not text_mask.any():
        return {"cleaned": img.copy(), "overlay": make_overlay(img, labels), "mask": text_mask}

    max_lbl = int(labels.max())
    text_label = max_lbl + 1
    labels_with_text = labels.copy()
    labels_with_text[text_mask] = text_label
    cleaned_labels = remove_labels_by_ids(labels_with_text, [text_label], fill="nearest")

    cleaned = cv2.inpaint(
        img,
        text_mask.astype(np.uint8) * 255,
        inpaintRadius=11,
        flags=cv2.INPAINT_TELEA,
    )

    colors = compute_label_colors(cleaned, cleaned_labels, text_mask)
    for lbl in np.unique(cleaned_labels):
        m = text_mask & (cleaned_labels == lbl)
        if m.any():
            cleaned[m] = colors[int(lbl)]

    return {
        "cleaned": cleaned,
        "overlay": make_overlay(cleaned, cleaned_labels),
        "mask": text_mask,
    }


# --------------------------------------------------------------------------- #
# Algorithm E: v5 per-label text-like subcomponent + dark component removal
# --------------------------------------------------------------------------- #
def remove_text_like_subcomponents(
    labels: np.ndarray,
    img: np.ndarray,
    min_area: int = 5,
    max_area: int = 500,
    ratio_thresh: float = 25.0,
) -> np.ndarray:
    result = labels.copy()
    gray = img.mean(axis=2).astype(np.float32)

    for lbl in sorted(set(labels.flatten()) - {0}):
        mask = labels == lbl
        cc = sklabel(mask, connectivity=2)
        regions = regionprops(cc)

        for r in regions:
            area = r.area
            if area < min_area or area > max_area:
                continue
            ratio = (r.perimeter**2) / max(area, 1)
            if ratio < ratio_thresh:
                continue
            comp_mask = cc == r.label
            comp_gray_mean = gray[comp_mask].mean()
            dilated = ndimage.binary_dilation(comp_mask, iterations=2)
            neighbor_mask = dilated & ~comp_mask & mask
            if neighbor_mask.any():
                neighbor_gray_mean = gray[neighbor_mask].mean()
                if comp_gray_mean >= neighbor_gray_mean - 10:
                    continue
            dilated = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool))
            neighbors = result[dilated & ~comp_mask]
            neighbors = neighbors[neighbors != 0]
            if len(neighbors) > 0:
                vals, counts = np.unique(neighbors, return_counts=True)
                result[comp_mask] = vals[counts.argmax()]
    return result


def remove_dark_components_v5(labels: np.ndarray, img: np.ndarray) -> np.ndarray:
    result = labels.copy()
    gray = img.mean(axis=2).astype(np.float32)
    dark = gray < 75
    cc = sklabel(dark, connectivity=2)
    for r in regionprops(cc):
        area = r.area
        if area < 3 or area > 400:
            continue
        ratio = (r.perimeter**2) / max(area, 1)
        if not (ratio > 20 or area < 30):
            continue
        comp_mask = cc == r.label
        dilated = ndimage.binary_dilation(comp_mask, iterations=3)
        border = dilated & ~comp_mask
        border_labels = result[border]
        border_labels = border_labels[border_labels != 0]
        if len(border_labels) == 0:
            continue
        vals, counts = np.unique(border_labels, return_counts=True)
        dominant = vals[counts.argmax()]
        if counts.max() / counts.sum() > 0.5:
            result[comp_mask] = dominant
    return result


def run_v5_shape(img: np.ndarray, labels: np.ndarray) -> dict:
    result = remove_text_like_subcomponents(labels, img, min_area=5, max_area=500, ratio_thresh=20.0)
    result = remove_dark_components_v5(result, img)
    return {
        "cleaned": img.copy(),
        "overlay": make_overlay(img, result),
        "mask": None,
    }


# --------------------------------------------------------------------------- #
# Algorithm F: v6 dark+edge small components, overlay median fill
# --------------------------------------------------------------------------- #
def detect_text_mask_v6(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark = gray < 80
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    high_edge = lap > np.percentile(lap, 75)
    text_candidates = dark & high_edge
    cc, num = ndimage.label(text_candidates)
    text_mask = np.zeros((h, w), dtype=bool)
    for r in regionprops(cc):
        if 3 <= r.area <= 500:
            text_mask[cc == r.label] = True
    text_mask = ndimage.binary_dilation(text_mask, iterations=1)
    return text_mask


def create_overlay_median_v6(
    img: np.ndarray, labels: np.ndarray, text_mask: np.ndarray, alpha: float = 0.65
) -> np.ndarray:
    overlay = _create_overlay(
        img,
        labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=alpha,
        boundary_mode="thin",
        skip_background=True,
        min_area_frac=0.001,
        fill_mode="blend",
    )
    h, w = labels.shape
    result = overlay.copy().astype(np.float32)
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        lbl_mask = labels == lbl
        lbl_text = text_mask & lbl_mask
        if not lbl_text.any():
            continue
        ys, xs = np.where(lbl_text)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 3), min(h, y + 4)
            x0, x1 = max(0, x - 3), min(w, x + 4)
            neighbor_mask = (~text_mask[y0:y1, x0:x1]) & lbl_mask[y0:y1, x0:x1]
            if neighbor_mask.any():
                result[y, x] = overlay[y0:y1, x0:x1][neighbor_mask].mean(axis=0)
    return np.clip(result, 0, 255).astype(np.uint8)


def run_v6_overlay_median(img: np.ndarray, labels: np.ndarray) -> dict:
    text_mask = detect_text_mask_v6(img, labels)
    overlay = create_overlay_median_v6(img, labels, text_mask)
    return {
        "cleaned": img.copy(),
        "overlay": overlay,
        "mask": text_mask,
    }


# --------------------------------------------------------------------------- #
# Algorithm G: v3 overlay nearest fill
# --------------------------------------------------------------------------- #
def detect_text_mask_v3(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark = gray < 55
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    lap_mask = lap > np.percentile(lap, 80)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=15, C=3
    )
    text_mask = dark | (lap_mask & (adaptive > 0))
    text_mask = ndimage.binary_dilation(text_mask, iterations=1)
    return text_mask


def overlay_nearest_fill(img: np.ndarray, labels: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    overlay = _create_overlay(
        img,
        labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=0.65,
        boundary_mode="thin",
        skip_background=True,
        min_area_frac=0.001,
        fill_mode="blend",
    )
    boundaries = segmentation.find_boundaries(labels, mode="thin")
    valid_mask = (~text_mask) & (~boundaries)
    if not valid_mask.any():
        return overlay
    dist, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
    result = overlay.copy()
    ys, xs = np.where(text_mask & ~boundaries)
    if len(ys) > 0:
        result[ys, xs] = overlay[indices[0][ys, xs], indices[1][ys, xs]]
    return result


def run_v3_overlay_nearest(img: np.ndarray, labels: np.ndarray) -> dict:
    text_mask = detect_text_mask_v3(img)
    overlay = overlay_nearest_fill(img, labels, text_mask)
    return {
        "cleaned": img.copy(),
        "overlay": overlay,
        "mask": text_mask,
    }


# --------------------------------------------------------------------------- #
# Algorithm H: v4 conservative small component + nearest fill
# --------------------------------------------------------------------------- #
def detect_text_mask_v4(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    very_dark = gray < 45
    cc, num = ndimage.label(very_dark)
    text_mask = np.zeros((h, w), dtype=bool)
    for r in regionprops(cc):
        if 5 <= r.area <= 300:
            comp_mask = cc == r.label
            dilated = ndimage.binary_dilation(comp_mask, iterations=2)
            border = dilated & ~comp_mask
            border_labels = labels[border]
            border_labels = border_labels[border_labels != 0]
            if len(border_labels) > 0:
                vals, counts = np.unique(border_labels, return_counts=True)
                if counts.max() / counts.sum() > 0.6:
                    text_mask[comp_mask] = True
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    high_edge = lap > np.percentile(lap, 92)
    small_high_edge = high_edge & (gray < 80)
    cc2, num2 = ndimage.label(small_high_edge)
    for r in regionprops(cc2):
        if 3 <= r.area <= 150:
            text_mask[cc2 == r.label] = True
    text_mask = ndimage.binary_dilation(text_mask, iterations=1)
    return text_mask


def run_v4_conservative(img: np.ndarray, labels: np.ndarray) -> dict:
    text_mask = detect_text_mask_v4(img, labels)
    overlay = overlay_nearest_fill(img, labels, text_mask)
    return {
        "cleaned": img.copy(),
        "overlay": overlay,
        "mask": text_mask,
    }


# --------------------------------------------------------------------------- #
# Algorithm I: v7 final dark-pixel removal within labels
# --------------------------------------------------------------------------- #
def remove_dark_pixels_median(
    labels: np.ndarray, img: np.ndarray, dark_threshold: int = 55, neighbor_radius: int = 4
) -> np.ndarray:
    h, w = labels.shape
    gray = img.mean(axis=2).astype(np.float32)
    dark_mask = gray < dark_threshold
    result = img.copy().astype(np.float32)

    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        lbl_dark = dark_mask & lbl_mask
        if not lbl_dark.any():
            continue
        ys, xs = np.where(lbl_dark)
        for y, x in zip(ys, xs):
            for radius in (neighbor_radius, 8):
                y0, y1 = max(0, y - radius), min(h, y + radius + 1)
                x0, x1 = max(0, x - radius), min(w, x + radius + 1)
                neighbor_mask = lbl_mask[y0:y1, x0:x1] & ~dark_mask[y0:y1, x0:x1]
                if neighbor_mask.sum() >= 3:
                    result[y, x] = np.median(img[y0:y1, x0:x1][neighbor_mask], axis=0)
                    break
    return np.clip(result, 0, 255).astype(np.uint8)


def remove_dark_pixels_distance(labels: np.ndarray, img: np.ndarray, dark_threshold: int = 55) -> np.ndarray:
    h, w = labels.shape
    gray = img.mean(axis=2).astype(np.float32)
    dark_mask = gray < dark_threshold
    result = img.copy()
    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        lbl_dark = dark_mask & lbl_mask
        if not lbl_dark.any():
            continue
        valid = lbl_mask & ~dark_mask
        if not valid.any():
            continue
        dist, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
        ys, xs = np.where(lbl_dark)
        result[ys, xs] = img[indices[0][ys, xs], indices[1][ys, xs]]
    return result


def remove_dark_pixels_inpaint(labels: np.ndarray, img: np.ndarray, dark_threshold: int = 55) -> np.ndarray:
    h, w = labels.shape
    gray = img.mean(axis=2).astype(np.float32)
    dark_mask = gray < dark_threshold
    result = img.copy()
    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        lbl_dark = dark_mask & lbl_mask
        if not lbl_dark.any():
            continue
        mask = lbl_dark.astype(np.uint8) * 255
        ys, xs = np.where(lbl_mask)
        y0, y1 = max(0, ys.min() - 3), min(h, ys.max() + 4)
        x0, x1 = max(0, xs.min() - 3), min(w, xs.max() + 4)
        roi_img = result[y0:y1, x0:x1].copy()
        roi_mask = mask[y0:y1, x0:x1]
        if roi_mask.sum() > 0:
            inpainted = cv2.inpaint(roi_img, roi_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
            result[y0:y1, x0:x1] = inpainted
    return result


def make_dark_pixel_runner(method: str, dark_threshold: int) -> Callable:
    def runner(img: np.ndarray, labels: np.ndarray) -> dict:
        if method == "median":
            cleaned = remove_dark_pixels_median(labels, img, dark_threshold=dark_threshold)
        elif method == "distance":
            cleaned = remove_dark_pixels_distance(labels, img, dark_threshold=dark_threshold)
        else:
            cleaned = remove_dark_pixels_inpaint(labels, img, dark_threshold=dark_threshold)
        return {
            "cleaned": cleaned,
            "overlay": make_overlay(cleaned, labels),
            "mask": None,
        }

    runner.__name__ = f"dark_pixels_{method}_t{dark_threshold}"
    return runner


# --------------------------------------------------------------------------- #
# Master runner
# --------------------------------------------------------------------------- #
ALGORITHMS: list[tuple[str, Callable]] = [
    ("A_text_removal_py", run_text_removal_py),
    ("B_row_median_5", make_row_median_runner(5)),
    ("B_row_median_7", make_row_median_runner(7)),
    ("B_row_median_9", make_row_median_runner(9)),
    ("C_adaptive_blur", run_adaptive_blur),
    ("D_v7_outlier", run_v7_outlier),
    ("E_v5_shape", run_v5_shape),
    ("F_v6_overlay_median", run_v6_overlay_median),
    ("G_v3_overlay_nearest", run_v3_overlay_nearest),
    ("H_v4_conservative", run_v4_conservative),
    ("I_dark_median_t55", make_dark_pixel_runner("median", 55)),
    ("I_dark_distance_t55", make_dark_pixel_runner("distance", 55)),
    ("I_dark_inpaint_t55", make_dark_pixel_runner("inpaint", 55)),
]


def build_comparison_grid(
    original: np.ndarray,
    results: list[tuple[str, np.ndarray]],
    max_width: int = 2400,
) -> np.ndarray:
    """Stack method overlays vertically with labels."""
    n = len(results)
    h, w = original.shape[:2]
    label_h = 30
    thumb_h = h
    thumb_w = w

    scale = min(1.0, max_width / thumb_w)
    if scale < 1.0:
        thumb_h = int(h * scale)
        thumb_w = int(w * scale)

    def resize(arr: np.ndarray) -> np.ndarray:
        if scale >= 1.0:
            return arr
        return cv2.resize(arr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

    canvas_h = (thumb_h + label_h) * n
    canvas = np.full((canvas_h, thumb_w, 3), 255, dtype=np.uint8)

    for i, (name, img) in enumerate(results):
        y0 = i * (thumb_h + label_h)
        canvas[y0 + label_h : y0 + label_h + thumb_h] = resize(img)
        cv2.putText(
            canvas,
            name,
            (10, y0 + label_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for panel_id in PANELS:
        print(f"\n=== {panel_id} ===")
        img, labels = load_panel(panel_id)
        h, w = img.shape[:2]
        panel_dir = OUT_DIR / panel_id
        panel_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(img).save(panel_dir / "00_original.jpg", quality=95)
        np.savez_compressed(panel_dir / "labels.npz", labels=labels)
        Image.fromarray(make_overlay(img, labels)).save(panel_dir / "00_original_overlay.jpg", quality=95)

        grid_entries: list[tuple[str, np.ndarray]] = [("ORIGINAL", img)]
        panel_summary: dict[str, dict] = {}

        for name, runner in ALGORITHMS:
            print(f"  running {name} ...")
            try:
                result = runner(img, labels)
            except Exception as e:
                print(f"    FAILED: {e}")
                panel_summary[name] = {"status": "failed", "error": str(e)}
                continue

            cleaned = result["cleaned"]
            overlay = result.get("overlay")
            mask = result.get("mask")

            save_result(panel_dir / name, name, cleaned, overlay, mask)
            grid_entries.append((name, overlay if overlay is not None else cleaned))

            panel_summary[name] = {
                "status": "ok",
                "mask_pixels": int(mask.sum()) if mask is not None else None,
                "mask_percent": round(float(mask.sum() / (h * w) * 100), 2) if mask is not None else None,
            }

        grid = build_comparison_grid(img, grid_entries, max_width=2400)
        Image.fromarray(grid).save(panel_dir / "_comparison_grid.jpg", quality=90)
        summary[panel_id] = panel_summary
        print(f"  grid saved: {panel_dir / '_comparison_grid.jpg'}")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAll done. Summary: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
