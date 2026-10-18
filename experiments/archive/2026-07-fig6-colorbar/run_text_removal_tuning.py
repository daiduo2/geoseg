#!/usr/bin/env python3
"""Parameter-tuning sweep for text removal, driven by visual audit feedback.

Focus areas after baseline audit:
1. row_median_filter sweet spot (size 5-9, especially 6-8).
2. dark-pixel threshold within labels (35-65).
3. v7 outlier with adjusted thresholds / dilation.
4. combos: row_median + dark_pixel, row_median + v7_outlier.
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
from skimage.color import rgb2lab

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseg.modules.post_process.merge import remove_labels_by_ids
from geoseg.modules.segment_engines._shared import row_median_filter
from geoseg.modules.segment_engines.regional_fusion import (
    generate_overlay_with_legend,
)

PANELS = [f"fig6_profile_{i:02d}" for i in range(3, 8)]
ORIG_DIR = Path("runs/feng_fig6_final_v4/crop_tests")
LABELS_DIR = Path("runs/feng_fig6_final_v5")
OUT_DIR = Path("runs/feng_fig6_text_remove_audit/tuning")


def load_panel(panel_id: str) -> tuple[np.ndarray, np.ndarray]:
    img = np.array(
        Image.open(ORIG_DIR / f"{panel_id}_cropped.jpg").convert("RGB")
    )
    labels = np.load(LABELS_DIR / panel_id / "labels.npz")["labels"]
    return img, labels


def make_overlay(panel_rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return generate_overlay_with_legend(panel_rgb, labels)


def detect_text_mask_v7(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    rgb_dist_thresh: float = 30.0,
    l_diff_thresh: float = 20.0,
    white_sat_drop: float = 40.0,
    dilation_iterations: int = 4,
    min_area: int = 30,
) -> np.ndarray:
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
    dark_mask = (rgb_dist > rgb_dist_thresh) | (
        (np.abs(L - L_med) > l_diff_thresh) & (L < 120)
    )

    sat = pf.max(axis=2) - pf.min(axis=2)
    sat_med = ndimage.median_filter(sat, size=7)
    bright = pf.max(axis=2)
    white_mask = (sat_med - sat > white_sat_drop) & (bright > 150) & (sat < 60)

    text_mask = (dark_mask | white_mask) & (labels != 0)
    struct = np.ones((5, 5), dtype=bool)
    text_mask = ndimage.binary_dilation(text_mask, structure=struct, iterations=dilation_iterations)

    cc, num = ndimage.label(text_mask)
    for i in range(1, num + 1):
        comp = cc == i
        area = int(comp.sum())
        if area < min_area or area > max(15000, h * w // 4):
            text_mask[comp] = False
    return text_mask


def inpaint_with_nearest_label_fill(
    img: np.ndarray, labels: np.ndarray, text_mask: np.ndarray
) -> np.ndarray:
    """Remove text labels and inpaint; then back-fill with per-label median color."""
    if not text_mask.any():
        return img.copy()

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

    colors: dict[int, np.ndarray] = {}
    for lbl in sorted(np.unique(cleaned_labels)):
        mask = (cleaned_labels == lbl) & (~text_mask)
        if not mask.any():
            continue
        pixels = cleaned[mask].astype(np.float32)
        q_low = np.percentile(pixels, 10, axis=0)
        q_high = np.percentile(pixels, 90, axis=0)
        pixels = np.clip(pixels, q_low, q_high)
        colors[int(lbl)] = np.median(pixels, axis=0).astype(np.uint8)

    for lbl in np.unique(cleaned_labels):
        m = text_mask & (cleaned_labels == lbl)
        if m.any():
            cleaned[m] = colors[int(lbl)]
    return cleaned


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


def build_runners() -> list[tuple[str, Callable]]:
    runners: list[tuple[str, Callable]] = []

    # Row-median sweep.
    for size in (5, 6, 7, 8, 9):
        def make_row_median(size: int):
            def runner(img: np.ndarray, labels: np.ndarray) -> dict:
                cleaned = row_median_filter(img, size=size)
                return {"cleaned": cleaned, "overlay": make_overlay(cleaned, labels)}
            return runner
        runners.append((f"row_median_{size}", make_row_median(size)))

    # Dark-pixel sweep.
    for thresh in (35, 45, 55, 65):
        def make_dark(thresh: int):
            def runner(img: np.ndarray, labels: np.ndarray) -> dict:
                cleaned = remove_dark_pixels_median(labels, img, dark_threshold=thresh)
                return {"cleaned": cleaned, "overlay": make_overlay(cleaned, labels)}
            return runner
        runners.append((f"dark_median_t{thresh}", make_dark(thresh)))

    # v7 outlier sweep.
    v7_configs = [
        {"name": "v7_conservative", "rgb": 40, "ldiff": 25, "dilate": 3, "min_area": 50},
        {"name": "v7_default", "rgb": 30, "ldiff": 20, "dilate": 4, "min_area": 30},
        {"name": "v7_aggressive", "rgb": 20, "ldiff": 15, "dilate": 5, "min_area": 20},
        {"name": "v7_white_focus", "rgb": 30, "ldiff": 20, "dilate": 4, "min_area": 30, "white_drop": 25},
    ]
    for cfg in v7_configs:
        def make_v7(cfg: dict):
            def runner(img: np.ndarray, labels: np.ndarray) -> dict:
                mask = detect_text_mask_v7(
                    img,
                    labels,
                    rgb_dist_thresh=cfg["rgb"],
                    l_diff_thresh=cfg["ldiff"],
                    white_sat_drop=cfg.get("white_drop", 40.0),
                    dilation_iterations=cfg["dilate"],
                    min_area=cfg["min_area"],
                )
                cleaned = inpaint_with_nearest_label_fill(img, labels, mask)
                return {"cleaned": cleaned, "overlay": make_overlay(cleaned, labels), "mask": mask}
            return runner
        runners.append((cfg["name"], make_v7(cfg)))

    # Combinations.
    def combo_row7_dark45(img: np.ndarray, labels: np.ndarray) -> dict:
        cleaned = row_median_filter(img, size=7)
        cleaned = remove_dark_pixels_median(labels, cleaned, dark_threshold=45)
        return {"cleaned": cleaned, "overlay": make_overlay(cleaned, labels)}

    def combo_row7_v7default(img: np.ndarray, labels: np.ndarray) -> dict:
        cleaned = row_median_filter(img, size=7)
        mask = detect_text_mask_v7(cleaned, labels, rgb_dist_thresh=30, l_diff_thresh=20, dilation_iterations=4)
        cleaned2 = inpaint_with_nearest_label_fill(cleaned, labels, mask)
        return {"cleaned": cleaned2, "overlay": make_overlay(cleaned2, labels), "mask": mask}

    runners.append(("combo_row7_dark45", combo_row7_dark45))
    runners.append(("combo_row7_v7default", combo_row7_v7default))

    return runners


def build_grid(
    original: np.ndarray,
    results: list[tuple[str, np.ndarray]],
    max_width: int = 1800,
) -> np.ndarray:
    n = len(results)
    h, w = original.shape[:2]
    label_h = 28
    scale = min(1.0, max_width / w)
    thumb_h = int(h * scale)
    thumb_w = int(w * scale)

    def resize(arr: np.ndarray) -> np.ndarray:
        if scale >= 1.0:
            return arr
        return cv2.resize(arr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

    canvas = np.full(((thumb_h + label_h) * n, thumb_w, 3), 255, dtype=np.uint8)
    for i, (name, img) in enumerate(results):
        y0 = i * (thumb_h + label_h)
        canvas[y0 + label_h : y0 + label_h + thumb_h] = resize(img)
        cv2.putText(
            canvas, name, (8, y0 + label_h - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
        )
    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runners = build_runners()
    summary: dict[str, dict] = {}

    for panel_id in PANELS:
        print(f"\n=== {panel_id} ===")
        img, labels = load_panel(panel_id)
        h, w = img.shape[:2]
        panel_dir = OUT_DIR / panel_id
        panel_dir.mkdir(parents=True, exist_ok=True)

        grid_entries = [("ORIGINAL", img)]
        panel_summary: dict[str, dict] = {}

        for name, runner in runners:
            try:
                result = runner(img, labels)
            except Exception as e:
                print(f"  {name}: FAILED {e}")
                panel_summary[name] = {"status": "failed", "error": str(e)}
                continue

            cleaned = result["cleaned"]
            overlay = result.get("overlay", cleaned)
            mask = result.get("mask")

            Image.fromarray(cleaned).save(panel_dir / f"{name}_cleaned.jpg", quality=95)
            Image.fromarray(overlay).save(panel_dir / f"{name}_overlay.jpg", quality=95)
            if mask is not None:
                Image.fromarray((mask * 255).astype(np.uint8)).save(
                    panel_dir / f"{name}_mask.jpg", quality=95
                )

            grid_entries.append((name, overlay))
            panel_summary[name] = {
                "status": "ok",
                "mask_pixels": int(mask.sum()) if mask is not None else None,
                "mask_percent": round(float(mask.sum() / (h * w) * 100), 2) if mask is not None else None,
            }
            print(f"  {name}: ok")

        grid = build_grid(img, grid_entries, max_width=1800)
        Image.fromarray(grid).save(panel_dir / "_tuning_grid.jpg", quality=90)
        summary[panel_id] = panel_summary
        print(f"  grid -> {panel_dir / '_tuning_grid.jpg'}")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone. {OUT_DIR}")


if __name__ == "__main__":
    main()
