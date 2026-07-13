#!/usr/bin/env python3
"""Focused tuning round: combine row median + dark pixel removal + white-text catch.

Audit take-aways from round 1:
- row_median_7 is the best single method for black horizontal text.
- dark_median_t45 removes residual dark specks that row median misses.
- White/light annotations ("sediment") need the white-text branch from v7.
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
from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend

PANELS = [f"fig6_profile_{i:02d}" for i in range(3, 8)]
ORIG_DIR = Path("runs/feng_fig6_final_v4/crop_tests")
LABELS_DIR = Path("runs/feng_fig6_final_v5")
OUT_DIR = Path("runs/feng_fig6_text_remove_audit/final")


def load_panel(panel_id: str) -> tuple[np.ndarray, np.ndarray]:
    img = np.array(Image.open(ORIG_DIR / f"{panel_id}_cropped.jpg").convert("RGB"))
    labels = np.load(LABELS_DIR / panel_id / "labels.npz")["labels"]
    return img, labels


def make_overlay(panel_rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return generate_overlay_with_legend(panel_rgb, labels)


def remove_dark_pixels_median(
    labels: np.ndarray, img: np.ndarray, dark_threshold: int = 45, neighbor_radius: int = 4
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


def detect_white_text_mask(img: np.ndarray, labels: np.ndarray, sat_drop: float = 40.0) -> np.ndarray:
    pf = img.astype(np.float32)
    sat = pf.max(axis=2) - pf.min(axis=2)
    sat_med = ndimage.median_filter(sat, size=7)
    bright = pf.max(axis=2)
    white_mask = (sat_med - sat > sat_drop) & (bright > 150) & (sat < 60) & (labels != 0)
    white_mask = ndimage.binary_dilation(white_mask, structure=np.ones((5, 5), dtype=bool), iterations=3)
    return white_mask


def fill_mask_with_label_median(img: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return img.copy()
    max_lbl = int(labels.max())
    text_label = max_lbl + 1
    labels_with_text = labels.copy()
    labels_with_text[mask] = text_label
    cleaned_labels = remove_labels_by_ids(labels_with_text, [text_label], fill="nearest")

    cleaned = cv2.inpaint(img, mask.astype(np.uint8) * 255, inpaintRadius=11, flags=cv2.INPAINT_TELEA)

    colors: dict[int, np.ndarray] = {}
    for lbl in sorted(np.unique(cleaned_labels)):
        m = (cleaned_labels == lbl) & (~mask)
        if not m.any():
            continue
        pixels = cleaned[m].astype(np.float32)
        q_low = np.percentile(pixels, 10, axis=0)
        q_high = np.percentile(pixels, 90, axis=0)
        pixels = np.clip(pixels, q_low, q_high)
        colors[int(lbl)] = np.median(pixels, axis=0).astype(np.uint8)

    for lbl in np.unique(cleaned_labels):
        m = mask & (cleaned_labels == lbl)
        if m.any():
            cleaned[m] = colors[int(lbl)]
    return cleaned


def build_runners() -> list[tuple[str, Callable]]:
    runners: list[tuple[str, Callable]] = []

    # Row median + dark pixel combos.
    for rm_size in (6, 7):
        for dark_t in (35, 40, 45):
            def make_combo(rm_size: int, dark_t: int):
                def runner(img: np.ndarray, labels: np.ndarray) -> dict:
                    cleaned = row_median_filter(img, size=rm_size)
                    cleaned = remove_dark_pixels_median(labels, cleaned, dark_threshold=dark_t)
                    return {"cleaned": cleaned, "overlay": make_overlay(cleaned, labels)}
                return runner
            runners.append((f"rm{rm_size}_dark{dark_t}", make_combo(rm_size, dark_t)))

    # Add white-text catch on top of best combo candidate.
    def runner_rm7_dark40_white(img: np.ndarray, labels: np.ndarray) -> dict:
        cleaned = row_median_filter(img, size=7)
        cleaned = remove_dark_pixels_median(labels, cleaned, dark_threshold=40)
        white_mask = detect_white_text_mask(cleaned, labels, sat_drop=35.0)
        cleaned = fill_mask_with_label_median(cleaned, labels, white_mask)
        return {"cleaned": cleaned, "overlay": make_overlay(cleaned, labels), "mask": white_mask}

    def runner_rm7_dark45_white(img: np.ndarray, labels: np.ndarray) -> dict:
        cleaned = row_median_filter(img, size=7)
        cleaned = remove_dark_pixels_median(labels, cleaned, dark_threshold=45)
        white_mask = detect_white_text_mask(cleaned, labels, sat_drop=35.0)
        cleaned = fill_mask_with_label_median(cleaned, labels, white_mask)
        return {"cleaned": cleaned, "overlay": make_overlay(cleaned, labels), "mask": white_mask}

    runners.append(("rm7_dark40_white", runner_rm7_dark40_white))
    runners.append(("rm7_dark45_white", runner_rm7_dark45_white))

    # Stronger row median baseline for comparison.
    def runner_rm7(img: np.ndarray, labels: np.ndarray) -> dict:
        cleaned = row_median_filter(img, size=7)
        return {"cleaned": cleaned, "overlay": make_overlay(cleaned, labels)}

    runners.insert(0, ("rm7_baseline", runner_rm7))
    return runners


def build_grid(original: np.ndarray, results: list[tuple[str, np.ndarray]], max_width: int = 1800) -> np.ndarray:
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
                Image.fromarray((mask * 255).astype(np.uint8)).save(panel_dir / f"{name}_mask.jpg", quality=95)

            grid_entries.append((name, overlay))
            panel_summary[name] = {
                "status": "ok",
                "mask_pixels": int(mask.sum()) if mask is not None else None,
                "mask_percent": round(float(mask.sum() / (h * w) * 100), 2) if mask is not None else None,
            }
            print(f"  {name}: ok")

        grid = build_grid(img, grid_entries, max_width=1800)
        Image.fromarray(grid).save(panel_dir / "_final_grid.jpg", quality=90)
        summary[panel_id] = panel_summary
        print(f"  grid -> {panel_dir / '_final_grid.jpg'}")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone. {OUT_DIR}")


if __name__ == "__main__":
    main()
