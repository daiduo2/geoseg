#!/usr/bin/env python3
"""Final text-removal delivery: apply the audit-selected best method to all panels.

Selected strategy per visual audit:
- row_median_filter(size=7) to suppress horizontal black text strokes and scatter dots.
- remove_dark_pixels_median(dark_threshold=40) to clean residual dark specks inside labels.

This is intentionally conservative on background/legend text to avoid damaging
colorbar/legend regions. It is applied to the original cropped panel image only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseg.modules.segment_engines._shared import row_median_filter
from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend

PANELS = [f"fig6_profile_{i:02d}" for i in range(3, 8)]
ORIG_DIR = Path("runs/feng_fig6_final_v4/crop_tests")
LABELS_DIR = Path("runs/feng_fig6_final_v5")
OUT_DIR = Path("runs/feng_fig6_text_remove_audit/delivery")


def remove_dark_pixels_median(
    labels: np.ndarray,
    img: np.ndarray,
    dark_threshold: int = 40,
    neighbor_radius: int = 4,
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


def remove_text_best(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Apply audit-selected text removal pipeline."""
    cleaned = row_median_filter(img, size=7)
    cleaned = remove_dark_pixels_median(labels, cleaned, dark_threshold=40)
    return cleaned


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for panel_id in PANELS:
        img = np.array(Image.open(ORIG_DIR / f"{panel_id}_cropped.jpg").convert("RGB"))
        labels = np.load(LABELS_DIR / panel_id / "labels.npz")["labels"]

        cleaned = remove_text_best(img, labels)
        overlay = generate_overlay_with_legend(cleaned, labels)

        panel_dir = OUT_DIR / panel_id
        panel_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cleaned).save(panel_dir / "panel_cleaned.jpg", quality=95)
        Image.fromarray(overlay).save(panel_dir / "overlay_cleaned.jpg", quality=95)
        np.savez_compressed(panel_dir / "labels.npz", labels=labels)

        summary[panel_id] = {
            "method": "row_median_filter(size=7) + remove_dark_pixels_median(dark_threshold=40)",
            "original_path": str(ORIG_DIR / f"{panel_id}_cropped.jpg"),
            "cleaned_path": str(panel_dir / "panel_cleaned.jpg"),
            "overlay_path": str(panel_dir / "overlay_cleaned.jpg"),
        }
        print(f"{panel_id}: cleaned -> {panel_dir / 'panel_cleaned.jpg'}")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDelivered to {OUT_DIR}")


if __name__ == "__main__":
    main()
