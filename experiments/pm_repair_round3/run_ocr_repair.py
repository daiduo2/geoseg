"""Wrapper to run pm_repair.py on fig6_profile_05 with an OCR-derived ROI.

Usage:
    python experiments/pm_repair_round3/run_ocr_repair.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure imports resolve against the repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "pm_repair_round3"))

from geoseg.modules.segment_engines._shared import _create_overlay
from pm_repair import (
    assign_label_to_background,
    repair_pm_artifact,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PANEL_ID = "fig6_profile_05"
OCR_ROI = [108, 39, 150, 66]  # [x1, y1, x2, y2] from OCR

ORIG_DIR = ROOT / "runs/feng_fig6_final_v4/crop_tests"
LABELS_DIR = ROOT / "runs/feng_fig6_comparisons_v7"
OUT_DIR = ROOT / "runs/pm_repair_ocr_experiment" / PANEL_ID


def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img = np.array(
        Image.open(ORIG_DIR / f"{PANEL_ID}_cropped.jpg").convert("RGB")
    )
    labels = np.load(LABELS_DIR / PANEL_ID / "labels.npz")["labels"]

    labels = assign_label_to_background(labels)
    result = repair_pm_artifact(img, labels, roi=OCR_ROI)

    # 1. cleaned_rgb.jpg
    Image.fromarray(result["cleaned_rgb"]).save(
        OUT_DIR / "cleaned_rgb.jpg", quality=95
    )

    # 2. overlay_repaired.jpg
    Image.fromarray(result["overlay"]).save(
        OUT_DIR / "overlay_repaired.jpg", quality=95
    )

    # 3. labels_repaired.npz
    np.savez_compressed(OUT_DIR / "labels_repaired.npz", labels=result["labels"])

    # 4. comparison.jpg
    orig_overlay = _create_overlay(
        img,
        labels,
        np.zeros((1, 3), dtype=np.uint8),
        alpha=0.65,
        boundary_mode="thin",
        skip_background=True,
        min_area_frac=0.001,
        fill_mode="blend",
    )
    mask_overlay = _create_overlay(
        result["cleaned_rgb"],
        result["labels"],
        np.zeros((1, 3), dtype=np.uint8),
        alpha=1.0,
        boundary_mode="thin",
        skip_background=True,
        min_area_frac=0.001,
        fill_mode="mask",
    )

    h, w = img.shape[:2]
    gap = 10
    comparison = np.full((h, w * 3 + gap * 2, 3), 32, dtype=np.uint8)
    comparison[:, :w] = img
    comparison[:, w + gap : 2 * w + gap] = orig_overlay
    comparison[:, 2 * w + gap * 2 :] = mask_overlay
    Image.fromarray(comparison).save(OUT_DIR / "comparison.jpg", quality=95)

    summary = {
        "panel_id": PANEL_ID,
        "roi_used": list(result["roi"]),
        "text_mask_pixels": int(result["text_mask"].sum()),
        "comparison_path": str(OUT_DIR / "comparison.jpg"),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Output directory: {OUT_DIR}")
    print(f"Comparison: {OUT_DIR / 'comparison.jpg'}")
    print(f"ROI used: {result['roi']}")
    print(f"Text mask pixels: {result['text_mask'].sum()}")

    return summary


if __name__ == "__main__":
    main()
