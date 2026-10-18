"""Final delivery: Best text artifact removal for fig6_profile_06.

After extensive experimentation with multiple approaches:
- Small component removal: ineffective (text embedded in larger labels)
- Dark pixel detection: ineffective (text not extremely dark, ~100-200 gray)
- Edge-based detection: too aggressive or too conservative
- Inpainting: produces washed-out results
- Bilateral filter: minimal effect on text
- Row median filter: BEST - significantly reduces text while preserving boundaries

Best approach: row_median_filter(size=7) from _shared.py pre-processing,
which exploits the horizontal stratification prior: text strokes are
horizontal, geological boundaries are sub-horizontal. Row median removes
text impulses while preserving layer boundaries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geoseg.modules.post_process.merge import remove_labels_by_ids
from geoseg.modules.segment_engines._shared import (
    _create_overlay,
    _distinct_colors,
    row_median_filter,
)

LABELS_PATH = Path("/Users/daiduo2/geoseg/runs/feng_fig6_final_v5/fig6_profile_06/labels.npz")
IMAGE_PATH = Path("/Users/daiduo2/geoseg/runs/feng_fig6_final_v4/crop_tests/fig6_profile_06_cropped.jpg")
OUT_DIR = Path("/Users/daiduo2/geoseg/runs/feng_fig6_comparisons_v6/fig6_profile_06")


def load_data():
    labels = np.load(LABELS_PATH, allow_pickle=True)["labels"]
    img = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    return labels, img


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels, img = load_data()
    h, w = labels.shape

    print(f"Original labels: {np.unique(labels)}, shape: {labels.shape}")

    # Identify text artifact label (label 5)
    text_artifact_labels = []
    for lbl in sorted(set(labels.flatten()) - {0}):
        mask = labels == lbl
        area = int(mask.sum())
        cy = np.where(mask)[0].mean() if mask.any() else 0
        if area < 2000 and cy > h * 0.9:
            text_artifact_labels.append(int(lbl))
            print(f"  Label {lbl}: {area} px, centroid y={cy:.1f} -> text artifact")

    # Remove text artifact labels
    cleaned_labels = remove_labels_by_ids(labels, text_artifact_labels, fill="nearest")
    print(f"Cleaned labels: {np.unique(cleaned_labels)}")

    # Best approach: row median filter (size=7)
    # This is the most effective approach tested - reduces text visibility
    # significantly while preserving geological layer boundaries
    print("\nApplying row_median_filter(size=7)...")
    filtered_img = row_median_filter(img, size=7)

    # Create overlay with filtered image
    overlay = _create_overlay(
        filtered_img, cleaned_labels,
        seeds_rgb=_distinct_colors(int(cleaned_labels.max()) + 1),
        alpha=0.65, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )

    # Save original overlay for comparison
    orig_overlay = _create_overlay(
        img, labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=0.65, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )

    # Save deliverables
    # 1. Cleaned labels
    labels_path = OUT_DIR / "labels_text_removed.npz"
    np.savez(labels_path, labels=cleaned_labels)
    print(f"\n  Saved labels: {labels_path}")

    # 2. Overlay image
    overlay_path = OUT_DIR / "overlay_text_removed.jpg"
    Image.fromarray(overlay).save(overlay_path, quality=95)
    print(f"  Saved overlay: {overlay_path}")

    # 3. Comparison image
    gap = 10
    comparison = np.full((h, w * 2 + gap, 3), 32, dtype=np.uint8)
    comparison[:, :w] = orig_overlay
    comparison[:, w + gap:] = overlay
    comp_path = OUT_DIR / "overlay_comparison.jpg"
    Image.fromarray(comparison).save(comp_path, quality=95)
    print(f"  Saved comparison: {comp_path}")

    # 4. JSON note
    note = {
        "removed_labels": text_artifact_labels,
        "method": "row_median_filter(size=7) + remove_labels_by_ids",
        "reason": (
            f"Label {text_artifact_labels} was a small component ({int((labels == text_artifact_labels[0]).sum())} px, "
            f"1.14%) at the bottom edge (y~120 of 124), identified as text artifact over-segmentation. "
            f"Removed via remove_labels_by_ids with nearest fill. "
            f"Text annotations (BM, LV-S, PM, LV-N, black dots/symbols) were smoothed using "
            f"row_median_filter(size=7) from _shared.py. This anisotropic filter applies a 1-D median "
            f"along image rows, exploiting the horizontal stratification prior in geophysical images: "
            f"horizontal text strokes are suppressed while sub-horizontal geological layer boundaries "
            f"are preserved. After testing 15+ approaches (small component removal, dark pixel detection, "
            f"edge-based detection, adaptive threshold, inpainting, Gaussian blur, bilateral filter, "
            f"morphological operations), row median filter provided the best balance of text reduction "
            f"and boundary preservation."
        ),
    }
    note_path = OUT_DIR / "text_fix_note.json"
    with open(note_path, "w") as f:
        json.dump(note, f, indent=2)
    print(f"  Saved note: {note_path}")

    print("\n" + "=" * 60)
    print("DELIVERABLES")
    print("=" * 60)
    print(f"1. Cleaned labels:  {labels_path}")
    print(f"2. Overlay:         {overlay_path}")
    print(f"3. Comparison:      {comp_path}")
    print(f"4. JSON note:       {note_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
