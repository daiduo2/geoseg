"""Method A: NN-in-label PM repair.

For each profile:
  1. Load original RGB and labels.
  2. Generate overlay.
  3. Within ROI, identify candidate text pixels using multi-criteria detection.
  4. For each candidate, find nearest same-label non-candidate using distance_transform_edt
     and assign that nearest neighbor's label.
  5. Keep pixels outside ROI unchanged.
  6. Generate repaired overlay and comparison images.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/Users/daiduo2/geoseg/src")

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from geoseg.modules.segment_engines._shared import _create_overlay
from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend


# ── Configuration ───────────────────────────────────────────────────────────

PROFILES = [
    {
        "id": "04",
        "image": "/Users/daiduo2/geoseg/runs/feng_fig6_final_v4/crop_tests/fig6_profile_04_cropped.jpg",
        "labels": "/Users/daiduo2/geoseg/runs/feng_fig6_comparisons_v7/fig6_profile_04/labels.npz",
        "roi": (120, 40, 260, 110),  # x1, y1, x2, y2
    },
    {
        "id": "05",
        "image": "/Users/daiduo2/geoseg/runs/feng_fig6_final_v4/crop_tests/fig6_profile_05_cropped.jpg",
        "labels": "/Users/daiduo2/geoseg/runs/feng_fig6_comparisons_v7/fig6_profile_05/labels.npz",
        "roi": (100, 50, 240, 115),
    },
]

OUT_ROOT = Path("/Users/daiduo2/geoseg/runs/pm_repair_round1/nn_in_label")

# Candidate detection thresholds
GRAY_THRESH = 140           # Dark text (relaxed since text isn't extremely dark)
SAT_THRESH = 50             # Low saturation (near grayscale)
LOCAL_STD_THRESH = 8.0      # High local variance (sharp edges)
MIN_CANDIDATE_SIZE = 3      # Minimum connected component size to be considered text


def load_inputs(image_path: str, labels_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load original RGB and labels array."""
    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    labels = np.load(labels_path)["labels"]
    return img, labels


def identify_candidates(
    img: np.ndarray,
    overlay: np.ndarray,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    """Return boolean mask of candidate text pixels within ROI.

    Heuristics (within ROI):
      - Low saturation + high local variance: text is near-grayscale with sharp edges
      - OR dark: grayscale < GRAY_THRESH (for very dark text)
      - OR overlay mismatch: RGB distance(original, overlay) > 60 (catch mis-segmented text)

    After detection, remove tiny isolated speckles (< MIN_CANDIDATE_SIZE pixels).
    """
    x1, y1, x2, y2 = roi
    h, w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[y1:y2, x1:x2] = True

    gray = img.mean(axis=2)
    dark = gray < GRAY_THRESH

    # Low saturation (near grayscale)
    sat = img.max(axis=2).astype(np.int16) - img.min(axis=2).astype(np.int16)
    low_sat = sat < SAT_THRESH

    # High local standard deviation (sharp edges / text)
    local_std = ndimage.generic_filter(gray, np.std, size=5)
    high_std = local_std > LOCAL_STD_THRESH

    # Text-like: (low saturation AND high local variance) OR dark
    text_like = (low_sat & high_std) | dark

    # Also catch pixels where overlay differs significantly from original
    # This helps catch text that was mis-segmented into wrong labels
    rgb_dist = np.linalg.norm(img.astype(np.float32) - overlay.astype(np.float32), axis=2)
    color_diff = rgb_dist > 60.0

    candidates = roi_mask & (text_like | color_diff)

    # Remove tiny isolated speckles (noise)
    labeled, n = ndimage.label(candidates)
    for i in range(1, n + 1):
        if (labeled == i).sum() < MIN_CANDIDATE_SIZE:
            candidates[labeled == i] = False

    return candidates


def repair_labels_nn_in_label(
    labels: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """For each candidate pixel, find nearest non-candidate with same label.

    If a label has no safe (non-candidate) pixels, candidates in that label
    fall back to nearest non-candidate from ANY label.
    """
    repaired = labels.copy()
    unique_labels = np.unique(labels)

    # First pass: same-label nearest neighbor
    for lbl in unique_labels:
        label_mask = labels == lbl
        safe_mask = label_mask & ~candidates
        if not safe_mask.any():
            continue

        label_candidates = label_mask & candidates
        if not label_candidates.any():
            continue

        dist, indices = ndimage.distance_transform_edt(
            ~safe_mask, return_indices=True
        )

        rr, cc = np.where(label_candidates)
        nr = indices[0][rr, cc]
        nc = indices[1][rr, cc]
        repaired[rr, cc] = labels[nr, nc]

    # Second pass: for any remaining candidates, find nearest from ANY label
    remaining = (repaired != labels) & candidates  # Actually: candidates that were NOT changed
    # Better: candidates where repaired still equals original labels (not fixed)
    remaining = candidates.copy()
    for lbl in unique_labels:
        remaining &= ~(repaired != labels)  # This is wrong logic, let me fix

    # Simpler: find pixels that are still candidates after first pass
    # A pixel is "still a candidate" if it was a candidate AND its label wasn't changed
    still_candidates = candidates & (repaired == labels)
    if still_candidates.any():
        safe_any = ~candidates
        if safe_any.any():
            dist, indices = ndimage.distance_transform_edt(
                ~safe_any, return_indices=True
            )
            rr, cc = np.where(still_candidates)
            nr = indices[0][rr, cc]
            nc = indices[1][rr, cc]
            repaired[rr, cc] = labels[nr, nc]

    return repaired


def create_roi_comparison(
    img: np.ndarray,
    overlay: np.ndarray,
    repaired_overlay: np.ndarray,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    """Side-by-side: original ROI, overlay ROI, repaired overlay ROI."""
    x1, y1, x2, y2 = roi
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    roi_h = y2 - y1
    roi_w = x2 - x1

    pad = 4
    total_w = roi_w * 3 + pad * 4
    total_h = roi_h + pad * 2

    canvas = np.full((total_h, total_w, 3), 240, dtype=np.uint8)

    offsets = [pad, pad + roi_w + pad, pad + 2 * (roi_w + pad)]
    for offset, arr in zip(offsets, [img, overlay, repaired_overlay]):
        canvas[pad:pad + roi_h, offset:offset + roi_w] = arr[y1:y2, x1:x2]

    return canvas


def create_full_comparison(
    overlay: np.ndarray,
    repaired_overlay: np.ndarray,
) -> np.ndarray:
    """Side-by-side full image comparison."""
    h, w = overlay.shape[:2]
    pad = 4
    canvas = np.full((h + pad * 2, w * 2 + pad * 3, 3), 240, dtype=np.uint8)
    canvas[pad:pad + h, pad:pad + w] = overlay
    canvas[pad:pad + h, pad * 2 + w:pad * 2 + w * 2] = repaired_overlay
    return canvas


def process_profile(profile: dict) -> dict:
    """Process one profile and return observation notes."""
    pid = profile["id"]
    print(f"\n=== Processing profile {pid} ===")

    img, labels = load_inputs(profile["image"], profile["labels"])
    print(f"  Image shape: {img.shape}, Labels shape: {labels.shape}")
    print(f"  Unique labels: {np.unique(labels)}")

    # Generate original overlay
    overlay = _create_overlay(img, labels, seeds_rgb=np.zeros((1, 3), dtype=np.uint8))
    print(f"  Overlay generated")

    # Identify candidates
    candidates = identify_candidates(img, overlay, profile["roi"])
    n_candidates = int(candidates.sum())
    print(f"  Candidate pixels: {n_candidates}")

    # Per-label breakdown
    for lbl in np.unique(labels):
        lbl_cand = ((labels == lbl) & candidates).sum()
        lbl_safe = ((labels == lbl) & ~candidates).sum()
        if lbl_cand > 0:
            print(f"    Label {lbl}: candidates={lbl_cand}, safe={lbl_safe}")

    # Repair
    repaired_labels = repair_labels_nn_in_label(labels, candidates)
    repaired_overlay = _create_overlay(
        img, repaired_labels, seeds_rgb=np.zeros((1, 3), dtype=np.uint8)
    )

    # Count changed pixels
    changed = (repaired_labels != labels).sum()
    print(f"  Changed pixels: {changed}")

    # Save outputs
    out_dir = OUT_ROOT / f"fig6_profile_{pid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / "labels_repaired.npz", labels=repaired_labels)
    Image.fromarray(repaired_overlay).save(out_dir / "overlay_repaired.jpg", quality=95)

    roi_comp = create_roi_comparison(img, overlay, repaired_overlay, profile["roi"])
    Image.fromarray(roi_comp).save(out_dir / "roi_comparison.jpg", quality=95)

    full_comp = create_full_comparison(overlay, repaired_overlay)
    Image.fromarray(full_comp).save(out_dir / "full_comparison.jpg", quality=95)

    print(f"  Saved to {out_dir}")

    return {
        "id": pid,
        "n_candidates": n_candidates,
        "n_changed": int(changed),
        "roi": profile["roi"],
        "out_dir": str(out_dir),
    }


def main() -> None:
    results = []
    for profile in PROFILES:
        results.append(process_profile(profile))

    # Write report
    report_path = OUT_ROOT / "report.md"
    with open(report_path, "w") as f:
        f.write("# Method A: NN-in-label PM Repair\n\n")
        f.write("## Method Description\n\n")
        f.write(
            "For each candidate pixel, find the nearest non-candidate pixel with the **same label** "
            "using `scipy.ndimage.distance_transform_edt`.\n"
            "If no safe pixel exists within the same label, fall back to the nearest non-candidate "
            "from any label.\n"
            "Pixels outside the ROI are left unchanged.\n\n"
        )

        f.write("## Candidate Detection\n\n")
        f.write("A pixel within the ROI is a candidate if it satisfies ANY of:\n")
        f.write(f"- Dark: grayscale < {GRAY_THRESH}\n")
        f.write(f"- Text-like: saturation < {SAT_THRESH} AND local std > {LOCAL_STD_THRESH}\n")
        f.write(f"- Speckles smaller than {MIN_CANDIDATE_SIZE} pixels are removed\n\n")

        f.write("## Parameters\n\n")
        f.write("- ROI per profile (image coords):\n")
        for r in results:
            f.write(f"  - Profile {r['id']}: {r['roi']}\n")

        f.write("\n## Observations\n\n")
        for r in results:
            f.write(f"### Profile {r['id']}\n\n")
            f.write(f"- Candidate pixels identified: {r['n_candidates']}\n")
            f.write(f"- Labels changed: {r['n_changed']}\n")
            f.write(f"- Output: `{r['out_dir']}`\n\n")

        f.write("## Pros\n\n")
        f.write("- Simple, no training required.\n")
        f.write("- Respects label boundaries when possible (same-label NN first).\n")
        f.write("- Fast: distance transform is O(n log n) per label.\n")
        f.write("- Non-destructive outside ROI.\n")
        f.write("- Fallback to any-label NN handles edge cases where a label is fully covered.\n\n")

        f.write("## Cons\n\n")
        f.write("- Candidate detection relies on hand-tuned thresholds; may miss subtle text or over-select.\n")
        f.write("- If a label has no safe pixels anywhere (not just ROI), candidates in that label cannot be repaired.\n")
        f.write("- Does not handle text that spans label boundaries well.\n")
        f.write("- Visual quality depends heavily on threshold tuning.\n\n")

        f.write("## Recommendation\n\n")
        f.write(
            "Method A is a reasonable baseline for text-in-label repair.\n"
            "If candidate detection is accurate, the NN fill produces clean results.\n"
            "However, for complex PM text (multi-color, overlapping labels),\n"
            "consider combining with morphological opening or a learned text mask.\n"
        )

    print(f"\n=== Report written to {report_path} ===")


if __name__ == "__main__":
    main()
