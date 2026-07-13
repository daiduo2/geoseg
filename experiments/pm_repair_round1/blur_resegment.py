#!/usr/bin/env python3
"""Method C: Blur + local resegment PM repair.

For each profile:
1. Load original RGB and labels.
2. Crop the ROI, apply Gaussian blur with sigma=1.5.
3. Determine n_layers from global labels (max label id, excluding 0).
4. Run v4_kmeans.segment on the blurred ROI with n_layers=n_layers.
5. Fuse the new ROI labels back into the global label map.
6. Generate overlay and comparison images.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, "/Users/daiduo2/geoseg/src")

import numpy as np
from PIL import Image
from scipy import ndimage

from geoseg.modules.segment_engines.v4_kmeans import segment as v4_segment
from geoseg.modules.segment_engines._shared import _create_overlay


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/Users/daiduo2/geoseg")
INPUT_IMG_DIR = BASE / "runs/feng_fig6_final_v4/crop_tests"
INPUT_LABELS_DIR = BASE / "runs/feng_fig6_comparisons_v7"
OUTPUT_DIR = BASE / "runs/pm_repair_round1/blur_resegment"

PROFILES = [
    {
        "id": "fig6_profile_04",
        "roi": {"x1": 120, "y1": 40, "x2": 260, "y2": 110},
    },
    {
        "id": "fig6_profile_05",
        "roi": {"x1": 100, "y1": 50, "x2": 240, "y2": 115},
    },
]

BLUR_SIGMA = 1.5


def load_rgb(path: Path) -> np.ndarray:
    """Load image as RGB uint8 array."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def load_labels(path: Path) -> np.ndarray:
    """Load labels from .npz file."""
    data = np.load(path)
    # Common keys: "labels", "arr_0"
    for key in ("labels", "arr_0"):
        if key in data:
            return data[key].astype(np.int32)
    raise KeyError(f"No labels key found in {path}; keys={list(data.keys())}")


def save_labels(path: Path, labels: np.ndarray) -> None:
    """Save labels to .npz file."""
    np.savez_compressed(path, labels=labels)


def determine_n_layers(labels: np.ndarray) -> int:
    """Determine number of layers from global labels (max label id, excluding 0)."""
    unique = np.unique(labels)
    non_bg = unique[unique != 0]
    if len(non_bg) == 0:
        return 1
    return int(non_bg.max()) + 1  # labels are 0-indexed, so max+1 = count


def apply_blur(roi_rgb: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to ROI RGB image."""
    blurred = np.zeros_like(roi_rgb, dtype=np.float32)
    for c in range(3):
        blurred[:, :, c] = ndimage.gaussian_filter(
            roi_rgb[:, :, c].astype(np.float32), sigma=sigma
        )
    return np.clip(blurred, 0, 255).astype(np.uint8)


def fuse_labels(
    global_labels: np.ndarray,
    roi_labels: np.ndarray,
    x1: int,
    y1: int,
) -> np.ndarray:
    """Fuse ROI labels back into global label map.

    Replaces global labels inside ROI with resegmented ROI labels.
    """
    fused = global_labels.copy()
    h, w = roi_labels.shape
    fused[y1 : y1 + h, x1 : x1 + w] = roi_labels
    return fused


def create_comparison_image(
    original_rgb: np.ndarray,
    original_labels: np.ndarray,
    repaired_labels: np.ndarray,
    roi: dict,
    seeds_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Create side-by-side comparison: original vs repaired overlay."""
    h, w, _ = original_rgb.shape

    # Original overlay
    if seeds_rgb is not None:
        orig_overlay = _create_overlay(original_rgb, original_labels, seeds_rgb)
    else:
        orig_overlay = _create_overlay(
            original_rgb, original_labels, np.zeros((1, 3), dtype=np.uint8)
        )

    # Repaired overlay
    if seeds_rgb is not None:
        rep_overlay = _create_overlay(original_rgb, repaired_labels, seeds_rgb)
    else:
        rep_overlay = _create_overlay(
            original_rgb, repaired_labels, np.zeros((1, 3), dtype=np.uint8)
        )

    # Side by side
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
    comparison[:, :w] = orig_overlay
    comparison[:, w:] = rep_overlay

    return comparison


def create_roi_comparison(
    roi_rgb: np.ndarray,
    original_roi_labels: np.ndarray,
    repaired_roi_labels: np.ndarray,
    seeds_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Create side-by-side comparison for ROI only."""
    h, w, _ = roi_rgb.shape

    if seeds_rgb is not None:
        orig_overlay = _create_overlay(roi_rgb, original_roi_labels, seeds_rgb)
        rep_overlay = _create_overlay(roi_rgb, repaired_roi_labels, seeds_rgb)
    else:
        orig_overlay = _create_overlay(
            roi_rgb, original_roi_labels, np.zeros((1, 3), dtype=np.uint8)
        )
        rep_overlay = _create_overlay(
            roi_rgb, repaired_roi_labels, np.zeros((1, 3), dtype=np.uint8)
        )

    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
    comparison[:, :w] = orig_overlay
    comparison[:, w:] = rep_overlay

    return comparison


def process_profile(profile: dict) -> dict:
    """Process a single profile: blur + resegment + fuse + save."""
    prof_id = profile["id"]
    roi = profile["roi"]
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]

    print(f"\n{'='*60}")
    print(f"Processing {prof_id}")
    print(f"ROI: ({x1}, {y1}) -> ({x2}, {y2})")
    print(f"{'='*60}")

    # Load inputs
    img_path = INPUT_IMG_DIR / f"{prof_id}_cropped.jpg"
    labels_path = INPUT_LABELS_DIR / prof_id / "labels.npz"

    print(f"  Image: {img_path}")
    print(f"  Labels: {labels_path}")

    original_rgb = load_rgb(img_path)
    original_labels = load_labels(labels_path)

    print(f"  Image shape: {original_rgb.shape}")
    print(f"  Labels shape: {original_labels.shape}")
    print(f"  Unique labels: {np.unique(original_labels)}")

    # Determine n_layers from global labels
    n_layers = determine_n_layers(original_labels)
    print(f"  Detected n_layers: {n_layers}")

    # Crop ROI
    roi_rgb = original_rgb[y1:y2, x1:x2].copy()
    roi_labels_orig = original_labels[y1:y2, x1:x2].copy()
    print(f"  ROI shape: {roi_rgb.shape}")

    # Apply Gaussian blur
    roi_blurred = apply_blur(roi_rgb, BLUR_SIGMA)
    print(f"  Applied Gaussian blur (sigma={BLUR_SIGMA})")

    # Run v4_kmeans.segment on blurred ROI
    print(f"  Running v4_kmeans.segment with n_layers={n_layers} ...")
    result = v4_segment(roi_blurred, n_layers=n_layers)
    roi_labels_new = result["labels"]
    seeds_rgb = np.array(result["seeds"], dtype=np.uint8)

    print(f"  Result labels unique: {np.unique(roi_labels_new)}")
    print(f"  Result seeds: {seeds_rgb.tolist()}")
    print(f"  Result meta: {result['meta']}")

    # Fuse back into global labels
    repaired_labels = fuse_labels(original_labels, roi_labels_new, x1, y1)
    print(f"  Fused ROI labels back into global map")
    print(f"  Repaired labels unique: {np.unique(repaired_labels)}")

    # Create output directory
    out_dir = OUTPUT_DIR / prof_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save repaired labels
    labels_path_out = out_dir / "labels_repaired.npz"
    save_labels(labels_path_out, repaired_labels)
    print(f"  Saved: {labels_path_out}")

    # Save repaired overlay (full image)
    overlay_repaired = _create_overlay(original_rgb, repaired_labels, seeds_rgb)
    overlay_path = out_dir / "overlay_repaired.jpg"
    Image.fromarray(overlay_repaired).save(overlay_path, quality=95)
    print(f"  Saved: {overlay_path}")

    # Save ROI comparison
    roi_comparison = create_roi_comparison(
        roi_rgb, roi_labels_orig, roi_labels_new, seeds_rgb
    )
    roi_comp_path = out_dir / "roi_comparison.jpg"
    Image.fromarray(roi_comparison).save(roi_comp_path, quality=95)
    print(f"  Saved: {roi_comp_path}")

    # Save full comparison
    full_comparison = create_comparison_image(
        original_rgb, original_labels, repaired_labels, roi, seeds_rgb
    )
    full_comp_path = out_dir / "full_comparison.jpg"
    Image.fromarray(full_comparison).save(full_comp_path, quality=95)
    print(f"  Saved: {full_comp_path}")

    return {
        "profile_id": prof_id,
        "n_layers": n_layers,
        "roi_shape": roi_rgb.shape,
        "original_unique_labels": np.unique(original_labels).tolist(),
        "repaired_unique_labels": np.unique(repaired_labels).tolist(),
        "engine_path": result["meta"].get("path", "unknown"),
        "output_dir": str(out_dir),
    }


def main() -> None:
    """Run blur + resegment PM repair for all profiles."""
    print("Method C: Blur + local resegment PM repair")
    print(f"Output directory: {OUTPUT_DIR}")

    results = []
    for profile in PROFILES:
        try:
            result = process_profile(profile)
            results.append(result)
        except Exception as e:
            print(f"ERROR processing {profile['id']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "profile_id": profile["id"],
                "error": str(e),
            })

    # Write report
    report_path = OUTPUT_DIR / "report.md"
    write_report(report_path, results)
    print(f"\nReport saved: {report_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


def write_report(path: Path, results: list[dict]) -> None:
    """Write experiment report."""
    lines = [
        "# Method C: Blur + Local Resegment PM Repair — Report",
        "",
        f"Date: 2026-06-23",
        f"Blur sigma: {BLUR_SIGMA}",
        "",
        "## Profiles Processed",
        "",
    ]

    for r in results:
        if "error" in r:
            lines.append(f"### {r['profile_id']} — ERROR")
            lines.append(f"- Error: {r['error']}")
        else:
            lines.append(f"### {r['profile_id']}")
            lines.append(f"- ROI shape: {r['roi_shape']}")
            lines.append(f"- n_layers: {r['n_layers']}")
            lines.append(f"- Engine path: {r['engine_path']}")
            lines.append(f"- Original labels: {r['original_unique_labels']}")
            lines.append(f"- Repaired labels: {r['repaired_unique_labels']}")
            lines.append(f"- Output: {r['output_dir']}")
        lines.append("")

    lines.extend([
        "## Observations",
        "",
        "- TBD: inspect overlay outputs visually",
        "",
        "## Recommendation",
        "",
        "- TBD: based on visual inspection",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
