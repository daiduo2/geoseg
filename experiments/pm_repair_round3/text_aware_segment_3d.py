"""Text-aware segmentation experiment for 3D schematic panels.

Pipeline:
    1. Detect text boxes with PaddleOCR.
    2. Inpaint text regions to remove them from the image.
    3. Run v4_kmeans segmentation on the cleaned image.
    4. Compare against baseline segmentation on the original image.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoseg.modules.segment_engines.v4_kmeans import segment as v4_segment

OCR_VENV_PYTHON = ROOT / ".venv_ocr" / "bin" / "python"
OCR_ROI_SCRIPT = ROOT / "experiments" / "pm_repair_round3" / "ocr_roi.py"


def detect_text_rois_subprocess(image_path: Path) -> list[dict]:
    """Run OCR in the isolated .venv_ocr via subprocess."""
    cmd = [
        str(OCR_VENV_PYTHON),
        str(OCR_ROI_SCRIPT),
        "--all",
        str(image_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def build_text_mask(
    shape: tuple[int, int],
    matches: list[dict],
    dilate_iters: int = 3,
) -> np.ndarray:
    """Build a binary mask from OCR text ROIs."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for match in matches:
        x1, y1, x2, y2 = match["roi"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = 255

    if dilate_iters > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iters)
    return mask


def inpaint_text(
    panel_rgb: np.ndarray,
    matches: list[dict],
    radius: int = 3,
) -> np.ndarray:
    """Inpaint text regions identified by OCR."""
    mask = build_text_mask(panel_rgb.shape[:2], matches)
    cleaned = cv2.inpaint(
        panel_rgb,
        mask,
        radius,
        cv2.INPAINT_NS,
    )
    return cleaned


def compute_label_difference(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    """Return a mask where the two label maps disagree."""
    return labels_a != labels_b


def process_panel(
    image_path: Path,
    output_dir: Path,
    n_layers: int = 5,
) -> dict:
    """Run baseline vs text-aware segmentation and save comparisons."""
    panel_id = image_path.stem
    out = output_dir / panel_id
    out.mkdir(parents=True, exist_ok=True)

    panel_rgb = np.array(Image.open(image_path).convert("RGB"))
    h, w = panel_rgb.shape[:2]

    # --- 1. OCR text detection ---
    matches = detect_text_rois_subprocess(image_path)
    text_mask = build_text_mask((h, w), matches)
    cleaned_rgb = inpaint_text(panel_rgb, matches)

    Image.fromarray(cleaned_rgb).save(out / "01_inpainted.jpg", quality=95)
    Image.fromarray(text_mask).save(out / "02_text_mask.jpg", quality=95)

    # --- 2. Baseline segmentation on original ---
    baseline_result = v4_segment(panel_rgb, n_layers=n_layers)
    baseline_labels = baseline_result["labels"]
    baseline_overlay = baseline_result["overlay"]

    np.savez_compressed(out / "labels_baseline.npz", labels=baseline_labels)
    Image.fromarray(baseline_overlay).save(out / "03_baseline_overlay.jpg", quality=95)

    # --- 3. Text-aware segmentation on cleaned image ---
    cleaned_result = v4_segment(cleaned_rgb, n_layers=n_layers)
    cleaned_labels = cleaned_result["labels"]
    cleaned_overlay_on_cleaned = cleaned_result["overlay"]

    # Overlay cleaned labels on original RGB for fair visual comparison
    from geoseg.modules.segment_engines._shared import _create_overlay
    cleaned_overlay_on_orig = _create_overlay(
        panel_rgb,
        cleaned_labels,
        np.zeros((1, 3), dtype=np.uint8),
        alpha=0.65,
        boundary_mode="thin",
        skip_background=True,
        min_area_frac=0.001,
        fill_mode="blend",
    )

    np.savez_compressed(out / "labels_text_aware.npz", labels=cleaned_labels)
    Image.fromarray(cleaned_overlay_on_cleaned).save(
        out / "04_text_aware_overlay_on_cleaned.jpg", quality=95
    )
    Image.fromarray(cleaned_overlay_on_orig).save(
        out / "05_text_aware_overlay_on_orig.jpg", quality=95
    )

    # --- 4. Difference map ---
    diff_mask = compute_label_difference(baseline_labels, cleaned_labels)
    diff_vis = np.full((h, w, 3), 32, dtype=np.uint8)
    diff_vis[diff_mask] = [255, 0, 0]  # red where labels differ
    Image.fromarray(diff_vis).save(out / "06_label_difference.jpg", quality=95)

    diff_frac = float(diff_mask.sum() / diff_mask.size)

    summary = {
        "panel_id": panel_id,
        "image_path": str(image_path),
        "n_text_boxes": len(matches),
        "text_boxes": matches,
        "diff_fraction": round(diff_frac, 4),
        "outputs": {
            "inpainted": str(out / "01_inpainted.jpg"),
            "text_mask": str(out / "02_text_mask.jpg"),
            "baseline_overlay": str(out / "03_baseline_overlay.jpg"),
            "text_aware_overlay_cleaned": str(out / "04_text_aware_overlay_on_cleaned.jpg"),
            "text_aware_overlay_orig": str(out / "05_text_aware_overlay_on_orig.jpg"),
            "label_difference": str(out / "06_label_difference.jpg"),
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n{panel_id}: diff_fraction={diff_frac:.4f}")
    print(f"  outputs: {out}")
    return summary


def main() -> int:
    base = Path("/Users/daiduo2/geoseg/docs/best_overlays_3d_schematic")
    output_dir = Path("/Users/daiduo2/geoseg/runs/pm_repair_ocr_experiment/text_aware_segmentation")
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        (base / "01_panel_1_original.png", 5),
        (base / "02_panel_2_original.png", 5),
        (base / "03_panel_3_original.png", 6),
    ]

    summaries = []
    for path, n_layers in panels:
        if not path.exists():
            print(f"SKIP: {path} not found")
            continue
        summary = process_panel(path, output_dir, n_layers=n_layers)
        summaries.append(summary)

    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nAll outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
