"""Text-aware post-segmentation repair for 3D schematic panels.

Strategy:
    1. Segment the original image with v4_kmeans (baseline).
    2. Detect text boxes with PaddleOCR.
    3. For each text ROI, locally repair labels using the same B->C pipeline
       as pm_repair.py (inpaint text pixels + nearest-color relabel).
    4. Compare repaired labels against baseline.

This avoids global re-segmentation after inpainting, which introduced severe
artifacts on 3D schematic panels.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "pm_repair_round3"))

from geoseg.modules.segment_engines._shared import _create_overlay
from geoseg.modules.segment_engines.v4_kmeans import segment as v4_segment
from pm_repair import repair_pm_artifact

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


def repair_text_rois(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    matches: list[dict],
) -> np.ndarray:
    """Apply local repair to each OCR text ROI in sequence."""
    repaired = labels.copy()
    for match in matches:
        roi = match["roi"]
        try:
            result = repair_pm_artifact(
                panel_rgb,
                repaired,
                roi=roi,
                dark_threshold=55,
                median_diff=25,
                inpaint_radius=3,
            )
            repaired = result["labels"]
        except Exception as e:
            print(f"  WARNING: failed to repair ROI {roi} ({match['text']}): {e}")
    return repaired


def process_panel(
    image_path: Path,
    output_dir: Path,
    n_layers: int = 5,
) -> dict:
    """Run baseline segmentation and post-segmentation text-aware repair."""
    panel_id = image_path.stem
    out = output_dir / panel_id
    out.mkdir(parents=True, exist_ok=True)

    panel_rgb = np.array(Image.open(image_path).convert("RGB"))
    h, w = panel_rgb.shape[:2]

    # --- 1. OCR text detection ---
    matches = detect_text_rois_subprocess(image_path)

    # --- 2. Baseline segmentation on original ---
    baseline_result = v4_segment(panel_rgb, n_layers=n_layers)
    baseline_labels = baseline_result["labels"]
    baseline_overlay = baseline_result["overlay"]

    np.savez_compressed(out / "labels_baseline.npz", labels=baseline_labels)
    Image.fromarray(baseline_overlay).save(out / "01_baseline_overlay.jpg", quality=95)

    # --- 3. Post-segmentation repair of text ROIs ---
    repaired_labels = repair_text_rois(panel_rgb, baseline_labels, matches)

    repaired_overlay = _create_overlay(
        panel_rgb,
        repaired_labels,
        np.zeros((1, 3), dtype=np.uint8),
        alpha=0.65,
        boundary_mode="thin",
        skip_background=True,
        min_area_frac=0.001,
        fill_mode="blend",
    )

    np.savez_compressed(out / "labels_text_repaired.npz", labels=repaired_labels)
    Image.fromarray(repaired_overlay).save(out / "02_text_repaired_overlay.jpg", quality=95)

    # --- 4. Difference map ---
    diff_mask = baseline_labels != repaired_labels
    diff_vis = np.full((h, w, 3), 32, dtype=np.uint8)
    diff_vis[diff_mask] = [255, 0, 0]
    Image.fromarray(diff_vis).save(out / "03_label_difference.jpg", quality=95)

    diff_frac = float(diff_mask.sum() / diff_mask.size)

    summary = {
        "panel_id": panel_id,
        "image_path": str(image_path),
        "n_text_boxes": len(matches),
        "diff_fraction": round(diff_frac, 4),
        "outputs": {
            "baseline_overlay": str(out / "01_baseline_overlay.jpg"),
            "text_repaired_overlay": str(out / "02_text_repaired_overlay.jpg"),
            "label_difference": str(out / "03_label_difference.jpg"),
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
    output_dir = Path("/Users/daiduo2/geoseg/runs/pm_repair_ocr_experiment/text_aware_post_segment")
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
