#!/usr/bin/env python3
"""Batch Agent review with VLM classifier: run pipeline on VLM-identified targets only.

Usage:
    cd /Users/daiduo2/geoseg
    .venv/bin/python scripts/batch_agent_review_vlm.py

Prerequisite: scripts/batch_vlm_classify.py must have been run to produce
    runs/agent_review/vlm_classification_results.json

Outputs per figure (targets only):
    runs/agent_review_vlm/{paper}/{fig_name}/
        01_original.jpg          — raw extracted figure
        02_classification.json   — VLM classification result
        03_panels_detected.jpg   — original image with panel bboxes drawn
        04_segmentation_overlay.jpg — segmentation result overlay
        05_summary.json          — pipeline summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.controller import run_pipeline
from geoseg.modules.cv_detect.panel_detector import detect_panels

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PAPERS: dict[str, str] = {
    "ph01": "runs/M0.5/images",
    "gras2019": "runs/literature_test/gras2019/mineru/extracted/images",
    "ma_2022": "runs/literature_test/ma_2022/mineru/extracted/images",
    "zailac2023": "runs/literature_test/zailac2023/mineru/extracted/images",
    "silixa2021": "runs/literature_test/silixa2021/mineru/extracted/images",
}

OUTPUT_ROOT = Path("runs/agent_review_vlm")
VLM_RESULTS = Path("runs/agent_review/vlm_classification_results.json")
TARGET_TYPES = {"velocity_model", "geological_cross_section"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def draw_panel_bboxes(img_rgb: np.ndarray, bboxes: list[dict]) -> Image.Image:
    """Draw panel bounding boxes on the original image."""
    img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img)
    colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]
    for i, pb in enumerate(bboxes):
        x, y, w, h = pb["bbox"]
        color = colors[i % len(colors)]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        draw.text((x + 4, y + 4), f"P{i}", fill=color)
    return img


def save_figure_result(
    paper: str,
    fig_name: str,
    img_rgb: np.ndarray,
    cls: dict,
    panels: list[dict],
    pipeline_result: dict | None,
) -> Path:
    """Save all per-stage artifacts for a single figure."""
    out_dir = OUTPUT_ROOT / paper / fig_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Original
    Image.fromarray(img_rgb).save(out_dir / "01_original.jpg", quality=90)

    # 2. Classification
    (out_dir / "02_classification.json").write_text(
        json.dumps(cls, indent=2, default=str), encoding="utf-8"
    )

    # 3. Panel detect visualization
    panels_img = draw_panel_bboxes(img_rgb, panels)
    panels_img.save(out_dir / "03_panels_detected.jpg", quality=90)

    # 4. Segmentation overlay (for the first ok panel)
    if pipeline_result and pipeline_result.get("status") == "ok":
        panel = pipeline_result["panels"][0]
        if panel.get("status") == "ok" and "overlay_path" in panel:
            overlay_src = Path(panel["overlay_path"])
            if overlay_src.exists():
                import shutil
                shutil.copy(overlay_src, out_dir / "04_segmentation_overlay.jpg")

    # 5. Summary
    summary = {
        "figure_name": fig_name,
        "classification": cls,
        "n_panels_detected": len(panels),
        "pipeline_status": pipeline_result.get("status") if pipeline_result else "not_run",
    }
    if pipeline_result:
        summary["pipeline_summary"] = pipeline_result.get("summary", {})
        summary["n_panels_processed"] = pipeline_result.get("summary", {}).get(
            "n_panels_processed", 0
        )
    (out_dir / "05_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    return out_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not VLM_RESULTS.exists():
        print(f"VLM results not found: {VLM_RESULTS}")
        print("Run: .venv/bin/python scripts/batch_vlm_classify.py")
        sys.exit(1)

    vlm_data = json.loads(VLM_RESULTS.read_text(encoding="utf-8"))
    targets = [r for r in vlm_data if r.get("vlm_type") in TARGET_TYPES]
    print(f"Total VLM results: {len(vlm_data)}")
    print(f"Targets to process: {len(targets)}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    total = 0
    processed = 0
    errors = 0

    for record in targets:
        paper = record["paper"]
        fig_name = record["fig_name"]
        img_dir = PAPERS.get(paper)
        if not img_dir:
            continue

        # Find image file
        img_path = None
        for ext in (".png", ".jpg"):
            candidate = Path(img_dir) / f"{fig_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if not img_path:
            print(f"[SKIP] {paper}/{fig_name}: image not found")
            continue

        total += 1
        try:
            img_rgb = np.array(Image.open(img_path).convert("RGB"))

            # VLM classification (from pre-computed results)
            cls = {
                "figure_type": record["vlm_type"],
                "confidence": record["vlm_confidence"],
                "reason": record["vlm_reason"],
            }

            # Detect panels
            panels = detect_panels(img_rgb)

            # Run full pipeline
            fig_out = OUTPUT_ROOT / paper / fig_name
            pipeline_result = run_pipeline(
                img_rgb,
                caption="",
                n_layers=5,
                skip_non_velocity_model=False,
                use_vlm=False,
                output_dir=str(fig_out),
                save_intermediates=True,
            )
            processed += 1

            # Save artifacts
            save_figure_result(paper, fig_name, img_rgb, cls, panels, pipeline_result)

            status = pipeline_result.get("status", "unknown")
            print(f"  {paper}/{fig_name}: panels={len(panels)}, status={status}")

        except Exception as exc:
            errors += 1
            print(f"  {paper}/{fig_name}: ERROR {exc}")

    print(f"\n{'='*50}")
    print(f"Total targets: {total}")
    print(f"Processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
