#!/usr/bin/env python3
"""Batch Agent review: run pipeline on all literature figures and save per-stage artifacts.

Usage:
    cd /Users/daiduo2/geoseg
    .venv/bin/python scripts/batch_agent_review.py

Outputs per figure:
    runs/agent_review/{paper}/{fig_name}/
        01_original.jpg          — raw extracted figure
        02_classification.json   — CV + optional VLM classification
        03_panels_detected.jpg   — original image with panel bboxes drawn
        04_segmentation_overlay.jpg — segmentation result overlay
        05_summary.json          — pipeline summary (status, layers, warnings)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.controller import run_pipeline
from geoseg.experiments import classify_figure
from geoseg.experiments import detect_panels

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

OUTPUT_ROOT = Path("runs/agent_review")
VLM_CACHE = Path("runs/agent_review/vlm_classification_results.json")

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

    # 4. Segmentation overlay + summary
    if pipeline_result and pipeline_result.get("status") == "ok":
        # overlay was saved by run_pipeline into a temp dir; copy it here
        # We re-run segment on the first panel to get overlay for visualization
        panel = pipeline_result["panels"][0]
        if panel.get("status") == "ok":
            # Find the overlay in the output directory (set by run_pipeline)
            # Since we didn't pass output_dir to run_pipeline in the fast path,
            # overlay is not saved. We'll handle this by passing output_dir.
            pass

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
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    total = 0
    processed = 0
    skipped = 0
    errors = 0

    # Load cached VLM results if available
    vlm_cache: dict[str, dict] = {}
    if VLM_CACHE.exists():
        cache_data = json.loads(VLM_CACHE.read_text(encoding="utf-8"))
        for record in cache_data:
            key = f"{record['paper']}/{record['fig_name']}"
            vlm_cache[key] = record
        print(f"Loaded {len(vlm_cache)} cached VLM classifications from {VLM_CACHE}")

    # Collect VLM review findings
    vlm_findings: list[dict] = []

    for paper, img_dir in PAPERS.items():
        img_path = Path(img_dir)
        if not img_path.exists():
            print(f"[SKIP] {paper}: directory not found: {img_dir}")
            continue

        images = sorted(img_path.glob("*.png")) + sorted(img_path.glob("*.jpg"))
        print(f"\n[{paper}] {len(images)} figures found")

        for img_file in images:
            fig_name = img_file.stem
            total += 1
            try:
                img_rgb = np.array(Image.open(img_file).convert("RGB"))

                # Step 1: VLM classify (use cache if available)
                cache_key = f"{paper}/{fig_name}"
                cached = vlm_cache.get(cache_key)
                if cached:
                    cls = {
                        "figure_type": cached["vlm_type"],
                        "confidence": cached["vlm_confidence"],
                        "reason": cached["vlm_reason"],
                    }
                else:
                    try:
                        vlm_result = classify_figure(img_rgb, mode="auto")
                        cls = {
                            "figure_type": vlm_result.figure_type,
                            "confidence": vlm_result.confidence,
                            "reason": vlm_result.reason,
                        }
                    except Exception as exc:
                        cls = {
                            "figure_type": "other",
                            "confidence": 0.0,
                            "reason": f"VLM error: {exc}",
                        }
                figure_type = cls.get("figure_type", "other")

                # Step 2: Detect panels
                panels = detect_panels(img_rgb)

                # Step 3: Run full pipeline if VLM says it's a velocity model or cross-section
                pipeline_result = None
                if figure_type in ("velocity_model", "geological_cross_section"):
                    fig_out = OUTPUT_ROOT / paper / fig_name
                    pipeline_result = run_pipeline(
                        img_rgb,
                        caption="",
                        n_layers=5,
                        skip_non_velocity_model=False,  # Already filtered by VLM above
                        use_vlm=True,   # Enable VLM review at every stage
                        output_dir=str(fig_out),
                        save_intermediates=True,
                    )
                    processed += 1
                else:
                    skipped += 1

                # Save artifacts
                save_figure_result(paper, fig_name, img_rgb, cls, panels, pipeline_result)

                status = pipeline_result.get("status", "skipped") if pipeline_result else "cv_skipped"
                print(f"  {fig_name}: type={figure_type}, panels={len(panels)}, status={status}")

                # Collect VLM review warnings for targets
                if pipeline_result and pipeline_result.get("status") == "ok":
                    summary = pipeline_result.get("summary", {})
                    warnings = summary.get("review_warnings", [])
                    if warnings:
                        vlm_findings.append({
                            "paper": paper,
                            "fig_name": fig_name,
                            "panels_detected": len(panels),
                            "n_panels_processed": summary.get("n_panels_processed", 0),
                            "warnings": warnings,
                            "engines_used": summary.get("engines_used", []),
                            "vlm_has_colorbar": summary.get("vlm_has_colorbar", False),
                            "vlm_target_panel_id": summary.get("vlm_target_panel_id", -1),
                        })

            except Exception as exc:
                errors += 1
                print(f"  {fig_name}: ERROR {exc}")

    print(f"\n{'='*50}")
    print(f"Total figures: {total}")
    print(f"Processed (non-observational): {processed}")
    print(f"Skipped (observational/other): {skipped}")
    print(f"Errors: {errors}")
    print(f"Output: {OUTPUT_ROOT}")

    # -----------------------------------------------------------------
    # VLM Review Report
    # -----------------------------------------------------------------
    if vlm_findings:
        print(f"\n{'='*50}")
        print("VLM REVIEW FINDINGS")
        print(f"{'='*50}")

        # Categorize findings
        panel_mismatches = []
        empty_segs = []
        under_segs = []
        other_warnings = []

        for f in vlm_findings:
            for w in f["warnings"]:
                if "panel_mismatch" in w:
                    panel_mismatches.append((f, w))
                elif "empty_segmentation" in w:
                    empty_segs.append((f, w))
                elif "under_segmented" in w:
                    under_segs.append((f, w))
                else:
                    other_warnings.append((f, w))

        if panel_mismatches:
            print(f"\n[Panel Mismatch] {len(panel_mismatches)} cases:")
            for f, w in panel_mismatches:
                print(f"  {f['paper']}/{f['fig_name']}: {w}")

        if empty_segs:
            print(f"\n[Empty Segmentation] {len(empty_segs)} cases:")
            for f, w in empty_segs:
                print(f"  {f['paper']}/{f['fig_name']}: {w}")

        if under_segs:
            print(f"\n[Under-segmentation] {len(under_segs)} cases:")
            for f, w in under_segs:
                print(f"  {f['paper']}/{f['fig_name']}: {w}")

        if other_warnings:
            print(f"\n[Other Warnings] {len(other_warnings)} cases:")
            for f, w in other_warnings:
                print(f"  {f['paper']}/{f['fig_name']}: {w}")

        # Save detailed report
        report_path = OUTPUT_ROOT / "VLM_REVIEW_REPORT.json"
        report_path.write_text(
            json.dumps({
                "total_targets_processed": processed,
                "targets_with_warnings": len(vlm_findings),
                "panel_mismatch_count": len(panel_mismatches),
                "empty_segmentation_count": len(empty_segs),
                "under_segmented_count": len(under_segs),
                "other_warnings_count": len(other_warnings),
                "findings": vlm_findings,
            }, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nDetailed report saved: {report_path}")
    else:
        print("\nNo VLM review warnings found.")


if __name__ == "__main__":
    main()
