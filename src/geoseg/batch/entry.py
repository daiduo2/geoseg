"""Processing for one batch session entry."""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from geoseg.batch.audit import run_visual_audit
from geoseg.controller import run_pipeline


def process_entry(
    entry,
    output_dir: Path,
    n_layers: int,
    quality_preference: str,
    skip_non_velocity_model: bool,
    use_vlm: bool,
    properties_map: dict[str, dict] | None,
) -> dict[str, Any]:
    """Run the full pipeline for a single figure entry."""
    img_path = Path(entry.source_path)
    t0 = time.perf_counter()

    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)

        img_out_dir = output_dir / entry.figure_id
        result = run_pipeline(
            arr,
            n_layers=n_layers,
            quality_preference=quality_preference,
            skip_non_velocity_model=skip_non_velocity_model,
            use_vlm=use_vlm,
            properties_map=properties_map,
            output_dir=img_out_dir,
            save_intermediates=True,
        )
        elapsed = time.perf_counter() - t0

        seg_record = None
        if result["status"] in ("ok", "empty") and result.get("panels"):
            panel = result["panels"][0] if result["panels"] else {}
            if panel.get("status") == "ok":
                from geoseg.session_state import SegmentationAttempt, SegmentationRecord

                panel_id = panel.get("panel_id", 0)
                panel_dir = img_out_dir / f"panel{panel_id}"
                seg_record = SegmentationRecord(
                    result_dir=str(panel_dir),
                    engine=panel.get("engines_used", "unknown"),
                    n_layers=panel.get("n_layers", n_layers),
                    quality_score=0.0,
                    overlay_path=str(panel_dir / "overlay.jpg"),
                    labels_path=str(panel_dir / "labels.npz"),
                    attempts=[
                        SegmentationAttempt(
                            engine=panel.get("engines_used", "unknown"),
                            n_layers=panel.get("n_layers", n_layers),
                            quality_score=0.0,
                        )
                    ],
                )

                try:
                    labels = np.load(seg_record.labels_path)["labels"]
                    bbox = panel.get("bbox", [0, 0, arr.shape[1], arr.shape[0]])
                    x, y, w, h = bbox
                    panel_rgb = arr[y : y + h, x : x + w]
                    panel3_mode = (
                        "panel3" in entry.figure_id.lower()
                        or "panel_3" in entry.figure_id.lower()
                    )
                    audit_dir = panel_dir / "visual_audit"
                    report = run_visual_audit(
                        labels,
                        panel_rgb,
                        audit_dir,
                        panel3_mode=panel3_mode,
                        labels_path=seg_record.labels_path,
                    )
                    seg_record.audit = report
                    if report.get("rejected"):
                        return {
                            "status": "audit_failed",
                            "reason": "; ".join(report.get("reasons", [])),
                            "classification": result["classification"],
                            "elapsed": time.perf_counter() - t0,
                            "seg_record": seg_record,
                        }
                except Exception as audit_exc:
                    return {
                        "status": "error",
                        "reason": f"visual_audit_failed: {audit_exc}",
                        "classification": result["classification"],
                        "elapsed": time.perf_counter() - t0,
                        "seg_record": seg_record,
                    }

        return {
            "status": result["status"],
            "reason": result.get("reason", ""),
            "classification": result["classification"],
            "elapsed": elapsed,
            "seg_record": seg_record,
        }

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "status": "error",
            "reason": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed": elapsed,
            "seg_record": None,
        }


__all__ = ["process_entry"]
