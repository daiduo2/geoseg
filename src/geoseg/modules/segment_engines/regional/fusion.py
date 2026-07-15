"""Regional fusion orchestration."""

from __future__ import annotations

import numpy as np

from geoseg.modules.post_process.split import split_label_by_color_components
from geoseg.modules.segment_engines.regional.models import FusionConfig, RegionalAudit
from geoseg.modules.segment_engines.regional.overlay import generate_overlay_with_legend
from geoseg.modules.segment_engines.regional.split_merge import fuse_with_freeze
from geoseg.modules.segment_engines.regions import reorder_labels_top_to_bottom


def _run_engine_by_name(
    engine: str,
    panel_rgb: np.ndarray,
    n_layers: int,
    reps: list[dict] | None = None,
    colorbar_rgb: np.ndarray | None = None,
) -> dict:
    """Run a segmentation engine by name."""
    from geoseg.modules.segment_engines.runner import run_engine

    return run_engine(engine, panel_rgb, reps, colorbar_rgb, n_layers)


def _apply_local_fixes(
    labels: np.ndarray,
    panel_rgb: np.ndarray,
    local_fixes: list[dict],
) -> np.ndarray:
    fixed = labels
    if not local_fixes:
        return fixed

    from geoseg.modules.post_process.merge import merge_labels_by_ids

    for fix in local_fixes:
        action = fix.get("action")
        if action == "merge_labels":
            fixed = merge_labels_by_ids(
                fixed,
                fix["label_ids"],
                target_id=fix.get("target_id", fix["label_ids"][0]),
            )
        elif action == "split_label_by_color_components":
            fixed = split_label_by_color_components(
                fixed,
                panel_rgb,
                target_label=fix["label_id"],
                color_space=fix.get("color_space", "LAB"),
                k=fix.get("k", 3),
                min_component_area=fix.get("min_component_area", 300),
            )
    return fixed


def regional_segment(
    panel_rgb: np.ndarray,
    n_layers: int,
    *,
    reps: list[dict] | None = None,
    colorbar_rgb: np.ndarray | None = None,
    primary_result: dict | None = None,
    audit: RegionalAudit | None = None,
    config: FusionConfig | None = None,
) -> dict:
    """Region-level multi-engine segmentation with freeze fusion.

    Experimental. If primary_result and audit are provided and retry_labels
    is non-empty, performs freeze+patch fusion with a secondary engine.
    Otherwise returns primary engine result only.
    """
    config = config or FusionConfig()

    no_audit = audit is None or not audit.retry_labels
    if primary_result is None or no_audit:
        result = _run_engine_by_name(
            config.primary_engine, panel_rgb, n_layers, reps, colorbar_rgb
        )
        overlay = (
            generate_overlay_with_legend(panel_rgb, result["labels"])
            if config.enable_legend
            else result.get("overlay")
        )
        return {
            "labels": result["labels"],
            "overlay": overlay,
            "meta": {
                "engine": config.primary_engine,
                "path": "primary_only",
                "fusion_applied": False,
            },
        }

    labels_a = primary_result["labels"]
    labels_a = reorder_labels_top_to_bottom(labels_a)
    labels_a = _apply_local_fixes(labels_a, panel_rgb, audit.local_fixes)

    freeze_mask = np.zeros(labels_a.shape, dtype=bool)
    for lbl in audit.frozen_labels:
        freeze_mask |= labels_a == lbl

    secondary = (
        audit.secondary_engine
        if audit and audit.secondary_engine
        else (config.secondary_engines[0] if config.secondary_engines else "v4_kmeans")
    )
    result_b = _run_engine_by_name(secondary, panel_rgb, n_layers, reps, colorbar_rgb)
    labels_b = reorder_labels_top_to_bottom(result_b["labels"])

    fused = fuse_with_freeze(
        labels_a,
        labels_b,
        freeze_mask,
        seam_width=config.seam_smooth_width,
    )

    overlay = generate_overlay_with_legend(panel_rgb, fused) if config.enable_legend else None

    return {
        "labels": fused,
        "overlay": overlay,
        "meta": {
            "engine": f"{config.primary_engine}+{secondary}",
            "path": "regional_fusion",
            "fusion_applied": True,
            "primary_engine": config.primary_engine,
            "secondary_engine": secondary,
            "repair_strategy": audit.repair_strategy if audit else "regional_fusion",
            "frozen_labels": audit.frozen_labels,
            "retry_labels": audit.retry_labels,
            "local_fixes": audit.local_fixes,
            "fusion_notes": audit.notes,
            "iteration": audit.iteration,
        },
    }


__all__ = ["regional_segment"]
