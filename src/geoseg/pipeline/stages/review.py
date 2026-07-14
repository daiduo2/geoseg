"""Figure review and target-panel resolution stages."""

from __future__ import annotations

from typing import Any

import numpy as np

from geoseg.core.models import make_whole_image_panel
from geoseg.pipeline.stages.detect import panel_complexity_score


def review_figure_stage(
    img_rgb: np.ndarray,
    *,
    caption: str = "",
    text_blocks: list[dict] | None = None,
    panel_bboxes: list[dict[str, Any]],
    target_panel_id: int = -1,
    use_vlm: bool = True,
    review_warnings: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict], bool, int]:
    """Run figure-level VLM review and resolve shared segmentation hints."""
    review_warnings = review_warnings if review_warnings is not None else []
    color_zones: list[dict] = []
    has_colorbar_hint = False
    resolved_target_panel_id = target_panel_id

    if use_vlm:
        try:
            from geoseg.modules.vlm_client import review_page_overview

            text_payload = text_blocks or []
            if caption and not any(tb.get("text") == caption for tb in text_payload):
                text_payload = text_payload + [
                    {"type": "caption", "text": caption, "bbox": []}
                ]
            overview = review_page_overview(
                img_rgb,
                text_blocks=text_payload,
                page_idx=0,
                mode="auto",
                min_confidence=0.7,
            )

            n_vlm_panels = len(overview.panels) if hasattr(overview, "panels") else 0
            n_cv_panels = len(panel_bboxes)
            if n_vlm_panels > 0 and n_vlm_panels != n_cv_panels:
                review_warnings.append(
                    f"panel_mismatch: vlm_sees_{n_vlm_panels}_panels "
                    f"cv_detects_{n_cv_panels}_panels"
                )
                if n_vlm_panels > n_cv_panels:
                    review_warnings.append("fallback_whole_image_due_to_missed_panels")
                    panel_bboxes = [make_whole_image_panel(img_rgb)]
                    if resolved_target_panel_id >= 0:
                        panel_bboxes[0]["id"] = resolved_target_panel_id

            if hasattr(overview, "color_zones"):
                color_zones = [cz.model_dump() for cz in overview.color_zones]
            if hasattr(overview, "has_colorbar"):
                has_colorbar_hint = overview.has_colorbar
            if hasattr(overview, "target_panel_id"):
                resolved_target_panel_id = overview.target_panel_id

        except Exception as exc:
            review_warnings.append(f"review_failed: {exc}")

    return panel_bboxes, color_zones, has_colorbar_hint, resolved_target_panel_id


def resolve_target_panel_stage(
    img_rgb: np.ndarray,
    panel_bboxes: list[dict[str, Any]],
    target_panel_id: int,
    review_warnings: list[str],
) -> int:
    """Resolve stale or mismatched target_panel_id values."""
    if target_panel_id < 0:
        return target_panel_id

    matching = [pb for pb in panel_bboxes if pb["id"] == target_panel_id]
    if matching or not panel_bboxes:
        return target_panel_id

    best = max(
        panel_bboxes,
        key=lambda pb: panel_complexity_score(
            img_rgb[
                pb["bbox"][1] : pb["bbox"][1] + pb["bbox"][3],
                pb["bbox"][0] : pb["bbox"][0] + pb["bbox"][2],
            ]
        ),
    )
    review_warnings.append(
        f"target_panel_fallback: requested_{target_panel_id}_not_found, "
        f"using_panel_{best['id']}"
    )
    return best["id"]


__all__ = ["resolve_target_panel_stage", "review_figure_stage"]
