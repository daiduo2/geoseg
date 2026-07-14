"""Figure classification stage helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from geoseg.modules.cv_detect.figure_classifier import classify


MIN_AUTO_SEGMENT_WIDTH = 300
MIN_AUTO_SEGMENT_HEIGHT = 200


def maybe_skip_tiny_image(
    img_rgb: np.ndarray,
    review_warnings: list[str],
) -> dict[str, Any] | None:
    """Return a skipped result for images too small for automatic segmentation."""
    h, w = img_rgb.shape[:2]
    if w >= MIN_AUTO_SEGMENT_WIDTH and h >= MIN_AUTO_SEGMENT_HEIGHT:
        return None
    return {
        "classification": classify(img_rgb),
        "panels": [],
        "summary": {
            "status": "skipped",
            "reason": (
                f"too_small_for_auto_segmentation "
                f"({w}x{h} < {MIN_AUTO_SEGMENT_WIDTH}x{MIN_AUTO_SEGMENT_HEIGHT})"
            ),
            "review_warnings": review_warnings,
        },
    }


def classify_figure_stage(
    img_rgb: np.ndarray,
    *,
    skip_non_velocity_model: bool = True,
    use_vlm: bool = True,
    review_warnings: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Classify a figure and optionally return a skipped pipeline result."""
    review_warnings = review_warnings if review_warnings is not None else []
    cls = classify(img_rgb)
    figure_type = cls["figure_type"]

    if skip_non_velocity_model and use_vlm:
        try:
            from geoseg.modules.vlm_client import classify_figure

            vlm_cls = classify_figure(img_rgb, mode="auto", min_confidence=0.7)
            vlm_type = vlm_cls.figure_type
            vlm_rec = getattr(vlm_cls, "segmentation_recommendation", None)

            should_skip = False
            skip_reason = ""
            if vlm_rec == "skip":
                should_skip = True
                skip_reason = f"vlm_recommended_skip: {vlm_type}"
            elif vlm_rec == "manual_review":
                should_skip = True
                skip_reason = f"vlm_recommended_manual_review: {vlm_type}"
            elif vlm_rec is None and vlm_type not in (
                "velocity_model",
                "geological_cross_section",
            ):
                should_skip = True
                skip_reason = f"vlm_not_target_type: {vlm_type}"

            vlm_payload = {
                "vlm_classification": vlm_type,
                "vlm_confidence": vlm_cls.confidence,
                "vlm_reason": vlm_cls.reason,
                "vlm_segmentation_recommendation": vlm_rec,
                "vlm_visual_features": getattr(vlm_cls, "visual_features", None),
                "vlm_primary_evidence": getattr(vlm_cls, "primary_evidence", None),
                "vlm_conflicting_evidence": getattr(
                    vlm_cls, "conflicting_evidence", None
                ),
            }

            if should_skip:
                return {
                    **cls,
                    **vlm_payload,
                }, {
                    "classification": {
                        **cls,
                        **vlm_payload,
                    },
                    "panels": [],
                    "summary": {
                        "status": "skipped",
                        "reason": skip_reason,
                        "review_warnings": review_warnings,
                    },
                }

            return {
                **cls,
                "figure_type": "conceptual_model",
                **vlm_payload,
            }, None
        except Exception as exc:
            if figure_type in ("observational_data", "other"):
                return {**cls, "vlm_error": str(exc)}, {
                    "classification": {**cls, "vlm_error": str(exc)},
                    "panels": [],
                    "summary": {
                        "status": "skipped",
                        "reason": f"figure_type={figure_type}",
                        "review_warnings": review_warnings,
                    },
                }
            return {**cls, "vlm_error": str(exc)}, None

    if skip_non_velocity_model and figure_type in ("observational_data", "other"):
        return cls, {
            "classification": cls,
            "panels": [],
            "summary": {
                "status": "skipped",
                "reason": f"figure_type={figure_type}",
                "review_warnings": review_warnings,
            },
        }

    return cls, None


__all__ = [
    "MIN_AUTO_SEGMENT_HEIGHT",
    "MIN_AUTO_SEGMENT_WIDTH",
    "classify_figure_stage",
    "maybe_skip_tiny_image",
]
