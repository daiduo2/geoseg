"""Stage helpers for the legacy full segmentation pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from geoseg.core.models import make_whole_image_panel
from geoseg.modules.cv_detect.colorbar_extractor import extract_colorbar, extract_colorbar_bbox
from geoseg.modules.cv_detect.figure_classifier import classify
from geoseg.modules.cv_detect.panel_detector import detect_panels
from geoseg.modules.segment_engines.internal.shared import saturation_ratio
from geoseg.modules.segment_engines.router import route_and_segment
from geoseg.modules.segment_engines.vlm_reps import color_zones_to_reps


MIN_AUTO_SEGMENT_WIDTH = 300
MIN_AUTO_SEGMENT_HEIGHT = 200


def panel_complexity_score(panel_rgb: np.ndarray) -> float:
    """Score a panel by structural complexity to avoid simple gradients."""
    from skimage.color import rgb2gray
    from skimage.filters import sobel

    gray = rgb2gray(panel_rgb)
    h, w = gray.shape
    if h < 10 or w < 10:
        return 0.0

    edges = sobel(gray)
    edge_dens = float((np.abs(edges) > 0.05).mean())

    gy, gx = np.gradient(gray.astype(np.float64))
    mag = np.sqrt(gx**2 + gy**2)
    mean_mag = mag.mean()
    if mean_mag < 1e-9:
        grad_uniformity = 1.0
    else:
        grad_uniformity = float(np.clip(mag.std() / mean_mag / 3.0, 0.0, 1.0))

    sat = saturation_ratio(panel_rgb)
    return edge_dens * 0.5 + sat * 0.3 + (1.0 - grad_uniformity) * 0.2


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


def detect_panels_stage(img_rgb: np.ndarray) -> list[dict[str, Any]]:
    """Detect panels, falling back to a whole-image panel."""
    panel_bboxes = detect_panels(img_rgb)
    if not panel_bboxes:
        panel_bboxes = [make_whole_image_panel(img_rgb)]
    return panel_bboxes


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


def crop_panel_for_segmentation(
    img_rgb: np.ndarray,
    panel_bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray | None, list[int] | None, str | None]:
    """Crop a panel and remove edge colorbar regions when detected."""
    h, w = img_rgb.shape[:2]
    x, y, pw, ph = panel_bbox
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    pw = min(pw, w - x)
    ph = min(ph, h - y)
    if pw < 20 or ph < 20:
        return None, None, "too_small"

    panel_img = img_rgb[y : y + ph, x : x + pw]

    colorbar_result = extract_colorbar_bbox(panel_img)
    if colorbar_result is not None:
        cx, cy, cw, ch, orient = colorbar_result
        orig_pw, orig_ph = pw, ph
        if orient == "vertical":
            if cx < pw * 0.15:
                panel_img = panel_img[:, cw:]
                x += cw
                pw -= cw
            elif cx + cw > pw * 0.85:
                panel_img = panel_img[:, :cx]
                pw = cx
        elif orient == "horizontal":
            if cy < ph * 0.15:
                panel_img = panel_img[ch:, :]
                y += ch
                ph -= ch
            elif cy + ch > ph * 0.85:
                panel_img = panel_img[:cy, :]
                ph = cy
        if pw < 30 or ph < 30:
            return (
                None,
                None,
                f"too_small_after_colorbar_crop: {orig_pw}x{orig_ph} -> {pw}x{ph}",
            )

    return panel_img, [x, y, pw, ph], None


def segment_panel_stage(
    img_rgb: np.ndarray,
    panel: dict[str, Any],
    *,
    panel_bboxes: list[dict[str, Any]],
    target_panel_id: int,
    color_zones: list[dict],
    n_layers: int,
    quality_preference: str,
    review_warnings: list[str],
) -> tuple[dict[str, Any] | None, int, str | None]:
    """Segment one panel and return panel result, layer count, and engine name."""
    is_target = (target_panel_id < 0) or (panel["id"] == target_panel_id)
    if not is_target and len(panel_bboxes) == 1:
        is_target = True
        review_warnings.append(
            f"panel_{panel['id']}_forced_target: "
            f"only_panel_with_mismatched_target_id={target_panel_id}"
        )
    if not is_target:
        x, y, pw, ph = panel["bbox"]
        review_warnings.append(
            f"panel_{panel['id']}_skipped_non_target: target_id={target_panel_id}"
        )
        return {
            "panel_id": panel["id"],
            "bbox": [x, y, pw, ph],
            "classification": classify(img_rgb[y : y + ph, x : x + pw]),
            "segmentation": None,
            "review": {
                "n_layers_found": 0,
                "is_target_panel": False,
                "skipped_reason": "non_target_panel",
            },
        }, 0, None

    panel_img, bbox, crop_reason = crop_panel_for_segmentation(img_rgb, panel["bbox"])
    if panel_img is None or bbox is None:
        if crop_reason and crop_reason.startswith("too_small_after_colorbar_crop"):
            review_warnings.append(f"panel_{panel['id']}_{crop_reason}")
        return None, 0, None

    x, y, pw, ph = bbox
    panel_cls = classify(panel_img)

    colorbar_rgb = extract_colorbar(img_rgb) if len(panel_bboxes) == 1 else extract_colorbar(panel_img)

    reps = None
    sat = saturation_ratio(panel_img)
    if sat >= 0.5:
        reps = color_zones_to_reps(
            panel_img,
            color_zones,
            colorbar_rgb=colorbar_rgb,
            n_layers=n_layers,
        )

    n_color_zones = len(color_zones) if color_zones else 0
    seg = route_and_segment(
        panel_img,
        reps=reps,
        colorbar_rgb=colorbar_rgb,
        n_layers=n_layers,
        quality_preference=quality_preference,
        is_velocity_model=True,
        n_color_zones=n_color_zones,
    )

    labels = seg["labels"]
    unique_labels = set(labels.flatten())
    n_layers_found = len(unique_labels - {0})

    if n_layers_found == 0:
        review_warnings.append(f"panel_{panel['id']}_empty_segmentation: no_layers_found")
    elif n_layers_found < 2:
        if seg["meta"].get("retry_from"):
            review_warnings.append(
                f"panel_{panel['id']}_under_segmented: "
                f"only_{n_layers_found}_layer(s)_after_retry"
            )
        else:
            review_warnings.append(
                f"panel_{panel['id']}_under_segmented: only_{n_layers_found}_layer(s)"
            )
    elif seg["meta"].get("retry_from"):
        review_warnings.append(
            f"panel_{panel['id']}_retry_fixed: "
            f"{seg['meta']['retry_from']}_to_{seg['meta']['engine']}_"
            f"now_{n_layers_found}_layers"
        )

    return {
        "panel_id": panel["id"],
        "bbox": [x, y, pw, ph],
        "classification": panel_cls,
        "segmentation": seg,
        "review": {
            "n_layers_found": n_layers_found,
            "is_target_panel": panel["id"] == target_panel_id,
        },
    }, n_layers_found, seg["meta"]["engine"]


def summarize_pipeline_result(
    img_rgb: np.ndarray,
    *,
    classification: dict[str, Any],
    panel_results: list[dict[str, Any]],
    total_layers: int,
    engines_used: set[str],
    review_warnings: list[str],
    has_colorbar_hint: bool,
    target_panel_id: int,
) -> dict[str, Any]:
    """Build the legacy process_figure result payload."""
    return {
        "classification": classification,
        "panels": panel_results,
        "summary": {
            "status": "ok",
            "n_panels": len(panel_results),
            "total_layers": total_layers,
            "engines_used": sorted(engines_used),
            "saturation_ratio": round(saturation_ratio(img_rgb), 4),
            "review_warnings": review_warnings,
            "vlm_has_colorbar": has_colorbar_hint,
            "vlm_target_panel_id": target_panel_id,
        },
    }


__all__ = [
    "classify_figure_stage",
    "crop_panel_for_segmentation",
    "detect_panels_stage",
    "maybe_skip_tiny_image",
    "panel_complexity_score",
    "resolve_target_panel_stage",
    "review_figure_stage",
    "segment_panel_stage",
    "summarize_pipeline_result",
]
