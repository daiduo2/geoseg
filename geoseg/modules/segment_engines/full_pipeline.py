"""Full segmentation pipeline: classify -> review -> detect panels -> segment.

NOTE: This module is now a LOW-LEVEL TOOL LIBRARY for agent-driven segmentation.
The high-level orchestration has moved to `.claude/skills/geo-segment` and
`.claude/skills/sandbox-segment`, where agents autonomously decide strategy.

`process_figure()` remains available for backward-compatible script usage,
but new agent-native workflows should use skills instead.

VLM calls via `vlm_client/client.py` are DEPRECATED (subprocess `claude -p`).
Semantic reasoning now happens inside Claude Code agent sessions.

Usage (legacy):
    from geoseg.modules.segment_engines.full_pipeline import process_figure
    result = process_figure(img_rgb, caption="Figure 2. Vs model", n_layers=5)
"""

from __future__ import annotations

import numpy as np

from geoseg.modules.cv_detect.figure_classifier import classify
from geoseg.modules.cv_detect.panel_detector import detect_panels
from geoseg.modules.cv_detect.colorbar_extractor import extract_colorbar, extract_colorbar_bbox
from geoseg.modules.segment_engines.router import route_and_segment
from geoseg.modules.segment_engines._shared import saturation_ratio
from geoseg.modules.segment_engines.vlm_reps import color_zones_to_reps


def _panel_complexity_score(panel_rgb: np.ndarray) -> float:
    """Score a panel by structural complexity to avoid picking "Initial model" gradients.

    Simple linear gradients (Initial models) have very low edge density and
    near-uniform gradient magnitude distribution. Real velocity models have
    higher edge density and more non-uniform gradients (edges concentrated at
    layer boundaries).
    """
    from skimage.color import rgb2gray
    from skimage.filters import sobel

    gray = rgb2gray(panel_rgb)
    h, w = gray.shape
    if h < 10 or w < 10:
        return 0.0

    # Edge density
    edges = sobel(gray)
    edge_dens = float((np.abs(edges) > 0.05).mean())

    # Gradient uniformity: high = smooth gradient (bad), low = complex structure (good)
    gy, gx = np.gradient(gray.astype(np.float64))
    mag = np.sqrt(gx**2 + gy**2)
    mean_mag = mag.mean()
    if mean_mag < 1e-9:
        grad_uniformity = 1.0
    else:
        grad_uniformity = float(np.clip(mag.std() / mean_mag / 3.0, 0.0, 1.0))

    # Color saturation: velocity models usually have vivid colors
    from geoseg.modules.segment_engines._shared import saturation_ratio
    sat = saturation_ratio(panel_rgb)

    # Score: reward edges and saturation, penalize uniform gradients
    return edge_dens * 0.5 + sat * 0.3 + (1.0 - grad_uniformity) * 0.2


def process_figure(
    img_rgb: np.ndarray,
    caption: str = "",
    text_blocks: list[dict] | None = None,
    n_layers: int = 5,
    quality_preference: str = "balanced",
    skip_non_velocity_model: bool = True,
    use_vlm: bool = True,
    target_panel_id: int = -1,
) -> dict:
    """Process a raw extracted figure image through the full pipeline.

    Steps:
    1. Classify figure type (CV heuristic + VLM semantic filter)
    2. Detect panels with CV
    3. VLM figure-level review: validate panel count, get color_zones
    4. Segment each panel (with fallback / retry on poor results)
    5. Return combined results with review audit trail

    Args:
        img_rgb: RGB uint8 array of the extracted figure.
        caption: Optional figure caption / text from PDF extraction.
        n_layers: Number of layers to extract per panel.
        quality_preference: "fast", "balanced", or "best".
        skip_non_velocity_model: If True, skip observational_data and other types.
        use_vlm: If False, skip VLM calls and use colorbar fallback only.
        target_panel_id: If >= 0, only segment this panel ID (overrides VLM target).

    Returns:
        dict with keys:
            classification: figure_classifier result
            panels: list of panel results
            summary: dict with aggregate stats + review_warnings
    """
    h, w = img_rgb.shape[:2]
    review_warnings: list[str] = []

    # ── Size gate: skip tiny inline images ──────────────────────────
    MIN_W, MIN_H = 300, 200
    if w < MIN_W or h < MIN_H:
        return {
            "classification": classify(img_rgb),
            "panels": [],
            "summary": {
                "status": "skipped",
                "reason": f"too_small_for_auto_segmentation ({w}x{h} < {MIN_W}x{MIN_H})",
                "review_warnings": review_warnings,
            },
        }

    # ── Step 1: Figure classification ───────────────────────────────
    cls = classify(img_rgb)
    figure_type = cls["figure_type"]

    if skip_non_velocity_model and use_vlm:
        try:
            from geoseg.modules.vlm_client import classify_figure
            vlm_cls = classify_figure(img_rgb, mode="auto", min_confidence=0.7)
            vlm_type = vlm_cls.figure_type
            vlm_rec = getattr(vlm_cls, "segmentation_recommendation", None)

            # Primary routing: use explicit VLM recommendation if available
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
                # Fallback to figure_type check for backward compatibility
                should_skip = True
                skip_reason = f"vlm_rejected: {vlm_type}"

            if should_skip:
                return {
                    "classification": {
                        **cls,
                        "vlm_classification": vlm_type,
                        "vlm_confidence": vlm_cls.confidence,
                        "vlm_reason": vlm_cls.reason,
                        "vlm_segmentation_recommendation": vlm_rec,
                        "vlm_visual_features": getattr(
                            vlm_cls, "visual_features", None
                        ),
                        "vlm_primary_evidence": getattr(
                            vlm_cls, "primary_evidence", None
                        ),
                        "vlm_conflicting_evidence": getattr(
                            vlm_cls, "conflicting_evidence", None
                        ),
                    },
                    "panels": [],
                    "summary": {
                        "status": "skipped",
                        "reason": skip_reason,
                        "review_warnings": review_warnings,
                    },
                }
            cls = {
                **cls,
                "figure_type": "conceptual_model",
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
        except Exception as exc:
            if figure_type in ("observational_data", "other"):
                return {
                    "classification": {**cls, "vlm_error": str(exc)},
                    "panels": [],
                    "summary": {
                        "status": "skipped",
                        "reason": f"figure_type={figure_type}",
                        "review_warnings": review_warnings,
                    },
                }
            cls = {**cls, "vlm_error": str(exc)}
    elif skip_non_velocity_model and figure_type in ("observational_data", "other"):
        return {
            "classification": cls,
            "panels": [],
            "summary": {
                "status": "skipped",
                "reason": f"figure_type={figure_type}",
                "review_warnings": review_warnings,
            },
        }

    # ── Step 2: Detect panels ───────────────────────────────────────
    panel_bboxes = detect_panels(img_rgb)

    if not panel_bboxes:
        from geoseg.pipeline_interfaces import make_whole_image_panel
        panel_bboxes = [make_whole_image_panel(img_rgb)]

    # ── Step 3: VLM figure-level review ─────────────────────────────
    overview = None
    color_zones: list[dict] = []
    has_colorbar_hint = False
    # target_panel_id comes from the caller parameter; VLM may override it.
    _target_panel_id = target_panel_id

    if use_vlm:
        try:
            from geoseg.modules.vlm_client import review_page_overview
            _tb = text_blocks or []
            if caption and not any(tb.get("text") == caption for tb in _tb):
                _tb = _tb + [{"type": "caption", "text": caption, "bbox": []}]
            overview = review_page_overview(
                img_rgb,
                text_blocks=_tb,
                page_idx=0,
                mode="auto",
                min_confidence=0.7,
            )

            # Panel count validation: VLM panels vs CV detected panels
            n_vlm_panels = len(overview.panels) if hasattr(overview, "panels") else 0
            n_cv_panels = len(panel_bboxes)
            if n_vlm_panels > 0 and n_vlm_panels != n_cv_panels:
                review_warnings.append(
                    f"panel_mismatch: vlm_sees_{n_vlm_panels}_panels "
                    f"cv_detects_{n_cv_panels}_panels"
                )
                # If VLM sees more panels than CV, CV likely missed gaps
                # (colorbar/annotation overlap can hide panel boundaries).
                # Fallback to whole-image mode for manual panel selection.
                if n_vlm_panels > n_cv_panels:
                    review_warnings.append("fallback_whole_image_due_to_missed_panels")
                    from geoseg.pipeline_interfaces import make_whole_image_panel
                    panel_bboxes = [make_whole_image_panel(img_rgb)]
                    # Ensure the whole-image panel matches VLM's target_panel_id
                    # so the target panel filter doesn't skip it.
                    if _target_panel_id >= 0:
                        panel_bboxes[0]["id"] = _target_panel_id

            # Extract shared hints from overview
            if hasattr(overview, "color_zones"):
                color_zones = [cz.model_dump() for cz in overview.color_zones]
            if hasattr(overview, "has_colorbar"):
                has_colorbar_hint = overview.has_colorbar
            if hasattr(overview, "target_panel_id"):
                _target_panel_id = overview.target_panel_id

        except Exception as exc:
            review_warnings.append(f"review_failed: {exc}")

    # ── Step 3b: Resolve stale/mismatched target_panel_id ───────────
    # If the requested target_panel_id does not match any CV-detected panel,
    # fall back to the most structurally complex panel (avoid "Initial model"
    # gradients that have no useful layer structure).
    if _target_panel_id >= 0:
        _matching = [pb for pb in panel_bboxes if pb["id"] == _target_panel_id]
        if not _matching and len(panel_bboxes) > 0:
            _best = max(
                panel_bboxes,
                key=lambda pb: _panel_complexity_score(
                    img_rgb[
                        pb["bbox"][1] : pb["bbox"][1] + pb["bbox"][3],
                        pb["bbox"][0] : pb["bbox"][0] + pb["bbox"][2],
                    ]
                ),
            )
            review_warnings.append(
                f"target_panel_fallback: requested_{_target_panel_id}_not_found, "
                f"using_panel_{_best['id']}"
            )
            _target_panel_id = _best["id"]

    # ── Step 4: Segment each panel ──────────────────────────────────
    panel_results = []
    total_layers = 0
    engines_used = set()

    for pb in panel_bboxes:
        x, y, pw, ph = pb["bbox"]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        pw = min(pw, w - x)
        ph = min(ph, h - y)
        if pw < 20 or ph < 20:
            continue

        # ── Target panel filter ───────────────────────────────────────
        # If VLM identified a specific target panel, skip non-target panels.
        # Exception: if there's only one panel and it doesn't match the target_id,
        # treat it as the target (VLM may have miscounted panels).
        is_target = (_target_panel_id < 0) or (pb["id"] == _target_panel_id)
        if not is_target and len(panel_bboxes) == 1:
            is_target = True
            review_warnings.append(
                f"panel_{pb['id']}_forced_target: only_panel_with_mismatched_target_id={_target_panel_id}"
            )
        if not is_target:
            panel_results.append({
                "panel_id": pb["id"],
                "bbox": [x, y, pw, ph],
                "classification": classify(img_rgb[y : y + ph, x : x + pw]),
                "segmentation": None,
                "review": {
                    "n_layers_found": 0,
                    "is_target_panel": False,
                    "skipped_reason": "non_target_panel",
                },
            })
            review_warnings.append(
                f"panel_{pb['id']}_skipped_non_target: target_id={_target_panel_id}"
            )
            continue

        panel_img = img_rgb[y : y + ph, x : x + pw]

        # ── Colorbar spatial cropping ──────────────────────────────────
        # Exclude colorbar region from the panel so it is not segmented as a layer.
        _cb_result = extract_colorbar_bbox(panel_img)
        if _cb_result is not None:
            _cx, _cy, _cw, _ch, _c_orient = _cb_result
            _orig_pw, _orig_ph = pw, ph
            if _c_orient == "vertical":
                if _cx < pw * 0.15:
                    panel_img = panel_img[:, _cw:]
                    x += _cw
                    pw -= _cw
                elif _cx + _cw > pw * 0.85:
                    panel_img = panel_img[:, :_cx]
                    pw = _cx
            elif _c_orient == "horizontal":
                if _cy < ph * 0.15:
                    panel_img = panel_img[_ch:, :]
                    y += _ch
                    ph -= _ch
                elif _cy + _ch > ph * 0.85:
                    panel_img = panel_img[:_cy, :]
                    ph = _cy
            if pw < 30 or ph < 30:
                review_warnings.append(
                    f"panel_{pb['id']}_too_small_after_colorbar_crop: "
                    f"{_orig_pw}x{_orig_ph} -> {pw}x{ph}"
                )
                continue

        panel_cls = classify(panel_img)

        # Colorbar extraction (always try; VLM hint only affects whether we
        # trust the result for rep generation, not whether we attempt it)
        colorbar_rgb = None
        if len(panel_bboxes) == 1:
            colorbar_rgb = extract_colorbar(img_rgb)
        else:
            colorbar_rgb = extract_colorbar(panel_img)

        # Reps for vivid panels
        reps = None
        sat = saturation_ratio(panel_img)
        if sat >= 0.5:
            reps = color_zones_to_reps(
                panel_img,
                color_zones,
                colorbar_rgb=colorbar_rgb,
                n_layers=n_layers,
            )

        # Segmentation with fallback on empty / poor results
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

        # ── Review: check segmentation quality ──────────────────────
        labels = seg["labels"]
        unique_labels = set(labels.flatten())
        n_layers_found = len(unique_labels - {0})

        if n_layers_found == 0:
            review_warnings.append(
                f"panel_{pb['id']}_empty_segmentation: no_layers_found"
            )
        elif n_layers_found < 2:
            if seg["meta"].get("retry_from"):
                review_warnings.append(
                    f"panel_{pb['id']}_under_segmented: only_{n_layers_found}_layer(s)_after_retry"
                )
            else:
                review_warnings.append(
                    f"panel_{pb['id']}_under_segmented: only_{n_layers_found}_layer(s)"
                )
        else:
            if seg["meta"].get("retry_from"):
                review_warnings.append(
                    f"panel_{pb['id']}_retry_fixed: "
                    f"{seg['meta']['retry_from']}_to_{seg['meta']['engine']}_"
                    f"now_{n_layers_found}_layers"
                )

        panel_results.append({
            "panel_id": pb["id"],
            "bbox": [x, y, pw, ph],
            "classification": panel_cls,
            "segmentation": seg,
            "review": {
                "n_layers_found": n_layers_found,
                "is_target_panel": pb["id"] == _target_panel_id,
            },
        })
        total_layers += n_layers_found
        engines_used.add(seg["meta"]["engine"])

    return {
        "classification": cls,
        "panels": panel_results,
        "summary": {
            "status": "ok",
            "n_panels": len(panel_results),
            "total_layers": total_layers,
            "engines_used": sorted(engines_used),
            "saturation_ratio": round(saturation_ratio(img_rgb), 4),
            "review_warnings": review_warnings,
            "vlm_has_colorbar": has_colorbar_hint,
            "vlm_target_panel_id": _target_panel_id,
        },
    }
