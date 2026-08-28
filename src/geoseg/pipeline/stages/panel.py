"""Panel crop and segmentation stage helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from geoseg.core.image_ops import saturation_ratio
from geoseg.modules.cv_detect.colorbar_extractor import (
    extract_colorbar,
    extract_colorbar_bbox,
)
from geoseg.modules.cv_detect.figure_classifier import classify
from geoseg.modules.segment_engines import route_and_segment
from geoseg.modules.segment_engines.vlm_reps import color_zones_to_reps
from geoseg.modules.post_process.split import (
    split_label_by_color_components,
    split_labels_by_red_boundaries,
)


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
    boundary_mode: str = "none",
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

    colorbar_rgb = (
        extract_colorbar(img_rgb)
        if len(panel_bboxes) == 1
        else extract_colorbar(panel_img)
    )

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

    if boundary_mode == "red":
        from geoseg.preprocessing.absorption import absorb_artifacts
        from geoseg.preprocessing.detectors import (
            detect_red_boundaries,
            detect_text,
        )

        boundary_mask = detect_red_boundaries(panel_img)
        text_mask = detect_text(
            panel_img,
            min_area=20,
            min_width=8,
            min_aspect=2.0,
        )
        cleaned_for_color = absorb_artifacts(
            panel_img,
            boundary_mask | text_mask,
            inpaint_radius=5,
            dilate_iters=1,
        )
        color_labels = split_label_by_color_components(
            np.ones(panel_img.shape[:2], dtype=np.int32),
            cleaned_for_color,
            target_label=1,
            k=max(2, n_layers),
            min_component_area=max(300, int(panel_img.shape[0] * panel_img.shape[1] * 0.005)),
        )
        parent_color_names = [
            f"color_region_{label_id}"
            for label_id in sorted(set(color_labels.flatten()) - {0})
        ]
        refined_labels, parent_map, boundary_mask = split_labels_by_red_boundaries(
            color_labels,
            panel_img,
            boundary_mask=boundary_mask,
        )
        seg["labels"] = refined_labels
        seg["color_partition"] = color_labels
        seg["boundary_mask"] = boundary_mask
        seg["meta"]["boundary_mode"] = "red"
        seg["meta"]["boundary_pixels"] = int(boundary_mask.sum())
        seg["meta"]["parent_labels"] = parent_map
        seg["meta"]["color_names"] = [
            parent_color_names[parent_map[label_id] - 1]
            if parent_map[label_id] - 1 < len(parent_color_names)
            else f"color_region_{parent_map[label_id]}"
            for label_id in sorted(parent_map)
        ]
        from geoseg.core.image_ops import create_overlay

        seg["overlay"] = create_overlay(panel_img, refined_labels, None)
        review_warnings.append(
            f"panel_{panel['id']}_red_boundary_split: "
            f"{len(parent_map)}_connected_regions"
        )
    elif boundary_mode != "none":
        raise ValueError(f"Unsupported boundary_mode: {boundary_mode!r}")

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


__all__ = ["crop_panel_for_segmentation", "segment_panel_stage"]
