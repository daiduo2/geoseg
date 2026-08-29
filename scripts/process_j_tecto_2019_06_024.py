#!/usr/bin/env python3
"""Prepare and segment Figures 6 and 7 from j.tecto.2019.06.024.

This driver contains only paper-specific geometry and annotation masks. It
delegates cleaning, palette segmentation, edge-guided segmentation, and overlay
rendering to the existing geoseg implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from geoseg.modules.segment_engines.edge_guided import segment as segment_edge_guided
from geoseg.modules.segment_engines.mask_aware import _fill_text_nearest
from geoseg.modules.segment_engines.horizon_refinement import refine_label_blur
from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend
from geoseg.modules.segment_engines.v4.colorbar_guided import segment_colorbar_guided
from geoseg.modules.segment_engines.v4.palette import _sample_colorbar_seeds
from geoseg.experiments import compute_all
from geoseg.modules.post_process.merge import (
    filter_small_components,
    remove_labels_by_ids,
)
from geoseg.modules.text_removal import remove_text
from geoseg.modules.visual_audit.semantic import compute_semantic_fidelity
from geoseg.modules.visual_audit.report import _sanitize_for_json
from geoseg.modules.visual_audit.views import create_audit_views, save_views
from geoseg.preprocessing.absorption import (
    fill_mask_nearest_along_axis,
    visualize_mask_on_image,
)
from geoseg.preprocessing.geometry import rectify_quadrilateral


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/figures/j_tecto_2019_06_024/source"
OUTPUT = ROOT / "runs/j_tecto_2019_06_024"


def _polygon_mask(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
    return mask > 0


def _annotation_masks(
    name: str, image: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = {
        "brightness_thresh": 0,
        "max_area": 900 if name == "fig7" else 1200,
        "lap_threshold": 32 if name == "fig7" else 28,
        "dilate_iter": 1,
        "inpaint_radius": 3,
        "residual_lap_threshold": 20 if name == "fig7" else 18,
        "residual_median_ksize": 41 if name == "fig7" else 51,
    }
    _, text_mask = remove_text(image, **params)
    text_mask = text_mask > 0
    cleaned = _fill_text_nearest(image, text_mask)
    horizontal_mask = text_mask.copy()
    vertical_mask = np.zeros(text_mask.shape, dtype=bool)

    if name == "fig6c":
        # White-backed crustal labels need their full rectangular backing
        # covered; the generic text detector primarily finds their glyphs.
        label_boxes = [
            (960, 110, 1070, 170),
            (925, 175, 1030, 235),
            (920, 235, 1030, 295),
            (1110, 335, 1215, 395),
            (1080, 405, 1250, 470),
            (1080, 535, 1215, 610),
            (1030, 640, 1325, 720),
            (1660, 80, 1780, 145),
            (2730, 125, 2825, 185),
            (2740, 200, 2845, 260),
            (2500, 255, 2610, 315),
            (2260, 310, 2375, 370),
            (2510, 375, 2700, 440),
        ]
        for x0, y0, x1, y1 in label_boxes:
            horizontal_mask[y0:y1, x0:x1] = True

        # Embedded colorbar obscures the lower-mantle color field.
        colorbar_mask = np.zeros(text_mask.shape, dtype=bool)
        colorbar_mask[845:930, 1840:3372] = True
        cleaned = fill_mask_nearest_along_axis(
            cleaned, colorbar_mask, axis="vertical"
        )
        vertical_mask |= colorbar_mask
    elif name == "fig7":
        # Tight masks for the large schematic symbols. Thin text and linework
        # remain covered by the detector mask above.
        polygons = [
            [(115, 135), (205, 135), (238, 168), (205, 202), (115, 202)],
            [(105, 270), (165, 270), (165, 247), (220, 300),
             (165, 353), (165, 330), (105, 330)],
            [(286, 104), (380, 104), (380, 285), (286, 285)],
            [(355, 145), (455, 145), (455, 235), (355, 235)],
            [(445, 88), (565, 88), (565, 205), (445, 205)],
            [(626, 204), (722, 204), (722, 270), (626, 270)],
            [(900, 135), (1015, 135), (1015, 305), (900, 305)],
            [(982, 92), (1095, 92), (1095, 265), (982, 265)],
            [(1092, 103), (1185, 103), (1185, 268), (1092, 268)],
            [(1195, 115), (1320, 115), (1320, 232), (1195, 232)],
            [(1290, 160), (1415, 160), (1415, 275), (1290, 275)],
            [(1340, 95), (1425, 95), (1425, 240), (1340, 240)],
            [(1405, 175), (1545, 175), (1545, 290), (1405, 290)],
        ]
        symbol_mask = np.zeros(text_mask.shape, dtype=np.uint8)
        for points in polygons:
            cv2.fillPoly(symbol_mask, [np.asarray(points, dtype=np.int32)], 1)
        # Adjacent symbols form two dense annotation clusters. Connecting the
        # tight masks prevents an unmasked symbol edge becoming a fill source.
        symbol_mask[90:350, :250] = 1
        symbol_mask[95:320, 750:1600] = 1
        symbol_mask[130:290, 1600:2050] = 1
        symbol_mask = symbol_mask.astype(bool)
        cleaned = fill_mask_nearest_along_axis(
            cleaned, symbol_mask, axis="horizontal"
        )
        horizontal_mask |= symbol_mask
    return horizontal_mask, vertical_mask, cleaned


def _body_mask(name: str, image: np.ndarray) -> np.ndarray:
    if name == "fig6b":
        spread = image.max(axis=2).astype(np.int16) - image.min(axis=2).astype(np.int16)
        mask = (spread > 12) & (image.min(axis=2) < 245)
        mask = ndimage.binary_closing(mask, iterations=3)
        components, count = ndimage.label(mask)
        if count:
            sizes = ndimage.sum(mask, components, range(1, count + 1))
            mask = components == int(np.argmax(sizes) + 1)
        mask = ndimage.binary_fill_holes(mask)
        # The embedded scale is colored but is not part of the model body.
        mask[115:158, 2575:3300] = False
        return mask
    if name == "fig6c":
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[20:980, :3358] = True
        return mask
    if name == "fig7":
        # Reviewed top boundary of the rectified front face.
        points = [
            (105, 105), (400, 96), (800, 85), (1200, 75),
            (1600, 65), (2000, 50), (2399, 35),
            (2399, 699), (105, 699),
        ]
        return _polygon_mask(image.shape[:2], points)
    return np.ones(image.shape[:2], dtype=bool)


def _repair_labels(
    name: str,
    labels: np.ndarray,
    body: np.ndarray,
    horizontal_mask: np.ndarray,
    vertical_mask: np.ndarray,
) -> np.ndarray:
    """Restore obscured geology in label space without inventing new classes."""
    result = labels.astype(np.int32, copy=True)
    if horizontal_mask.any():
        result = fill_mask_nearest_along_axis(
            result, horizontal_mask | ~body, axis="horizontal"
        )
    if vertical_mask.any():
        result = fill_mask_nearest_along_axis(
            result, vertical_mask | ~body, axis="vertical"
        )
    result[~body] = -1

    # Reuse the repository's established nearest-label component cleanup.
    shifted = result + 1
    shifted[~body] = 0
    shifted = filter_small_components(
        shifted, min_area_ratio=0.00035, fill="nearest"
    )
    if name == "fig6b":
        # Palette class 5 occurs only in disconnected numeric/contour strokes,
        # not as a resolved velocity region in this shallow section.
        shifted = remove_labels_by_ids(shifted, [6], fill="nearest")
    elif name == "fig7":
        shifted = refine_label_blur(shifted, sigma=5.0)
    result = shifted - 1
    result[~body] = -1
    return result


def _reconstruct(labels: np.ndarray, palette: np.ndarray, body: np.ndarray) -> np.ndarray:
    result = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    for label_id in np.unique(labels):
        if label_id < 0 or label_id >= len(palette):
            continue
        result[(labels == label_id) & body] = palette[label_id]
    return result


def _save_candidate(
    output_dir: Path,
    candidate_name: str,
    original: np.ndarray,
    labels: np.ndarray,
    palette: np.ndarray,
    body: np.ndarray,
    horizontal_mask: np.ndarray,
    vertical_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = _repair_labels(
        output_dir.name, labels, body, horizontal_mask, vertical_mask
    )
    labels = ndimage.median_filter(labels, size=(5, 15))
    labels[~body] = -1
    candidate_dir = output_dir / candidate_name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(candidate_dir / "labels.npz", labels=labels, body_mask=body)
    reconstructed = _reconstruct(labels, palette, body)
    Image.fromarray(reconstructed).save(candidate_dir / "reconstructed.png")
    audit_labels = labels + 1
    audit_labels[~body] = 0
    overlay = generate_overlay_with_legend(original, audit_labels)
    Image.fromarray(overlay).save(candidate_dir / "overlay_legend.png")
    return labels, reconstructed


def _save_final(
    name: str,
    panel_dir: Path,
    original: np.ndarray,
    cleaned: np.ndarray,
    annotation_mask: np.ndarray,
    labels: np.ndarray,
    reconstructed: np.ndarray,
    body: np.ndarray,
) -> dict[str, object]:
    final_dir = panel_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(final_dir / "labels.npz", labels=labels, body_mask=body)
    Image.fromarray(reconstructed).save(final_dir / "clean_basemap.png")

    gap = np.full((original.shape[0], 12, 3), 255, dtype=np.uint8)
    comparison = np.concatenate([original, gap, reconstructed], axis=1)
    Image.fromarray(comparison).save(final_dir / "original_vs_clean.png")

    audit_labels = labels + 1
    audit_labels[~body] = 0
    audit_dir = final_dir / "visual_audit"
    views = create_audit_views(
        audit_labels,
        original,
        no_text_rgb=reconstructed,
        text_mask=annotation_mask,
    )
    view_paths = save_views(views, str(audit_dir / "views"))
    diagnostics = {
        **compute_all(audit_labels, original),
        **compute_semantic_fidelity(audit_labels, original),
    }
    report_path = audit_dir / "report.json"
    report_path.write_text(
        json.dumps(
            _sanitize_for_json({
                "diagnostic_signals": diagnostics,
                "view_paths": view_paths,
                "note": "Independent views are used because a single tiled summary exceeds JPEG dimension limits for this wide panel.",
            }),
            indent=2,
            default=lambda value: value.item() if hasattr(value, "item") else value,
        )
        + "\n",
        encoding="utf-8",
    )

    palette_labels = sorted(int(value) for value in np.unique(labels) if value >= 0)
    notes = {
        "fig6b": (
            "The shallow velocity body and its two dominant velocity zones are "
            "retained; embedded scale and numeric-stroke islands were removed."
        ),
        "fig6c": (
            "Major crustal, Moho, mantle, and lower-zone geometries are retained. "
            "White-backed labels and the embedded colorbar were restored from "
            "neighboring labels."
        ),
        "fig7": (
            "Major stratigraphic bands, Moho geometry, orange mantle, and pink "
            "lower zone are retained. Boundaries inside the densely annotated "
            "x=750..2050, y=95..320 region are inferred from left/right labels "
            "and remain the principal uncertainty."
        ),
    }
    audit = {
        "accepted_candidate": "colorbar_guided",
        "frozen_labels": palette_labels,
        "retry_labels": [],
        "notes": notes[name],
        "repair_strategy": "accept_with_documented_inference",
        "rejected_candidate": "edge_guided",
        "rejection_reason": "excessive fragmentation at the requested class count",
        "local_fixes": [
            "axis-constrained label fill inside reviewed annotation masks",
            "existing nearest-label small-component filtering",
            *( ["existing horizon label-blur refinement (sigma=5)"] if name == "fig7" else [] ),
        ],
        "iteration": 3,
    }
    (final_dir / "regional_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "accepted_candidate": "colorbar_guided",
        "final_dir": str(final_dir.relative_to(ROOT)),
        "palette_labels_present": palette_labels,
        "audit_report": str(report_path.relative_to(ROOT)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig6 = np.asarray(Image.open(SOURCE / "fig6_source.png").convert("RGB"))
    fig7 = np.asarray(Image.open(SOURCE / "fig7_source.png").convert("RGB"))

    fig7_points = np.array(
        [[530, 655], [2900, 623], [2900, 1045], [530, 1430]],
        dtype=np.float32,
    )
    fig7_rectified, homography = rectify_quadrilateral(
        fig7, fig7_points, (2400, 700)
    )
    panels = {
        "fig6b": fig6[650:850, 137:3509],
        "fig6c": fig6[850:1831, 137:3509],
        "fig7": fig7_rectified,
    }
    colorbars = {
        "fig6b": fig6[775:790, 2750:3400],
        "fig6c": fig6[1705:1765, 2010:3490],
        "fig7": fig6[1705:1765, 2010:3490],
    }
    layer_counts = {"fig6b": 8, "fig6c": 20, "fig7": 20}

    summary: dict[str, object] = {
        "source": "j.tecto.2019.06.024.pdf",
        "fig7_source_points_tl_tr_br_bl": fig7_points.astype(int).tolist(),
        "fig7_homography": homography.tolist(),
        "panels": {},
    }
    for name, panel in panels.items():
        panel_dir = OUTPUT / name
        panel_dir.mkdir(parents=True, exist_ok=True)
        horizontal_mask, vertical_mask, cleaned = _annotation_masks(name, panel)
        mask = horizontal_mask | vertical_mask
        body = _body_mask(name, panel)
        colorbar = colorbars[name]
        n_layers = layer_counts[name]
        palette, _ = _sample_colorbar_seeds(colorbar, n_layers)

        Image.fromarray(panel).save(panel_dir / "01_panel.png")
        Image.fromarray(colorbar).save(panel_dir / "02_colorbar.png")
        Image.fromarray(visualize_mask_on_image(panel, mask)).save(
            panel_dir / "03_annotation_mask_overlay.png"
        )
        Image.fromarray(cleaned).save(panel_dir / "04_cleaned.png")
        Image.fromarray(body.astype(np.uint8) * 255).save(panel_dir / "05_body_mask.png")

        guided = segment_colorbar_guided(cleaned, colorbar, n_layers=n_layers)
        guided_palette = np.asarray(guided["seeds"], dtype=np.uint8)
        guided_labels, guided_reconstructed = _save_candidate(
            panel_dir,
            "colorbar_guided",
            panel,
            guided["labels"],
            guided_palette,
            body,
            horizontal_mask,
            vertical_mask,
        )

        edge = segment_edge_guided(cleaned, reps=None, n_layers=n_layers)
        edge_palette = np.asarray(edge["seeds"], dtype=np.uint8)
        _save_candidate(
            panel_dir,
            "edge_guided",
            panel,
            edge["labels"],
            edge_palette,
            body,
            horizontal_mask,
            vertical_mask,
        )

        final_summary = _save_final(
            name,
            panel_dir,
            panel,
            cleaned,
            mask,
            guided_labels,
            guided_reconstructed,
            body,
        )

        summary["panels"][name] = {
            "shape": list(panel.shape),
            "n_layers_requested": n_layers,
            "annotation_mask_fraction": round(float(mask.mean()), 5),
            "body_fraction": round(float(body.mean()), 5),
            "candidates": ["colorbar_guided", "edge_guided"],
            **final_summary,
        }

    (OUTPUT / "run_config.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
