"""End-to-end pipeline controller.

Assembles the full backend chain:
    figure image -> classify -> segment -> post-process -> export SPECFEM

Public API:
    run_pipeline(img_rgb, config) -> dict
    run_post_process_and_export(labels, ...) -> dict

Test scenario:
    >>> import numpy as np
    >>> img = np.full((100, 200, 3), 128, dtype=np.uint8)
    >>> result = run_pipeline(img, n_layers=3)
    >>> assert result["status"] in ("ok", "skipped")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from geoseg.modules.exporter.specfem import (
    labels_to_grids,
    write_parfile_snippet,
    write_tomography_file,
)
from geoseg.modules.post_process.polygon import extract_components, labels_to_polygons, save_geojson
from geoseg.modules.post_process.properties import (
    assign_properties,
    generate_properties_for_layers,
    inherit_properties_for_new_labels,
)
from geoseg.modules.segment_engines.full_pipeline import process_figure


def run_post_process_and_export(
    labels: np.ndarray,
    color_names: list[str] | None = None,
    properties_map: dict[str, dict] | None = None,
    output_dir: str | Path | None = None,
    save_intermediates: bool = True,
    overlay: np.ndarray | None = None,
) -> dict[str, Any]:
    """Post-process a single label array and export SPECFEM artifacts.

    This is the standalone export entry-point used both by the full pipeline
    and by the napari-review resume flow.

    Args:
        labels: (H, W) int array. 0 = background/boundary.
        color_names: Optional list indexed by label id (label 0 ignored).
        properties_map: Optional custom {color_name: {"Vp", "Vs", "rho"}} map.
        output_dir: Directory to save artifacts.
        save_intermediates: Whether to save overlays, labels, geojson, etc.
        overlay: Optional overlay image for display.

    Returns:
        dict with keys:
            status: "ok" | "skipped"
            n_components: int
            n_polygons: int
            color_names: list[str]
            properties: dict
    """
    if (labels != 0).sum() == 0:
        return {
            "status": "skipped",
            "reason": "empty_segmentation",
            "n_components": 0,
            "n_polygons": 0,
            "color_names": [],
            "properties": {},
        }

    if not color_names:
        color_names = [f"layer_{i}" for i in sorted(set(labels.flatten()) - {0})]

    # 2a: Polygon extraction
    geojson = labels_to_polygons(labels, color_names=color_names)
    components = extract_components(labels)

    # 2b: Property assignment (auto-generate fallback if unknown)
    try:
        props = assign_properties(color_names, custom_map=properties_map)
    except ValueError:
        # Napari editing may have created new labels — try inheritance first
        try:
            base_props = assign_properties(
                color_names,
                custom_map=properties_map,
            ) if properties_map else {}
        except ValueError:
            base_props = {}
        props = inherit_properties_for_new_labels(labels, base_props, color_names)
        if not props:
            props = generate_properties_for_layers(color_names)

    # 2c: Build property grids
    vp, vs, rho = labels_to_grids(labels, props, color_names=color_names)

    # Save artifacts
    out_dir = Path(output_dir) if output_dir else None
    if out_dir and save_intermediates:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "labels.npz", labels=labels)
        if overlay is not None:
            from PIL import Image
            Image.fromarray(overlay).save(out_dir / "overlay.jpg", quality=90)
        save_geojson(geojson, out_dir / "polygons.geojson")
        (out_dir / "properties.json").write_text(
            json.dumps(props, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # SPECFEM export
        h, w = labels.shape
        x_coords = np.linspace(0, w - 1, w)
        z_coords = np.linspace(0, h - 1, h)
        write_tomography_file(vp, vs, rho, x_coords, z_coords, out_dir / "tomo.xyz")
        write_parfile_snippet(color_names, props, out_dir / "parfile_snippet.txt", nx=w, nz=h)

    return {
        "status": "ok",
        "n_components": len(components),
        "n_polygons": len(geojson["features"]),
        "color_names": color_names,
        "properties": props,
    }


def run_pipeline(
    img_rgb: np.ndarray,
    caption: str = "",
    text_blocks: list[dict] | None = None,
    n_layers: int = 5,
    quality_preference: str = "balanced",
    skip_non_velocity_model: bool = True,
    use_vlm: bool = True,
    properties_map: dict[str, dict] | None = None,
    output_dir: str | Path | None = None,
    save_intermediates: bool = True,
) -> dict[str, Any]:
    """Run the full geoseg pipeline on a single figure image.

    Args:
        img_rgb: RGB uint8 array.
        caption: Optional figure caption from PDF extraction.
        n_layers: Number of layers to extract per panel.
        quality_preference: "fast", "balanced", or "best".
        skip_non_velocity_model: If True, skip observational_data and other types.
        use_vlm: Whether to use VLM for rep generation.
        properties_map: Optional custom {color_name: {"Vp", "Vs", "rho"}} map.
        output_dir: If given, save all artifacts here.
        save_intermediates: Whether to save overlays, labels, geojson, etc.

    Returns:
        dict with keys:
            status: "ok" | "skipped"
            classification: figure classifier result
            panels: list of panel result dicts
            summary: aggregate stats
    """
    # Step 1: Segment
    seg_result = process_figure(
        img_rgb,
        caption=caption,
        text_blocks=text_blocks,
        n_layers=n_layers,
        quality_preference=quality_preference,
        skip_non_velocity_model=skip_non_velocity_model,
        use_vlm=use_vlm,
    )

    if seg_result["summary"]["status"] == "skipped":
        return {
            "status": "skipped",
            "reason": seg_result["summary"]["reason"],
            "classification": seg_result["classification"],
            "panels": [],
            "summary": seg_result["summary"],
        }

    # Prepare output directory
    out_dir = Path(output_dir) if output_dir else None

    # Step 2: Post-process and export each panel
    panel_outputs = []
    for p in seg_result["panels"]:
        seg = p.get("segmentation")
        if seg is None:
            panel_outputs.append({
                "panel_id": p["panel_id"],
                "bbox": p["bbox"],
                "status": "skipped",
                "reason": "no_segmentation",
            })
            continue

        labels = seg["labels"]
        if (labels != 0).sum() == 0:
            panel_outputs.append({
                "panel_id": p["panel_id"],
                "bbox": p["bbox"],
                "status": "skipped",
                "reason": "empty_segmentation",
            })
            continue

        color_names = seg["meta"].get("color_names")
        if not color_names:
            color_names = [f"layer_{i}" for i in sorted(set(labels.flatten()) - {0})]

        panel_out_dir = out_dir / f"panel{p['panel_id']}" if out_dir else None

        export_result = run_post_process_and_export(
            labels,
            color_names=color_names,
            properties_map=properties_map,
            output_dir=panel_out_dir,
            save_intermediates=save_intermediates,
            overlay=seg.get("overlay"),
        )

        panel_outputs.append({
            "panel_id": p["panel_id"],
            "bbox": p["bbox"],
            "status": export_result["status"],
            "n_components": export_result.get("n_components", 0),
            "n_polygons": export_result.get("n_polygons", 0),
            "color_names": export_result.get("color_names", []),
            "properties": export_result.get("properties", {}),
            "engines_used": seg["meta"]["engine"],
        })

    n_processed = sum(1 for po in panel_outputs if po["status"] == "ok")
    n_skipped = sum(1 for po in panel_outputs if po["status"] == "skipped")

    summary = {
        **seg_result["summary"],
        "n_panels_processed": n_processed,
        "n_panels_skipped": n_skipped,
    }

    # If figure passed classifier but no panels were actually processed,
    # mark as empty rather than ok (no artifacts to export).
    if n_processed == 0 and panel_outputs:
        return {
            "status": "empty",
            "reason": "all_panels_skipped_or_no_segmentation",
            "classification": seg_result["classification"],
            "panels": panel_outputs,
            "summary": summary,
        }

    return {
        "status": "ok",
        "classification": seg_result["classification"],
        "panels": panel_outputs,
        "summary": summary,
    }


__all__ = ["run_pipeline", "run_post_process_and_export"]
