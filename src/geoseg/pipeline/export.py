"""Post-process and export pipeline stages."""

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


def _color_names_from_labels(labels: np.ndarray) -> list[str]:
    return [f"layer_{i}" for i in sorted(set(labels.flatten()) - {0})]


def run_post_process_and_export(
    labels: np.ndarray,
    color_names: list[str] | None = None,
    properties_map: dict[str, dict] | None = None,
    output_dir: str | Path | None = None,
    save_intermediates: bool = True,
    overlay: np.ndarray | None = None,
) -> dict[str, Any]:
    """Post-process a single label array and export SPECFEM artifacts."""
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
        color_names = _color_names_from_labels(labels)

    geojson = labels_to_polygons(labels, color_names=color_names)
    components = extract_components(labels)

    try:
        props = assign_properties(color_names, custom_map=properties_map)
    except ValueError:
        try:
            base_props = (
                assign_properties(color_names, custom_map=properties_map)
                if properties_map
                else {}
            )
        except ValueError:
            base_props = {}
        props = inherit_properties_for_new_labels(labels, base_props, color_names)
        if not props:
            props = generate_properties_for_layers(color_names)

    vp, vs, rho = labels_to_grids(labels, props, color_names=color_names)

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


def export_segmented_panels(
    seg_result: dict[str, Any],
    *,
    properties_map: dict[str, dict] | None = None,
    output_dir: str | Path | None = None,
    save_intermediates: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Export all segmented panels from a segmentation-stage result."""
    out_dir = Path(output_dir) if output_dir else None
    panel_outputs: list[dict[str, Any]] = []

    for panel in seg_result["panels"]:
        seg = panel.get("segmentation")
        if seg is None:
            panel_outputs.append({
                "panel_id": panel["panel_id"],
                "bbox": panel["bbox"],
                "status": "skipped",
                "reason": "no_segmentation",
            })
            continue

        labels = seg["labels"]
        if (labels != 0).sum() == 0:
            panel_outputs.append({
                "panel_id": panel["panel_id"],
                "bbox": panel["bbox"],
                "status": "skipped",
                "reason": "empty_segmentation",
            })
            continue

        color_names = seg["meta"].get("color_names") or _color_names_from_labels(labels)
        panel_out_dir = out_dir / f"panel{panel['panel_id']}" if out_dir else None

        export_result = run_post_process_and_export(
            labels,
            color_names=color_names,
            properties_map=properties_map,
            output_dir=panel_out_dir,
            save_intermediates=save_intermediates,
            overlay=seg.get("overlay"),
        )

        if panel_out_dir and save_intermediates:
            color_partition = seg.get("color_partition")
            if color_partition is not None:
                np.savez_compressed(
                    panel_out_dir / "color_partition.npz",
                    labels=color_partition,
                )
            boundary_mask = seg.get("boundary_mask")
            if boundary_mask is not None:
                from PIL import Image

                Image.fromarray(
                    np.asarray(boundary_mask, dtype=np.uint8) * 255
                ).save(panel_out_dir / "red_boundary_mask.png")

        panel_outputs.append({
            "panel_id": panel["panel_id"],
            "bbox": panel["bbox"],
            "status": export_result["status"],
            "n_components": export_result.get("n_components", 0),
            "n_polygons": export_result.get("n_polygons", 0),
            "color_names": export_result.get("color_names", []),
            "properties": export_result.get("properties", {}),
            "engines_used": seg["meta"]["engine"],
        })

    n_processed = sum(1 for panel in panel_outputs if panel["status"] == "ok")
    n_skipped = sum(1 for panel in panel_outputs if panel["status"] == "skipped")
    summary = {
        **seg_result["summary"],
        "n_panels_processed": n_processed,
        "n_panels_skipped": n_skipped,
    }
    return panel_outputs, summary


__all__ = [
    "export_segmented_panels",
    "run_post_process_and_export",
]
