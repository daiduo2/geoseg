"""End-to-end pipeline controller.

Assembles the full backend chain:
    figure image -> classify -> segment -> post-process -> export SPECFEM

Public API:
    run_pipeline(img_rgb, config) -> dict

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
)
from geoseg.modules.segment_engines.full_pipeline import process_figure


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
    if out_dir and save_intermediates:
        out_dir.mkdir(parents=True, exist_ok=True)

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
            # Build fallback color names from unique labels
            color_names = [f"layer_{i}" for i in sorted(set(labels.flatten()) - {0})]

        # 2a: Polygon extraction
        geojson = labels_to_polygons(labels, color_names=color_names)
        components = extract_components(labels)

        # 2b: Property assignment (auto-generate fallback if unknown)
        try:
            props = assign_properties(color_names, custom_map=properties_map)
        except ValueError:
            props = generate_properties_for_layers(color_names)

        # 2c: Build property grids
        vp, vs, rho = labels_to_grids(labels, props, color_names=color_names)

        # Save artifacts
        if out_dir and save_intermediates:
            base = out_dir / f"panel{p['panel_id']}"
            np.savez_compressed(f"{base}_labels.npz", labels=labels)
            if "overlay" in seg:
                from PIL import Image
                Image.fromarray(seg["overlay"]).save(f"{base}_overlay.jpg", quality=90)
            save_geojson(geojson, f"{base}_polygons.geojson")
            json.dump(props, open(f"{base}_properties.json", "w"), indent=2)

            # SPECFEM export
            h, w = labels.shape
            x_coords = np.linspace(0, w - 1, w)
            z_coords = np.linspace(0, h - 1, h)
            write_tomography_file(vp, vs, rho, x_coords, z_coords, f"{base}_tomo.xyz")
            write_parfile_snippet(color_names, props, f"{base}_parfile_snippet.txt", nx=w, nz=h)

        panel_outputs.append({
            "panel_id": p["panel_id"],
            "bbox": p["bbox"],
            "status": "ok",
            "n_components": len(components),
            "n_polygons": len(geojson["features"]),
            "color_names": color_names,
            "properties": props,
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


__all__ = ["run_pipeline"]
