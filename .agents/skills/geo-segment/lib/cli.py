"""Orchestration CLI for geo-segment.

Entry point:
    python3 -m lib.cli <image_path> --vlm-json <json> [options]

Options:
    --panel N|auto      Which panel to segment (default 0)
    --zones N           Number of color zones (default 5)
    --out-dir DIR       Output directory (default ./out/<stem>/)
    --interactive       Emit review.json for human correction
    --apply-review FILE Re-segment using corrected points from review.json
    --vlm-json PATH     Load VLM results produced by the current Claude session

Pipeline:
    load image + VLM data
    → crop panel (bbox refinement)
    → build noise mask (VLM + CV)
    → dual-path segmentation
    → polygon extraction
    → save PNG / GeoJSON / report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .clean import compose_clean_mask
from .crop import crop_panel
from .interactive import apply_corrections, write_review_prompts
from .polygon import labels_to_polygons, render_label_overlay, save_geojson
from .mesh import label_grid_to_properties, rasterize_polygons
from .properties import assign_properties, build_properties_template, load_properties_json, save_properties_json
from .segment import JET_VIVID_RATIO, saturation_ratio, segment, segment_jet_vivid, segment_colorbar_guided
from .specfem import write_parfile_snippet, write_tomography_file
from .vlm import load_vlm_json


def _inpaint_nearest(panel_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked pixels with nearest non-masked pixel (scipy EDT)."""
    from scipy import ndimage

    out = panel_rgb.copy().astype(np.float32)
    for c in range(3):
        ch = out[..., c].copy()
        ch[mask] = np.nan
        ind = ndimage.distance_transform_edt(np.isnan(ch), return_distances=False, return_indices=True)
        out[..., c] = ch[tuple(ind)]
    return out.clip(0, 255).astype(np.uint8)


def _zones_to_panel_local(zones: list[dict], origin: tuple[int, int]) -> list[dict]:
    """Shift representative_point from image-global to panel-local coords."""
    ox, oy = origin
    out = []
    for z in zones:
        rp = z["representative_point"]
        out.append({
            **z,
            "representative_point": {
                "x": int(rp["x"]) - ox,
                "y": int(rp["y"]) - oy,
            },
        })
    return out


def _extract_panels(data: dict) -> list[dict]:
    if "T1_panels" in data:
        return data["T1_panels"].get("panels", [])
    if "panels" in data:
        return data["panels"]
    raise ValueError("VLM JSON missing panels")


def _extract_noise(data: dict, panel_idx: int) -> list[dict]:
    for key, val in data.items():
        if key.startswith("T2_noise_elements"):
            if val.get("panel_id") == panel_idx:
                return val.get("noise_elements", [])
    # Fallback: try generic key
    return data.get("noise_elements", [])


def _extract_zones(data: dict, panel_idx: int) -> list[dict]:
    for key, val in data.items():
        if key.startswith("T3_color_zones"):
            if val.get("panel_id") == panel_idx:
                return val.get("zones", [])
    return data.get("zones", [])


def _save_png(arr: np.ndarray, path: Path) -> None:
    if arr.dtype == bool:
        arr = (arr.astype(np.uint8) * 255)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)


def _run(
    image_path: Path,
    vlm_data: dict,
    panel_idx: int,
    k: int,
    out_dir: Path,
    interactive: bool,
    dx: float,
    dz: float,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    props_override: dict[str, dict] | None,
    max_auto_k: int = 3,
    colorbar_path: Path | None = None,
):
    panels = _extract_panels(vlm_data)
    if panel_idx < 0 or panel_idx >= len(panels):
        raise SystemExit(f"Panel index {panel_idx} out of range (0–{len(panels) - 1})")
    p = panels[panel_idx]
    bbox = (int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"]))

    # Crop
    panel_rgb, refined_bbox = crop_panel(image_path, bbox)
    _save_png(panel_rgb, out_dir / f"panel_{panel_idx}_crop.png")

    # Noise mask
    noise = _extract_noise(vlm_data, panel_idx)
    mask = compose_clean_mask(panel_rgb, noise, panel_origin=(refined_bbox[0], refined_bbox[1]))
    _save_png(mask, out_dir / f"panel_{panel_idx}_noise_mask.png")

    # Inpainting is Phase 2; for Phase 1 just darken masked pixels for visual QA
    darkened = panel_rgb.copy()
    darkened[mask] = (darkened[mask].astype(np.uint16) * 0.3).astype(np.uint8)
    _save_png(darkened, out_dir / f"panel_{panel_idx}_clean.png")

    # Segment (on noise-cleaned image so rep-point sampling is not corrupted)
    zones = _extract_zones(vlm_data, panel_idx)
    zones_local = _zones_to_panel_local(zones, (refined_bbox[0], refined_bbox[1]))
    cleaned_rgb = _inpaint_nearest(panel_rgb, mask)
    sat = saturation_ratio(cleaned_rgb)

    # Load colorbar if provided
    colorbar_rgb = None
    if colorbar_path is not None and colorbar_path.exists():
        colorbar_rgb = np.array(Image.open(colorbar_path).convert("RGB"))

    result = segment(
        cleaned_rgb,
        kimi_reps=zones_local,
        colorbar_rgb=colorbar_rgb,
        k=k,
        max_auto_k=max_auto_k,
    )

    # If jet_vivid but saturation is marginal, still trust VLM reps
    if sat < JET_VIVID_RATIO and zones_local and (colorbar_rgb is None or colorbar_rgb.size == 0):
        # Force jet path anyway when user gave explicit rep points and no colorbar
        result = segment_jet_vivid(cleaned_rgb, zones_local)

    overlay = render_label_overlay(panel_rgb, result.labels, result.palette, alpha=0.5)
    _save_png(overlay, out_dir / f"panel_{panel_idx}_segmentation.png")

    # Polygons
    features = labels_to_polygons(result.labels, result.color_names)
    save_geojson(features, out_dir / f"panel_{panel_idx}_polygons.geojson")

    # Report
    report = {
        "image": str(image_path),
        "panel_id": panel_idx,
        "refined_bbox": refined_bbox,
        "saturation_ratio": sat,
        "segment_path": result.path,
        "palette_rgb": result.palette.tolist(),
        "color_names": result.color_names,
        "noise_elements_count": len(noise),
        "zones_count": len(zones),
        "notes": result.notes,
    }
    (out_dir / f"panel_{panel_idx}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Interactive review stub
    if interactive:
        write_review_prompts(out_dir, result, panel_idx)

    # ── Phase 2: properties → mesh → SPECFEM ──
    props = assign_properties(result.color_names, custom_map=props_override)
    save_properties_json(props, out_dir / f"panel_{panel_idx}_properties.json")

    grid_result = rasterize_polygons(
        features,
        result.color_names,
        dx=dx,
        dz=dz,
        x_range=x_range,
        z_range=z_range,
    )
    vp, vs, rho = label_grid_to_properties(grid_result.labels, result.color_names, props)

    write_tomography_file(
        vp, vs, rho,
        grid_result.x_coords,
        grid_result.z_coords,
        out_dir / f"panel_{panel_idx}_tomo.xyz",
    )
    write_parfile_snippet(
        result.color_names, props,
        out_dir / f"panel_{panel_idx}_Par_file_snippet.txt",
        nx=grid_result.labels.shape[1],
        nz=grid_result.labels.shape[0],
        dx=dx,
        dz=dz,
    )

    print(f"✓ panel {panel_idx} done → {out_dir}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="geo-segment: figure → SPECFEM zones")
    parser.add_argument("image", type=Path, help="Input figure image")
    parser.add_argument("--vlm-json", type=Path, help="Pre-computed VLM JSON")
    parser.add_argument("--panel", type=str, default="0", help="Panel index or 'auto'")
    parser.add_argument("--zones", type=int, default=5)
    parser.add_argument("--max-auto-k", type=int, default=3, help="Max extra auto-detected color seeds (jet_vivid path)")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--apply-review", type=Path, help="review.json with corrected points")
    # Grid / model params
    parser.add_argument("--dx", type=float, default=1.0, help="Horizontal grid spacing in km (default 1.0)")
    parser.add_argument("--dz", type=float, default=1.0, help="Vertical grid spacing in km (default 1.0)")
    parser.add_argument("--x-range", type=str, default="0,100", help="Horizontal range 'xmin,xmax' in km")
    parser.add_argument("--z-range", type=str, default="0,50", help="Depth range 'zmin,zmax' in km")
    parser.add_argument("--properties-json", type=Path, help="Custom color->Vp/Vs/rho mapping JSON")
    parser.add_argument("--colorbar", type=Path, help="Path to colorbar crop image (enables colorbar-guided segmentation)")
    # NOTE: There is no --api flag.  All vision reasoning happens inside the
    # running Claude Code session.  Save the session reply as JSON and pass
    # it via --vlm-json.
    args = parser.parse_args(argv)

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    # Resolve output directory
    out_dir = args.out_dir or Path("out") / args.image.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load / call VLM data
    if args.apply_review:
        review = json.loads(Path(args.apply_review).read_text(encoding="utf-8"))
        # Load existing crop for re-segmentation
        crop_path = out_dir / f"panel_{review['panel_id']}_crop.png"
        if not crop_path.exists():
            raise SystemExit(f"Missing crop for review: {crop_path}")
        panel_rgb = np.array(Image.open(crop_path).convert("RGB"))
        result = apply_corrections(panel_rgb, review)
        overlay = render_label_overlay(panel_rgb, result.labels, result.palette, alpha=0.5)
        _save_png(overlay, out_dir / f"panel_{review['panel_id']}_segmentation.png")
        features = labels_to_polygons(result.labels, result.color_names)
        save_geojson(features, out_dir / f"panel_{review['panel_id']}_polygons.geojson")
        print(f"✓ review applied → {out_dir}")
        return

    if not args.vlm_json:
        raise SystemExit(
            "This skill does not call external LLM APIs.\n"
            "Run T1/T2/T3 in the current Claude Code session, save the JSON reply, "
            "then pass it with --vlm-json <path>."
        )
    vlm_data = load_vlm_json(args.vlm_json)

    x_range = tuple(float(v) for v in args.x_range.split(","))
    z_range = tuple(float(v) for v in args.z_range.split(","))
    props_override = None
    if args.properties_json:
        props_override = load_properties_json(args.properties_json)

    panel_idx = int(args.panel) if args.panel != "auto" else 0
    _run(
        args.image, vlm_data, panel_idx, args.zones, out_dir, args.interactive,
        dx=args.dx, dz=args.dz, x_range=x_range, z_range=z_range,
        props_override=props_override, max_auto_k=args.max_auto_k,
        colorbar_path=args.colorbar,
    )


if __name__ == "__main__":
    main()
