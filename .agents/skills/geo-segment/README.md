# geo-segment

Half-automatic conversion of geophysics interpretation figures into SPECFEM-ready velocity zone models.

## Quick Start

```bash
# Use project uv environment from the geoseg root
uv run scripts/env_check.py
uv run python -m geoseg.controller_demo
```

See `geo-segment/SKILL.md` for the active end-to-end skill.

## Segmentation Algorithms

The skill implements multiple segmentation strategies:

### `segment_colorbar_guided()` — Default for colorbar-available panels

Proven in experiment e026. Uses K-means with colorbar-extracted RGB seeds as initial centroids.

Key characteristics:
- **No bilateral denoising** — avoids boundary artifacts
- **No keep_largest_component** — preserves valid fracture/fault regions
- **Reorder by median_y** — top=high velocity (low label), bottom=low velocity
- **Fill holes only** — physically reasonable (no voids in rock)
- **Remove small components** — merge fragments < 0.1% panel area into neighbors
- **Enhance close boundaries** — re-classify boundary pixels between adjacent layers with similar seed colors (distance < 55)

### `segment_jet_vivid()` — VLM rep-point driven

For high-saturation jet/rainbow colormap panels. Uses multi-source seed refinement with CV fallback, then nearest-median classification + shape filter.

### `segment_pastel_faded()` — Legacy K-means

Kept for backward compatibility and no-colorbar fallback. Uses K-means in LAB space with optional colorbar seeds, followed by perimeter^2/area shape filter.

## Pipeline

```
[crop]   VLM bbox + gutter refinement          lib/crop.py
[clean]  VLM noise list + CV backup             lib/clean.py
[seg]    Dual-path color segmentation           lib/segment.py
[poly]   Polygon fitting + smoothing            lib/polygon.py
[prop]   Color zone -> Vp/Vs/rho assignment     lib/properties.py
[mesh]   Polygon rasterization -> regular grid  lib/mesh.py
[spec]   SPECFEM2D tomography_file output       lib/specfem.py
```

## Output Files

```
panel_0_crop.png              # Cropped panel
panel_0_noise_mask.png        # Noise mask (white = to be cleared)
panel_0_clean.png             # Noise-removed preview
panel_0_segmentation.png      # Color zone label map
panel_0_polygons.geojson      # Polygon vertices
panel_0_properties.json       # Color -> Vp/Vs/rho mapping
panel_0_tomo.xyz              # SPECFEM2D tomography_file (x z Vp Vs rho)
panel_0_Par_file_snippet.txt  # Par_file configuration snippet
panel_0_report.json           # Metadata and algorithm notes
review.json                   # Human correction stub (with --interactive)
```

## Environment

> **DEPRECATED**: This README describes the old e026 standalone skill layout. The active end-to-end skill is `geo-segment/SKILL.md`.
> Use `uv` to manage the environment:

```bash
uv sync
uv run scripts/env_check.py
uv run python -m geoseg.modules.<module>.demo
```

Do **not** use `python3 -m venv`/`pip` or `npm`/`yarn` in this project.
