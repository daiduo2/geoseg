# Layer 2 experiment contract

This directory is a shared, read-only harness for parallel experiments.  Each
experiment gets its own subdirectory and output directory.

## Dataset

- `validation/curated/manifest.jsonl`, 11 panels.
- `ph01_page8_panel3` is included but its labels are known under-segmented;
  report it separately, do not use it to tune parameters.
- Retention is fixed by Layer 1: `keep_paths()` in `common.py` returns only
  `action == "keep"` exact boundary objects.
- `keep_paths()` now also prunes sub-5 px endpoint-to-branch skeleton spurs and
  drops sub-8 px open junction-cluster edges whose ends touch longer paths.
  Closed loops and short paths with a free end are retained.  Dropped pixels
  remain in the Layer 1 exact boundary; they are simply not independent curves.
  This is path graph v2: 376 → 242 paths, but the non-ph01 smoothable set stays
  at 150 paths.

## Shared API

```python
from experiments.layer2.common import (
    load_panels, keep_paths, resample, boundary_metrics,
    render_overlay, contact_sheet, recolour_labels, cross_intersections,
)
```

- `Panel`: panel_id, labels, image_path.
- `BoundaryPath`: boundary_id, labels_pair, points (x, y, ordered), closed,
  exact_length.
- Open paths: pin both endpoints.  Most are junctions or panel-border
  endpoints; when unsure, pinning is the conservative choice.
- Closed paths: periodic smoothing; do not let them shrink silently.

## Metrics

`boundary_metrics(smoothed_points, exact_path)` reports:

- mean/p95/max displacement to the exact centerline;
- max/p95 curvature, exact max curvature;
- exact/smoothed length; endpoint drift;
- self-intersections;
- area/area change for closed paths.

Also report panel-level `cross_intersections(paths)` for smoothed paths.

## Required output per experiment

Under `runs/layer2_experiments/<experiment>/`:

- `metrics.csv` or `metrics.json`: one row per panel/boundary/variant;
- `summary.md`: aggregate table + interpretation + failure cases;
- `contact.png`: recoloured mask + exact thin white + smoothed thick green;
- per-panel overlays if useful.

Use fixed global parameters, not per-panel tuning.  If a parameter is selected,
select it on synthetic cases or one global grid, then report the same setting
for all panels.  Do not delete a boundary because smoothing failed; fall back
to exact geometry and mark it.

## Constraints

- Do not edit `src/`, `tests/`, `docs/`, or `common.py`.
- Do not smooth across a junction or fault: smooth each `BoundaryPath` (or
  explicitly split it) independently.
- Do not use global image diffusion, level sets, graph cuts, or re-segmentation.
- Do not use the original photo as the audit baseline; use the recoloured mask.
- Every smoothed result must be checked for displacement, self-intersection,
  cross-intersection, endpoint drift, and closed-body area change.
- If a method cannot run on a panel, record the error; do not silently skip it.

## Directions

1. `a_segmented_spline`: per-path constrained smoothing spline/P-spline with
   endpoint pinning, tension/curvature penalty, displacement corridor, and
   iterative fallback.
2. `b_kernel_benchmark`: compare global spline, per-path spline, P-spline,
   Savitzky–Golay, and knot/quintic on the same paths and metrics.
3. `c_fault_topology`: fault/junction-aware path handling plus a global
   topology validator and fallback; test on synthetic fault cases and curated
   panels.
4. `d_closed_aniso`: closed-body area/width preservation and optional
   image-dip anisotropy as a weak prior.
5. `e_endpoint_tangent`: soft endpoint tangent constraints to reduce the
   junction-corner curvature spikes seen in experiment A.
