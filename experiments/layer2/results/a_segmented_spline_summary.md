# Layer 2 experiment A — per-boundary-path constrained smoothing spline

Run: 2026-10-21 13:30:36 local
Command: `uv run python experiments/layer2/a_segmented_spline/run.py`
Runtime: 292.7s

## Method and fixed parameters

- 11 curated panels, **242 keep paths** from **158 exact boundary objects**
  (Layer 1 keep set unchanged; no re-segmentation).
- Every `BoundaryPath` is fitted independently in arc length.  No path is
  smoothed across a junction or fault.
- Cubic B-spline P-spline with knots every **8px** of exact arc length.
  The objective is
  `min_c ||B c - y||² + λ (||D2 c||² + 0.25 ||D3 c||²)`.
  `D2` is a tension/bending penalty on the control polygon; `D3` penalizes
  curvature variation.  Both use circular differences for closed paths.
- **Open paths:** clamped basis, first and last coefficients eliminated and set
  exactly to the two exact centerline endpoints (hard endpoint pinning).
  `boundary_metrics` reports endpoint drift ≈ 1e-13 px (floating point).
- **Closed paths:** periodic basis; closure is built into the basis, and the
  exact-to-smoothed area change is checked before accepting a fit.
- **Corridor:** max displacement to the exact centerline ≤ **3px**.
  Closed area change ≤ **2%**; self-intersections must be zero;
  open endpoint drift ≤ **0.001px**.
- **Iterative fallback:** if a fit violates a constraint, λ is halved and the
  path is refitted, up to **6** times.  If it still fails, the
  exact geometry is returned and marked `fallback_constraint`.
- **Eligibility:** at least 4 points, ≥6px arc length,
  and (for closed paths) ≥25px² exact area.  Shorter skeleton
  fragments cannot support a cubic fit; they are returned exact and marked
  `fallback_short` — recorded, never dropped.
- One global λ is selected for all panels.  Per-path fallback is the only
  place where the effective λ differs; `lambda_used` and `refits` are recorded
  per path.

## Global setting and selection criterion

**Selected global setting: λ = 1** (`lam1`).

The rule was fixed before examining grid outcomes.  Using eligible paths from
all panels **except `ph01_page8_panel3`**, a setting is a candidate only if:

1. constraint-fallback rate ≤ 2%;
2. path-reduction rate ≤ 10%;
3. median p95 displacement ≤ 1px;
4. 90th-percentile p95 displacement ≤ 2px;
5. median max-curvature reduction ≥ 50%.

Among candidates, the **largest λ** (strongest smoothing) wins.  If none pass,
the setting with the smallest median p95 displacement is used.

Selection for λ = 1: **pass**.

## Global grid (non-ph01, eligible paths only)

`smoothed` is the number of paths successfully fitted (including paths that
needed λ reduction); `reduced` counts paths that required at least one fallback
reduction; `fb constraint` counts paths that exhausted reductions and returned
exact geometry.  Displacements are in px.

| λ | smoothed | reduced | fb constraint | reduction rate | fallback rate | med mean disp | med p95 disp | p90 p95 disp | med max disp | p95 max disp | med abs ΔL/L |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 150 | 0 | 0 | 0.00% | 0.00% | 0.356 | 0.590 | 0.877 | 0.833 | 1.442 | 3.35% |
| 0.03 | 150 | 0 | 0 | 0.00% | 0.00% | 0.361 | 0.612 | 1.030 | 0.904 | 1.578 | 3.45% |
| 0.1 | 150 | 0 | 0 | 0.00% | 0.00% | 0.376 | 0.640 | 1.179 | 1.025 | 1.921 | 3.55% |
| 0.3 | 150 | 0 | 0 | 0.00% | 0.00% | 0.394 | 0.711 | 1.457 | 1.202 | 2.264 | 3.70% |
| 1 | 150 | 3 | 0 | 2.00% | 0.00% | 0.423 | 0.801 | 1.780 | 1.412 | 2.691 | 3.80% |
| 3 | 150 | 12 | 0 | 8.00% | 0.00% | 0.453 | 0.933 | 2.121 | 1.669 | 2.879 | 4.04% |
| 10 | 150 | 43 | 0 | 28.67% | 0.00% | 0.504 | 1.079 | 2.186 | 2.137 | 2.900 | 4.26% |

| λ | med max-curv reduction | med p95-curv reduction | max closed abs ΔA/A | self-int | cross-int | selection |
|---|---|---|---|---|---|---|
| 0.01 | 66.0% | 86.8% | 0.019% | 0 | 0 | pass |
| 0.03 | 70.1% | 88.1% | 0.028% | 0 | 0 | pass |
| 0.1 | 74.3% | 89.9% | 0.059% | 0 | 0 | pass |
| 0.3 | 78.6% | 91.1% | 0.179% | 0 | 0 | pass |
| 1 | 84.0% | 93.0% | 0.426% | 0 | 0 | **selected** |
| 3 | 88.1% | 94.8% | 0.547% | 0 | 0 | fail: p90 p95>2px |
| 10 | 91.7% | 96.3% | 0.734% | 0 | 0 | fail: reductions>10%, median p95>1px, p90 p95>2px |

## Aggregate results at the selected setting

Definitions: `smoothed` statistics use successful fits only; `eligible` excludes
short skeleton fragments.  Final-output fallbacks are exact geometry.  Rows for
exact geometry (`baseline`, `fallback_short`, `fallback_constraint`) report
zero displacement, length change and area change by definition.  Smoothed rows
still include the contract metric's independent-resampling floor (scoring exact
geometry against itself gives ~0.27px mean / 0.51px p95 on a long path), so the
reported displacement values are conservative rather than exact
point-to-segment distances.

- Eligible paths (non-ph01): **150**; successfully smoothed:
  **150** (3 after λ reduction);
  constraint fallbacks: **0**.
- Median mean / p95 / max displacement: **0.423 /
  0.801 / 1.412 px**;
  p90 of p95 and p95 of max: **1.780 /
  2.691 px**.
- Median max-curvature reduction: **84.0%**;
  median p95-curvature reduction: **93.0%**.
- Median |length change| / exact length: **3.80%**
  (uniform smoothing removes pixel-staircase length; signed changes are in
  `metrics.csv`).
- Maximum closed-body |area change| / exact area: **0.426%**.
- Self-intersections at selected setting: **0**.
- Cross-intersections (all panels): exact **0** → selected
  **0** (non-ph01: **0**).
- Errors: **0**.

## Per-panel results at the selected setting

`objects` is the number of distinct exact boundary objects; skeleton
fragmentation can yield more paths than objects.  `med p95 (smoothed)` and
`med max-curv red.` use successful fits; `fb short` paths are exact.

| panel | paths | objects | eligible | ok | reduced | fb short | fb constraint | med p95 (smoothed) | med max-curv red. | max abs ΔA/A | self-int | cross-int | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| newimage_p0 | 3 | 3 | 3 | 3 | 0 | 0 | 0 | 1.103 | 86.8% | n/a | 0 | 0 |  |
| newimage_p1 | 8 | 8 | 8 | 8 | 0 | 0 | 0 | 0.709 | 90.4% | 0.033% | 0 | 0 |  |
| newimage_p2 | 8 | 7 | 7 | 7 | 0 | 1 | 0 | 0.743 | 84.9% | 0.426% | 0 | 0 |  |
| newimage_p3 | 4 | 4 | 4 | 4 | 0 | 0 | 0 | 0.837 | 66.0% | 0.004% | 0 | 0 |  |
| newimage_p4 | 7 | 5 | 7 | 7 | 0 | 0 | 0 | 0.841 | 77.0% | n/a | 0 | 0 |  |
| paper22_fig5a | 13 | 13 | 13 | 13 | 0 | 0 | 0 | 0.590 | 90.0% | n/a | 0 | 0 |  |
| paper22_fig6 | 9 | 7 | 9 | 9 | 0 | 0 | 0 | 0.835 | 96.0% | n/a | 0 | 0 |  |
| ph01_page8_panel3 | 85 | 22 | 34 | 33 | 0 | 51 | 1 | 0.586 | 85.6% | 0.000% | 0 | 0 | excluded from selection |
| paper23_fig7a | 34 | 31 | 34 | 34 | 0 | 0 | 0 | 0.657 | 81.6% | n/a | 0 | 0 |  |
| schematic_panel2 | 40 | 27 | 34 | 34 | 0 | 6 | 0 | 0.979 | 83.5% | 0.099% | 0 | 0 |  |
| paper23_fault_guided | 31 | 31 | 31 | 28 | 3 | 0 | 0 | 0.677 | 80.3% | n/a | 0 | 0 |  |

## `ph01_page8_panel3` (under-segmented; excluded from selection)

This panel is reported separately and was **not** used for parameter selection.
Its labels split into **167 skeleton paths from only 22 boundary objects**
(62 fragments from one object), and 108 paths are the same label pair (1,2);
the boundaries are strongly under-segmented and fragmented.

| item | value |
|---|---|
| paths | 85 |
| exact boundary objects | 22 |
| fit-eligible | 34 |
| fallback_short (too short / tiny closed) | 51 |
| ok / ok_reduced | 33 / 0 |
| fallback_constraint | 1 |
| cross intersections (exact -> selected) | 0 -> 0 |
| median mean / p95 displacement (smoothed paths, px) | 0.309 / 0.586 |
| median max-curvature reduction (smoothed paths) | 85.6% |

The panel therefore illustrates the failure mode of treating a pixel skeleton
as a geological curve: many 1–2px fragments cannot be smoothed, while the few
long boundaries do smooth normally.  Its metrics do not affect the global λ or
the aggregate selection tables.

## Failure cases and warnings

### Constraint fallbacks at selected λ

- `ph01_page8_panel3` path 73 (boundary 39, 3-5, L=55.5px): requested λ=1, used λ=0.015625; area_change_rel=-0.0481>0.02

At the selected setting the only constraint fallback is a small closed loop in
the under-segmented `ph01_page8_panel3` (boundary 39, L≈55px).  At the fixed
8px knot spacing its periodic basis has only 7 control coefficients; even the
unpenalised projection onto that basis changes its area by ≈2.6%, so no amount
of λ reduction can satisfy the 2% area corridor.  Returning exact geometry is
the intended resolution limit for a 55px loop, not a silent skip.

### Short/ineligible skeleton fragments (`fallback_short`)

- Total: **58** of **242** paths.
- By panel: `newimage_p2` 1, `ph01_page8_panel3` 51, `schematic_panel2` 6.
- These are mostly 1–2px skeleton edges produced by `trace_skeleton` at branch
  nodes, plus one tiny 9px closed loop.  They are genuine Layer 1 keep objects,
  but a cubic fit is undefined or meaningless at this size.  Returning the
  exact geometry is the conservative outcome; they are not removed.

### Paths whose max curvature increased after smoothing

Endpoint pinning is a hard positional constraint; at a junction corner the
cubic must transition from the pinned endpoint into the interior, which can
create a local boundary-layer curvature spike.  This is the main qualitative
failure of the method.

- `schematic_panel2` path 37 (boundary 34, 5-7, L=2819.9px, n=2451): max curvature 0.817 -> 1.903 (ok, λ_used=1, p95 disp=0.98, max disp=2.45)
- `schematic_panel2` path 24 (boundary 30, 3-7, L=7.7px, n=7): max curvature 2.374 -> 4.278 (ok, λ_used=1, p95 disp=0.98, max disp=1.03)
- `paper23_fig7a` path 13 (boundary 25, 5-8, L=260.0px, n=225): max curvature 0.887 -> 1.547 (ok, λ_used=1, p95 disp=1.87, max disp=2.51)
- `newimage_p4` path 4 (boundary 5, 0-3, L=14.3px, n=12): max curvature 2.282 -> 2.737 (ok, λ_used=1, p95 disp=1.37, max disp=1.37)
- `schematic_panel2` path 36 (boundary 33, 5-7, L=127.4px, n=106): max curvature 1.649 -> 1.795 (ok, λ_used=1, p95 disp=1.67, max disp=2.26)

### Largest displacements at selected λ

- `paper23_fault_guided` path 18 (boundary 23, 5-10, L=850.2px): mean/p95/max = 1.18/2.29/3.00px, λ_used=1
- `paper23_fault_guided` path 16 (boundary 20, 5-6, L=593.3px): mean/p95/max = 0.90/2.33/2.95px, λ_used=1
- `paper23_fig7a` path 15 (boundary 25, 5-8, L=299.7px): mean/p95/max = 1.34/2.48/2.89px, λ_used=1
- `newimage_p3` path 0 (boundary 1, 0-1, L=1459.7px): mean/p95/max = 0.52/1.34/2.87px, λ_used=1
- `paper23_fault_guided` path 17 (boundary 21, 5-8, L=761.7px): mean/p95/max = 1.07/2.08/2.74px, λ_used=0.5
- `paper23_fig7a` path 5 (boundary 16, 1-7, L=573.1px): mean/p95/max = 0.92/2.11/2.74px, λ_used=1
- `paper23_fault_guided` path 19 (boundary 24, 5-11, L=87.2px): mean/p95/max = 0.95/2.18/2.70px, λ_used=1
- `paper23_fault_guided` path 13 (boundary 17, 3-7, L=1713.8px): mean/p95/max = 0.89/1.90/2.69px, λ_used=0.5
- `paper23_fig7a` path 21 (boundary 30, 6-12, L=449.7px): mean/p95/max = 1.24/2.17/2.69px, λ_used=1
- `paper23_fault_guided` path 14 (boundary 18, 3-9, L=624.2px): mean/p95/max = 1.10/2.11/2.62px, λ_used=0.5

### Closed-body area changes

All closed paths at selected λ (including exact fallbacks; sorted by |relative
area change|).  Closed bodies are periodic-smoothed, and a fit is rejected if
it exceeds the 2% area tolerance.

- `newimage_p2` path 2 (boundary 7, 0-1, L=357.1px): area 4490.9 -> 4471.8px² (-0.43%), status=ok, λ_used=1.0
- `schematic_panel2` path 20 (boundary 28, 3-7, L=934.5px): area 17627.0 -> 17609.6px² (-0.10%), status=ok, λ_used=1.0
- `newimage_p1` path 0 (boundary 1, 0-1, L=1272.9px): area 56525.4 -> 56506.7px² (-0.03%), status=ok, λ_used=1.0
- `newimage_p3` path 0 (boundary 1, 0-1, L=1459.7px): area 56513.7 -> 56511.6px² (-0.00%), status=ok, λ_used=1.0
- `newimage_p2` path 1 (boundary 6, 0-1, L=9.1px): area 5.4 -> 5.4px² (+0.00%), status=fallback_short, λ_used=n/a
- `ph01_page8_panel3` path 73 (boundary 39, 3-5, L=55.5px): area 113.4 -> 113.4px² (+0.00%), status=fallback_constraint, λ_used=0.015625
- `schematic_panel2` path 13 (boundary 21, 3-7, L=3.4px): area 0.5 -> 0.5px² (+0.00%), status=fallback_short, λ_used=n/a

### Largest length changes

- `schematic_panel2` path 24 (boundary 30, L=7.7px): length 7.4 -> 3.1px (-58.14%), status=ok
- `newimage_p4` path 4 (boundary 5, L=14.3px): length 13.9 -> 7.4px (-46.87%), status=ok
- `paper23_fig7a` path 15 (boundary 25, L=299.7px): length 290.8 -> 219.6px (-24.48%), status=ok
- `paper22_fig6` path 5 (boundary 11, L=91.5px): length 88.3 -> 69.6px (-21.23%), status=ok
- `paper23_fig7a` path 17 (boundary 26, L=234.7px): length 227.5 -> 180.1px (-20.85%), status=ok
- `paper23_fig7a` path 21 (boundary 30, L=449.7px): length 435.9 -> 346.9px (-20.42%), status=ok
- `paper23_fig7a` path 6 (boundary 17, L=461.7px): length 449.5 -> 361.1px (-19.68%), status=ok
- `paper23_fig7a` path 16 (boundary 26, L=128.3px): length 124.4 -> 101.2px (-18.62%), status=ok

## Interpretation / sensitivity

- The grid shows a monotone trade-off: λ = 0.01 → 10 moves median p95
  displacement from 0.590 to
  1.079px and median max-curvature reduction
  from 66.0% to
  91.7%.
- λ = 1 is the strongest setting that keeps the aggregate p95 displacement
  distribution inside the selection corridor (p90 p95 =
  1.780px) with only
  3 path(s) needing reduction.  λ = 3 already pushes the
  p90 p95 to 2.121px
  and 12
  paths require reduction; λ = 10 triggers reductions on
  43 paths.
- Most of the curvature gain is already obtained at λ = 0.1–0.3
  (p95-curvature reduction
  91.1%);
  larger λ mainly shortens and simplifies the curves, not the noise scale.
- The method is conservative by construction: corridor + iterative reduction
  guarantee no accepted path moves more than 3px, but the corridor
  also caps how much a sharp junction can be rounded.  When the exact centerline
  has a corner exactly at a pinned endpoint, the spline's transition can raise
  the reported max curvature even though the p95 curvature drops.
- Cross-intersections do not increase in this run
  (0 → 0); smoothing each path
  independently with pinned endpoints preserved the junction topology at the
  resolution of the validator.

## Outputs

- `runs/layer2_experiments/a_segmented_spline/metrics.csv` — 1936 rows
  (required columns followed by diagnostic columns).
- `runs/layer2_experiments/a_segmented_spline/panel_metrics.csv` — 88 rows
  (per panel × variant, including cross-intersections).
- `runs/layer2_experiments/a_segmented_spline/summary.md` — this file.
- `runs/layer2_experiments/a_segmented_spline/contact.png` — 11 rows,
  recoloured labels + exact thin white + smoothed thick green.
- `runs/layer2_experiments/a_segmented_spline/panels/<panel_id>.png` — full-resolution
  per-panel overlays at the selected setting.

## Sanity checks

- Keep paths in manifest: **242** from **158** exact boundary objects (skeleton fragmentation creates repeated boundary ids).
- Rows in `metrics.csv`: **1936** = 242 paths × 8 variants (exact + 7 λ settings).
- Every variant contains one record for every keep path: **yes**.
- Status counts sum to path counts in every panel row: **yes**.
- Runtime errors at selected setting: **0**; across all variants: **0**.
- Fit-eligible paths: **184**; short/ineligible returned exact and marked `fallback_short`: **58**.
- No sanity problems detected.
