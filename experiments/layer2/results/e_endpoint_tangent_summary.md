# Layer 2 experiment E — endpoint tangent constraints

Run: 2026-10-21 13:53:17 local
Command: `uv run python experiments/layer2/e_endpoint_tangent/run.py`
Runtime: 280.2s
Repo root: `/Users/daiduo2/geoseg`

## Question and method

Experiment A pins both endpoints of every open path and fits a cubic P-spline.  Its reported max-curvature increases were attributed to boundary-layer spikes at pinned junction corners.  This experiment tests whether a soft endpoint **tangent** penalty removes those spikes while keeping the hard endpoint positions, the 3 px corridor, and the topology guarantees.

Each open path is fitted independently with the same basis as A:

```text
min_c  ||B c - y||^2
       + lambda * (||D2 c||^2 + 0.25 ||D3 c||^2)
       + beta * (||f'(0) - t_start||^2 + ||f'(1) - t_end||^2)
```

- Cubic clamped B-spline, knots every **8 px**, lambda = **1** (A's selected value, frozen), alpha3 = **0.25**.
- The tangent rows are added to the normal equations (`beta * d0^T d0`), using the exact B-spline derivative basis.  The curve is not rotated or post-processed.
- First and last coefficients are eliminated and fixed to the exact centerline endpoints, so **hard endpoint positions remain exact**.
- Corridor **3 px**; closed area |ΔA|/A ≤ **0.02**; endpoint drift ≤ **0.001 px**; self-intersections must be 0.  On violation lambda is halved up to **6** times; if it still fails, exact geometry is returned and marked `fallback_constraint`.
- Eligibility: ≥4 points, ≥6 px, closed area ≥25 px².  Shorter fragments are returned exact and marked `fallback_short`.

## Endpoint tangent estimator

- Window: **10 px** of exact centerline from the endpoint (within the requested 5–12 px range).
- Total-least-squares line through the endpoint (PCA of endpoint-relative offsets), trimmed 3 times at `median + 2.5*MAD` with a 0.75 px floor, then re-fitted.
- Oriented into the curve.  For the end, the forward derivative target is the **negative** of the into-curve vector; this sign is essential.
- Skipped when the endpoint is within 1.5 px of the panel border, when the window is shorter than 5 px, when fewer than 4 points or less than 4 px projected support remain, or when the line-fit eigenvalue ratio is below 2.
- A skipped end contributes no tangent row; the other end still does.

Endpoint support counts (eligible paths, both ends):

| status | endpoints |
|---|---|
| applied | 306 |
| border | 50 |
| closed | 10 |
| weak_support | 2 |

## Beta grid and selection

The grid was evaluated on **non-ph01 smoothable paths** (A's eligible set), with one global beta.  The rule was fixed before inspecting grid outcomes:

1. zero fit errors, zero constraint fallbacks, zero self-intersections, zero panel cross-intersections, endpoint drift ≤ 0.001 px;
2. median p95 displacement ≤ hard_only median + 0.01 px, and p90 p95 displacement ≤ hard_only p90 + 0.01 px;
3. median endpoint-region max curvature ≤ 98% of hard_only, and p90 endpoint-region max curvature ≤ 102% of hard_only.

Endpoint region = first/last `min(15 px, 0.5*L)`.  A wider `max(15 px, 10% of L)` diagnostic is also reported as `endpoint_wide_max_curv` in `metrics.csv`; selection uses the fixed 15 px region.  Displacement stats use all eligible non-ph01 paths; endpoint-curvature stats use eligible open paths.  Among passing settings, the **largest beta** wins (same preference for the strongest accepted constraint as experiment A).

| beta | smoothed | reduced | fb constraint | med mean | med p95 | p90 p95 | med max | med endpoint max | p90 endpoint max | med endpoint red. | med tangent err (deg) | self-int | cross-int | selection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 150 | 3 | 0 | 0.423 | 0.801 | 1.780 | 1.412 | 0.0690 | 0.2285 | 0.0% | 8.29 | 0 | 0 | baseline |
| 0.01 | 150 | 3 | 0 | 0.423 | 0.801 | 1.780 | 1.412 | 0.0690 | 0.2285 | 0.0% | 8.29 | 0 | 0 | fail: median_endpoint_curv=0.0690>0.0677 |
| 0.1 | 150 | 3 | 0 | 0.423 | 0.801 | 1.780 | 1.413 | 0.0690 | 0.2282 | 0.1% | 8.28 | 0 | 0 | fail: median_endpoint_curv=0.0690>0.0677 |
| 1 | 150 | 3 | 0 | 0.423 | 0.803 | 1.781 | 1.422 | 0.0684 | 0.2270 | 1.0% | 8.15 | 0 | 0 | fail: median_endpoint_curv=0.0684>0.0677 |
| 10 | 150 | 3 | 0 | 0.424 | 0.806 | 1.788 | 1.465 | 0.0632 | 0.2161 | 8.5% | 6.92 | 0 | 0 | **selected** |
| 100 | 150 | 3 | 0 | 0.429 | 0.838 | 1.809 | 1.499 | 0.0461 | 0.1790 | 33.2% | 2.86 | 0 | 0 | fail: median_p95=0.838>0.801+0.01; p90_p95=1.809>1.780+0.01 |

**Selected: beta = 10** (largest passing beta); the fit is exposed as `tangent_soft`.

## Aggregate results and comparison

Non-ph01: **150** eligible paths (146 open, 4 closed).  Values are final outputs; fallback paths are exact geometry.

| variant | med mean disp | med p95 disp | p90 p95 disp | med max disp | p95 max disp | med endpoint max | p90 endpoint max | med endpoint p95 | med tangent err | med max curv | med max-curv red. | self-int | cross-int | max drift | max |ΔA|/A | fallbacks | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hard_only | 0.423 | 0.801 | 1.780 | 1.412 | 2.691 | 0.0690 | 0.2285 | 0.0687 | 8.29 | 0.145 | 0.840 | 0 | 0 | 0.000 | 0.00426 | 0 | 0 |
| tangent_soft (β=10) | 0.424 | 0.806 | 1.788 | 1.465 | 2.711 | 0.0632 | 0.2161 | 0.0629 | 6.92 | 0.128 | 0.856 | 0 | 0 | 0.000 | 0.00426 | 0 | 0 |
| tangent_soft_guard (β=10) | 0.424 | 0.806 | 1.788 | 1.465 | 2.711 | 0.0632 | 0.2161 | 0.0629 | 6.92 | 0.128 | 0.856 | 0 | 0 | 0.000 | 0.00426 | 0 | 0 |

Endpoint-region max curvature improved on **112** open paths, worsened on **17**, unchanged on **17** (guard: 112/17/17).  Median of per-path relative changes: **-8.0%** (the grid's 8.5% is the change of the aggregate medians, not the median of per-path changes).

Longer-path sensitivity (`L >= 30 px`, n=139): median endpoint max 0.0695 → 0.0633 (9.0% lower), p90 0.2257 → 0.2067.  The improvement is not an artefact of the smallest paths.

Cross-intersections (non-ph01 panels): exact **0**, hard **0**, tangent **0**, guard **0**.  Self-intersections: **0** at all settings.  Maximum open endpoint drift: **4.687e-13 px**.

Closed-body maximum |ΔA|/A over all variants: **0.00426** (0.43%; tolerance 0.02).  Width-guard fallbacks: **0**.
The closed-body width guard did not trigger; `tangent_soft_guard` returns the same geometry and status as `tangent_soft` on all 242 paths.  Open-path metric rows are byte-identical (verified row-by-row, 0 differing rows); closed rows add the width diagnostics.

## Per-panel results

| panel | paths | objects | eligible | ok/red | fb short | fb constraint | fb width | cross exact→hard→soft→guard | med p95 hard→soft | med endpoint max hard→soft | med tangent err soft | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| newimage_p0 | 3 | 3 | 3 | 3/0 | 0 | 0 | 0 | 0→0→0→0 | 1.103→1.098 | 0.068→0.063 | 10.96 |  |
| newimage_p1 | 8 | 8 | 8 | 8/0 | 0 | 0 | 0 | 0→0→0→0 | 0.709→0.710 | 0.069→0.065 | 6.06 |  |
| newimage_p2 | 8 | 7 | 7 | 7/0 | 1 | 0 | 0 | 0→0→0→0 | 0.743→0.724 | 0.116→0.103 | 12.35 |  |
| newimage_p3 | 4 | 4 | 4 | 4/0 | 0 | 0 | 0 | 0→0→0→0 | 0.837→0.849 | 0.122→0.110 | 12.05 |  |
| newimage_p4 | 7 | 5 | 7 | 7/0 | 0 | 0 | 0 | 0→0→0→0 | 0.841→0.847 | 0.156→0.143 | 9.99 |  |
| paper22_fig5a | 13 | 13 | 13 | 13/0 | 0 | 0 | 0 | 0→0→0→0 | 0.590→0.594 | 0.041→0.038 | 4.10 |  |
| paper22_fig6 | 9 | 7 | 9 | 9/0 | 0 | 0 | 0 | 0→0→0→0 | 0.835→0.847 | 0.030→0.029 | 3.39 |  |
| ph01_page8_panel3 | 85 | 22 | 34 | 33/0 | 51 | 1 | 0 | 0→0→0→0 | 0.586→0.604 | 0.047→0.041 | 4.05 | excluded from selection (under-segmented) |
| paper23_fig7a | 34 | 31 | 34 | 34/0 | 0 | 0 | 0 | 0→0→0→0 | 0.657→0.678 | 0.072→0.064 | 7.51 |  |
| schematic_panel2 | 40 | 27 | 34 | 34/0 | 6 | 0 | 0 | 0→0→0→0 | 0.979→0.978 | 0.069→0.068 | 6.53 |  |
| paper23_fault_guided | 31 | 31 | 31 | 28/3 | 0 | 0 | 0 | 0→0→0→0 | 0.677→0.670 | 0.076→0.068 | 8.35 |  |

`ph01_page8_panel3` is reported but never used to select beta.

## Endpoint-spike cases and the main finding

The table reports A's cited cases plus the strongest baseline endpoint regions.  `contract max` is A's `boundary_metrics` max curvature; `dense max` is the analytic B-spline curvature on a dense grid.  `at end dist` is the distance from the dense maximum to the nearer endpoint, so interior maxima are visible.

| case | L | contract max hard→soft | dense max hard→soft | at end dist (px) | start region hard→soft | end region hard→soft | tangent start/end | soft tangent err | guard status | tag |
|---|---|---|---|---|---|---|---|---|---|---|
| schematic_panel2 p37 (b34, 5-7) | 2819.9px | 1.903→1.924 | 12.440→12.440 | 312.0 | 0.069→0.070 | 0.142→0.142 | applied/border | 7.58 | ok | A case |
| paper23_fig7a p13 (b25, 5-8) | 260.0px | 1.547→1.526 | 1.659→1.659 | 128.6 | 0.128→0.118 | 0.008→0.009 | applied/applied | 10.49 | ok | A case |
| schematic_panel2 p36 (b33, 5-7) | 127.4px | 1.795→1.791 | 2.013→2.014 | 22.6 | 0.068→0.068 | 0.053→0.052 | border/applied | 2.00 | ok | A case |
| schematic_panel2 p24 (b30, 3-7) | 7.7px | 4.278→4.278 | 5.176→5.176 | 3.6 | 4.879→4.879 | 5.176→5.176 | weak_support/weak_support | n/a | ok | A case |
| newimage_p4 p4 (b5, 0-3) | 14.3px | 2.737→2.645 | 2.705→2.595 | 6.4 | 2.705→2.595 | 2.705→2.595 | applied/applied | 2.26 | ok | A case |
| schematic_panel2 p12 (b21, 3-7) | 574.2px | 1.267→1.322 | 1.930→1.930 | 154.6 | 0.073→0.064 | 0.803→0.725 | applied/applied | 11.67 | ok | baseline spike |
| paper23_fig7a p18 (b27, 5-11) | 43.6px | 0.642→0.684 | 0.665→0.601 | 5.0 | 0.006→0.006 | 0.665→0.601 | applied/applied | 7.12 | ok | baseline spike |
| newimage_p4 p1 (b3, 0-3) | 350.5px | 0.476→0.476 | 0.452→0.452 | 5.8 | 0.452→0.452 | 0.000→0.000 | border/applied | 0.00 | ok | baseline spike |
| paper23_fig7a p21 (b30, 6-12) | 449.7px | 0.442→0.414 | 0.446→0.405 | 5.8 | 0.446→0.405 | 0.165→0.147 | applied/applied | 13.98 | ok | baseline spike |
| newimage_p4 p3 (b5, 0-3) | 711.9px | 0.536→0.531 | 0.594→0.594 | 146.6 | 0.153→0.154 | 0.432→0.455 | applied/applied | 9.99 | ok | baseline spike |

**Main finding.**  The two cited A failures are not endpoint spikes.  For `schematic_panel2` p37 the dense maximum is an interior hairpin: the exact centerline reverses direction over ~2 px and the spline rounds it at a radius well below one pixel; the max is hundreds of pixels from either endpoint.  For `paper23_fig7a` p13 the maximum is also at mid-path.  For the cited failures, the endpoint regions stay near 0.07–0.14 px⁻¹ in hard_only, and the tangent penalty changes them only modestly.  It cannot, and does not, move the interior hairpin.  A's `max_curvature` undersamples long paths (at most 1000 samples), which is why the contract value (1.903) is smaller than the dense value (~12.4).  The right remedy for the interior maxima is path splitting/curvature-capping or a different interior model, not endpoint tangents.

## Largest endpoint-region improvements and regressions

| case | L | hard endpoint max | soft endpoint max | relative change | tangent start/end |
|---|---|---|---|---|---|
| newimage_p4 p4 (b5) | 14.3px | 2.7050 | 2.5947 | -4.1% | applied/applied |
| newimage_p2 p7 (b14) | 283.2px | 0.3125 | 0.2253 | -27.9% | applied/applied |
| schematic_panel2 p12 (b21) | 574.2px | 0.8028 | 0.7250 | -9.7% | applied/applied |
| paper23_fig7a p18 (b27) | 43.6px | 0.6650 | 0.6012 | -9.6% | applied/applied |
| paper23_fig7a p21 (b30) | 449.7px | 0.4465 | 0.4055 | -9.2% | applied/applied |
| paper23_fault_guided p19 (b24) | 87.2px | 0.3337 | 0.3018 | -9.5% | applied/applied |
| newimage_p4 p5 (b7) | 1086.7px | 0.1395 | 0.1090 | -21.9% | applied/applied |
| paper23_fault_guided p5 (b6) | 306.8px | 0.2309 | 0.2068 | -10.4% | applied/applied |

Worst regressions at the selected beta:

| case | L | hard endpoint max | soft endpoint max | relative change | endpoint flips hard→soft | tangent start/end |
|---|---|---|---|---|---|---|
| newimage_p4 p3 (b5) | 711.9px | 0.4323 | 0.4552 | 5.3% | 0→0 | applied/applied |
| paper23_fault_guided p14 (b18) | 624.2px | 0.1464 | 0.1541 | 5.3% | 2→2 | applied/applied |
| schematic_panel2 p11 (b20) | 672.1px | 0.1383 | 0.1406 | 1.7% | 0→0 | applied/applied |
| newimage_p1 p1 (b3) | 774.1px | 0.1075 | 0.1091 | 1.5% | 1→1 | applied/applied |
| paper23_fault_guided p2 (b3) | 405.1px | 0.2256 | 0.2289 | 1.5% | 1→1 | applied/applied |
| paper23_fault_guided p8 (b11) | 241.8px | 0.1582 | 0.1600 | 1.1% | 1→1 | applied/applied |
| newimage_p0 p0 (b1) | 1378.9px | 0.0626 | 0.0633 | 1.0% | 1→1 | applied/applied |
| newimage_p3 p1 (b3) | 940.8px | 0.0044 | 0.0044 | 0.9% | 1→1 | applied/applied |
| schematic_panel2 p15 (b23) | 318.6px | 0.0307 | 0.0309 | 0.7% | 0→0 | applied/applied |
| paper22_fig5a p3 (b4) | 216.9px | 0.0235 | 0.0236 | 0.6% | 1→1 | applied/border |

## Failure cases and warnings

### Beta-grid failures

- beta=0.01: median_endpoint_curv=0.0690>0.0677
- beta=0.1: median_endpoint_curv=0.0690>0.0677
- beta=1: median_endpoint_curv=0.0684>0.0677
- beta=100: median_p95=0.838>0.801+0.01; p90_p95=1.809>1.780+0.01

### Selected-setting warnings

- Endpoint curvature worsened on 17 open paths; the worst are listed above.  This is a real local cost of the tangent penalty, not hidden.
- Oscillation suspects (endpoint curvature sign flips increased by ≥2 and endpoint max rose >0.001): **0**.
  - None under this criterion.  Dense curvature sign flips and self-intersections were still checked on every path.
- Constraint fallbacks on non-ph01 smoothable paths at the selected setting: **0**.
- Width-guard fallbacks: **0**.
- Border endpoints have no tangent target.  In the cited anchor cases, `schematic_panel2` p37 ends on the bottom panel border and `schematic_panel2` p36 starts on the top border, so only one end of each can be constrained.
- Runtime/cross-intersection errors: none.
- Render errors: none.

## `ph01_page8_panel3` (under-segmented; excluded from selection)

- 85 paths, 22 exact boundary objects, 34 fit-eligible, 51 `fallback_short`.
- hard: ok/reduced 33/0, fallback constraint 1; median p95 0.586 px, median endpoint max 0.0469.
- soft: ok/reduced 33/0, fallback constraint 1; median p95 0.604 px, median endpoint max 0.0414.
- cross-intersections: exact/hard/soft/guard 0/0/0/0.
- Its metrics are reported and checked, but it did not contribute to beta selection.

## Baseline parity with experiment A

Compared 242 of 242 A `lam1` rows by `(panel_id, path_index)`; missing: 0; status mismatches: 0.

| field | max |hard_only - A lam1| |
|---|---|
| mean_disp | 4.97558e-07 |
| p95_disp | 4.95963e-07 |
| max_disp | 4.9861e-07 |
| max_curvature | 4.93504e-07 |
| p95_curvature | 4.99487e-07 |
| endpoint_drift | 4.68743e-13 |
| self_intersections | 0 |
| length_smooth | 4.96516e-07 |
| area_change | 4.10789e-07 |
| lambda_used | 0 |

A's CSV is rounded to 6 decimals; differences at the 1e-6 level are formatting, not method differences.  Contract fields and statuses match A's selected lambda=1 baseline.

## Sanity checks

| check | value | result |
|---|---|---|
| keep paths | 242 (expected 242) | pass |
| exact boundary objects | 158 (expected 158) | pass |
| fit-eligible paths | 184 (expected 184) | pass |
| short/ineligible paths | 58 (expected 58) | pass |
| non-ph01 eligible | 150 (expected 150) | pass |
| metrics rows | 1936 = 242 x 8 | pass |
| one row per path per variant | True | pass |
| status totals sum to rows | 1936 | pass |
| open endpoint drift <= 1e-3 px | 4.687e-13 | pass |
| self-intersections total | 0 | pass |
| cross-intersections (exact) | 0 | pass |
| cross-intersections (hard) | 0 | pass |
| cross-intersections (tangent) | 0 | pass |
| cross-intersections (guard) | 0 | pass |
| closed |area change| max | 0.00426 | pass |
| selected constraint fallbacks | 0 | pass |
| width guard fallbacks | 0 | pass |
| fit errors | 0 | pass |
| open guard identity | 0 differing rows | pass |
| tangent_soft alias identity | 0 differing rows | pass |
| runtime errors | 0 | pass |
| render errors | 0 | pass |

Status totals across all variants: `fallback_constraint` 8, `fallback_short` 464, `ok` 1440, `ok_reduced` 24.

## Interpretation and answer

1. **Does the tangent constraint improve endpoint curvature?** At the selected beta=10, median endpoint-region max curvature moves 0.0690 → 0.0632 (8.5% lower), with 112 paths improved and 17 worsened.  The median tangent error drops from 8.29° to 6.92°.  So yes, it improves endpoint curvature and tangent alignment, but the gain is modest and not universal.
2. **Does it preserve fidelity and topology?** Median p95 displacement moves 0.8011 → 0.8062 px and p90 p95 1.7800 → 1.7882 px.  Self-intersections are 0, cross-intersections are 0, endpoint drift is ≤4.7e-13 px, closed |ΔA|/A ≤ 0.00426 (0.43%), and there are 0 constraint fallbacks on non-ph01 paths at the selected setting.  Fidelity and topology are preserved within the pre-registered tolerance.
3. **Which beta is safest?** beta=10 is the largest grid value that passes the pre-registered fidelity/topology rule, so it is the recommended operating point.  beta=100 gives a larger endpoint reduction (median 0.0461) but its median p95 displacement is 0.8382 px vs 0.8011 px at baseline and it fails: median_p95=0.838>0.801+0.01; p90_p95=1.809>1.780+0.01; it is the least safe of the tested values.  beta=0.01/0.1 are nearly a no-op; beta=1 reduces the median by only 1.0% (below the 2% rule).  If strict fidelity is the priority, beta=0.1 is the conservative choice.
4. **Does it fix experiment A's main failure?** Not the cited max-curvature failures.  The dense maxima are interior: `schematic_panel2` p37: 12.440 → 12.440 (maximum 312.0 px from the nearer endpoint); `paper23_fig7a` p13: 1.659 → 1.659 (128.6 px from the nearer endpoint).  A's contract metric (at most 1000 samples) reported 1.903 and 1.547, but the true maxima are interior hairpins far from either endpoint.  The tangent penalty changes endpoint regions, not those interior maxima.  It is a valid endpoint-shape tool, not a cure for interior path hairpins.  That is the key negative result of this experiment.

Recommended next step for the interior failures: detect high-turn hairpins/spurs in the exact centerline and split or simplify them before fitting, or add an interior curvature cap/short-segment model.  More pruning of short junction edges is unlikely to help (the path graph v2 result already showed that the smoothable set and its metrics did not change).

## Outputs

- `/Users/daiduo2/geoseg/runs/layer2_experiments/e_endpoint_tangent/metrics.csv` — 11 panels × 242 paths × 8 variants (grid betas, hard_only, selected alias and guard; see the file).
- `/Users/daiduo2/geoseg/runs/layer2_experiments/e_endpoint_tangent/panel_metrics.csv` — per panel × variant, including cross-intersections and status counts.
- `/Users/daiduo2/geoseg/runs/layer2_experiments/e_endpoint_tangent/summary.md` — this file.
- `/Users/daiduo2/geoseg/runs/layer2_experiments/e_endpoint_tangent/contact.png` — 11 panel rows, hard_only (left) vs tangent_soft_guard (right), recoloured mask + exact thin white + smoothed thick green.
- `/Users/daiduo2/geoseg/runs/layer2_experiments/e_endpoint_tangent/panels/<panel_id>.png` — full-resolution guard overlays for every panel.
- `/Users/daiduo2/geoseg/runs/layer2_experiments/e_endpoint_tangent/endpoint_zoom.png` — 8 zoomed endpoint comparisons (anchor cases, baseline spikes, worst regressions).

## Runtime and commands

From `/Users/daiduo2/geoseg`:

```bash
uv run python experiments/layer2/e_endpoint_tangent/run.py
```

Total runtime: **280.2s**.

| stage | time |
|---|---|
| build | 35.2s |
| guard | 28.7s |
| grid_fit | 214.7s |
| metrics_rows | 0.0s |
| render | 1.5s |

No per-panel or per-path tuning: lambda=1 and the selected beta are global; only the iterative lambda fallback changes a path's effective lambda when a constraint fails.
