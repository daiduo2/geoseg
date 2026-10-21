# Layer 2 experiment B: smoothing-kernel benchmark

All five kernels run on the **same Layer 1 keep set**, the same arc-length resampling, the same open/closed convention, and the same post-fit endpoint reset.  No image diffusion, level set, graph cut or re-segmentation is used.

## Setup

- Panels: **11**, keep paths: **242** (235 open, 7 closed).
- Resampling: uniform arc length, `round(length)` samples clamped to [8, 1200] (about 1 sample/pixel).  Open paths include both endpoints; closed paths are periodic and do not duplicate the endpoint.
- Paths shorter than **8 px** cannot be smoothed meaningfully; they are passed through and marked `short_passthrough` (58 paths).  They remain in every table; nothing is silently skipped.
- Open-path endpoints are reset to the exact resampled endpoints after every variant.  `splprep` additionally uses endpoint weights 100; `savgol` uses edge padding/mode `interp`; `knot_quintic` uses truncated edge windows.  Closed paths use periodic/wrap handling and are never endpoint-pinned.
- `knot_quintic` is a sliding **quintic local regression on x(s), y(s)**.  A true y(x) quintic smoother is not applicable to arbitrary curves (closed loops, overhangs, vertical segments), so this is the explicit arc-length equivalent.  It is not a y(x) method.
- Metric note: the harness `displacement` is nearest-vertex distance, not point-to-segment distance, and the exact/smoothed curves are resampled at different densities.  The `exact` rows therefore have a small non-zero displacement baseline (typically a few tenths of a pixel).  This offset is common to all kernels and the exact rows are retained in `metrics.csv`; compare kernels against each other and against the exact row, not against a literal zero.
- Structural constants: `bspline_lsq` knot spacing 24 px; `pspline_diff2` knot spacing 4 px; Savitzky-Golay order 3; quintic degree 5.

## Global parameter selection

One parameter per kernel was selected on a single global grid pooled over all **non-`ph01_page8_panel3`** smoothable paths (150 paths).  The selection rule is global and data-independent per panel: among configs with mean path p95 displacement <= 1.5 px, p95 of path max displacement <= 4.5 px and no error fallback, with at most 2% of paths degenerate-restored; choose the one with the largest mean p95 curvature reduction.  Length change is reported as an outcome rather than used as a selection gate, because percentage changes on very short paths would dominate.  If no config is inside the corridor, choose the one closest to it.  The selected value is then frozen for all 11 panels.  `ph01_page8_panel3` is excluded from selection and reported separately below.

| kernel | selected parameter | eligible | selection rule | final mean p95 disp | final curv. reduction |
| --- | --- | --- | --- | --- | --- |
| splprep | s_per_point=0.8 | 6/8 | max curvature reduction inside displacement corridor (<= 3 fallbacks allowed) | 1.376 | 0.866 |
| bspline_lsq | lambda=100;knot=24px | 10/10 | max curvature reduction inside displacement corridor (<= 3 fallbacks allowed) | 1.197 | 0.861 |
| pspline_diff2 | lambda=100;knot=4px | 9/10 | max curvature reduction inside displacement corridor (<= 3 fallbacks allowed) | 1.360 | 0.847 |
| savgol | window=61;order=3 | 7/9 | max curvature reduction inside displacement corridor (<= 3 fallbacks allowed) | 1.342 | 0.831 |
| knot_quintic | half_window=64;degree=5 | 8/9 | max curvature reduction inside displacement corridor (<= 3 fallbacks allowed) | 1.261 | 0.860 |

The values in the last two columns are measured on the **ranking pool**: non-`ph01_page8_panel3` paths whose fit status is `ok`.  This is the same population used for parameter selection.  The all-panel/all-path table below additionally includes short passthrough fragments, which are identical for every kernel and dilute absolute displacement/curvature values without changing the ordering.

## Aggregate ranking

Lower is better for displacement, endpoint drift, topology violations, area change and runtime; higher is better for curvature reduction.  Topology violations count increases in cross-intersections plus increases in self-intersections over the exact skeleton.  Endpoint drift is zero for every kernel by construction (endpoints are pinned), so that criterion is a tie by design rather than a measured advantage.

| criterion | ranking (best to worst) |
| --- | --- |
| Displacement | bspline_lsq (1.197) > knot_quintic (1.261) > savgol (1.342) > pspline_diff2 (1.360) > splprep (1.376) |
| Curvature reduction | splprep (0.866) > bspline_lsq (0.861) > knot_quintic (0.860) > pspline_diff2 (0.847) > savgol (0.831) |
| Endpoint drift | splprep (0.000) > bspline_lsq (0.000) > pspline_diff2 (0.000) > savgol (0.000) > knot_quintic (0.000) |
| Topology violations | bspline_lsq (0) > knot_quintic (0) > savgol (2) > splprep (4) > pspline_diff2 (7) |
| Area change (closed) | splprep (0.13) > bspline_lsq (0.21) > knot_quintic (0.31) > savgol (0.32) > pspline_diff2 (0.48) |
| Runtime | bspline_lsq (0.04) > savgol (0.04) > splprep (0.06) > pspline_diff2 (0.18) > knot_quintic (0.48) |

## Interpretation

- **Most conservative:** `bspline_lsq` has the smallest mean path p95 displacement (1.197 px) at its selected global setting.  It is the least disruptive to the Layer 1 geometry.
- **Smooths most:** `splprep` has the largest mean p95 curvature reduction (0.866) at a mean p95 displacement of 1.376 px.  Aggressive settings were allowed by the common displacement corridor, so this is a smoothing-vs-fidelity trade-off, not a free win.
- **Area preservation (non-ph01 pool):** `splprep` has the smallest mean absolute closed-body area change (0.13 %), while the largest is `pspline_diff2` (0.48 %).  This pool excludes ph01, whose large closed loop is a much more severe area failure (see the special-panel and failure-case sections): the closed-body ranking is therefore not a global area guarantee.
- **Topology:** `bspline_lsq` has the fewest added cross/self intersections (0); the worst is `pspline_diff2` (7).
- **Runtime:** on the ranking pool, `bspline_lsq` is fastest at 0.04 s; `knot_quintic` is slowest at 0.48 s.  Runtime is measured for fitting only, not for metric or rendering work.
- **Endpoint drift:** every kernel reports exactly zero because open endpoints are pinned after fitting.  This metric checks the harness, not the kernel; the meaningful endpoint question is shape distortion just inside the ends, visible in the composite images.

## Per-variant aggregate (all 11 panels, all paths)

| kernel | paths | measured | short | degen | mean p95 disp | mean disp | max disp | curv. reduction | max curv. ratio | \|len change\| % | endpoint drift | self-int | cross delta | \|area\| % | runtime s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| splprep | 242 | 241 | 58 | 1 | 0.970 | 0.443 | 5.070 | 0.815 | 0.298 | 3.87 | 0.0000 | 0 | 4 | 1.04 | 0.06 |
| bspline_lsq | 242 | 241 | 58 | 1 | 0.852 | 0.412 | 4.748 | 0.791 | 5.063 | 4.20 | 0.0000 | 0 | 2 | 1.79 | 0.04 |
| pspline_diff2 | 242 | 241 | 58 | 1 | 0.994 | 0.456 | 4.816 | 0.745 | 3.753 | 4.34 | 0.0000 | 0 | 11 | 15.94 | 0.19 |
| savgol | 242 | 241 | 58 | 1 | 0.973 | 0.455 | 8.422 | 0.774 | 2.257 | 4.75 | 0.0000 | 2 | 5 | 11.82 | 0.05 |
| knot_quintic | 242 | 241 | 58 | 1 | 0.895 | 0.417 | 9.112 | 0.792 | 0.805 | 4.33 | 0.0000 | 0 | 4 | 13.96 | 0.53 |

`measured` excludes degenerate/error fallbacks from the metric means; `degen` counts paths where a fit collapsed the geometry and the exact polyline was restored.  `curv. reduction` is `1 - p95 curvature(smoothed)/p95 curvature(exact)` on the same resampled points (negative means the smoother added curvature).  `max curv. ratio` is the per-path `max_curvature` ratio averaged over paths; it is outlier-sensitive and is reported for completeness, while the p95 reduction is the more stable curvature summary.  `|area| %` is the mean absolute area change over closed paths with exact area >= 25 px^2; small degenerate loops are excluded from that average but remain in the CSV.

## Per-variant aggregate (ranking pool)

Non-`ph01_page8_panel3` smoothable paths with `status == ok`: **150 smoothable paths** per kernel, of which **149 are measured** after the shared tiny-loop degenerate fallback.  This is the population used for the ranking and interpretation above.  Cross-intersection deltas here exclude ph01; ph01 is tabulated separately.

| kernel | paths | mean p95 disp | mean disp | max disp | curv. reduction | \|len change\| % | self-int | cross delta | \|area\| % | runtime s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| splprep | 149 | 1.376 | 0.621 | 5.070 | 0.866 | 5.31 | 0 | 4 | 0.13 | 0.056 |
| bspline_lsq | 149 | 1.197 | 0.578 | 4.748 | 0.861 | 5.88 | 0 | 0 | 0.21 | 0.035 |
| pspline_diff2 | 149 | 1.360 | 0.626 | 4.816 | 0.847 | 5.86 | 0 | 7 | 0.48 | 0.183 |
| savgol | 149 | 1.342 | 0.627 | 8.422 | 0.831 | 6.42 | 2 | 0 | 0.32 | 0.043 |
| knot_quintic | 149 | 1.261 | 0.584 | 9.112 | 0.860 | 5.88 | 0 | 0 | 0.31 | 0.476 |

The all-path table and this ranking-pool table use the same per-path metrics; the difference is only the inclusion of short passthrough fragments and the special panel.  Rankings are reported on the pool because short fragments cannot be smoothed by any kernel and would otherwise dominate the average with identical zero-displacement rows.

## Per-panel displacement and topology

Mean path p95 displacement (px):

| panel | splprep | bspline_lsq | pspline_diff2 | savgol | knot_quintic |
| --- | --- | --- | --- | --- | --- |
| newimage_p0 | 1.698 | 1.446 | 1.605 | 1.549 | 1.619 |
| newimage_p1 | 1.079 | 0.998 | 1.084 | 1.064 | 1.022 |
| newimage_p2 | 1.376 | 1.136 | 1.260 | 1.250 | 1.245 |
| newimage_p3 | 1.455 | 1.159 | 1.434 | 1.269 | 1.269 |
| newimage_p4 | 1.193 | 1.183 | 1.346 | 1.383 | 1.206 |
| paper22_fig5a | 1.097 | 0.729 | 0.830 | 0.840 | 0.699 |
| paper22_fig6 | 1.284 | 0.961 | 1.139 | 1.004 | 0.907 |
| ph01_page8_panel3 | 0.336 | 0.313 | 0.431 | 0.402 | 0.322 |
| paper23_fig7a | 1.344 | 1.187 | 1.347 | 1.276 | 1.169 |
| schematic_panel2 | 1.180 | 1.198 | 1.358 | 1.372 | 1.351 |
| paper23_fault_guided | 1.585 | 1.259 | 1.429 | 1.461 | 1.351 |

Cross-intersection delta vs exact (smoothed minus exact):

| panel | splprep | bspline_lsq | pspline_diff2 | savgol | knot_quintic |
| --- | --- | --- | --- | --- | --- |
| newimage_p0 | 0 (+0) | 0 (+0) | 0 (+0) | 0 (+0) | 0 (+0) |
| newimage_p1 | 0 (+0) | 0 (+0) | 0 (+0) | 0 (+0) | 0 (+0) |
| newimage_p2 | 0 (+0) | 0 (+0) | 2 (+2) | 0 (+0) | 0 (+0) |
| newimage_p3 | 0 (+0) | 0 (+0) | 2 (+2) | 0 (+0) | 0 (+0) |
| newimage_p4 | 0 (+0) | 0 (+0) | 0 (+0) | 0 (+0) | 0 (+0) |
| paper22_fig5a | 0 (+0) | 0 (+0) | 0 (+0) | 0 (+0) | 0 (+0) |
| paper22_fig6 | 0 (+0) | 0 (+0) | 0 (+0) | 0 (+0) | 0 (+0) |
| ph01_page8_panel3 | 0 (+0) | 2 (+2) | 4 (+4) | 5 (+5) | 4 (+4) |
| paper23_fig7a | 0 (+0) | 0 (+0) | 0 (+0) | 0 (+0) | 0 (+0) |
| schematic_panel2 | 0 (+0) | 0 (+0) | 1 (+1) | 0 (+0) | 0 (+0) |
| paper23_fault_guided | 4 (+4) | 0 (+0) | 2 (+2) | 0 (+0) | 0 (+0) |

## Special panel: `ph01_page8_panel3`

This panel is known under-segmented.  It contributes **85** keep paths, of which **51** are shorter than 8 px and are passed through.  It is excluded from parameter selection; the values below use the same global settings.

| kernel | paths | short | mean p95 disp | curv. reduction | \|area\| % | cross exact->smooth | flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| splprep | 85 | 51 | 0.336 | 0.647 | 4.70 | 0->0 | 2 |
| bspline_lsq | 85 | 51 | 0.313 | 0.539 | 8.09 | 0->2 | 3 |
| pspline_diff2 | 85 | 51 | 0.431 | 0.352 | 77.74 | 0->4 | 4 |
| savgol | 85 | 51 | 0.402 | 0.578 | 57.82 | 0->5 | 4 |
| knot_quintic | 85 | 51 | 0.322 | 0.546 | 68.53 | 0->4 | 3 |

Because most boundaries are tiny fragments between under-segmented labels, displacement and curvature on this panel are dominated by passthrough paths and a few long boundaries; it is useful as a stress case, not as a quality target.

The one large closed loop on this panel (path `p73`, 113.4 px^2) loses up to **77.7% of its area** under `pspline_diff2`.  This is a genuine closed-body failure on under-segmented geometry, not a metric artifact.

## Failure cases and diagnostics

- **Closed loops:** 7 closed keep paths.  Periodic handling is used for all kernels, so endpoints cannot drift, but a smoother can still shrink or inflate the enclosed body.  The worst absolute area changes are listed below.

| panel | path | labels | kernel | exact area px^2 | area change % | max disp px |
| --- | --- | --- | --- | --- | --- | --- |
| ph01_page8_panel3 | p73 | 3-5 | pspline_diff2 | 113.4 | -77.74 | 2.981 |
| ph01_page8_panel3 | p73 | 3-5 | knot_quintic | 113.4 | -68.53 | 2.901 |
| ph01_page8_panel3 | p73 | 3-5 | savgol | 113.4 | -57.82 | 3.038 |
| ph01_page8_panel3 | p73 | 3-5 | bspline_lsq | 113.4 | -8.09 | 1.236 |
| ph01_page8_panel3 | p73 | 3-5 | splprep | 113.4 | -4.70 | 1.339 |
| newimage_p2 | p2 | 0-1 | pspline_diff2 | 4490.9 | -1.55 | 3.818 |
| newimage_p2 | p2 | 0-1 | savgol | 4490.9 | -1.03 | 3.530 |
| newimage_p2 | p2 | 0-1 | knot_quintic | 4490.9 | -1.01 | 3.884 |

- **Degenerate-collapse fallbacks:** 5 path/kernel fits collapsed the geometry and were restored to the exact polyline.  These rows are excluded from the metric means and are visible here:

| panel | path | labels | kernel | closed | length px | exact area px^2 |
| --- | --- | --- | --- | --- | --- | --- |
| newimage_p2 | p1 | 0-1 | splprep | yes | 9.1 | 5.40 |
| newimage_p2 | p1 | 0-1 | bspline_lsq | yes | 9.1 | 5.40 |
| newimage_p2 | p1 | 0-1 | pspline_diff2 | yes | 9.1 | 5.40 |
| newimage_p2 | p1 | 0-1 | savgol | yes | 9.1 | 5.40 |
| newimage_p2 | p1 | 0-1 | knot_quintic | yes | 9.1 | 5.40 |

  The dominant case is a tiny closed loop (`newimage_p2:b6:p5`, about 9 px long and 5 px^2 in area): any global smoothing length larger than the loop itself collapses it.  The fallback is intentional and explicit; it is not a silent deletion and it does not change the Layer 1 keep set.

- **Thin / highly curved paths:** the largest maximum displacements are:

| panel | path | labels | kernel | exact length px | max disp | p95 disp | flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| schematic_panel2 | p19 | 3-7 | knot_quintic | 2572.1 | 9.112 | 4.402 | displacement_gt_5px;length_change_gt_8pct |
| schematic_panel2 | p19 | 3-7 | savgol | 2572.1 | 8.422 | 4.262 | displacement_gt_5px;length_change_gt_8pct |
| schematic_panel2 | p37 | 5-7 | knot_quintic | 2819.9 | 6.611 | 2.865 | displacement_gt_5px |
| schematic_panel2 | p37 | 5-7 | savgol | 2819.9 | 6.181 | 2.827 | displacement_gt_5px |
| schematic_panel2 | p39 | 5-7 | knot_quintic | 2092.3 | 5.935 | 2.819 | displacement_gt_5px |
| paper23_fault_guided | p13 | 3-7 | savgol | 1713.8 | 5.860 | 3.381 | displacement_gt_5px;length_change_gt_8pct |
| schematic_panel2 | p39 | 5-7 | savgol | 2092.3 | 5.746 | 2.740 | displacement_gt_5px |
| paper23_fault_guided | p13 | 3-7 | knot_quintic | 1713.8 | 5.511 | 3.708 | displacement_gt_5px;length_change_gt_8pct |

- **Topology:** 2 path/variant rows gained at least one self-intersection.  Cross-intersection changes by panel are tabulated above; the largest positive deltas are:

| panel | kernel | exact | smoothed | delta |
| --- | --- | --- | --- | --- |
| ph01_page8_panel3 | savgol | 0 | 5 | +5 |
| ph01_page8_panel3 | pspline_diff2 | 0 | 4 | +4 |
| ph01_page8_panel3 | knot_quintic | 0 | 4 | +4 |
| paper23_fault_guided | splprep | 0 | 4 | +4 |
| newimage_p2 | pspline_diff2 | 0 | 2 | +2 |
| newimage_p3 | pspline_diff2 | 0 | 2 | +2 |
| ph01_page8_panel3 | bspline_lsq | 0 | 2 | +2 |
| paper23_fault_guided | pspline_diff2 | 0 | 2 | +2 |

  On the faulted `paper23_fault_guided` panel, `splprep` adds 4, `pspline_diff2` adds 2 cross-intersections between nearby fault blocks.  This is the most important non-ph01 topology failure: smoothing is per path, but paths from different label pairs can still be pushed through each other by aggressive corner cutting.

- **Short fragments:** paths below the smoothing length threshold (8 px) are identity passthrough for every kernel. They bias the displacement aggregate toward zero by the same amount for all kernels, so comparisons remain fair, but panel-level means on fragment-heavy panels should be read with that in mind.

- **`ph01_page8_panel3`:** under-segmented labels produce many 2-3 px skeleton fragments and a handful of long boundaries; it is the clearest failure case for any smoothing benchmark and is therefore not used for tuning.

## Parameter sensitivity (global grid)

All values below are pooled over the same non-ph01 smoothable paths.  The selected row is marked with `*`.  This is the sensitivity report; there are no per-panel parameter values.

### `splprep`

| parameter | paths | mean p95 disp | p95 max disp | mean curv. reduction | mean \|len change\| % | errors | degen | grid s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s_per_point=0.02 | 150 | 0.282 | 0.606 | 0.530 | 2.28 | 0 | 0 | 0.36 |
| s_per_point=0.05 | 150 | 0.427 | 0.915 | 0.690 | 3.04 | 0 | 0 | 0.20 |
| s_per_point=0.1 | 150 | 0.592 | 1.437 | 0.756 | 3.63 | 0 | 0 | 0.14 |
| s_per_point=0.2 | 150 | 0.818 | 1.895 | 0.793 | 4.21 | 0 | 0 | 0.12 |
| s_per_point=0.4 | 150 | 1.079 | 2.650 | 0.821 | 4.91 | 0 | 0 | 0.11 |
| s_per_point=0.8* | 150 | 1.386 | 3.292 | 0.866 | 5.30 | 0 | 1 | 0.10 |
| s_per_point=1.6 | 150 | 1.761 | 4.271 | 0.894 | 6.06 | 0 | 1 | 0.09 |
| s_per_point=3.2 | 150 | 2.156 | 5.328 | 0.922 | 6.66 | 0 | 1 | 0.08 |

### `bspline_lsq`

| parameter | paths | mean p95 disp | p95 max disp | mean curv. reduction | mean \|len change\| % | errors | degen | grid s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lambda=0.001;knot=24px | 150 | 1.153 | 3.961 | 0.856 | 5.51 | 0 | 0 | 0.07 |
| lambda=0.01;knot=24px | 150 | 1.154 | 3.961 | 0.856 | 5.51 | 0 | 0 | 0.07 |
| lambda=0.1;knot=24px | 150 | 1.155 | 3.961 | 0.856 | 5.54 | 0 | 0 | 0.07 |
| lambda=1;knot=24px | 150 | 1.157 | 3.961 | 0.855 | 5.66 | 0 | 0 | 0.07 |
| lambda=3;knot=24px | 150 | 1.158 | 3.961 | 0.852 | 5.83 | 0 | 0 | 0.07 |
| lambda=10;knot=24px | 150 | 1.163 | 3.964 | 0.859 | 5.64 | 0 | 1 | 0.07 |
| lambda=30;knot=24px | 150 | 1.167 | 3.964 | 0.858 | 5.72 | 0 | 1 | 0.07 |
| lambda=100;knot=24px* | 150 | 1.176 | 3.962 | 0.861 | 5.84 | 0 | 1 | 0.07 |
| lambda=300;knot=24px | 150 | 1.200 | 3.975 | 0.861 | 5.93 | 0 | 1 | 0.07 |
| lambda=1000;knot=24px | 150 | 1.268 | 4.125 | 0.861 | 5.96 | 0 | 1 | 0.07 |

### `pspline_diff2`

| parameter | paths | mean p95 disp | p95 max disp | mean curv. reduction | mean \|len change\| % | errors | degen | grid s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lambda=0.01;knot=4px | 150 | 0.325 | 0.689 | 0.636 | 2.84 | 0 | 0 | 0.19 |
| lambda=0.03;knot=4px | 150 | 0.349 | 0.748 | 0.676 | 3.13 | 0 | 0 | 0.21 |
| lambda=0.1;knot=4px | 150 | 0.402 | 0.894 | 0.718 | 3.54 | 0 | 0 | 0.19 |
| lambda=0.3;knot=4px | 150 | 0.477 | 1.141 | 0.753 | 4.04 | 0 | 0 | 0.19 |
| lambda=1;knot=4px | 150 | 0.595 | 1.484 | 0.785 | 4.77 | 0 | 0 | 0.19 |
| lambda=3;knot=4px | 150 | 0.744 | 1.980 | 0.804 | 5.49 | 0 | 0 | 0.19 |
| lambda=10;knot=4px | 150 | 0.932 | 2.585 | 0.834 | 5.61 | 0 | 1 | 0.19 |
| lambda=30;knot=4px | 150 | 1.132 | 3.296 | 0.843 | 5.74 | 0 | 1 | 0.19 |
| lambda=100;knot=4px* | 150 | 1.379 | 4.092 | 0.847 | 5.75 | 0 | 1 | 0.20 |
| lambda=300;knot=4px | 150 | 1.653 | 4.965 | 0.848 | 5.81 | 0 | 1 | 0.18 |

### `savgol`

| parameter | paths | mean p95 disp | p95 max disp | mean curv. reduction | mean \|len change\| % | errors | degen | grid s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| window=9;order=3 | 150 | 0.304 | 0.657 | 0.590 | 2.84 | 0 | 0 | 0.08 |
| window=13;order=3 | 150 | 0.399 | 0.902 | 0.670 | 3.72 | 0 | 0 | 0.08 |
| window=17;order=3 | 150 | 0.489 | 1.221 | 0.733 | 3.89 | 0 | 1 | 0.08 |
| window=21;order=3 | 150 | 0.578 | 1.478 | 0.755 | 4.39 | 0 | 1 | 0.08 |
| window=29;order=3 | 150 | 0.759 | 2.067 | 0.751 | 5.11 | 0 | 1 | 0.08 |
| window=41;order=3 | 150 | 1.033 | 2.994 | 0.827 | 5.74 | 0 | 1 | 0.08 |
| window=61;order=3* | 150 | 1.357 | 4.348 | 0.831 | 6.39 | 0 | 1 | 0.08 |
| window=81;order=3 | 150 | 1.615 | 5.186 | 0.876 | 6.46 | 0 | 1 | 0.08 |
| window=101;order=3 | 150 | 1.890 | 6.079 | 0.892 | 6.97 | 0 | 1 | 0.09 |

### `knot_quintic`

| parameter | paths | mean p95 disp | p95 max disp | mean curv. reduction | mean \|len change\| % | errors | degen | grid s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| half_window=6;degree=5 | 150 | 0.224 | 0.514 | 0.470 | 2.10 | 0 | 0 | 0.14 |
| half_window=8;degree=5 | 150 | 0.280 | 0.607 | 0.567 | 2.60 | 0 | 0 | 0.15 |
| half_window=11;degree=5 | 150 | 0.348 | 0.775 | 0.643 | 3.27 | 0 | 0 | 0.17 |
| half_window=15;degree=5 | 150 | 0.432 | 1.056 | 0.715 | 3.41 | 0 | 1 | 0.20 |
| half_window=22;degree=5 | 150 | 0.581 | 1.651 | 0.773 | 4.13 | 0 | 1 | 0.24 |
| half_window=32;degree=5 | 150 | 0.804 | 2.527 | 0.815 | 4.96 | 0 | 1 | 0.30 |
| half_window=45;degree=5 | 150 | 1.035 | 3.634 | 0.842 | 5.46 | 0 | 1 | 0.39 |
| half_window=64;degree=5* | 150 | 1.251 | 4.430 | 0.860 | 5.85 | 0 | 1 | 0.52 |
| half_window=90;degree=5 | 150 | 1.510 | 5.296 | 0.875 | 6.19 | 0 | 1 | 0.72 |

## Sanity checks

- Expected keep paths: **242** per variant; see `sanity.json` for the exact key-set comparison.  The run aborts if any variant is missing or has extra paths.
- Every variant produces one row per path per panel in `metrics.csv`; `panel_metrics.csv` has one row per panel and variant.
- Errors are never silently dropped: a failed fit falls back to the exact geometry and is marked `error_passthrough:*`.  Fits that collapse the geometry are likewise restored and marked `degenerate_passthrough`.  Both statuses are counted in `sanity.json`; only `short_passthrough`, `ok`, `degenerate_passthrough` and `exact` should appear, and every variant must cover the full path key set.

## Files

- `metrics.csv` - one row per panel/path/variant.
- `panel_metrics.csv` - one row per panel/variant, including cross-intersections.
- `sensitivity.csv` - the global parameter grid.
- `selected_params.json` - frozen parameters and selection notes.
- `sanity.json` - path coverage and status counts.
- `contact.png` - representative panel grid plus aggregate rankings.
- `panels/<panel_id>.png` - full per-panel composite for all 11 panels.
