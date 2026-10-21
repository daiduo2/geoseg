# Layer 2 experiment F — interior hairpins / direction reversals

Run: 2026-10-21 14:48:05 local
Command: `uv run python experiments/layer2/f_hairpin/run.py`
Runtime: **238.3s**
Repo root: `/Users/daiduo2/geoseg`

## Question and method

Experiments A and E showed that the worst dense curvature is interior: `schematic_panel2` p37 reverses direction ~312 px from either endpoint (analytic baseline dense max 12.44; 12.39 on this experiment's 8 sample/px reporting grid) and `paper23_fig7a` p13 has its maximum ~128.6 px from the nearer endpoint.  This experiment detects those reversals on the exact 1 px centerline and compares four handlers on the same paths.

### Hairpin detector

- Resample the exact centerline at 1 px.  Tangent from a ±2 px chord; turn angle over ±3 px (`t3`, ~6 px total) and ±8 px (`t8`, ~16 px total).
- Dense local curvature `k5` from the circumradius of the 5-point stencil `i-2, i, i+2`, which is robust to the 45° skeleton staircase.
- Flag `t3 >= 140°`, or (`k5 >= 0.85 px⁻¹` and `t3 >= 90°`), or (`t8 >= 170°` **and** (`t3 >= 80°` or `k5 >= 0.30 px⁻¹`)).  The sharpness gate on the `t8` branch is important: a smooth 16 px U-bend also has `t8 ≈ 180°` but is not a curvature threat, and splitting it can increase the corner angle.  Such smooth bends are deliberately left to baseline A.
- Merge consecutive flagged pixels and groups separated by ≤ 8 px; the anchor is the pixel maximizing `t3 + t8 + 100 k5`.
- Ignore open-path anchors within 2 px of a free endpoint and paths shorter than 10 dense samples.  The same detector setting is used for every panel.

### Variants

All variants share the cubic clamped P-spline basis (knots every 8 px), `D2 + 0.25 D3` penalty, lambda = 1, hard original endpoints, 3 px corridor, 2% closed-area tolerance, 1e-3 px endpoint drift, zero self-intersections, and exact fallback on constraint failure.

- `baseline_A`: one A-style P-spline per path.
- `split`: cut at every anchor, pin both new endpoints to the exact centerline, fit each subpath independently.  The join is C0.
- `corner_knot`: the same pieces, with the exact one-sided tangents at each anchor imposed as hard linear constraints (KKT).  If a constrained piece cannot meet the corridor, it is refit without tangents and marked relaxed.
- `curvature_cap`: one path with a local Gaussian penalty boost (σ=6 px) around anchors, trying boosts (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0).  Accept the smallest boost whose local dense curvature is ≤ 2 px⁻¹ (radius ≥ 0.5 px) inside the corridor **and** whose local turn stays at least 50% of the exact turn; otherwise fall back to split, then exact.  The turn guard is essential: without it the cap can smooth a 141° hairpin down to 24° and still report 'cap met'.

## Detection counts

| setting | turn3° | turn8° | curv 1/px | anchors | paths | p37 | p13 | p4 |
|---|---|---|---|---|---|---|---|---|
| selected | 140 | 170 | 0.85 | 28 | 22 | 2 | 1 | 1 |
| turn3=120 | 120 | 170 | 0.85 | 40 | 26 | 3 | 1 | 1 |
| turn3=160 | 160 | 170 | 0.85 | 26 | 20 | 2 | 1 | 0 |
| turn8=150 | 140 | 150 | 0.85 | 44 | 28 | 3 | 1 | 1 |
| turn8=190 | 140 | 190 | 0.85 | 16 | 13 | 2 | 0 | 1 |
| curv=0.75 | 140 | 170 | 0.75 | 41 | 28 | 2 | 1 | 1 |
| curv=0.95 | 140 | 170 | 0.95 | 25 | 21 | 1 | 1 | 1 |

Selected setting: **t3=140°, t8=170° with (t3≥80° or k5≥0.3 px⁻¹), k5≥0.85 px⁻¹ (with t3≥90°)**.  Detection found **28 anchors on 22 of 184 eligible curated paths** (ph01 included in the count).  Lowering the turn thresholds adds ordinary skeleton corners; raising them drops real reversals.  The selected setting keeps all three named anchors.

## Variant comparison (non-ph01 eligible paths)

| variant | n | med mean | med p95 | p90 p95 | med max | p95 max | max drift | med dense max | med hairpin base | med hairpin var | max hairpin base | max hairpin var | turn preserved | self-int | fallbacks | errors | max |ΔA|/A | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_A | 150 | 0.423 | 0.801 | 1.780 | 1.412 | 2.691 | 0.00 | 0.105 | 0.982 | 0.982 | 12.386 | 12.386 | 82.6% | 0 | 0 | 0 | 0.4% | 7.3 |
| split | 150 | 0.423 | 0.801 | 1.779 | 1.412 | 2.603 | 0.00 | 0.103 | 0.982 | 0.205 | 12.386 | 0.721 | 93.4% | 0 | 0 | 0 | 0.4% | 8.0 |
| corner_knot | 150 | 0.423 | 0.801 | 1.780 | 1.412 | 2.619 | 0.00 | 0.105 | 0.982 | 0.191 | 12.386 | 0.826 | 92.6% | 0 | 0 | 0 | 0.4% | 8.0 |
| curvature_cap | 150 | 0.423 | 0.801 | 1.780 | 1.412 | 2.658 | 0.00 | 0.105 | 0.982 | 0.876 | 12.386 | 1.929 | 82.6% | 0 | 0 | 0 | 0.4% | 17.9 |

Displacements are px.  Dense curvature is evaluated at 8 samples/px (analytic spline derivatives for fitted pieces, finite differences for exact fallback), excluding ±1 px around deliberate C0 joins; `dense_raw_max_curvature` including the join is in `metrics.csv`.  `hairpin base` is the baseline fit's local maximum within ±12 px of the same anchor, so the comparison isolates the handling from the detector.  `turn preserved` is `1 - |after-before|/before`, median over hairpin paths.

## Curated anchor cases

| case | L px | hairpins | dense base | dense split | dense corner | dense cap | hairpin base | hairpin split | hairpin corner | hairpin cap | turn before | turn split | turn corner | max disp split | max disp corner | cap outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| schematic_panel2 p37 | 2819.9 | 2 | 12.39 | 0.29 | 0.34 | 0.29 | 12.39 | 0.20 | 0.19 | 0.20 | 163.4 | 162.1 | 158.3 | 2.46 | 2.46 | cap_to_split |
| paper23_fig7a p13 | 260.0 | 1 | 1.66 | 0.41 | 0.62 | 1.66 | 1.66 | 0.41 | 0.62 | 1.66 | 141.4 | 142.5 | 140.4 | 2.55 | 2.56 | baseline_within_cap |
| newimage_p4 p4 | 14.3 | 1 | 2.71 | 0.28 | 0.34 | 0.28 | 2.71 | 0.28 | 0.34 | 0.28 | 168.6 | 168.6 | 168.6 | 1.01 | 0.87 | cap_to_split |
| schematic_panel2 p36 | 127.4 | 1 | 2.01 | 0.21 | 0.21 | 1.70 | 2.01 | 0.13 | 0.16 | 1.70 | 157.0 | 144.2 | 138.2 | 1.94 | 2.10 | cap_met |
| schematic_panel2 p12 | 574.2 | 1 | 1.93 | 0.80 | 0.80 | 1.93 | 1.93 | 0.10 | 0.19 | 1.93 | 154.4 | 153.5 | 151.9 | 2.10 | 2.07 | baseline_within_cap |
| paper23_fig7a p21 | 449.7 | 1 | 0.45 | 0.18 | 0.17 | 0.45 | 0.45 | 0.18 | 0.14 | 0.45 | 104.3 | 117.0 | 119.9 | 2.68 | 2.71 | baseline_within_cap |
| newimage_p4 p2 | 1778.2 | 2 | 1.34 | 0.72 | 0.83 | 0.72 | 1.34 | 0.72 | 0.83 | 0.72 | 146.8 | 115.0 | 107.8 | 2.51 | 2.46 | cap_to_split |
| schematic_panel2 p22 | 599.8 | 1 | 0.90 | 0.34 | 0.46 | 0.90 | 0.90 | 0.19 | 0.20 | 0.90 | 145.4 | 134.4 | 129.7 | 2.03 | 2.08 | baseline_within_cap |
| newimage_p4 p3 | 711.9 | 1 | 0.59 | 0.43 | 0.43 | 0.59 | 0.59 | 0.05 | 0.17 | 0.59 | 135.5 | 126.6 | 125.1 | 2.47 | 2.47 | baseline_within_cap |
| schematic_panel2 p39 | 2092.3 | 1 | 0.98 | 0.29 | 0.55 | 0.98 | 0.98 | 0.26 | 0.55 | 0.98 | 139.4 | 130.0 | 123.1 | 2.17 | 2.32 | baseline_within_cap |

`dense *` is the whole-path maximum excluding deliberate joins; `hairpin *` is the local maximum within ±12 px of the anchor.  p37, p13 and p4 are the named cases from A/E; the others are the strongest curated detections.

## Synthetic cases

| case | class | description | hairpins | dense base | dense split | dense corner | dense cap | max disp base | max disp split | turn before | turn after | split status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn_v_cusp | true geometry corner | sharp V, ~150 deg turn in <=3 px | 1 | 1.35 | 0.03 | 0.06 | 1.35 | 0.97 | 0.57 | 146.1 | 146.2 | ok |
| syn_u_turn | skeleton artifact | U-turn around an 8 px protrusion | 1 | 0.52 | 0.18 | 0.24 | 0.52 | 2.07 | 1.06 | 125.2 | 126.9 | ok |
| syn_zigzag_junction | skeleton artifact | junction spur doubles back 6 px | 1 | 118.14 | 3.56 | 3.56 | 3.56 | 1.86 | 0.00 | 154.5 | 154.4 | fallback_constraint |
| syn_pinchout | true geology pinch-out | 6 px layer terminating at a rounded tip | 1 | 1.10 | 0.13 | 0.21 | 1.10 | 2.37 | 1.05 | 140.5 | 140.9 | ok |
| syn_smooth_fold | true geology control | smooth 30 px radius fold; no hairpin | 0 | 0.04 | 0.04 | 0.04 | 0.04 | 0.51 | 0.51 | n/a | n/a | ok |

**True geology:** V cusp (a real geometric corner), pinch-out (a genuine thin-layer tip / triple junction), smooth fold (control).  **Skeleton artifacts:** U-turn and junction spur double-back.  The V cusp is expected to be detected: it is a real corner, and the test is whether a handler preserves it rather than rounding it into a sub-pixel cusp.

`syn_zigzag_junction` is the expected limitation of splitting: the exact spur overlaps itself (zero-length retrace), so opening the loop at the anchor turns the overlap into a real crossing.  The assembled curve then self-intersects and the contract forces exact fallback for split, corner and cap.  Baseline A reporting 118 px⁻¹ is a smoothing cusp at that spur, not a usable result.  Exact fallback is the only topology-safe outcome for a self-overlapping artifact.

## Per-panel results (selected variant)

| panel | paths | eligible | hairpins | paths with | ok | fb short | fb constraint | med p95 | med hairpin | max hairpin | turn preserved | self-int | cross-int | max |ΔA|/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| newimage_p0 | 3 | 3 | 1 | 1 | 3 | 0 | 0 | 1.103 | 0.045 | 0.045 | 93.0% | 0 | 0 | n/a |
| newimage_p1 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0.709 | n/a | n/a | n/a | 0 | 0 | n/a |
| newimage_p2 | 8 | 7 | 1 | 1 | 7 | 1 | 0 | 0.743 | 0.267 | 0.267 | 95.6% | 0 | 0 | n/a |
| newimage_p3 | 4 | 4 | 4 | 2 | 4 | 0 | 0 | 0.840 | 0.341 | 0.643 | 87.8% | 0 | 0 | n/a |
| newimage_p4 | 7 | 7 | 4 | 3 | 7 | 0 | 0 | 0.841 | 0.282 | 0.721 | 93.4% | 0 | 0 | n/a |
| paper22_fig5a | 13 | 13 | 0 | 0 | 13 | 0 | 0 | 0.590 | n/a | n/a | n/a | 0 | 0 | n/a |
| paper22_fig6 | 9 | 9 | 0 | 0 | 9 | 0 | 0 | 0.835 | n/a | n/a | n/a | 0 | 0 | n/a |
| ph01_page8_panel3 | 85 | 34 | 3 | 3 | 33 | 51 | 1 | 0.585 | 0.370 | 7.876 | 91.7% | 0 | 0 | n/a |
| paper23_fig7a | 34 | 34 | 2 | 2 | 34 | 0 | 0 | 0.657 | 0.294 | 0.409 | 93.5% | 0 | 0 | n/a |
| schematic_panel2 | 40 | 34 | 10 | 8 | 34 | 6 | 0 | 1.019 | 0.160 | 0.271 | 94.3% | 0 | 0 | n/a |
| paper23_fault_guided | 31 | 31 | 3 | 2 | 31 | 0 | 0 | 0.677 | 0.326 | 0.341 | 91.4% | 0 | 0 | n/a |

`ph01_page8_panel3` is under-segmented and is excluded from the aggregate variant comparison; it is reported here only.  Its p73 is the small closed loop that also failed A's area corridor; split opens it at the anchor, the assembled area changes by >2%, and the whole result falls back to exact.  The 7.88 in its `max hairpin` column is the exact pixel centerline's local value, not a fitted spike.

## Cap sensitivity

| cap 1/px | paths with hairpins | cap met | fallback to split/exact |
|---|---|---|---|
| 1 | 22 | 12 | 10 |
| 2 | 22 | 16 | 6 |
| 4 | 22 | 16 | 6 |
| 8 | 22 | 16 | 6 |

Derived from the boost trials already run at cap=2; a lower cap forces more fallbacks to split, a higher cap accepts more locally boosted fits.  The cap is a single global setting, not a panel parameter.

## Baseline parity with experiment A

Compared 242/242 A `lam1` rows by `(panel_id, path_index)`; missing 0, status mismatches 0.

| field | max |baseline_A - A lam1| |
|---|---|
| mean_disp | 4.98e-07 |
| p95_disp | 4.96e-07 |
| max_disp | 4.99e-07 |
| max_curvature | 4.94e-07 |
| endpoint_drift | 4.69e-13 |
| length_smooth | 4.97e-07 |
| area_change | 4.11e-07 |

Differences are at the 1e-6 level because A's CSV is rounded to six decimals; the fitted geometry and statuses match.

## Failure cases and warnings

- No constraint fallbacks, errors, or relaxed corner pieces on non-ph01 paths.

Dense curvature at a deliberate C0 join is mathematically singular.  Reported dense maxima exclude ±1 px around joins; the join/tangent error is reported as `tangent_error_max` in `metrics.csv`.  This is the intended trade: split keeps the true corner instead of hiding it in a sub-pixel loop.  If a reader treats the corner itself as unacceptable curvature, no C0 method can satisfy both fidelity and finite curvature there.

## Answer

1. **Are hairpins real corners, skeleton artifacts, or genuine pinches?** They are a mix, and the recoloured mask must decide, not the skeleton alone. `schematic_panel2` p37 is a genuine narrow protrusion in the mask: a 2–3 px wide blue finger whose two sides the exact centerline must traverse.  It is sub-resolution at the 3480 px panel scale and may be label noise, but it is not a skeleton traversal through thick rock.  `paper23_fig7a` p13 is a small brown tongue tip/notch: the tip is a genuine pinch-out-like geometry, while the staircase notches along its sides are skeleton quantization. `newimage_p4` p4 is a 14 px junction loop around a tiny hole, a skeleton artifact.  In the synthetic set, V cusp, pinch-out and smooth fold are true geometry; U-turn and spur double-back are skeleton artifacts.  The detector is purely geometric and cannot make that geological call by itself.

2. **Which handling is safest?** `split` is the safest default.  It pins both new endpoints to the exact centerline, so corridor, endpoint drift, self-intersection, cross-intersection and closed-area checks are unchanged, and it makes no assumption about the tangents at the corner.  `corner_knot` is an equally safe upgrade when the exact tangents are trustworthy (piece length >= 6 px): median turn preservation is 92.6% versus 93.4% for split, and it has the lowest median local curvature (0.191 vs 0.205 px⁻¹), with no relaxed pieces on the curated paths.  It can over-constrain short noisy pieces; those are relaxed to plain split and marked.  `curvature_cap` is the least safe: the 3 px corridor prevents a low cap on a narrow finger, so it mostly falls back to split, and when it does apply the turn guard is what stops it from erasing the reversal.  Baseline A is not safe for these paths because it creates the sub-pixel cusp.

3. **Does it reduce dense curvature without topology/fidelity loss?** On non-ph01 hairpin paths, split lowers the local hairpin maximum by a median of 78.9% relative to baseline_A.  p37: whole-path dense max 12.39 → 0.29, local hairpin max 12.39 → 0.20.  p13: 1.66 → 0.41.  p4: 2.71 → 0.28.  Displacement stays inside the 3 px corridor, endpoint drift is exact, self/cross-intersections remain zero, and closed-body area change remains under 2%.  The price is a deliberate C0 corner: its angle is preserved, but its mathematical curvature is singular at exactly one point, so the reported dense maxima exclude ±1 px around joins and the join/tangent error is reported.  For transparency, A's contract max-curvature metric (at most 1000 samples, no join exclusion) is 1.90 for baseline p37 and 6.54 for split: the latter is the corner itself, not a smoothing spike, and the dense maximum away from the join is 0.29.  That is fidelity-preserving topology, not a hidden smoothing failure.

## Sanity checks

| check | value | result |
|---|---|---|
| keep paths | 242 | pass |
| exact boundary objects | 158 | pass |
| fit-eligible paths | 184 | pass |
| short/ineligible paths | 58 | pass |
| curated metrics rows | 968 | pass |
| one row per path per variant | 968 | pass |
| status totals sum to rows | 242 | pass |
| open endpoint drift <= 1e-3 px | 4.69e-13 | pass |
| self-intersections total | 0 | pass |
| closed |area change| max | 0.0042586 | pass |
| fit errors | 0 | pass |
| synthetic records | 5 | pass |
| cross-intersections panels >0 | 0 | pass |
| baseline parity matched | 242/242 | pass |
| baseline parity status mismatches | 0 | pass |

## Outputs

- `metrics.csv` — one row per panel/path/variant (curated + synthetic).
- `panel_metrics.csv` — one row per panel/variant, including cross-intersections.
- `hairpins.csv` — one row per detected anchor/variant.
- `pieces.csv` — split/corner piece diagnostics.
- `summary.md` — this file.
- `contact.png` — 11 curated panels, baseline left, selected variant right.
- `hairpin_zoom.png` — zoomed anchor comparisons.
- `synthetic.png` — five synthetic cases.
- `panels/<panel_id>.png` — full-resolution overlays.

## Runtime

| stage | seconds |
|---|---|
| build_records | 9.6 |
| panel_newimage_p0 | 2.2 |
| panel_newimage_p1 | 4.5 |
| panel_newimage_p2 | 6.3 |
| panel_newimage_p3 | 5.2 |
| panel_newimage_p4 | 5.6 |
| panel_paper22_fig5a | 8.6 |
| panel_paper22_fig6 | 4.0 |
| panel_ph01_page8_panel3 | 24.7 |
| panel_paper23_fig7a | 49.6 |
| panel_schematic_panel2 | 52.6 |
| panel_paper23_fault_guided | 56.5 |
| curated | 219.7 |
| synthetic | 6.6 |
| render | 2.4 |
