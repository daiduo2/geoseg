# Layer 2 experiments: cross-experiment analysis

> 2026-10-21. Four independent experiments on the same Layer 1 keep set.
> This is an analysis of the first pass, not a production decision.

## 0. What was run

| Direction | Implementation | Main question |
|---|---|---|
| A `a_segmented_spline` | per-path cubic P-spline, D2+D3 penalty, endpoint pinning, 3 px corridor, area fallback | Is a conservative segmented spline enough? |
| B `b_kernel_benchmark` | splprep / B-spline LSQ / P-spline D2 / Savitzky–Golay / knot-quintic on identical paths | Which smoothing kernel has the best fidelity/topology trade-off? |
| C `c_fault_topology` | global vs junction-pinned vs fault-aware vs fault-aware+fallback | Do junctions/faults require explicit handling, and can a validator repair violations? |
| D `d_closed_aniso` | periodic spline, area restoration, width guard, weak image anisotropy | Do closed bodies shrink/round, and does area/width preservation work? |

All experiments use `experiments/layer2/common.py`, `keep_paths()`, and the same recoloured-mask render style. No experiment re-segments or changes the Layer 1 keep set.

## 1. Headline results

### A. Segmented constrained spline

- 11 panels, **376 keep paths** from 158 exact boundary objects.
- **190 eligible**, 186 short/ineligible; of those, **157 paths are ≤2 px** and 186 are ≤9 px. They are skeleton spurs at junctions, not geological boundaries.
- Selected global setting **λ = 1**; 8 px knots; 3 px displacement corridor; 2% closed-area tolerance.
- Non-ph01, 150 smoothable paths:
  - median p95 displacement **0.801 px**, median max **1.412 px**;
  - p90 of p95 **1.780 px**, p95 of max **2.691 px**;
  - median max-curvature reduction **84.0%**, median p95-curvature reduction **93.0%**;
  - **0 self-intersections, 0 cross-intersections** on all panels;
  - maximum closed-body area change **0.426%**;
  - **0 constraint fallbacks** on non-ph01.
- 186 short paths returned exact and marked `fallback_short`; they are retained, not deleted.
- One ph01 closed loop fell back on area (2.6% raw loss at 8 px knots).

**Verdict:** the most conservative result and the only one with zero topology violations on all panels. It is the best production baseline so far. Its weakness is not topology but local shape: endpoint-pinned junction corners can raise max curvature (e.g. schematic p64 0.817→1.903), and very short zigzag paths can contract a lot (7.7 px path −58%) if they are allowed into the fit.

### B. Smoothing kernel benchmark

Identical 376 paths, 190 short passthrough, 149 measured non-ph01 paths per kernel.

| kernel | mean p95 disp (px) | max disp (px) | curvature reduction | cross Δ | area change | runtime |
|---|---:|---:|---:|---:|---:|---:|
| **bspline_lsq** | **1.197** | 4.748 | 0.861 | **0** | 0.21% | **0.034 s** |
| knot_quintic | 1.261 | 9.112 | 0.860 | 0 | 0.31% | 0.514 s |
| savgol | 1.342 | 8.422 | 0.831 | 0 | 0.32% | 0.045 s |
| pspline_diff2 | 1.360 | 4.816 | 0.847 | **7** | 0.48% | 0.165 s |
| splprep | 1.376 | 5.070 | **0.866** | **4** | **0.13%** | 0.059 s |

- `bspline_lsq` is the best trade-off: lowest displacement, zero topology violations in the ranking pool, near-best curvature reduction, fastest.
- `splprep` smooths slightly more but creates 4 cross-intersections on `paper23_fault_guided`; `pspline_diff2` creates 7 across panels; `savgol`/`knot_quintic` produce large max displacements (8–9 px).
- Every kernel collapses the tiny 9 px closed loop (`newimage_p2:b6:p5`); all fall back to exact. The ph01 closed loop loses 4.7–77.7% area depending on kernel.

**Verdict:** kernel choice is secondary to constraints. A's corridor + endpoint pinning + fallback is why A has lower max displacement than B's unconstrained `bspline_lsq`. If a kernel is swapped in, it must inherit A's corridor/fallback.

### C. Fault/junction/topology

Synthetic cases: normal fault, listric fault, three-layer thin bed, explicit fault zone, pinch-out.

| variant | synthetic p95 disp | synthetic violations | curated violations | curated cross-int | endpoint drift |
|---|---:|---:|---:|---:|---:|
| global | 0.563 | 11 | 103 | 8 | 2.925 px |
| junction_pinned | 0.561 | 5 | 1 | 0 | 0 |
| fault_aware | 0.571 | 0 | 3 | 0 | 0 |
| fault_aware_fallback | 0.571 | 0 | 0 | 0 | 0 |

- Global smoothing pulls open endpoints and creates 8 cross-intersections on curated panels.
- **Endpoint pinning alone eliminates all cross-intersections and endpoint drift** (1 violation remains).
- On synthetic faults, fault-aware handling preserves the fault corner exactly (max corner error 2.42 px → 0.00) and reduces mean offset error 0.368 → 0.310 px; fallback reduces max offset error 0.829 → 0.562 px.
- **No curated panel has an explicit fault label.** Curated fault-aware handling uses only junction/kink geometry, marks 50 paths `review`, and does not claim fault-offset accuracy. Geometry-only fault inference actually increased curated violations from 1 to 3, so it should not be automatic.
- The three-layer synthetic case shows a limitation: a short fault-bounded segment still smooths at the global λ, giving a 0.56 px offset error on the thin lower layer (global 0.27 px).

**Verdict:** junction pinning is mandatory. Explicit fault masks improve fault cases, but inferred fault handling is unsafe without a real fault label/mask. Topology fallback repairs the remaining violations.

### D. Closed bodies

- 6 closed keep paths; only 4 non-degenerate bodies.
- Large bodies are almost area-neutral under periodic smoothing: raw mean ΔA ≈ 0.000%, max |ΔA| 0.043%.
- Small loops are not safe: ph01 loop −3.769% raw; 4 px loop −63.8%, all variants fall back.
- Area restoration brings final ΔA to ≤0.002%, but can inflate width after the raw width guard; a final width check is needed.
- 26 open controls: endpoint drift 0, max displacement 1.85 px, 0 self/cross-intersections.
- Weak image anisotropy had negligible effect (mean ΔA +0.001 pp; max displacement 1.31→1.34 px).

**Verdict:** area/width preservation is cheap insurance for small/closed bodies, but not a major issue for normal-size lenses. Do not make image anisotropy a priority.

## 1.5 Path graph v2 rerun (after the first analysis)

`keep_paths()` now:

- prunes endpoint-to-branch skeleton spurs up to 5 px;
- drops short open junction-cluster edges up to 8 px whose two ends touch longer paths;
- keeps closed loops and short paths with a free end;
- never changes the Layer 1 keep set; dropped edges remain exact boundary pixels.

Effect:

| metric | v1 | v2 |
|---|---:|---:|
| keep paths | 376 | **242** |
| short/ineligible paths | 186 | **58** |
| short paths outside ph01 | 59 | **8** |
| non-ph01 smoothable paths | 150 | **150** |
| A median p95 / median max | 0.801 / 1.412 px | **identical** |
| B `bspline_lsq` mean p95 / cross Δ | 1.197 / 0 | **identical** |

The path graph fix cleaned the reporting and removed 51 ph01 fragments plus 8 other artifacts, but it did **not** change any smoothable path or any smoothing metric. That is a useful negative result: the remaining problem is not path count, it is shape control at endpoints/junctions. The next experiment should target endpoint tangents, not more pruning.



## 1.6 Experiment E: endpoint tangents (negative result)

A's max-curvature increases were assumed to come from endpoint-pinned junction corners. E added a soft endpoint tangent penalty to the same P-spline, keeping hard endpoint positions and the 3 px corridor.

- Selected **β = 10** globally: endpoint-region max curvature **0.0690 → 0.0632 (−8.5%)**, p90 **0.2285 → 0.2161**; 112 paths improved, 17 worsened, 17 unchanged.
- Fidelity cost: median p95 **0.8011 → 0.8062 px**, p90 p95 **1.7800 → 1.7882 px**; 0 topology violations, 0 fallbacks, area unchanged.
- β=100 reduced endpoint curvature by 33% but pushed median p95 to 0.838 px and failed the no-regression rule.

The important finding is negative: A's cited worst cases are **not endpoint spikes**. For `schematic_panel2` p37 the dense-curvature maximum (12.44) is an interior hairpin **312 px from the nearest endpoint**; for `paper23_fig7a` p13 it is 128.6 px from the endpoint. A's contract max (1.903 / 1.547) is an undersampled value on long paths. The tangent penalty cannot move these interior maxima.

So endpoint tangents are an optional small improvement (β=10), not the main fix. The next experiment must target interior hairpins and local corner preservation: split at high-curvature reversals, add a curvature cap/local knot, or treat the hairpin as a junction rather than smoothing through it.

## 1.7 Experiment F: interior hairpins (main win)

F detected 28 direction reversals on 22 of 184 eligible curated paths using a global turn/curvature detector, then compared four handlers on the same paths.

| variant | med p95 disp | med hairpin max | max hairpin | turn preserved | topology |
|---|---:|---:|---:|---:|---|
| baseline_A | 0.801 | 0.982 | 12.386 | 82.6% | 0 |
| **split** | **0.801** | **0.205** | **0.721** | **93.4%** | **0** |
| corner_knot | 0.801 | 0.191 | 0.826 | 92.6% | 0 |
| curvature_cap | 0.801 | 0.876 | 1.929 | 82.6% | 0 |

- `schematic_panel2` p37: whole-path dense max **12.39 → 0.29 (split), 0.34 (corner)**; local hairpin max **12.39 → 0.20 / 0.19**.
- `paper23_fig7a` p13: **1.66 → 0.41 / 0.62**.
- `newimage_p4` p4: **2.71 → 0.28 / 0.34**.
- Median local hairpin max reduction: **78.9%**.
- Displacement, endpoint drift, closed-area change and topology checks are unchanged; 0 fallbacks on non-ph01 paths.

Hairpins are a mix: p37 is a real 2–3 px narrow protrusion in the mask (possibly label noise), p13 is a genuine tongue tip plus staircase notches, p4 is a skeleton loop. Split is the safest default because it assumes nothing about the corner tangents and preserves the turn. `corner_knot` is a safe upgrade when the one-sided tangents are trustworthy, but can over-constrain short pieces. `curvature_cap` is unsafe: without a turn guard it can turn a 141° hairpin into 24° and still report success.

The price of split/corner is a deliberate C0 join. Dense maxima exclude ±1 px around joins; at p37 the join itself has infinite mathematical curvature. This is fidelity-preserving topology, not a smoothing failure. A self-overlapping synthetic spur correctly falls back to exact.

## 2. Cross-experiment findings

### 2.1 Path graph v2 cleaned the counts, not the smoothable set

v1 had 376 keep paths, 186 short and 157 ≤2 px.  v2 prunes sub-5 px skeleton spurs and sub-8 px open junction-cluster edges, reducing this to 242 paths and 58 short paths (51 of them on ph01).  The non-ph01 smoothable set stayed at 150 paths and every A/B metric stayed identical.

So the path-count problem was real for reporting and for the 186 exact fallbacks, but it was not the cause of the smoothing errors.  The real shape defect is interior direction reversals, not endpoint spurs.

### 2.2 Endpoint pinning is the highest-value constraint

C shows that without endpoint pinning, open paths drift up to 2.9 px and create 8 cross-intersections. A and C both show that pinning all open endpoints eliminates endpoint drift and cross-intersections. A's remaining local curvature spikes at pinned corners are the price of hard pinning; the next step is endpoint **tangent** constraints or junction-aware anchors, not releasing the points.

### 2.3 Global smoothers create topology violations; corridor/fallback fixes them

B's unconstrained kernels produce 4–7 cross-intersections or 8–9 px max displacement. A's 3 px corridor + iterative fallback produces 0 topology violations and 1.5 px mean max displacement. C's validator repairs the last self-intersection/closed-loop cases.

Conclusion: never accept a smoothing result without a per-path corridor and a panel-level intersection/area check. The validator can be simple and local; it does not need a global NP-hard topology optimization.

### 2.4 Fault handling must be evidence-based

C's synthetic results validate the principle: explicit fault splits preserve corners and reduce offset error. But geometry-only fault inference on curated panels made things worse (1 → 3 violations). Without a fault label/mask, the safe behavior is:

- pin junctions;
- mark kink/fault-like segments `review`;
- return ambiguous gaps exact;
- do not claim fault accuracy.

This is consistent with the survey: faults are hard discontinuities, not outliers.

### 2.5 Closed-body area is mostly safe; width is the hidden failure

Large closed bodies changed area by <0.05%; area restoration is almost cosmetic. The real risk is small/degenerate loops and final width inflation after area restoration. The width guard must run on the **final** curve, not only the raw spline.

### 2.6 Current metrics are fidelity/topology metrics, not geological truth

A/B measure distance to the exact centerline and curvature reduction. They do not prove the smoothed curve is more geologically correct. C's synthetic cases have ground truth for fault offset; the curated panels do not. The next benchmark needs:

- synthetic clean curves with ground-truth geometry;
- boundary-level visual audit on recoloured masks;
- held-out panels and fixed global parameters;
- protected boundaries (faults, junctions, thin beds) with violation rates.

## 3. Recommended next step

### Priority 1: candidate integration (A + F split + guards)

The experiments now support a concrete candidate:

- A's per-path P-spline with D2 + 0.25 D3, 8 px knots, λ=1;
- F's hairpin detector + split at reversals (safest); `corner_knot` only if one-sided tangents have enough support;
- E's β=10 endpoint tangent penalty is optional: it improves endpoint curvature by 8.5% at negligible cost, but it is not the main fix;
- B's `bspline_lsq` remains an alternative kernel for speed/comparison;
- 3 px corridor, 2% closed-area guard, final width guard;
- C's local topology validator + iterative λ reduction + exact fallback;
- D's area restoration only for small closed bodies; never for degenerate loops.

Run this candidate on all 11 panels + synthetic cases and compare against A/F per path. The expected result is F's geometry with A's metrics plus zero topology violations.

### Priority 2: decide hairpin semantics

The detector is geometric, not geological. For each anchor, the agent/human must decide:

- real corner/pinch-out → preserve with split or corner knot;
- skeleton artifact → prune/split;
- sub-resolution label noise → exact/review.

Do not let the detector alone decide. In particular, p37's 2–3 px finger may be label noise at panel scale.

### Priority 3: fault benchmark

- Either annotate explicit fault masks on the two faulted curated panels, or restrict fault-aware claims to synthetic cases.
- Do not infer a fault from high curvature alone; mark it `review`.
- Add fault offset/corner error to the benchmark.

### Priority 4: visual audit

Inspect the worst cases first:

- `newimage_p0` — long exterior boundary and small purple lens;
- `paper23_fault_guided` — B's `splprep`/`pspline` cross-intersections, A's junction smoothing;
- `schematic_panel2` — p37 hairpin, long thin structures;
- `paper23_fig7a` — p13 tongue tip;
- `newimage_p2` — the tiny closed loop collapse;
- synthetic `syn_normal_fault`, `syn_three_layer_fault`, and F's `syn_v_cusp`/`syn_pinchout`.

## 4. Artifacts

- A: `runs/layer2_experiments/a_segmented_spline/` (v1 archived at `a_segmented_spline_v1/`)
- B: `runs/layer2_experiments/b_kernel_benchmark/` (v1 archived at `b_kernel_benchmark_v1/`)
- C: `runs/layer2_experiments/c_fault_topology/`
- D: `runs/layer2_experiments/d_closed_aniso/`
- E: `runs/layer2_experiments/e_endpoint_tangent/`
- F: `runs/layer2_experiments/f_hairpin/`
- Shared harness: `experiments/layer2/common.py` (path graph v2)
- Experiment contract: `experiments/layer2/README.md`

Recommended visual entry points:

- A contact: `runs/layer2_experiments/a_segmented_spline/contact.png`
- B contact: `runs/layer2_experiments/b_kernel_benchmark/contact.png`
- C synthetic grids: `runs/layer2_experiments/c_fault_topology/panels/syn_normal_fault_grid.png`, `syn_three_layer_fault_grid.png`
- D contact: `runs/layer2_experiments/d_closed_aniso/contact.png`
- E endpoint zoom: `runs/layer2_experiments/e_endpoint_tangent/endpoint_zoom.png`
- F hairpin zoom: `runs/layer2_experiments/f_hairpin/hairpin_zoom.png`
- F synthetic cases: `runs/layer2_experiments/f_hairpin/synthetic.png`
