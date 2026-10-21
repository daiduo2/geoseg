# Layer 2 experiment results (tracked summaries)

This directory keeps the text summaries and cross-experiment analysis in git.
Full metrics, overlays, contact sheets and per-panel images are generated under
`runs/layer2_experiments/` (ignored by git; on this machine `runs` points to
the artifact archive).

- `analysis.md` — cross-experiment analysis, including the path-graph v2,
  endpoint-tangent and hairpin results.
- `a_segmented_spline_summary.md` — constrained per-boundary P-spline baseline.
- `b_kernel_benchmark_summary.md` — smoothing-kernel comparison.
- `c_fault_topology_summary.md` — junction/fault handling and topology fallback.
- `d_closed_aniso_summary.md` — closed-body area/width preservation.
- `e_endpoint_tangent_summary.md` — endpoint tangent constraints (small gain).
- `f_hairpin_summary.md` — interior direction reversals and split handling.

Reproduce from the repository root:

```bash
uv run python experiments/layer2/a_segmented_spline/run.py
uv run python experiments/layer2/b_kernel_benchmark/run.py
uv run python experiments/layer2/c_fault_topology/run_experiment.py
uv run python experiments/layer2/d_closed_aniso/run.py
uv run python experiments/layer2/e_endpoint_tangent/run.py
uv run python experiments/layer2/f_hairpin/run.py
```

The shared data/retention/centerline/metric/render contract is
`experiments/layer2/common.py`; the experiment rules are in
`experiments/layer2/README.md`.
