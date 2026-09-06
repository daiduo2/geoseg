"""Synthetic benchmark for geoseg segmentation engines.

Generates 2D conceptual-model panels with ground-truth labels, applies
realistic-looking noise (text artifacts, colorbars, contour lines, JPEG
compression), runs the existing segmentation engines on them, and reports
quantitative metrics (mIoU, Boundary F1, topology error, fragment count)
alongside the existing unsupervised diagnostics from
`geoseg.modules.segment_engines.diagnostics.metrics`.

Inspired by Wu et al. (Geophysics 2020) "Building a Structural Model" —
their procedural 3D generator became a unit-test gold mine for DL pipelines.
We adapt the spirit (parameterised synthesis with known ground truth) to
geoseg's 2D colour-clustering use case.

Why this lives under experiments/: per CLAUDE.md Promotion Rules, an
experimental feature only graduates into `src/geoseg/` after it has stable
function entry points, minimal tests, no hard-coded paths, and an updated
module CLAUDE.md or CODEBASE.md. Until then it stays here.

Import notes: this package is imported both as a top-level
`synthetic_benchmark.*` namespace (when running scripts/tests from the
project root) and as `experiments.synthetic_benchmark.*` (when sys.path
includes the project root). We lazy-import the submodules so neither
order is forced at package import time.
"""

from __future__ import annotations

import importlib as _il
import sys as _sys

__all__ = [
    "BenchmarkConfig",
    "PanelConfig",
    "generate_panel",
    "apply_noise",
    "evaluate",
    "run_benchmark",
    "default_benchmark",
    "synthesize_reps",
    "synthesize_noisy_reps",
]

_MODULES = ("configs", "generator", "noise", "evaluate", "runner", "vlm_seeds")


def _load(sub: str):
    """Load `synthetic_benchmark.<sub>` from the right namespace.

    When this package is imported as `experiments.synthetic_benchmark`,
    `synthetic_benchmark` is *not* on sys.path (only `experiments` is).
    When it's imported as a top-level package (e.g. by pytest from the
    project root), `synthetic_benchmark` *is* on sys.path. Try both.
    """
    try:
        return _il.import_module(f"synthetic_benchmark.{sub}")
    except ImportError:
        from pathlib import Path
        import importlib.util as _ilu
        import types as _ts

        here = Path(__file__).resolve().parent
        # First call: register a synthetic parent package so that
        # `from synthetic_benchmark.configs import ...` inside the
        # loaded submodules works.
        parent_name = "synthetic_benchmark"
        if parent_name not in _sys.modules:
            pkg = _ts.ModuleType(parent_name)
            pkg.__path__ = [str(here)]
            _sys.modules[parent_name] = pkg
        spec = _ilu.spec_from_file_location(
            f"{parent_name}.{sub}", here / f"{sub}.py"
        )
        if spec is None or spec.loader is None:
            raise
        mod = _ilu.module_from_spec(spec)
        _sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name in {"BenchmarkConfig", "PanelConfig"}:
        cfg = _load("configs")
        return {"BenchmarkConfig": cfg.BenchmarkConfig,
                "PanelConfig": cfg.PanelConfig}[name]
    if name == "generate_panel":
        return _load("generator").generate_panel
    if name == "apply_noise":
        return _load("noise").apply_noise
    if name == "evaluate":
        return _load("evaluate").evaluate
    if name == "run_benchmark":
        return _load("runner").run_benchmark
    if name == "default_benchmark":
        return _load("runner").default_benchmark
    if name in {"synthesize_reps", "synthesize_noisy_reps"}:
        return getattr(_load("vlm_seeds"), name)
    raise AttributeError(name)
