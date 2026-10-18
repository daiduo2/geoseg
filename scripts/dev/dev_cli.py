#!/usr/bin/env python3
"""Developer task entry points.

Wired up by the repository Makefile so contributors do not have to remember
which script implements which task:

    make segment IMAGE=path/to/panel.png N=5 OUT=runs/cli
    make map
    make bench

``map`` is generated from the live package (``geoseg.api`` and the engine
registry), so it cannot drift like a hand-written index.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def cmd_segment(args: argparse.Namespace) -> int:
    from geoseg import api

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"image not found: {image_path}", file=sys.stderr)
        return 2

    img = np.array(Image.open(image_path).convert("RGB"))
    out_dir = Path(args.out)
    result = api.run_pipeline(
        img,
        n_layers=args.n_layers,
        use_vlm=not args.no_vlm,
        skip_non_velocity_model=not args.force,
        output_dir=out_dir,
        save_intermediates=True,
    )
    print(f"status: {result['status']}")
    if result.get("reason"):
        print(f"reason: {result['reason']}")
    for panel in result.get("panels", []):
        status = panel.get("status")
        pid = panel.get("panel_id", "?")
        polygons = panel.get("n_polygons", "?")
        engine = panel.get("engines_used", "?")
        print(f"panel {pid}: {status}  polygons={polygons}  engine={engine}")
    if result["status"] == "ok":
        print(f"outputs: {out_dir}")
    return 0 if result["status"] in {"ok", "skipped"} else 1


def cmd_map(_: argparse.Namespace) -> int:
    from geoseg import api
    from geoseg.modules.segment_engines.registry import ENGINE_REGISTRY

    print("GEOSEG MAP\n")
    print("Entry points")
    print("  pipeline   geoseg.pipeline.run.run_pipeline   (classify → segment → export)")
    print("  stage      geoseg.pipeline.segment.run_segmentation_stage")
    print("  public API geoseg.api                        (scripts/skills import here)")
    print("  batch      geoseg.batch.process_directory    (python -m geoseg.batch)")
    print("  editor     geoseg.modules.editor.napari_app  (python -m ...)")
    print()
    print(f"Public API ({len(api.__all__)} names)")
    for name in sorted(api.__all__):
        print(f"  {name}")
    print()
    print("Segmentation engines (registry is the only dispatch source)")
    print(f"  {'name':24s} {'reps':5s} {'fallback':16s} callable")
    for name, spec in ENGINE_REGISTRY.items():
        reps = "yes" if spec.requires_reps else "no"
        fallback = spec.fallback_engine or "-"
        print(f"  {name:24s} {reps:5s} {fallback:16s} {spec.callable_path}")
    print()
    print("See docs/CODEBASE.md for module responsibilities and docs/ALGORITHM_FAMILY.md for engine history.")
    return 0


def cmd_tools(_: argparse.Namespace) -> int:
    import ast

    rows = []
    for path in sorted((REPO_ROOT / "scripts").glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            doc = (ast.get_docstring(tree) or "").strip().splitlines()
        except SyntaxError:
            doc = []
        first = doc[0] if doc else ""
        rows.append((path.name, first))
    width = max(len(name) for name, _ in rows)
    print("Maintained scripts (scripts/):")
    for name, first in rows:
        print(f"  {name:{width}s}  {first}")
    print("\nTask wrappers live in the Makefile: run `make help`.")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    from synthetic_benchmark.runner import default_benchmark, run_benchmark

    output = Path(args.out) if args.out else REPO_ROOT / "runs" / "benchmark" / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = run_benchmark(default_benchmark(), output_path=output)
    print(f"panels: {summary['config']['n_panels']}  engines: {', '.join(summary['config']['engines'])}")
    print(f"results: {output}")
    for engine, stats in sorted(summary["aggregated"].items()):
        print(
            f"  {engine:26s} mIoU mean={stats['mIoU_mean']:.3f} "
            f"median={stats['mIoU_median']:.3f}  boundary_f1={stats['boundary_f1_mean']:.3f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="geoseg developer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    segment = sub.add_parser("segment", help="run the full pipeline on one image")
    segment.add_argument("image")
    segment.add_argument("--n-layers", type=int, default=5)
    segment.add_argument("--out", default="runs/cli")
    segment.add_argument("--no-vlm", action="store_true")
    segment.add_argument("--force", action="store_true", help="process even if classified as observational data")
    segment.set_defaults(func=cmd_segment)

    sub.add_parser("map", help="print the public API and engine registry").set_defaults(
        func=cmd_map
    )

    bench = sub.add_parser("bench", help="run the synthetic engine benchmark")
    bench.add_argument("--out", default=None)
    bench.set_defaults(func=cmd_bench)

    sub.add_parser("tools", help="list maintained scripts with their purpose").set_defaults(
        func=cmd_tools
    )

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
