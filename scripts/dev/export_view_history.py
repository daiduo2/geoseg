#!/usr/bin/env python3
"""Rank panels by export and review evidence.

    python scripts/dev/export_view_history.py [--min-exports 1] [--limit 40]

Export evidence: tomo.xyz / Par_file snippet / polygons.geojson / properties.json.
View evidence: visual_audit directories, overlay-with-legend, comparison or
side-by-side renders, and mentions in run reports. Both are counted per
distinct run directory so repeated pipeline reruns do not inflate the score.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = Path.home() / "geoseg-artifacts" / "2026-10" / "runs"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "dev"))
import panel_history as ph  # noqa: E402

EXPORT_PATTERNS = ("*tomo.xyz", "*parfile*", "*polygons*.geojson", "*properties*.json")
AUDIT_DIR = "visual_audit"
LEGEND_PATTERNS = ("*overlay_legend*",)
COMPARE_PATTERNS = ("*comparison*", "*side_by_side*", "*numbered*")


def date_of(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="export/view ranking")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--min-exports", type=int, default=1)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    runs = args.runs

    # Map result directories to a panel key via their label files.
    dir_key: dict[Path, str] = {}
    for labels in runs.rglob("*labels*.npz"):
        rel = labels.relative_to(runs)
        key = ph.panel_key(rel, labels)
        dir_key.setdefault(labels.parent, key)
        if labels.parent.name == "visual_audit":
            dir_key.setdefault(labels.parent.parent, key)

    agg: dict[str, dict] = defaultdict(lambda: {
        "export_dirs": set(),
        "export_files": 0,
        "view_dirs": set(),
        "audit_dirs": set(),
        "legend": 0,
        "compare": 0,
        "dates": [],
    })

    def key_for(path: Path) -> str | None:
        for candidate in (
            path.parent,
            path.parent.parent,
            path.parent.parent.parent,
            path.parent.parent.parent.parent,
        ):
            if candidate in dir_key:
                return dir_key[candidate]
        return None

    for pattern in EXPORT_PATTERNS:
        for path in runs.rglob(pattern):
            key = key_for(path)
            if key is None:
                continue
            rec = agg[key]
            rec["export_dirs"].add(path.parent)
            rec["export_files"] += 1
            rec["dates"].append(date_of(path))

    for path in runs.rglob(AUDIT_DIR):
        if not path.is_dir():
            continue
        key = key_for(path)
        if key is None:
            continue
        rec = agg[key]
        rec["audit_dirs"].add(path)
        rec["view_dirs"].add(path)
        rec["dates"].append(date_of(path))

    for pattern in LEGEND_PATTERNS:
        for path in runs.rglob(pattern):
            key = key_for(path)
            if key is None:
                continue
            rec = agg[key]
            rec["legend"] += 1
            rec["view_dirs"].add(path.parent)
            rec["dates"].append(date_of(path))

    for pattern in COMPARE_PATTERNS:
        for path in runs.rglob(pattern):
            if path.is_dir():
                continue
            key = key_for(path)
            if key is None:
                continue
            rec = agg[key]
            rec["compare"] += 1
            rec["view_dirs"].add(path.parent)
            rec["dates"].append(date_of(path))

    rows = []
    for key, rec in agg.items():
        if len(rec["export_dirs"]) < args.min_exports:
            continue
        rows.append({
            "panel_key": key,
            "export_dirs": len(rec["export_dirs"]),
            "export_files": rec["export_files"],
            "view_dirs": len(rec["view_dirs"]),
            "audit_dirs": len(rec["audit_dirs"]),
            "legend": rec["legend"],
            "compare": rec["compare"],
            "first": min(rec["dates"]) if rec["dates"] else "",
            "last": max(rec["dates"]) if rec["dates"] else "",
        })

    rows.sort(key=lambda r: (r["export_dirs"], r["view_dirs"], r["last"]), reverse=True)
    rows = rows[: args.limit]

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    print(f"{'panel':58s} {'exp':>3s} {'files':>5s} {'view':>4s} {'audit':>5s} {'legend':>6s} {'cmp':>4s} {'last':10s}")
    for r in rows:
        print(
            f"{r['panel_key'][:58]:58s} {r['export_dirs']:3d} {r['export_files']:5d} "
            f"{r['view_dirs']:4d} {r['audit_dirs']:5d} {r['legend']:6d} {r['compare']:4d} {r['last']:10s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
