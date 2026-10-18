#!/usr/bin/env python3
"""Panel iteration history, grouped by campaign family.

    python scripts/dev/panel_history.py [--min-campaigns 2] [--limit 40]

Generic pipeline reruns (e2e_export_v2/v3, agent_review) are collapsed into one
family so they do not dominate the count. Panels that were revisited by several
campaigns late in the project are the mature cases worth using as the best set.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = Path.home() / "geoseg-artifacts" / "2026-10" / "runs"
HEX = re.compile(r"\b[0-9a-f]{8,64}\b")
FIG6 = re.compile(r"fig6_profile_(\d+)")


def campaign(rel: Path) -> str:
    parts = rel.parts
    root = parts[0] if parts else "unknown"
    if root.startswith("feng_fig6"):
        return root
    if root.startswith("fig6_colorbar_16zone"):
        return root
    if root.startswith("pm_repair"):
        return root
    if root.startswith("preprocess_newimage"):
        return root
    if root.startswith("horizon_refine"):
        return "horizon_refine"
    if root.startswith("j_tecto"):
        return f"{root}/{parts[2]}" if len(parts) > 2 else root
    if root.startswith("3d_schematic"):
        return "3d_schematic"
    if root in {"agent_review", "agent_review_vlm"}:
        return root
    if root == "literature_test":
        run = parts[2] if len(parts) > 2 else ""
        return "literature_test/e2e" if run.startswith("e2e_export") else f"literature_test/{run}"
    if root == "readme_examples_v2":
        return "readme_examples"
    if root == "self_heal_v1":
        return "self_heal"
    if root == "sandbox":
        return "sandbox"
    if root == "fig6_profile_all_best_summary":
        return "fig6_best_summary"
    return root


def panel_key(rel: Path, labels: Path) -> str:
    parts = rel.parts
    root = parts[0] if parts else ""
    text = str(rel)
    m = FIG6.search(text)
    if m:
        return f"feng:fig6_profile_{m.group(1)}"
    if root.startswith("j_tecto"):
        return f"jtecto:{parts[1] if len(parts) > 1 else labels.parent.name}"
    if root.startswith("preprocess_newimage"):
        pm = re.search(r"panel_(\d+)", text)
        return f"newimage:panel_{pm.group(1)}" if pm else "newimage:unknown"
    if root.startswith("sandbox"):
        return f"sandbox:{parts[1] if len(parts) > 1 else labels.parent.name}"
    if root == "self_heal_v1":
        return f"selfheal:{parts[1] if len(parts) > 1 else labels.parent.name}"
    if root == "readme_examples_v2":
        name = parts[1] if len(parts) > 1 else labels.parent.name
        paper = name.split("_")[0]
        return f"readme:{paper}:{name}"

    idx_m = re.search(r"panel(\d+)_", labels.name)
    idx = int(idx_m.group(1)) if idx_m else 0
    parent = labels.parent.name
    if HEX.fullmatch(parent):
        paper = parts[1] if len(parts) > 1 else "unknown"
        return f"{paper}:{parent}:p{idx}"
    paper = parts[1] if len(parts) > 1 else root
    return f"{paper}:{parent}:p{idx}"


def normalize_keys(rows: list[dict]) -> None:
    """Map short hash prefixes and known aliases onto the canonical key."""
    hashes: dict[tuple[str, str], str] = {}
    for r in rows:
        key = r["panel_key"]
        if ":" in key:
            parts = key.split(":")
            if len(parts) >= 2 and HEX.fullmatch(parts[1]):
                hashes[(parts[0], parts[1][:7])] = key
                hashes[(parts[0], parts[1][:8])] = key

    alias = {
        "silixa_page5": "silixa2021:page5_img0:p0",
        "silixa_page4": "silixa2021:page4_img0:p0",
        "ph01_page011": "ph01:page_011:p0",
    }
    for r in rows:
        key = r["panel_key"]
        if key.startswith("readme:"):
            _, paper, name = key.split(":", 2)
            for token in name.split("_"):
                if (paper, token[:7]) in hashes or (paper, token) in hashes:
                    r["panel_key"] = hashes[(paper, token)]
                    break
            else:
                for token, target in alias.items():
                    if token in name:
                        r["panel_key"] = target
                        break
        elif key.startswith("selfheal:"):
            name = key.split(":", 1)[1]
            for (paper, prefix), target in hashes.items():
                if prefix in name:
                    r["panel_key"] = target
                    break
        elif key.startswith(("merged:", "auto:", "visual:")) or key.startswith("panel_"):
            m = re.search(r"panel_(\d+)", key)
            if m:
                r["panel_key"] = f"newimage:panel_{m.group(1)}"


def scan(runs: Path) -> list[dict]:
    import numpy as np

    records = []
    for labels in runs.rglob("*labels*.npz"):
        rel = labels.relative_to(runs)
        try:
            with np.load(labels) as data:
                shape = data["labels"].shape
        except Exception:
            continue
        stat = labels.stat()
        records.append({
            "panel_key": panel_key(rel, labels),
            "campaign": campaign(rel),
            "run": str(rel.parent),
            "labels": str(labels),
            "shape": shape,
            "mtime": stat.st_mtime,
            "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d"),
        })
    normalize_keys(records)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="panel iteration history")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--min-campaigns", type=int, default=2)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    records = scan(args.runs)
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["panel_key"], []).append(r)

    rows = []
    for key, items in groups.items():
        campaigns = sorted({i["campaign"] for i in items})
        if len(campaigns) < args.min_campaigns:
            continue
        items.sort(key=lambda i: i["mtime"])
        latest = items[-1]
        rows.append({
            "panel_key": key,
            "n_campaigns": len(campaigns),
            "n_runs": len({i["run"] for i in items}),
            "n_files": len(items),
            "first": items[0]["date"],
            "last": latest["date"],
            "shape": list(latest["shape"]),
            "campaigns": campaigns,
            "latest_labels": latest["labels"],
        })

    rows.sort(key=lambda r: (r["n_campaigns"], r["n_runs"], r["last"]), reverse=True)
    rows = rows[: args.limit]

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    print(f"{'panel':46s} {'camp':>4s} {'runs':>4s} {'files':>5s}  {'first':10s} {'last':10s} shape")
    for r in rows:
        print(
            f"{r['panel_key']:46s} {r['n_campaigns']:4d} {r['n_runs']:4d} {r['n_files']:5d}  "
            f"{r['first']:10s} {r['last']:10s} {r['shape'][0]}x{r['shape'][1]}"
        )
        print(f"    campaigns: {', '.join(r['campaigns'])}")
        print(f"    latest: {r['latest_labels']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
