#!/usr/bin/env python3
"""Load and check the validation corpus.

    python scripts/dev/corpus.py stats
    python scripts/dev/corpus.py list --split dev
    python scripts/dev/corpus.py check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "validation"


def manifest(root: Path = DEFAULT_ROOT) -> list[dict]:
    """Return all usable corpus records."""
    return [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]


def labels_only(root: Path = DEFAULT_ROOT) -> list[dict]:
    path = root / "labels_only.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def iter_split(split: str, root: Path = DEFAULT_ROOT):
    """Yield records assigned to ``split`` (dev/holdout/rare/negative)."""
    for record in manifest(root):
        if record["split"] == split:
            yield record


def load_panel(panel_id: str, root: Path = DEFAULT_ROOT):
    """Return ``(image_rgb, labels, record)`` with shapes asserted equal."""
    record = next(r for r in manifest(root) if r["panel_id"] == panel_id)
    image = np.array(Image.open(root / record["files"]["image"]).convert("RGB"))
    labels = np.load(root / record["files"]["labels"])["labels"]
    if image.shape[:2] != labels.shape:
        raise ValueError(
            f"{panel_id}: image {image.shape[:2]} != labels {labels.shape}"
        )
    return image, labels, record


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_stats(args: argparse.Namespace) -> int:
    rows = manifest(args.root)
    from collections import Counter

    print(f"root: {args.root}")
    print(f"usable: {len(rows)}  labels_only: {len(labels_only(args.root))}")
    print("split:", dict(Counter(r["split"] for r in rows)))
    print("tier:", dict(Counter(r["tier"] for r in rows)))
    print("n_labels>=2:", sum(1 for r in rows if r["labels"]["n_labels"] >= 2))
    print("flags:", dict(Counter(f for r in rows for f in r["flags"])))
    print("sources:", dict(Counter(r["source"]["root"] for r in rows)))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    rows = manifest(args.root)
    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
    print(f"{'panel_id':44s} {'split':9s} {'tier':8s} {'n':>3s} {'shape':>12s} flags")
    for r in rows:
        shape = "x".join(str(v) for v in r["labels"]["shape"])
        print(
            f"{r['panel_id']:44s} {r['split']:9s} {r['tier']:8s} "
            f"{r['labels']['n_labels']:3d} {shape:>12s} {','.join(r['flags'])}"
        )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    rows = manifest(args.root)
    problems: list[str] = []
    for r in rows:
        for kind in ("image", "labels"):
            rel = r["files"][kind]
            path = args.root / rel
            if not path.exists():
                problems.append(f"{r['panel_id']}: missing {rel}")
                continue
            expected = r["files"][f"{kind}_sha256"]
            if args.hashes and _sha256(path) != expected:
                problems.append(f"{r['panel_id']}: {kind} hash mismatch")
        image = np.array(Image.open(args.root / r["files"]["image"]).convert("RGB"))
        labels = np.load(args.root / r["files"]["labels"])["labels"]
        if image.shape[:2] != labels.shape:
            problems.append(
                f"{r['panel_id']}: shape {image.shape[:2]} != {labels.shape}"
            )
    print(f"checked {len(rows)} records: {len(problems)} problems")
    for p in problems:
        print(" ", p)
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="validation corpus helper")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats").set_defaults(func=cmd_stats)

    listing = sub.add_parser("list")
    listing.add_argument("--split", choices=["dev", "holdout", "rare", "negative"])
    listing.set_defaults(func=cmd_list)

    check = sub.add_parser("check")
    check.add_argument("--hashes", action="store_true", help="also verify sha256")
    check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
