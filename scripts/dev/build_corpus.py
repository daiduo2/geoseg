#!/usr/bin/env python3
"""Build the validation corpus from accumulated runs.

Scans the run trees for label maps, resolves an aligned clean panel image for
each, deduplicates candidates that describe the same panel, and writes either a
report (``--report-only``) or the corpus:

    validation/
      manifest.jsonl        one record per canonical panel
      images/<id>.jpg       clean panel crop, pixel-aligned with labels
      labels/<id>.npz       int32 label map (0 = background unless flagged)
      overlays/<id>.jpg     overlay when available
      report.json

Run trees live in ~/geoseg-artifacts/2026-10/runs (the repo's runs/ symlink).
Nothing in the run trees is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = Path.home() / "geoseg-artifacts" / "2026-10" / "runs"
CROP_CACHE = Path(tempfile.mkdtemp(prefix="geoseg-corpus-"))

TIER_RANK = {"human": 3, "audited": 2, "auto": 1, "candidate": 0}
SKIP_NAME_PARTS = ("overlay", "detected", "comparison", "comparisons", "montage", "grid")


@dataclass
class Candidate:
    source_root: str
    mode: str
    tier: str
    paper: str
    figure_hash: str
    panel_index: int
    labels_path: Path
    image_path: Path | None = None
    image_source: str = "missing"
    page_path: Path | None = None
    crop_bbox: list[int] | None = None
    overlay_path: Path | None = None
    engine: str | None = None
    figure_type: str | None = None
    label_convention: str = "zero_is_background"
    label_stats: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    variants: list[dict] = field(default_factory=list)
    panel_id_override: str = ""

    @property
    def panel_id(self) -> str:
        if self.panel_id_override:
            return self.panel_id_override
        if self.mode == "fig6_summary":
            return f"feng_fig6_p{self.figure_hash}"
        if self.mode == "self_heal":
            return f"selfheal_{self.figure_hash}"
        if self.mode == "jtecto":
            return f"jtecto_{self.figure_hash}"
        return f"{self.paper}_{self.figure_hash[:8]}_p{self.panel_index}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def clean_sibling(labels_path: Path, shape: tuple[int, int]) -> Path | None:
    """Find a non-overlay image in the same directory whose size matches."""
    h, w = shape
    preferred = (
        "01_panel_crop.jpg", "01_panel_crop.png", "01_panel.png", "01_panel.jpg",
        "panel_cropped.png", "panel_cropped.jpg", "panel.png", "panel.jpg",
    )
    for name in preferred:
        p = labels_path.parent / name
        if p.exists() and image_size(p) == (w, h):
            return p
    for p in sorted(labels_path.parent.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if any(part in p.name.lower() for part in SKIP_NAME_PARTS):
            continue
        if image_size(p) == (w, h):
            return p
    return None


def crop_original_to_shape(
    original: Path, shape: tuple[int, int]
) -> tuple[Path, Path, list[int]] | None:
    """Detect panels on a page image and crop the one matching the label shape.

    Returns ``(crop_path, page_path, bbox)`` so provenance survives the build.
    """
    from geoseg.modules.cv_detect.panel_detector import detect_panels

    try:
        page = np.array(Image.open(original).convert("RGB"))
        panels = detect_panels(page)
    except Exception:
        return None
    h, w = shape
    exact = [p for p in panels if tuple(p["bbox"][2:]) == (w, h)]
    near = [
        p for p in panels
        if abs(p["bbox"][2] - w) <= 3 and abs(p["bbox"][3] - h) <= 3
    ]
    matches = exact or near
    if len(matches) != 1:
        return None
    x, y, pw, ph = matches[0]["bbox"]
    if x + w > page.shape[1] or y + h > page.shape[0]:
        return None
    crop = page[y : y + h, x : x + w]
    out = CROP_CACHE / f"crop_{original.stem}_{w}x{h}.png"
    Image.fromarray(crop).save(out)
    return out, original, [x, y, w, h]


def label_stats(labels: np.ndarray, convention: str) -> tuple[dict, list[str]]:
    unique = sorted(int(v) for v in np.unique(labels))
    h, w = labels.shape
    total = h * w
    if convention == "zero_is_background":
        bg = 0 if 0 in unique else None
    elif convention == "negative_is_background":
        bg = -1 if -1 in unique else None
    else:
        bg = None
    regions = [v for v in unique if v != bg and v >= 0]
    areas = {v: int((labels == v).sum()) for v in regions}
    tiny = 0
    thin = 0

    for v, area in areas.items():
        if area / total < 0.001:
            tiny += 1
        mask = labels == v
        ys = np.where(mask.any(axis=1))[0]
        if len(ys) and (ys[-1] - ys[0] + 1) / h < 0.02 and area / total > 0.001:
            thin += 1
    stats = {
        "n_labels": len(regions),
        "background_label": bg,
        "background_fraction": round(1.0 - sum(areas.values()) / total, 4) if bg is not None else 0.0,
        "tiny_fragments": tiny,
        "thin_layers": thin,
        "min_area_fraction": round(min(areas.values()) / total, 5) if areas else 0.0,
    }
    flags = []
    if tiny > 50:
        flags.append("many_tiny_fragments")
    if len(regions) == 0:
        flags.append("empty_segmentation")
    if len(regions) == 1:
        flags.append("single_region")
    if len(regions) > 20:
        flags.append("many_regions")
    if thin:
        flags.append("thin_layer_candidate")
    if bg is not None and stats["background_fraction"] > 0.6:
        flags.append("large_background")
    return stats, flags


def candidate_quality(c: Candidate) -> tuple:
    stats = c.label_stats
    return (
        TIER_RANK.get(c.tier, 0),
        c.image_path is not None,
        -stats.get("tiny_fragments", 0),
        -abs(stats.get("n_labels", 0) - 5),
        c.image_source == "clean_crop",
    )


def build_index(roots: list[Path]) -> dict[str, Path]:
    """Map figure hash -> page original across agent_review trees."""
    index: dict[str, Path] = {}
    for root in roots:
        for p in root.rglob("01_original.jpg"):
            index.setdefault(p.parent.name, p)
    return index


def build_mineru_index(runs: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    patterns = [
        "literature_test/*/mineru/extracted/images/*",
        "mineru/extracted/images/*",
        "mineru/*/images/*",
    ]
    for pattern in patterns:
        for p in runs.glob(pattern):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                index.setdefault(p.stem, p)
    return index


def scan(runs: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    originals = build_index([runs / "agent_review", runs / "agent_review_vlm"])
    mineru = build_mineru_index(runs)

    # 1. Curated fig6 best set
    fig6_summary = runs / "fig6_profile_all_best_summary" / "summary.json"
    if fig6_summary.exists():
        data = json.loads(fig6_summary.read_text())
        for profile in data.get("profiles", []):
            labels = Path(profile["labels_path"])
            image = Path(profile["image_path"])
            if not labels.exists() or not image.exists():
                continue
            lid = profile["panel_id"].replace("fig6_profile_", "")
            c = Candidate(
                "fig6_profile_all_best_summary", "fig6_summary", "audited", "feng", lid, 0,
                labels, image, "clean_crop",
                label_convention="zero_is_region",
            )
            candidates.append(c)

    # 2. self-heal finals
    for d in (runs / "self_heal_v1").glob("*"):
        labels = d / "final_labels.npz"
        image = d / "01_panel_crop.jpg"
        if labels.exists() and image.exists():
            candidates.append(Candidate(
                "self_heal_v1", "self_heal", "audited", "selfheal", d.name, 0,
                labels, image, "clean_crop",
            ))

    # 3. j_tecto curated panels
    jt = runs / "j_tecto_2019_06_024_direct_colorbar"
    for labels in jt.rglob("*labels*.npz"):
        panel_img = labels.parent.parent / "01_panel.png"
        if not panel_img.exists():
            continue
        candidates.append(Candidate(
            jt.name, "jtecto", "audited", "jtecto", labels.parent.parent.name, 0,
            labels, panel_img, "clean_crop",
            label_convention="negative_is_background",
        ))

    # 4. agent_review / agent_review_vlm / literature_test panel outputs
    for root_name, tier in (("agent_review_vlm", "audited"), ("agent_review", "auto"), ("literature_test", "auto")):
        for labels in (runs / root_name).rglob("panel*_labels.npz"):
            try:
                with np.load(labels) as data:
                    shape = data["labels"].shape
            except Exception:
                continue
            parent = labels.parent
            fig_hash = parent.name
            panel_idx = int(labels.stem.split("_")[0].replace("panel", ""))
            image = clean_sibling(labels, shape)
            image_source = "clean_crop" if image else "missing_no_original"
            if image is None:
                original = parent / "01_original.jpg"
                if not original.exists():
                    original = mineru.get(fig_hash)
                if original is None:
                    original = originals.get(fig_hash)
                if original is not None:
                    result = crop_original_to_shape(original, shape)
                    if result is not None:
                        image, page_path, crop_bbox = result
                        image_source = "redetected_crop"
            overlay = parent / f"panel{panel_idx}_overlay.jpg"
            rel = labels.relative_to(runs / root_name)
            paper = rel.parts[0] if len(rel.parts) > 1 else root_name
            candidate = Candidate(
                root_name, "panel_npz", tier, paper, fig_hash, panel_idx,
                labels, image, image_source,
                overlay_path=overlay if overlay.exists() else None,
            )
            if image_source == "redetected_crop":
                candidate.page_path = page_path
                candidate.crop_bbox = crop_bbox
            candidates.append(candidate)

    # 5. panel-level experiment directories
    for root_name, tier in (
        ("readme_examples_v2", "audited"),
        ("sandbox", "auto"),
        ("preprocess_newimage_merged", "auto"),
    ):
        root = runs / root_name
        if not root.exists():
            continue
        for labels in root.rglob("labels*.npz"):
            try:
                with np.load(labels) as data:
                    shape = data["labels"].shape
            except Exception:
                continue
            image = clean_sibling(labels, shape)
            image_source = "clean_crop" if image else "missing_no_original"
            if image is None:
                original = next(
                    (p for p in [labels.parent / "01_original.jpg", labels.parent.parent / "01_original.jpg", labels.parent.parent.parent / "01_original.jpg"] if p.exists()),
                    None,
                )
                if original is not None:
                    result = crop_original_to_shape(original, shape)
                    if result is not None:
                        image, page_path, crop_bbox = result
                        image_source = "redetected_crop"
            candidates.append(Candidate(
                root_name, "panel_dir", tier, root_name, labels.parent.name, 0,
                labels, image, image_source,
                panel_id_override=f"{root_name}_{labels.parent.name}_{labels.stem}",
            ))

    return candidates


def enrich(c: Candidate) -> None:
    with np.load(c.labels_path) as data:
        labels = data["labels"].astype(np.int32)
    c.label_stats, c.flags = label_stats(labels, c.label_convention)
    c.label_stats["shape"] = list(labels.shape)

    summary = c.labels_path.parent / "05_summary.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text())
            c.figure_type = (data.get("classification") or {}).get("figure_type")
            c.engine = (data.get("pipeline_summary") or {}).get("engines_used", [None])[0]
        except Exception:
            pass


def split_of(c: Candidate) -> str:
    if "single_region" in c.flags:
        return "negative"
    if "thin_layer_candidate" in c.flags:
        return "rare"
    digest = int(hashlib.md5(c.panel_id.encode()).hexdigest(), 16)
    return "dev" if digest % 10 < 7 else "holdout"


def build(candidates: list[Candidate], out: Path, report_only: bool) -> dict:
    for c in candidates:
        enrich(c)

    groups: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        if c.mode == "panel_npz":
            key = f"figure:{c.figure_hash}:p{c.panel_index}:{tuple(c.label_stats.get('shape', []))}"
        else:
            key = c.panel_id
        groups[key].append(c)

    canonical: list[Candidate] = []
    for key, group in groups.items():
        group.sort(key=candidate_quality, reverse=True)
        best = group[0]
        best.variants = [
            {
                "labels_path": str(v.labels_path),
                "root": v.source_root,
                "tier": v.tier,
                "n_labels": v.label_stats.get("n_labels"),
                "tiny_fragments": v.label_stats.get("tiny_fragments"),
            }
            for v in group[1:]
        ]
        if best.label_stats.get("n_labels", 0) == 0:
            continue
        canonical.append(best)

    # second pass: identical resolved images describe the same panel
    by_image: dict[str, list[Candidate]] = defaultdict(list)
    no_image: list[Candidate] = []
    for c in canonical:
        if c.image_path is not None:
            by_image[sha256_file(c.image_path)].append(c)
        else:
            no_image.append(c)
    merged: list[Candidate] = []
    for group in by_image.values():
        group.sort(key=candidate_quality, reverse=True)
        best = group[0]
        for other in group[1:]:
            if other.panel_id != best.panel_id:
                best.variants.append({
                    "labels_path": str(other.labels_path),
                    "root": other.source_root,
                    "tier": other.tier,
                    "n_labels": other.label_stats.get("n_labels"),
                    "tiny_fragments": other.label_stats.get("tiny_fragments"),
                })
            best.variants.extend(other.variants)
        merged.append(best)
    canonical = merged + no_image

    # unique ids for panels that only differ by crop shape
    id_counts = defaultdict(int)
    for c in canonical:
        id_counts[c.panel_id] += 1
    for c in canonical:
        if id_counts[c.panel_id] > 1:
            h, w = c.label_stats.get("shape", [0, 0])
            c.panel_id_override = f"{c.panel_id}_{h}x{w}"
    id_counts = defaultdict(int)
    for c in canonical:
        id_counts[c.panel_id] += 1
    for c in canonical:
        if id_counts[c.panel_id] > 1:
            c.panel_id_override = f"{c.panel_id}_{sha256_file(c.labels_path)[:8]}"

    if not report_only:
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        (out / "overlays").mkdir(parents=True, exist_ok=True)

    manifest = []
    labels_only = []
    for c in canonical:
        split = split_of(c)
        record = {
            "panel_id": c.panel_id,
            "tier": c.tier,
            "split": split,
            "source": {
                "root": c.source_root,
                "mode": c.mode,
                "paper": c.paper,
                "figure_hash": c.figure_hash,
                "panel_index": c.panel_index,
                "labels_path": str(c.labels_path),
                "image_path": str(c.page_path or c.image_path) if c.image_path else None,
                "crop_bbox": c.crop_bbox,
                "overlay_path": str(c.overlay_path) if c.overlay_path else None,
                "image_source": c.image_source,
            },
            "labels": {
                **c.label_stats,
                "convention": c.label_convention,
                "sha256": sha256_file(c.labels_path),
            },
            "engine": c.engine,
            "figure_type": c.figure_type,
            "flags": c.flags,
            "variants": c.variants,
        }
        if c.image_path is None:
            labels_only.append(record)
            continue
        if not report_only:
            img = Image.open(c.image_path).convert("RGB")
            img_path = out / "images" / f"{c.panel_id}.jpg"
            img.save(img_path, quality=92)
            with np.load(c.labels_path) as data:
                np.savez_compressed(out / "labels" / f"{c.panel_id}.npz", labels=data["labels"].astype(np.int32))
            if c.overlay_path is not None:
                Image.open(c.overlay_path).convert("RGB").save(out / "overlays" / f"{c.panel_id}.jpg", quality=90)
            with Image.open(img_path) as saved:
                if saved.size != (c.label_stats["shape"][1], c.label_stats["shape"][0]):
                    record["flags"].append("image_label_shape_mismatch")
            record["files"] = {
                "image": f"images/{c.panel_id}.jpg",
                "labels": f"labels/{c.panel_id}.npz",
                "overlay": f"overlays/{c.panel_id}.jpg" if c.overlay_path else None,
                "image_sha256": sha256_file(img_path),
                "labels_sha256": sha256_file(out / "labels" / f"{c.panel_id}.npz"),
            }
        manifest.append(record)

    usable = [c for c in canonical if c.image_path is not None]
    report = {
        "runs_root": str(runs),
        "n_candidates": len(candidates),
        "n_panels": len(canonical),
        "n_usable": len(usable),
        "n_labels_only": len(labels_only),
        "n_core": sum(1 for c in usable if c.label_stats.get("n_labels", 0) >= 2),
        "n_single_region": sum(1 for c in usable if "single_region" in c.flags),
        "by_tier": dict(sorted((t, sum(1 for c in usable if c.tier == t)) for t in TIER_RANK)),
        "by_image_source": dict(sorted((s, sum(1 for c in canonical if c.image_source == s)) for s in {c.image_source for c in canonical})),
        "by_source": dict(sorted((s, sum(1 for c in canonical if c.source_root == s)) for s in {c.source_root for c in canonical})),
        "by_split": dict(sorted((s, sum(1 for c in usable if split_of(c) == s)) for s in ("dev", "holdout", "rare", "negative"))),
        "flags": dict(sorted((f, sum(1 for c in usable if f in c.flags)) for f in {f for c in usable for f in c.flags})),
        "n_shape_mismatch": sum(1 for r in manifest if "image_label_shape_mismatch" in r["flags"]),
        "missing_image": [c.panel_id for c in canonical if c.image_path is None],
    }
    if not report_only:
        (out / "manifest.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest) + "\n",
            encoding="utf-8",
        )
        (out / "labels_only.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in labels_only) + "\n",
            encoding="utf-8",
        )
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        splits_dir = out / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        for split in ("dev", "holdout", "rare", "negative"):
            ids = [r["panel_id"] for r in manifest if r["split"] == split]
            (splits_dir / f"{split}.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the validation corpus")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "validation")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--clean", action="store_true", help="remove old images/labels/overlays/splits first")
    args = parser.parse_args(argv)

    global runs
    runs = args.runs
    if not runs.exists():
        print(f"runs root not found: {runs}", file=sys.stderr)
        return 2

    if args.clean and not args.report_only:
        for sub in ("images", "labels", "overlays", "splits"):
            shutil.rmtree(args.out / sub, ignore_errors=True)

    candidates = scan(runs)
    report = build(candidates, args.out, args.report_only)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
