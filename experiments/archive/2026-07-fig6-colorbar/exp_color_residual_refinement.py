"""Color residual-guided regional refinement experiment.

Loads existing fig6_profile_* segmentation results, computes per-pixel color
residuals, generates audit materials, and optionally re-segments high-deviation
regions. All refinement decisions are left to agent visual audit; this script
does not auto-accept or auto-reject.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Allow running from repo root without installation.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend
from geoseg.modules.segment_engines.regional_refinement import (
    RefinementConfig,
    refine_by_candidate_regions,
    refine_by_residual_mask,
)
from geoseg.modules.visual_audit import create_audit_report
from geoseg.modules.visual_audit.color_residual import (
    compute_color_residual_map,
    compute_label_representative_colors,
    compute_label_residual_stats,
    create_color_residual_overlay,
    estimate_text_mask,
    find_high_deviation_regions,
)


SUMMARY_PATH = repo_root / "runs" / "fig6_profile_all_best_summary" / "summary.json"
OUTPUT_ROOT = repo_root / "runs" / "exp_color_residual"


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def load_profile(profile_id: str) -> dict:
    """Load image and labels for a profile from the summary JSON."""
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    for entry in summary["profiles"]:
        if entry["panel_id"] == profile_id:
            return entry
    raise ValueError(f"Profile {profile_id} not found in {SUMMARY_PATH}")


def process_profile(
    profile_id: str,
    refine: bool = False,
    percentile: float = 95.0,
    min_area_frac: float = 0.005,
    filter_text: bool = False,
    secondary_engine: str = "edge_guided",
) -> Path:
    """Run color residual audit and optional refinement for one profile."""
    entry = load_profile(profile_id)
    image_path = Path(entry["image_path"])
    labels_path = Path(entry["labels_path"])

    panel_rgb = np.array(Image.open(image_path).convert("RGB"))
    labels = np.load(labels_path)["labels"]

    out_dir = OUTPUT_ROOT / profile_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Representative colors and residual map.
    representatives = compute_label_representative_colors(labels, panel_rgb)
    residual_map = compute_color_residual_map(labels, panel_rgb, representatives)

    # 1b. Optional text filtering: zero out residual in text regions so they
    # are not proposed as re-segmentation candidates.
    text_mask = None
    if filter_text:
        text_mask = estimate_text_mask(panel_rgb, dilation_iterations=2)
        residual_map = residual_map.copy()
        residual_map[text_mask] = 0.0

    stats = compute_label_residual_stats(labels, panel_rgb, residual_map)
    candidates = find_high_deviation_regions(
        labels,
        residual_map,
        min_area_frac=min_area_frac,
        deviation_percentile=percentile,
    )

    # 2. Save standalone residual heatmap.
    heatmap_overlay = create_color_residual_overlay(
        residual_map, panel_rgb, labels, alpha=0.5
    )
    heatmap_path = out_dir / "01_residual_heatmap.jpg"
    Image.fromarray(heatmap_overlay).save(heatmap_path, quality=90)

    # 3. Save candidate overlay with red boxes.
    candidate_overlay = create_color_residual_overlay(
        residual_map, panel_rgb, labels, candidates=candidates, alpha=0.5
    )
    candidate_path = out_dir / "02_residual_candidates.jpg"
    Image.fromarray(candidate_overlay).save(candidate_path, quality=90)

    if filter_text and text_mask is not None:
        text_view = panel_rgb.copy()
        text_view[text_mask] = (text_view[text_mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        text_path = out_dir / "02b_text_mask.jpg"
        Image.fromarray(text_view).save(text_path, quality=90)
        print(f"  text mask: {text_path}")

    # 4. Full audit report (includes color_residual view).
    audit_dir = out_dir / "audit"
    report = create_audit_report(labels, panel_rgb, str(audit_dir))

    # 5. Save residual diagnostics JSON.
    diagnostics = {
        "profile_id": profile_id,
        "representative_colors": {
            str(lbl): {
                "median_rgb": reps["median_rgb"].tolist(),
                "median_lab": reps["median_lab"].tolist(),
            }
            for lbl, reps in representatives.items()
        },
        "per_label_residual": stats,
        "high_deviation_regions": candidates,
    }
    diag_path = out_dir / "residual_diagnostics.json"
    diag_path.write_text(
        json.dumps(_to_serializable(diagnostics), indent=2), encoding="utf-8"
    )

    print(f"\n=== {profile_id} ===")
    print(f"  image: {image_path}")
    print(f"  labels: {labels_path}")
    print(f"  heatmap: {heatmap_path}")
    print(f"  candidates: {candidate_path}")
    print(f"  audit summary: {report['summary_image_path']}")
    print(f"  diagnostics: {diag_path}")
    print(f"  candidates found: {len(candidates)}")
    for i, cand in enumerate(candidates[:5], 1):
        print(f"    {i}. bbox={cand['bbox']} area={cand['area']} "
              f"mean_delta_e={cand['mean_delta_e']} max={cand['max_delta_e']}")

    if refine and candidates:
        # 6. Optional local refinement: process each candidate bbox independently
        # so unrelated areas stay frozen.
        config = RefinementConfig(secondary_engine=secondary_engine)
        refined = refine_by_candidate_regions(
            labels,
            panel_rgb,
            candidates,
            n_layers=max(3, len(np.unique(labels)) - 1),
            config=config,
        )

        refined_labels = refined["labels"]
        refined_overlay = refined["overlay"]

        refined_labels_path = out_dir / "labels_refined.npz"
        np.savez_compressed(refined_labels_path, labels=refined_labels)

        refined_overlay_path = out_dir / "03_overlay_refined.jpg"
        Image.fromarray(refined_overlay).save(refined_overlay_path, quality=90)

        # 7. Audit refined result.
        refined_audit_dir = out_dir / "audit_refined"
        create_audit_report(refined_labels, panel_rgb, str(refined_audit_dir))

        # 8. Side-by-side comparison.
        original_overlay = generate_overlay_with_legend(panel_rgb, labels)
        h, w = panel_rgb.shape[:2]
        gap = 10
        comparison = np.full((h, w * 3 + gap * 2, 3), 32, dtype=np.uint8)
        comparison[:, :w] = original_overlay
        comparison[:, w + gap : 2 * w + gap] = heatmap_overlay
        comparison[:, 2 * (w + gap) :] = refined_overlay
        compare_path = out_dir / "04_compare_original_refined.jpg"
        Image.fromarray(comparison).save(compare_path, quality=90)

        print(f"  refined labels: {refined_labels_path}")
        print(f"  refined overlay: {refined_overlay_path}")
        print(f"  comparison: {compare_path}")

        print(f"  refinement meta: {json.dumps(_to_serializable(refined['meta']), indent=2)}")

    return out_dir


def compare_engines(
    profile_id: str,
    engines: list[str] | None = None,
    percentile: float = 95.0,
    min_area_frac: float = 0.005,
    filter_text: bool = False,
) -> Path:
    """Run per-candidate refinement with multiple engines side-by-side.

    Produces a single comparison image with the original overlay, residual
    heatmap, and one refined overlay column per engine.
    """
    engines = engines or ["v4_kmeans", "kmeans_full", "edge_guided"]

    entry = load_profile(profile_id)
    image_path = Path(entry["image_path"])
    labels_path = Path(entry["labels_path"])

    panel_rgb = np.array(Image.open(image_path).convert("RGB"))
    labels = np.load(labels_path)["labels"]

    out_dir = OUTPUT_ROOT / profile_id / "engine_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    representatives = compute_label_representative_colors(labels, panel_rgb)
    residual_map = compute_color_residual_map(labels, panel_rgb, representatives)

    text_mask = None
    if filter_text:
        text_mask = estimate_text_mask(panel_rgb, dilation_iterations=2)
        residual_map = residual_map.copy()
        residual_map[text_mask] = 0.0

    candidates = find_high_deviation_regions(
        labels,
        residual_map,
        min_area_frac=min_area_frac,
        deviation_percentile=percentile,
    )

    original_overlay = generate_overlay_with_legend(panel_rgb, labels)
    heatmap_overlay = create_color_residual_overlay(
        residual_map, panel_rgb, labels, alpha=0.5
    )

    refined_overlays: dict[str, np.ndarray] = {}
    refined_meta: dict[str, dict] = {}
    n_layers = max(3, len(np.unique(labels)) - 1)

    for engine in engines:
        config = RefinementConfig(secondary_engine=engine)
        result = refine_by_candidate_regions(
            labels,
            panel_rgb,
            candidates,
            n_layers=n_layers,
            config=config,
        )
        refined_overlays[engine] = result["overlay"]
        refined_meta[engine] = result["meta"]

    # Build side-by-side comparison grid: original | heatmap | engine_0 | ...
    h, w = panel_rgb.shape[:2]
    gap = 10
    n_cols = 2 + len(engines)
    comparison = np.full((h, w * n_cols + gap * (n_cols - 1), 3), 32, dtype=np.uint8)
    comparison[:, :w] = original_overlay
    comparison[:, w + gap : 2 * w + gap] = heatmap_overlay
    for idx, engine in enumerate(engines):
        x0 = (2 + idx) * (w + gap)
        comparison[:, x0 : x0 + w] = refined_overlays[engine]

    compare_path = out_dir / "compare_engines.jpg"
    Image.fromarray(comparison).save(compare_path, quality=90)

    # Save per-engine refined labels and overlays for closer inspection.
    for engine in engines:
        engine_dir = out_dir / engine
        engine_dir.mkdir(parents=True, exist_ok=True)
        # Labels were not retained in the loop above, so re-run once to save.
        config = RefinementConfig(secondary_engine=engine)
        result = refine_by_candidate_regions(
            labels,
            panel_rgb,
            candidates,
            n_layers=n_layers,
            config=config,
        )
        np.savez_compressed(engine_dir / "labels_refined.npz", labels=result["labels"])
        Image.fromarray(result["overlay"]).save(
            engine_dir / "overlay_refined.jpg", quality=90
        )

    diagnostics = {
        "profile_id": profile_id,
        "engines": engines,
        "candidates": candidates,
        "refined_meta": refined_meta,
        "comparison_path": str(compare_path),
    }
    diag_path = out_dir / "comparison_diagnostics.json"
    diag_path.write_text(
        json.dumps(_to_serializable(diagnostics), indent=2), encoding="utf-8"
    )

    print(f"\n=== {profile_id} engine comparison ===")
    print(f"  candidates: {len(candidates)}")
    print(f"  comparison: {compare_path}")
    for engine in engines:
        print(f"  [{engine}] n_regions={len(refined_meta[engine]['refined_regions'])}")

    return out_dir


def assemble_five_figure_comparison(
    profile_ids: list[str] | None = None,
    target_width: int = 2000,
    output_path: Path | None = None,
) -> Path:
    """Stack per-profile engine-comparison images into one large figure.

    Layout: one row per profile. Each row contains the original overlay,
    residual heatmap, and refined overlays for the compared engines.
    """
    profile_ids = profile_ids or [
        "fig6_profile_03",
        "fig6_profile_04",
        "fig6_profile_05",
        "fig6_profile_06",
        "fig6_profile_07",
    ]

    rows: list[Image.Image] = []
    for profile_id in profile_ids:
        img_path = OUTPUT_ROOT / profile_id / "engine_comparison" / "compare_engines.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing comparison image: {img_path}")
        img = Image.open(img_path).convert("RGB")
        # Normalize width while keeping aspect ratio.
        aspect = img.height / img.width
        new_height = int(target_width * aspect)
        img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
        rows.append(img)

    row_height = rows[0].height
    header_height = 60
    label_width = 120
    gap = 10
    total_height = header_height + len(rows) * (row_height + gap)
    canvas_width = label_width + target_width + gap

    canvas = Image.new("RGB", (canvas_width, total_height), color=(32, 32, 32))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    # Column headers inferred from the first row image.
    # The comparison image has: original | heatmap | engine_0 | engine_1 | engine_2
    header_labels = ["original", "residual", "v4_kmeans", "kmeans_full", "edge_guided"]
    n_cols = len(header_labels)
    col_width = target_width // n_cols
    for idx, label in enumerate(header_labels):
        x = label_width + gap + idx * col_width + col_width // 2
        draw.text((x, header_height // 2 - 10), label, fill=(255, 255, 255), font=small_font, anchor="mm")

    for row_idx, (profile_id, row_img) in enumerate(zip(profile_ids, rows)):
        y = header_height + row_idx * (row_height + gap)
        # Row label.
        draw.text((label_width // 2, y + row_height // 2), profile_id, fill=(255, 255, 255), font=font, anchor="mm")
        # Row image.
        canvas.paste(row_img, (label_width + gap, y))

    if output_path is None:
        output_path = OUTPUT_ROOT / "five_figure_engine_comparison.jpg"
    canvas.save(output_path, quality=92)
    print(f"Five-figure comparison saved to: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Color residual-guided regional refinement experiment."
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="fig6_profile_03",
        help="Profile ID to process (default: fig6_profile_03).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all fig6_profile_* profiles in the summary.",
    )
    parser.add_argument(
        "--secondary-engine",
        type=str,
        default="edge_guided",
        choices=["v4_kmeans", "kmeans_full", "edge_guided", "edge_grow", "grayscale"],
        help="Secondary engine for local refinement (default: edge_guided).",
    )
    parser.add_argument(
        "--compare-engines",
        action="store_true",
        help="Run refinement with all engines and generate a comparison image.",
    )
    parser.add_argument(
        "--engines",
        type=str,
        nargs="+",
        default=["v4_kmeans", "kmeans_full", "edge_guided"],
        choices=["v4_kmeans", "kmeans_full", "edge_guided", "edge_grow", "grayscale"],
        help="Engines to compare when --compare-engines is set.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Run local refinement on high-deviation candidate regions.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Deviation percentile for candidate extraction (default: 95).",
    )
    parser.add_argument(
        "--min-area-frac",
        type=float,
        default=0.005,
        help="Minimum candidate area as fraction of image (default: 0.005).",
    )
    parser.add_argument(
        "--filter-text",
        action="store_true",
        help="Estimate and exclude text/annotation regions from candidates.",
    )
    parser.add_argument(
        "--assemble",
        action="store_true",
        help="Stack existing per-profile engine comparisons into a 5-figure composite.",
    )
    args = parser.parse_args()

    if args.assemble:
        assemble_five_figure_comparison()
        return

    if args.all:
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        profile_ids = [entry["panel_id"] for entry in summary["profiles"]]
    else:
        profile_ids = [args.profile]

    for profile_id in profile_ids:
        if args.compare_engines:
            compare_engines(
                profile_id,
                engines=args.engines,
                percentile=args.percentile,
                min_area_frac=args.min_area_frac,
                filter_text=args.filter_text,
            )
        else:
            process_profile(
                profile_id,
                refine=args.refine,
                percentile=args.percentile,
                min_area_frac=args.min_area_frac,
                filter_text=args.filter_text,
                secondary_engine=args.secondary_engine,
            )


if __name__ == "__main__":
    main()
