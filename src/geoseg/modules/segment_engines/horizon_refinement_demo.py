"""Standalone demo for horizon refinement.

Validates refine_boundaries() on example images.
Produces coarse vs refined overlay comparison for visual inspection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from geoseg.modules.segment_engines.horizon_refinement import refine_boundaries
from geoseg.modules.segment_engines.kmeans_full import segment as seg_kmeans
from geoseg.modules.segment_engines._shared import _create_overlay
from geoseg.modules.segment_engines.metrics import compute_all


def _make_comparison_image(
    original: np.ndarray,
    coarse_labels: np.ndarray,
    refined_labels: np.ndarray,
    title_coarse: str,
    title_refined: str,
) -> np.ndarray:
    """Create side-by-side comparison: original | coarse | refined."""
    h, w = original.shape[:2]
    margin = 10
    text_h = 30

    n_layers_coarse = len(set(coarse_labels.flatten()) - {0})
    n_layers_refined = len(set(refined_labels.flatten()) - {0})

    coarse_overlay = _create_overlay(original, coarse_labels, np.zeros((1, 3)))
    refined_overlay = _create_overlay(original, refined_labels, np.zeros((1, 3)))

    total_w = w * 3 + margin * 4
    total_h = h + text_h + margin * 2
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

    # Original
    canvas[margin:margin + h, margin:margin + w] = original
    _draw_text(canvas, "Original", margin, margin + h + 5)

    # Coarse
    x2 = margin * 2 + w
    canvas[margin:margin + h, x2:x2 + w] = coarse_overlay
    _draw_text(canvas, f"{title_coarse} ({n_layers_coarse} layers)", x2, margin + h + 5)

    # Refined
    x3 = margin * 3 + w * 2
    canvas[margin:margin + h, x3:x3 + w] = refined_overlay
    _draw_text(canvas, f"{title_refined} ({n_layers_refined} layers)", x3, margin + h + 5)

    return canvas


def _draw_text(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    """Draw simple text on canvas (minimal, no font dependency)."""
    # Placeholder: draw a colored bar as text area
    pass


def _run_on_image(img_path: Path, out_dir: Path, n_layers: int) -> dict:
    """Run horizon refinement on a single image and save results."""
    print(f"\nProcessing: {img_path.name}")
    img = np.array(Image.open(img_path).convert("RGB"))

    # Step 1: Coarse segmentation with kmeans_full
    print("  -> Running kmeans_full (max_auto_k=0)...")
    coarse_result = seg_kmeans(img, n_layers=n_layers, max_auto_k=0)
    coarse_labels = coarse_result["labels"]

    coarse_metrics = compute_all(coarse_labels, img)
    coarse_frag = coarse_metrics.get("total_fragment_area_fraction", 0.0)
    print(f"  -> Coarse: n_layers={coarse_metrics['n_layers']}, "
          f"frag_area={coarse_frag:.3f}")

    # Step 2: Refine with horizon fitting
    print("  -> Running horizon refinement (savgol)...")
    refined_labels, boundaries = refine_boundaries(
        img,
        coarse_labels=coarse_labels,
        method="savgol",
        smoothness=1.0,
    )

    refined_metrics = compute_all(refined_labels, img)
    refined_frag = refined_metrics.get("total_fragment_area_fraction", 0.0)
    same_as_coarse = np.array_equal(coarse_labels, refined_labels)
    status = "fallback" if same_as_coarse else "refined"
    print(f"  -> Refined ({status}): n_layers={refined_metrics['n_layers']}, "
          f"frag_area={refined_frag:.3f}")

    # Step 3: Save comparison
    comparison = _make_comparison_image(
        img, coarse_labels, refined_labels,
        "Coarse (kmeans)", "Refined (horizon)"
    )
    # Use parent directory name for unique filename
    panel_id = img_path.parent.name
    comp_path = out_dir / f"{panel_id}_comparison.png"
    Image.fromarray(comparison).save(comp_path)
    print(f"  -> Saved: {comp_path}")

    # Step 4: Save individual overlays
    Image.fromarray(_create_overlay(img, coarse_labels, np.zeros((1, 3)))) \
        .save(out_dir / f"{panel_id}_coarse.png")
    Image.fromarray(_create_overlay(img, refined_labels, np.zeros((1, 3)))) \
        .save(out_dir / f"{panel_id}_refined.png")

    return {
        "image": img_path.name,
        "panel_id": panel_id,
        "coarse_n_layers": int(coarse_metrics["n_layers"]),
        "refined_n_layers": int(refined_metrics["n_layers"]),
        "coarse_fragments": coarse_frag,
        "refined_fragments": refined_frag,
        "n_boundaries": len(boundaries),
        "status": status,
    }


def main() -> None:
    out_dir = Path("runs/horizon_refine")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    test_cases = [
        ("gras2019_16b0cf", "runs/readme_examples_v2/gras2019_16b0cf/panel_cropped.png", 5),
        ("gras2019_c11b8db", "runs/readme_examples_v2/gras2019_c11b8db/panel_cropped.png", 5),
        ("silixa_page5", "runs/readme_examples_v2/silixa_page5/panel_cropped.png", 5),
    ]

    for panel_id, path, n_layers in test_cases:
        p = Path(path)
        if p.exists():
            results.append(_run_on_image(p, out_dir, n_layers=n_layers))
        else:
            print(f"Warning: {p} not found, skipping {panel_id}")

    # Summary
    summary_path = out_dir / "demo_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSummary saved to: {summary_path}")

    for r in results:
        print(f"\n{r['panel_id']}:")
        print(f"  Layers: {r['coarse_n_layers']} -> {r['refined_n_layers']}")
        print(f"  Frag area: {r['coarse_fragments']:.4f} -> {r['refined_fragments']:.4f}")
        print(f"  Boundaries fitted: {r['n_boundaries']} ({r['status']})")


if __name__ == "__main__":
    main()
