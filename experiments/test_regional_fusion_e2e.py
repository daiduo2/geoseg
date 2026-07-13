"""End-to-end validation of regional fusion on a real panel.

Steps:
1. Run v4_kmeans as primary engine
2. Generate overlay with legend for agent audit
3. Compute per-label metrics
4. Simulate agent regional audit (manual frozen/retry selection)
5. Run regional fusion with secondary engine
6. Save comparison images
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from geoseg.modules.segment_engines.metrics import compute_all
from geoseg.modules.segment_engines.regional_fusion import (
    FusionConfig,
    RegionalAudit,
    generate_overlay_with_legend,
    regional_segment,
)
from geoseg.modules.segment_engines.v4_kmeans import segment as v4_segment
from geoseg.modules.segment_engines.edge_guided import segment as eg_segment


def main():
    # Use page_011 (larger, likely more complex)
    panel_path = Path("runs/test_panel_fix/page_011_img_0_panels.jpg")
    output_dir = Path("runs/regional_fusion_e2e")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading panel: {panel_path}")
    panel_rgb = np.array(Image.open(panel_path).convert("RGB"))
    print(f"Panel shape: {panel_rgb.shape}")

    n_layers = 5

    # Step 1: Primary engine (v4_kmeans)
    print("\n[Step 1] Running primary engine: v4_kmeans")
    result_a = v4_segment(panel_rgb, n_layers=n_layers)
    labels_a = result_a["labels"]
    overlay_a = result_a["overlay"]

    Image.fromarray(overlay_a).save(output_dir / "01_overlay_primary.jpg", quality=90)
    np.savez_compressed(output_dir / "labels_primary.npz", labels=labels_a)
    print(f"  Primary: {len(set(labels_a.flatten()) - {0})} layers")

    # Step 2: Generate overlay with legend
    print("\n[Step 2] Generating overlay with legend")
    overlay_legend = generate_overlay_with_legend(panel_rgb, labels_a)
    Image.fromarray(overlay_legend).save(output_dir / "02_overlay_legend.jpg", quality=90)

    # Step 3: Compute per-label metrics
    print("\n[Step 3] Computing per-label metrics")
    metrics = compute_all(labels_a, panel_rgb)
    per_label = metrics.get("per_label", {})

    print(f"  Overall: n_layers={metrics['n_layers']}, "
          f"boundary_alignment={metrics['boundary_alignment']}")
    print("  Per-label:")
    for lbl, m in sorted(per_label.items()):
        print(f"    Label {lbl}: ba={m['boundary_alignment']:.3f}, "
              f"area={m['area_fraction']:.3f}, tiny_fragments={m['has_tiny_fragments']}")

    # Step 4: Simulate agent regional audit
    # Based on per-label metrics: pick top 2 labels by boundary_alignment as frozen
    sorted_labels = sorted(
        per_label.items(),
        key=lambda x: x[1]["boundary_alignment"],
        reverse=True,
    )
    frozen = [lbl for lbl, _ in sorted_labels[:2]]
    retry = [lbl for lbl, _ in sorted_labels[2:]]

    print(f"\n[Step 4] Simulated agent audit:")
    print(f"  Frozen labels: {frozen} (best alignment)")
    print(f"  Retry labels: {retry} (need improvement)")

    audit = RegionalAudit(
        frozen_labels=frozen,
        retry_labels=retry,
        notes="Simulated: freeze top-2 aligned regions, retry rest",
        iteration=1,
    )

    # Step 5: Regional fusion
    print("\n[Step 5] Running regional fusion")
    result_fused = regional_segment(
        panel_rgb,
        n_layers=n_layers,
        primary_result={"labels": labels_a},
        audit=audit,
        config=FusionConfig(
            primary_engine="v4_kmeans",
            secondary_engines=["edge_guided"],
            seam_smooth_width=3,
        ),
    )

    labels_fused = result_fused["labels"]
    overlay_fused = result_fused["overlay"]

    Image.fromarray(overlay_fused).save(output_dir / "03_overlay_fused.jpg", quality=90)
    np.savez_compressed(output_dir / "labels_fused.npz", labels=labels_fused)

    meta = result_fused["meta"]
    print(f"  Fusion applied: {meta['fusion_applied']}")
    print(f"  Engines: {meta['engine']}")
    print(f"  Frozen: {meta['frozen_labels']}")
    print(f"  Retry: {meta['retry_labels']}")

    # Step 6: Also run secondary engine alone for comparison
    print("\n[Step 6] Running secondary engine alone for comparison")
    result_b = eg_segment(panel_rgb, n_layers=n_layers)
    overlay_b = result_b["overlay"]
    Image.fromarray(overlay_b).save(output_dir / "04_overlay_secondary.jpg", quality=90)

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"  01_overlay_primary.jpg    - v4_kmeans (primary)")
    print(f"  02_overlay_legend.jpg     - primary with legend")
    print(f"  03_overlay_fused.jpg      - regional fusion result")
    print(f"  04_overlay_secondary.jpg  - edge_guided (secondary, solo)")
    print(f"\nFusion config:")
    print(f"  Frozen: {frozen} (kept from v4_kmeans)")
    print(f"  Retry:  {retry} (replaced by edge_guided)")

    # Verify freeze mask preserved
    # NOTE: regional_segment internally reorders labels_a before fusion,
    # so we must compare against the reordered primary labels.
    from geoseg.modules.segment_engines.v4_kmeans import _reorder_labels_by_median_y

    labels_a_reordered = _reorder_labels_by_median_y(labels_a)
    freeze_mask = np.zeros(labels_a_reordered.shape, dtype=bool)
    for lbl in frozen:
        freeze_mask |= labels_a_reordered == lbl

    preserved = np.sum(
        (labels_a_reordered[freeze_mask] == labels_fused[freeze_mask]).astype(int)
    )
    total_frozen = freeze_mask.sum()
    print(f"\nFreeze preservation: {preserved}/{total_frozen} pixels "
          f"({preserved / total_frozen * 100:.1f}%)")


if __name__ == "__main__":
    main()
