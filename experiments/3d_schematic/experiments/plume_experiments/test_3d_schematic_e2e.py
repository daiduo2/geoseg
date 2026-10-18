"""E2E regional fusion on 3D schematic panels."""
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
from geoseg.modules.segment_engines.v4_kmeans import _reorder_labels_by_median_y


def process_panel(panel_path: Path, output_dir: Path, n_layers: int = 5):
    print(f"\n{'='*60}")
    print(f"Processing: {panel_path.name}")
    print(f"{'='*60}")

    panel_rgb = np.array(Image.open(panel_path).convert("RGB"))
    print(f"Shape: {panel_rgb.shape}")

    panel_id = panel_path.stem
    out = output_dir / panel_id
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Primary engine
    print("\n[1] Primary: v4_kmeans")
    result_a = v4_segment(panel_rgb, n_layers=n_layers)
    labels_a = result_a["labels"]
    overlay_a = result_a["overlay"]

    Image.fromarray(overlay_a).save(out / "01_primary.jpg", quality=90)
    np.savez_compressed(out / "labels_primary.npz", labels=labels_a)
    unique = set(labels_a.flatten()) - {0}
    print(f"  Labels: {sorted(unique)}, shape={labels_a.shape}")

    # Step 2: Legend overlay
    print("\n[2] Legend overlay")
    overlay_legend = generate_overlay_with_legend(panel_rgb, labels_a)
    Image.fromarray(overlay_legend).save(out / "02_legend.jpg", quality=90)

    # Step 3: Metrics
    print("\n[3] Per-label metrics")
    metrics = compute_all(labels_a, panel_rgb)
    per_label = metrics.get("per_label", {})

    print(f"  Overall: n_layers={metrics['n_layers']}, "
          f"boundary_alignment={metrics['boundary_alignment']:.3f}")
    for lbl, m in sorted(per_label.items()):
        print(f"    Label {lbl}: ba={m['boundary_alignment']:.3f}, "
              f"area={m['area_fraction']:.3f}, tiny={m['has_tiny_fragments']}")

    # Step 4: Simulate audit — freeze top half by boundary alignment
    sorted_labels = sorted(
        per_label.items(),
        key=lambda x: x[1]["boundary_alignment"],
        reverse=True,
    )
    n_freeze = max(1, len(sorted_labels) // 2)
    frozen = [lbl for lbl, _ in sorted_labels[:n_freeze]]
    retry = [lbl for lbl, _ in sorted_labels[n_freeze:]]

    print(f"\n[4] Audit: freeze={frozen}, retry={retry}")

    audit = RegionalAudit(
        frozen_labels=frozen,
        retry_labels=retry,
        notes=f"Simulated: freeze top-{n_freeze} aligned",
        iteration=1,
    )

    # Step 5: Regional fusion
    print("\n[5] Regional fusion with edge_guided")
    result_fused = regional_segment(
        panel_rgb,
        n_layers=n_layers,
        primary_result={"labels": labels_a},
        audit=audit,
        config=FusionConfig(
            primary_engine="v4_kmeans",
            secondary_engines=["edge_guided", "kmeans_full"],
            seam_smooth_width=3,
        ),
    )
    labels_fused = result_fused["labels"]
    overlay_fused = result_fused["overlay"]

    Image.fromarray(overlay_fused).save(out / "03_fused.jpg", quality=90)
    np.savez_compressed(out / "labels_fused.npz", labels=labels_fused)

    meta = result_fused["meta"]
    print(f"  Fusion: {meta['fusion_applied']}, engines={meta['engine']}")

    # Verify freeze preservation
    labels_a_reordered = _reorder_labels_by_median_y(labels_a)
    freeze_mask = np.zeros(labels_a_reordered.shape, dtype=bool)
    for lbl in frozen:
        freeze_mask |= labels_a_reordered == lbl

    if freeze_mask.any():
        preserved = np.sum(
            (labels_a_reordered[freeze_mask] == labels_fused[freeze_mask]).astype(int)
        )
        total_frozen = freeze_mask.sum()
        print(f"  Freeze preservation: {preserved}/{total_frozen} "
              f"({preserved/total_frozen*100:.1f}%)")
    else:
        print("  Freeze mask empty (no frozen labels)")

    return {
        "panel": panel_id,
        "n_layers": len(unique),
        "frozen": frozen,
        "retry": retry,
        "fusion_applied": meta["fusion_applied"],
        "output_dir": str(out),
    }


def main():
    base = Path("src/3d_schematic")
    output_dir = Path("runs/3d_schematic_e2e")
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        (base / "panel_1_front.png", 5),
        (base / "panel_2_front.png", 5),
        (base / "panel_3_front.png", 6),
    ]

    summaries = []
    for path, n in panels:
        if not path.exists():
            print(f"SKIP: {path} not found")
            continue
        try:
            summary = process_panel(path, output_dir, n_layers=n)
            summaries.append(summary)
        except Exception as e:
            print(f"ERROR processing {path.name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for s in summaries:
        print(f"\n{s['panel']}:")
        print(f"  n_layers={s['n_layers']}, frozen={s['frozen']}, retry={s['retry']}")
        print(f"  fusion={s['fusion_applied']}, dir={s['output_dir']}")

    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False)
    )
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
