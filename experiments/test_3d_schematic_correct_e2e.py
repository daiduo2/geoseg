"""E2E regional fusion on 3D schematic panels WITH proper preprocessing.

Uses the proven text-removal + enhancement pipeline from process_v4_unified.py
before segmentation. Regional fusion is applied on the CLEANED images.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from geoseg.modules.segment_engines.metrics import compute_all
from geoseg.modules.segment_engines.regional_fusion import (
    FusionConfig,
    RegionalAudit,
    generate_overlay_with_legend,
    regional_segment,
)
from geoseg.modules.segment_engines.v4_kmeans import segment as v4_segment
from geoseg.modules.segment_engines.v4_kmeans import _reorder_labels_by_median_y


# ---- Reuse proven preprocessing from process_v4_unified.py ----


def remove_text_v2(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive threshold + Laplacian + inpaint + median fill + Gaussian blend."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=25, C=-5
    )
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    laplacian = (laplacian > np.percentile(laplacian, 80)).astype(np.uint8) * 255
    text_mask = ((adaptive > 0) & (laplacian > 0))
    labeled, num = ndimage.label(text_mask)
    text_mask_clean = np.zeros_like(text_mask)
    for i in range(1, num + 1):
        comp = labeled == i
        if 8 < comp.sum() < 1200:
            text_mask_clean[comp] = True
    kernel = np.ones((5, 5), np.uint8)
    text_dilated = cv2.dilate(text_mask_clean.astype(np.uint8), kernel, iterations=2)
    inpainted = cv2.inpaint(image, text_dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    cleaned = inpainted.copy()
    mask_bool = text_dilated.astype(bool)
    for ch in range(3):
        channel = inpainted[:, :, ch].astype(np.float32)
        ys, xs = np.where(mask_bool)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 3), min(image.shape[0], y + 4)
            x0, x1 = max(0, x - 3), min(image.shape[1], x + 4)
            patch = channel[y0:y1, x0:x1]
            patch_mask = mask_bool[y0:y1, x0:x1]
            valid = patch[~patch_mask]
            if len(valid) > 0:
                cleaned[y, x, ch] = int(np.median(valid))
            else:
                cleaned[y, x, ch] = int(channel[y, x])
    blurred = cv2.GaussianBlur(cleaned, (7, 7), sigmaX=1.5)
    mask_f = text_dilated.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (5, 5), sigmaX=2)
    mask_3ch = np.stack([mask_f] * 3, axis=-1)
    cleaned = (blurred * mask_3ch + cleaned * (1 - mask_3ch)).astype(np.uint8)
    return cleaned, text_dilated


def enhance_v(image: np.ndarray) -> np.ndarray:
    """CLAHE enhancement on V channel."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ---- E2E pipeline ----


def process_panel(panel_path: Path, output_dir: Path, n_layers: int = 5):
    print(f"\n{'='*60}")
    print(f"Processing: {panel_path.name}")
    print(f"{'='*60}")

    panel_rgb = np.array(Image.open(panel_path).convert("RGB"))
    print(f"Original shape: {panel_rgb.shape}")

    panel_id = panel_path.stem
    out = output_dir / panel_id
    out.mkdir(parents=True, exist_ok=True)

    # Stage 1: Text removal (proven from process_v4_unified)
    print("\n[1] Text removal v2")
    cleaned, text_mask = remove_text_v2(panel_rgb)
    Image.fromarray(cleaned).save(out / "00_cleaned.jpg", quality=90)
    print(f"  Text mask pixels: {text_mask.sum()}")

    # Stage 2: Enhancement
    print("\n[2] CLAHE enhancement")
    enhanced = enhance_v(cleaned)
    Image.fromarray(enhanced).save(out / "00_enhanced.jpg", quality=90)

    # Stage 3: Primary engine on CLEANED image
    print("\n[3] Primary: v4_kmeans on cleaned image")
    result_a = v4_segment(enhanced, n_layers=n_layers)
    labels_a = result_a["labels"]
    overlay_a = result_a["overlay"]

    Image.fromarray(overlay_a).save(out / "01_primary.jpg", quality=90)
    np.savez_compressed(out / "labels_primary.npz", labels=labels_a)
    unique = set(labels_a.flatten()) - {0}
    print(f"  Labels: {sorted(unique)}, shape={labels_a.shape}")

    # Stage 4: Legend overlay
    print("\n[4] Legend overlay")
    overlay_legend = generate_overlay_with_legend(enhanced, labels_a)
    Image.fromarray(overlay_legend).save(out / "02_legend.jpg", quality=90)

    # Stage 5: Metrics
    print("\n[5] Per-label metrics")
    metrics = compute_all(labels_a, enhanced)
    per_label = metrics.get("per_label", {})

    print(f"  Overall: n_layers={metrics['n_layers']}, "
          f"boundary_alignment={metrics['boundary_alignment']:.3f}")
    for lbl, m in sorted(per_label.items()):
        print(f"    Label {lbl}: ba={m['boundary_alignment']:.3f}, "
              f"area={m['area_fraction']:.3f}, tiny={m['has_tiny_fragments']}")

    # Stage 6: Simulate audit
    sorted_labels = sorted(
        per_label.items(),
        key=lambda x: x[1]["boundary_alignment"],
        reverse=True,
    )
    n_freeze = max(1, len(sorted_labels) // 2)
    frozen = [lbl for lbl, _ in sorted_labels[:n_freeze]]
    retry = [lbl for lbl, _ in sorted_labels[n_freeze:]]

    print(f"\n[6] Audit: freeze={frozen}, retry={retry}")

    audit = RegionalAudit(
        frozen_labels=frozen,
        retry_labels=retry,
        notes=f"Simulated: freeze top-{n_freeze} aligned on cleaned image",
        iteration=1,
    )

    # Stage 7: Regional fusion
    print("\n[7] Regional fusion with edge_guided")
    result_fused = regional_segment(
        enhanced,
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
        print("  Freeze mask empty")

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
    output_dir = Path("runs/3d_schematic_correct_e2e")
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
