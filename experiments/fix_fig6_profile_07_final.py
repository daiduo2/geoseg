#!/usr/bin/env python3
"""Final selection and copy of best variant for fig6_profile_07."""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
from PIL import Image

OUTPUT_DIR = "/Users/daiduo2/geoseg/runs/feng_fig6_final_v2/fig6_profile_07"
PANEL_PATH = "/Users/daiduo2/geoseg/runs/feng_fig6_panels_v4/fig6_profile_07.png"


def main():
    # Best variant is B: v4_kmeans colorbar_guided + median with n_layers=7
    best_variant = "B"
    best_dir = os.path.join(OUTPUT_DIR, f"variant_{best_variant}")

    # Load best variant data
    labels = np.load(os.path.join(best_dir, "labels.npz"))["labels"]
    overlay = np.array(Image.open(os.path.join(best_dir, "overlay_mask.jpg")))
    with open(os.path.join(best_dir, "meta.json")) as f:
        meta = json.load(f)

    # Save to main output directory
    np.savez(os.path.join(OUTPUT_DIR, "labels.npz"), labels=labels)
    Image.fromarray(overlay).save(os.path.join(OUTPUT_DIR, "overlay_mask.jpg"))

    # Create overlay_legend.jpg (same as mask for pure mask mode)
    Image.fromarray(overlay).save(os.path.join(OUTPUT_DIR, "overlay_legend.jpg"))

    # Save meta.json
    meta["panel_id"] = "fig6_profile_07"
    meta["best_variant"] = best_variant
    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save regional_audit.json
    unique = np.unique(labels[labels >= 0])
    label_stats = {}
    for lbl in unique:
        count = int((labels == lbl).sum())
        ys, xs = np.where(labels == lbl)
        label_stats[int(lbl)] = {
            "pixels": count,
            "median_y": float(np.median(ys)) if len(ys) > 0 else 0,
            "median_x": float(np.median(xs)) if len(xs) > 0 else 0,
        }

    audit = {
        "panel_id": "fig6_profile_07",
        "best_variant": best_variant,
        "n_layers": len(unique),
        "label_stats": label_stats,
        "issues_fixed": [
            "Split previously over-merged yellow LV-N and greenish-teal layers (was single 48703 px label, now split into label 3 and label 4)",
            "Increased n_layers from 4 to 7, better matching 5-6 visible color bands in source",
            "Added median post-processing for smoother boundaries",
        ],
        "remaining_issues": [
            "Label 4 (34297 px) may still merge greenish-teal and cyan regions - boundary is gradual in source colorbar",
            "Top region (labels 1-2, ~5000 px combined) may be over-segmented into thin strips",
            "Bottom labels 5-6 may represent a single layer split by color gradient",
        ],
    }
    with open(os.path.join(OUTPUT_DIR, "regional_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    # Create comparison.jpg (original vs mask)
    img = Image.open(PANEL_PATH).convert("RGB")
    img_rgb = np.array(img)
    h, w, _ = img_rgb.shape
    # Extract panel (same logic as segmentation script)
    colorbar_height = min(40, h // 5)
    panel_rgb = img_rgb[:-colorbar_height, :]
    ph, pw, _ = panel_rgb.shape

    combined = np.zeros((ph, pw * 2, 3), dtype=np.uint8)
    combined[:, :pw, :] = panel_rgb
    combined[:, pw:, :] = overlay[:ph, :pw, :]
    Image.fromarray(combined).save(os.path.join(OUTPUT_DIR, "comparison.jpg"))

    print(f"Best variant {best_variant} saved to {OUTPUT_DIR}")
    print(f"  labels.npz: {labels.shape}, unique labels: {unique}")
    print(f"  overlay_mask.jpg: {overlay.shape}")
    print(f"  meta.json: engine={meta['engine']}, n_layers={meta['n_layers']}")
    print(f"  regional_audit.json: saved")
    print(f"  comparison.jpg: saved")

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Panel: fig6_profile_07")
    print(f"Best variant: {best_variant}")
    print(f"Engine: {meta['engine']}")
    print(f"n_layers: {len(unique)}")
    print(f"\nLabel distribution:")
    for lbl, s in sorted(label_stats.items()):
        print(f"  Label {lbl}: {s['pixels']} px, median_y={s['median_y']:.0f}")
    print(f"\nIssues fixed:")
    for issue in audit["issues_fixed"]:
        print(f"  - {issue}")
    print(f"\nRemaining issues:")
    for issue in audit["remaining_issues"]:
        print(f"  - {issue}")


if __name__ == "__main__":
    main()
