#!/usr/bin/env python3
"""Generate segmentation overlays for audit review.

Processes each target with >=2 layers and saves:
- original_panel.png: original panel image
- overlay.png: color-coded segmentation overlay
- labels.npy: raw label array
- meta.json: segmentation metadata

Output: runs/new_papers_vlm/audit_overlays/{fig_key}/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.pipeline.segment import run_segmentation_stage

RESULTS_FILE = Path("runs/new_papers_vlm/pipeline_results.json")
RETRY_FILE = Path("runs/new_papers_vlm/retry_results.json")
OUT_DIR = Path("runs/new_papers_vlm/audit_overlays")

# Color map for labels (similar to matplotlib tab10)
LABEL_COLORS = np.array([
    [0, 0, 0],         # 0 = background (black)
    [255, 0, 0],       # 1 = red
    [0, 255, 0],       # 2 = green
    [0, 0, 255],       # 3 = blue
    [255, 255, 0],     # 4 = yellow
    [255, 0, 255],     # 5 = magenta
    [0, 255, 255],     # 6 = cyan
    [255, 128, 0],     # 7 = orange
    [128, 0, 255],     # 8 = purple
    [0, 128, 128],     # 9 = teal
], dtype=np.uint8)


def create_overlay(panel_img: np.ndarray, labels: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create colored overlay from label array."""
    h, w = labels.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    unique_labels = sorted(set(labels.flatten()))
    for lbl in unique_labels:
        color = LABEL_COLORS[int(lbl) % len(LABEL_COLORS)]
        mask = labels == lbl
        overlay[mask] = color
    # Blend with original
    blended = (panel_img * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return blended


def main() -> None:
    pipeline = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    retry = json.loads(RETRY_FILE.read_text(encoding="utf-8")) if RETRY_FILE.exists() else []

    # Use best result per fig_key
    best = {r["fig_key"]: r for r in pipeline}
    for r in retry:
        orig = r.get("original_fig_key", r["fig_key"])
        if orig not in best or r.get("total_layers", 0) > best[orig].get("total_layers", 0):
            best[orig] = r

    targets = [r for r in best.values() if r.get("total_layers", 0) >= 2]
    print(f"Generating overlays for {len(targets)} targets\n")

    for i, target in enumerate(targets, 1):
        fig_key = target["fig_key"]
        paper = target.get("paper", "unknown")
        fig_name = target.get("fig_name", fig_key.split("/")[-1])

        # Find image path
        img_path = target.get("img_path", "")
        if not img_path or img_path == "N/A":
            # Try pipeline results
            p = [x for x in pipeline if x["fig_key"] == fig_key]
            if p:
                img_path = p[0].get("img_path", "")
        if not img_path:
            print(f"[{i}/{len(targets)}] SKIP {fig_key}: no image path")
            continue

        img_path = Path(img_path)
        if not img_path.exists():
            print(f"[{i}/{len(targets)}] SKIP {fig_key}: image not found")
            continue

        # Create output directory
        audit_dir = OUT_DIR / fig_key.replace("/", "_")
        audit_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(targets)}] {fig_key} -> {audit_dir}")

        # Run pipeline
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        result = run_segmentation_stage(
            img_rgb,
            caption="",
            n_layers=target.get("total_layers", 5),
            quality_preference="balanced",
            skip_non_velocity_model=True,
            use_vlm=True,
        )

        # Save whole figure info
        meta = {
            "fig_key": fig_key,
            "paper": paper,
            "status": result["summary"]["status"],
            "n_panels": result["summary"].get("n_panels", 0),
            "total_layers": result["summary"].get("total_layers", 0),
            "saturation_ratio": result["summary"].get("saturation_ratio", 0),
            "engines_used": result["summary"].get("engines_used", []),
            "review_warnings": result["summary"].get("review_warnings", []),
        }
        (audit_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Save original whole image
        Image.fromarray(img_rgb).save(audit_dir / "whole_image.png")

        # Process each panel
        for panel in result.get("panels", []):
            if panel["segmentation"] is None:
                continue

            seg = panel["segmentation"]
            labels = seg["labels"]
            panel_id = panel["panel_id"]
            x, y, pw, ph = panel["bbox"]

            panel_img = img_rgb[y:y+ph, x:x+pw]

            # Save panel original
            Image.fromarray(panel_img).save(audit_dir / f"panel_{panel_id}_original.png")

            # Save labels as npy
            np.save(audit_dir / f"panel_{panel_id}_labels.npy", labels)

            # Create and save overlay
            if labels.shape[:2] == panel_img.shape[:2]:
                overlay = create_overlay(panel_img, labels)
                Image.fromarray(overlay).save(audit_dir / f"panel_{panel_id}_overlay.png")

            # Save panel meta
            panel_meta = {
                "panel_id": panel_id,
                "bbox": [x, y, pw, ph],
                "n_layers_found": panel["review"].get("n_layers_found", 0),
                "engine": seg["meta"].get("engine", "unknown"),
                "is_target_panel": panel["review"].get("is_target_panel", False),
            }
            (audit_dir / f"panel_{panel_id}_meta.json").write_text(
                json.dumps(panel_meta, indent=2), encoding="utf-8"
            )

    print(f"\nDone. Overlays in: {OUT_DIR}")


if __name__ == "__main__":
    main()
