#!/usr/bin/env python3
"""Text-aware workflow for fig6_profile_05 — improved parameters."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path("/Users/daiduo2/geoseg/src")))

from geoseg.modules.text_removal import remove_text
from geoseg.modules.segment_engines.mask_aware import segment_with_text_mask

# Paths
base = Path("/Users/daiduo2/geoseg")
panel_path = base / "runs/feng_fig6_final_v4/crop_tests/fig6_profile_05_cropped.jpg"
out_dir = base / "runs/feng_fig6_workflow_v8/fig6_profile_05"
out_dir.mkdir(parents=True, exist_ok=True)

# Load panel
img_rgb = np.array(Image.open(panel_path).convert("RGB"))
print(f"Panel shape: {img_rgb.shape}")

# Step 1: Text removal with conservative parameters
print("\n=== Step 1: Text removal (conservative) ===")
cleaned, text_mask_raw = remove_text(
    img_rgb,
    brightness_thresh=200,      # Only very bright text
    max_area=500,               # Smaller max area
    dilate_iter=1,
    inpaint_radius=2,
)

# Binarize the mask properly
text_mask = (text_mask_raw > 128).astype(np.uint8) * 255
print(f"Text mask coverage: {(text_mask > 0).mean()*100:.2f}%")

# Save cleaned panel and mask
cleaned_path = out_dir / "panel_cleaned.jpg"
mask_path = out_dir / "text_mask.jpg"
Image.fromarray(cleaned).save(cleaned_path)
Image.fromarray(text_mask).save(mask_path)

# Step 2: Mask-aware segmentation with multiple engines
print("\n=== Step 2: Mask-aware segmentation ===")
engines = ["v4_kmeans", "edge_guided", "kmeans_full"]
n_layers = 5

results = {}
for engine in engines:
    print(f"\n  Running {engine}...")
    try:
        result = segment_with_text_mask(
            engine_name=engine,
            image_rgb=img_rgb,
            text_mask=text_mask.astype(bool),
            n_layers=n_layers,
        )
        results[engine] = result
        unique_labels = len(np.unique(result["labels"])) - (1 if 0 in np.unique(result["labels"]) else 0)
        print(f"    -> {unique_labels} non-zero labels, overlay shape {result['overlay'].shape}")

        # Save labels and overlay
        np.savez(out_dir / f"labels_{engine}.npz", labels=result["labels"])
        Image.fromarray(result["overlay"]).save(out_dir / f"overlay_{engine}.jpg")
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\n=== Completed {len(results)} engines ===")
for engine, result in results.items():
    unique = np.unique(result["labels"])
    print(f"  {engine}: labels {unique}")
