#!/usr/bin/env python3
"""Quick test to measure under-segmentation retry impact."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.pipeline.segment import run_segmentation_stage
from geoseg.core.image_ops import saturation_ratio

PAPERS: dict[str, str] = {
    "ph01": "runs/M0.5/images",
    "gras2019": "runs/literature_test/gras2019/mineru/extracted/images",
    "ma_2022": "runs/literature_test/ma_2022/mineru/extracted/images",
    "silixa2021": "runs/literature_test/silixa2021/mineru/extracted/images",
}

results = []
for paper, img_dir in PAPERS.items():
    img_path = Path(img_dir)
    if not img_path.exists():
        continue
    images = sorted(img_path.glob("*.png")) + sorted(img_path.glob("*.jpg"))
    print(f"\n[{paper}] {len(images)} figures")
    for img_file in images:
        fig_name = img_file.stem
        try:
            img_rgb = np.array(Image.open(img_file).convert("RGB"))
            result = run_segmentation_stage(img_rgb, n_layers=5, use_vlm=False, skip_non_velocity_model=False)
            status = result["summary"]["status"]
            warnings = result["summary"]["review_warnings"]
            engines = result["summary"].get("engines_used", [])
            # Only track targets (non-skipped)
            if status != "skipped":
                retry_fixed = [w for w in warnings if "retry_fixed" in w]
                under_segs = [w for w in warnings if "under_segmented" in w]
                results.append({
                    "paper": paper,
                    "fig": fig_name,
                    "status": status,
                    "engines": engines,
                    "retry_fixed": retry_fixed,
                    "under_segs": under_segs,
                    "all_warnings": warnings,
                })
                if retry_fixed or under_segs:
                    print(f"  {fig_name}: retry_fixed={len(retry_fixed)}, under_segs={len(under_segs)}, engines={engines}")
                    for w in retry_fixed + under_segs:
                        print(f"    {w}")
        except Exception as exc:
            print(f"  {fig_name}: ERROR {exc}")

# Summary
print(f"\n{'='*60}")
print(f"Total targets processed: {len(results)}")
print(f"Cases with retry_fixed: {sum(1 for r in results if r['retry_fixed'])}")
print(f"Cases still under-segmented: {sum(1 for r in results if r['under_segs'])}")
print(f"Total retry_fixed events: {sum(len(r['retry_fixed']) for r in results)}")
print(f"Total under_seg events: {sum(len(r['under_segs']) for r in results)}")
