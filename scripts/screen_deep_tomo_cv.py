#!/usr/bin/env python3
"""Quick CV screen of deep_tomo_2024 images to find velocity_model candidates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.cv_detect.figure_classifier import classify

PAPER_DIR = Path("papers_new/to_process/deep_tomo_2024")


def main() -> None:
    images = sorted(PAPER_DIR.glob("*.png"))
    print(f"Found {len(images)} images in {PAPER_DIR}")

    candidates = []
    for img_path in images:
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        h, w = img_rgb.shape[:2]

        # Skip tiny images
        if w < 300 or h < 200:
            continue

        cls = classify(img_rgb)
        fig_type = cls["figure_type"]
        sat = cls.get("saturation_ratio", 0)

        # Look for conceptual_model (velocity models are typically conceptual)
        if fig_type == "conceptual_model":
            candidates.append({
                "path": str(img_path),
                "name": img_path.stem,
                "size": f"{w}x{h}",
                "saturation": round(sat, 3),
                "features": {k: round(v, 3) if isinstance(v, float) else v
                             for k, v in cls.get("features", {}).items()},
            })
            print(f"  CANDIDATE: {img_path.stem:40s} | {w}x{h} | sat={sat:.3f}")
        else:
            print(f"  skip:      {img_path.stem:40s} | {w}x{h} | type={fig_type} | sat={sat:.3f}")

    print(f"\n{candidates=}")

    out_path = Path("runs/new_papers_vlm/deep_tomo_cv_screen.json")
    out_path.write_text(json.dumps(candidates, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path} ({len(candidates)} candidates)")


if __name__ == "__main__":
    main()
