#!/usr/bin/env python3
"""Classify figures for a single paper (parallel helper)."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.experiments import classify_figure

PAPER = sys.argv[1]
IMG_DIR = sys.argv[2]
CACHE_FILE = Path("runs/5_papers_vlm_test/vlm_classification_results.json")
MAX_CLASSIFY = 12

def load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return []

def main():
    img_path = Path(IMG_DIR)
    if not img_path.exists():
        print(f"[SKIP] {PAPER}: directory not found")
        return

    images = sorted(img_path.glob("*.png")) + sorted(img_path.glob("*.jpg"))
    print(f"[{PAPER}] {len(images)} total figures")

    images_with_size = [(p, p.stat().st_size) for p in images]
    images_with_size.sort(key=lambda x: x[1], reverse=True)
    to_classify = images_with_size[:MAX_CLASSIFY]

    cache_data = load_cache()
    cache = {f"{r['paper']}/{r['fig_name']}": r for r in cache_data}

    targets = []
    for img_file, size in to_classify:
        fig_name = img_file.stem
        cache_key = f"{PAPER}/{fig_name}"
        cached = cache.get(cache_key)

        if cached:
            cls_result = cached
            print(f"  {fig_name}: CACHED {cls_result['vlm_type']} (conf={cls_result['vlm_confidence']:.2f})")
        else:
            try:
                img_rgb = np.array(Image.open(img_file).convert("RGB"))
                vlm_result = classify_figure(img_rgb, mode="auto")
                cls_result = {
                    "paper": PAPER,
                    "fig_name": fig_name,
                    "vlm_type": vlm_result.figure_type,
                    "vlm_confidence": vlm_result.confidence,
                    "vlm_reason": vlm_result.reason,
                }
                cache_data.append(cls_result)
                cache[cache_key] = cls_result
                print(f"  {fig_name}: {cls_result['vlm_type']} (conf={cls_result['vlm_confidence']:.2f})")
            except Exception as exc:
                print(f"  {fig_name}: VLM ERROR {exc}")
                continue

        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache_data, indent=2, default=str), encoding="utf-8")

        if cls_result["vlm_type"] in ("velocity_model", "geological_cross_section"):
            targets.append({
                "paper": PAPER,
                "fig_name": fig_name,
                "path": str(img_file),
                "vlm_confidence": cls_result["vlm_confidence"],
                "vlm_reason": cls_result["vlm_reason"],
            })

    print(f"  -> {len(targets)} VLM-confirmed targets")

if __name__ == "__main__":
    main()
