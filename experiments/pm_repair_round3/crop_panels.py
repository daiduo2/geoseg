"""Crop the five tomography panels from newimage.jpg.

The default panel detector merges/overshoots on this figure, so this script
uses a robust colour-connected-component approach tuned for stacked
tomography panels with a topographic strip on top and a colorbar at the
bottom.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def detect_panels_colored_components(img_rgb: np.ndarray, margin: int = 5):
    """Detect the five data panels as the largest colored CCs.

    Returns a list of dicts with ``bbox`` as (x, y, w, h), sorted top-to-bottom.
    """
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.int16)
    val = hsv[..., 2].astype(np.int16)

    # Exclude topographic gray and white margins; keep the colored tomography.
    grayish = (sat < 40) & (val > 60) & (val < 180)
    colored = (sat > 35) & (val > 50) & ~grayish

    labeled, n = ndimage.label(colored)
    stats = []
    for i in range(1, n + 1):
        comp = labeled == i
        ys, xs = np.where(comp)
        area = int(comp.sum())
        if area < 5000:
            continue
        stats.append((area, int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))

    stats.sort(reverse=True)
    top5 = stats[:5]
    top5.sort(key=lambda s: s[2])  # top-to-bottom

    panels = []
    for _, x0, y0, x1, y1 in top5:
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(w - 1, x1 + margin)
        y1 = min(h - 1, y1 + margin)
        panels.append({"bbox": (x0, y0, x1 - x0 + 1, y1 - y0 + 1)})
    return panels


def main() -> int:
    image_path = Path("/Users/daiduo2/geoseg/newimage.jpg")
    output_dir = Path("/Users/daiduo2/geoseg/runs/panel_crops")
    output_dir.mkdir(parents=True, exist_ok=True)

    img_rgb = np.array(Image.open(image_path).convert("RGB"))
    panels = detect_panels_colored_components(img_rgb)

    # Save each crop and a visualization.
    vis = img_rgb.copy()
    records = []
    for idx, panel in enumerate(panels):
        x, y, pw, ph = panel["bbox"]
        crop = img_rgb[y : y + ph, x : x + pw]
        crop_path = output_dir / f"panel_{idx}.png"
        Image.fromarray(crop).save(crop_path)
        records.append({"panel": idx, "bbox": panel["bbox"], "path": str(crop_path)})
        cv2.rectangle(vis, (x, y), (x + pw, y + ph), (255, 0, 0), 2)
        cv2.putText(
            vis, str(idx), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
        )

    Image.fromarray(vis).save(output_dir / "panel_bboxes.jpg", quality=95)
    (output_dir / "bboxes.json").write_text(
        json.dumps({"image": str(image_path), "panels": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved {len(panels)} panels to {output_dir}")
    for r in records:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
