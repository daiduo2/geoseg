"""Create side-by-side comparison: baseline vs unified."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_results(path: Path) -> list[dict]:
    """Load results from a saved comparison figure."""
    img = np.array(Image.open(path).convert("RGB"))
    # Figure layout: header_h=35, label_w=100, 4 columns per row
    header_h = 35
    label_w = 100
    h = (img.shape[0] - header_h) // 3
    w = (img.shape[1] - label_w) // 4
    results = []
    for r in range(3):
        y0 = header_h + r * h
        results.append({
            "original": img[y0:y0+h, label_w:label_w+w],
            "cleaned": img[y0:y0+h, label_w+w:label_w+2*w],
            "fill": img[y0:y0+h, label_w+2*w:label_w+3*w],
            "boundaries": img[y0:y0+h, label_w+3*w:label_w+4*w],
        })
    return results


def create_comparison(baseline: list[dict], unified: list[dict]) -> np.ndarray:
    n = 3
    h, w = baseline[0]["original"].shape[:2]
    # Layout per panel: [Original, Baseline Fill, Baseline Bound, Unified Fill, Unified Bound]
    cols = ["Original", "Baseline Fill", "Baseline Bound", "Unified Fill", "Unified Bound"]
    header_h = 35
    label_w = 100
    cell_h, cell_w = h, w
    canvas = np.ones((n * cell_h + header_h, label_w + len(cols) * cell_w, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    for c, title in enumerate(cols):
        x = label_w + c * cell_w + cell_w // 2 - len(title) * 5
        cv2.putText(canvas, title, (x, 25), font, 0.6, (0, 0, 0), 2)

    row_titles = ["Panel 1", "Panel 2", "Panel 3"]
    for r in range(n):
        y = header_h + r * cell_h
        cv2.putText(canvas, row_titles[r], (10, y + 30), font, 0.6, (0, 0, 0), 2)

        # Original
        x = label_w
        canvas[y:y+cell_h, x:x+cell_w] = baseline[r]["original"]

        # Baseline fill
        x = label_w + cell_w
        canvas[y:y+cell_h, x:x+cell_w] = baseline[r]["fill"]

        # Baseline boundaries
        x = label_w + 2 * cell_w
        canvas[y:y+cell_h, x:x+cell_w] = baseline[r]["boundaries"]

        # Unified fill
        x = label_w + 3 * cell_w
        canvas[y:y+cell_h, x:x+cell_w] = unified[r]["fill"]

        # Unified boundaries
        x = label_w + 4 * cell_w
        canvas[y:y+cell_h, x:x+cell_w] = unified[r]["boundaries"]

    return canvas


def main():
    base = Path(__file__).parent.parent.parent
    baseline = load_results(base / "result_v4_two_stage.png")
    unified = load_results(base / "result_v4_unified.png")

    fig = create_comparison(baseline, unified)
    out = base / "result_comparison_baseline_vs_unified.png"
    Image.fromarray(fig).save(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
