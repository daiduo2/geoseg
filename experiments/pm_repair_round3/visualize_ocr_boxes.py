"""Visualize PaddleOCR text detection boxes on 3D schematic panels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Ensure ocr_roi import resolves
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "pm_repair_round3"))
from ocr_roi import detect_text_rois


def visualize_boxes(
    image_path: str | Path,
    matches: list[dict],
    output_path: str | Path,
) -> None:
    """Draw OCR detection boxes and text labels on the image."""
    img = np.array(Image.open(image_path).convert("RGB"))
    vis = img.copy()

    for i, match in enumerate(matches):
        x1, y1, x2, y2 = match["roi"]
        color = (
            (0, 255, 0) if i % 2 == 0 else (0, 200, 255)
        )  # green / orange-ish
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{match['text']} ({match['confidence']:.2f})"
        cv2.putText(
            vis,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    Image.fromarray(vis).save(output_path, quality=95)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize PaddleOCR text boxes on 3D schematic panels."
    )
    parser.add_argument("image", help="Input image path")
    parser.add_argument("output", help="Output visualization path")
    parser.add_argument(
        "--json", help="Optional path to save OCR results as JSON"
    )
    args = parser.parse_args()

    matches = detect_text_rois(args.image, keywords=None)
    visualize_boxes(args.image, matches, args.output)

    if args.json:
        Path(args.json).write_text(
            json.dumps(matches, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Saved visualization to {args.output}")
    print(f"Detected {len(matches)} text boxes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
