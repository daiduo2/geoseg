"""PaddleOCR-based ROI detection for text artifacts (experimental).

This module is intentionally separate from the existing ROI detection in
pm_repair.py. It lives in experiments/ only and is not meant for production
use.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# PaddleOCR lives in a dedicated venv because the current project venv uses
# Python 3.14, which PaddlePaddle does not yet support.


def normalize_text(text: str) -> str:
    """Lowercase and strip whitespace/punctuation for fuzzy matching."""
    return re.sub(r"\W+", "", text.lower())


def box_to_roi(box: list[tuple[float, float]]) -> tuple[int, int, int, int]:
    """Convert a 4-point polygon to an axis-aligned bounding box."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def detect_text_rois(
    image_path: str | Path,
    keywords: list[str] | None = None,
) -> list[dict]:
    """Run PaddleOCR and return ROIs matching the given keywords.

    Args:
        image_path: Path to the image to OCR.
        keywords: List of keywords to match (case-insensitive, punctuation
            stripped). If None, returns all detected text boxes.

    Returns:
        List of dicts with keys: text, confidence, roi (x1,y1,x2,y2).
    """
    from paddleocr import PaddleOCR

    # Disable doc orientation/unwarping/textline orientation so coordinates
    # map directly to the input image.
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    result = ocr.predict(str(image_path))
    if not result:
        return []

    rec_texts = result[0].get("rec_texts", [])
    rec_scores = result[0].get("rec_scores", [])
    rec_polys = result[0].get("rec_polys", [])

    matches = []
    for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
        norm = normalize_text(text)
        if keywords is not None:
            if not any(normalize_text(kw) in norm for kw in keywords):
                continue
        x1, y1, x2, y2 = box_to_roi(poly.tolist())
        matches.append(
            {
                "text": text,
                "confidence": float(score),
                "roi": [x1, y1, x2, y2],
                "poly": poly.tolist(),
            }
        )
    return matches


def detect_pm_roi(image_path: str | Path) -> tuple[int, int, int, int] | None:
    """Convenience wrapper: return the highest-confidence PM ROI or None."""
    matches = detect_text_rois(image_path, keywords=["PM"])
    if not matches:
        return None
    best = max(matches, key=lambda m: m["confidence"])
    return tuple(best["roi"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect text ROIs with PaddleOCR (experimental)."
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["PM"],
        help="Keywords to filter (default: PM)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Return all detected boxes, no keyword filtering",
    )
    args = parser.parse_args()

    keywords = None if args.all else args.keywords
    matches = detect_text_rois(args.image, keywords=keywords)
    print(json.dumps(matches, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
