"""Batch VLM review of top-saturation images from literature test sets."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoseg.experiments import review_page_overview


def main():
    results = {}
    for paper in ["gras2019", "zailac2023", "ma_2022"]:
        summary_path = Path(f"runs/literature_test/{paper}/segment_results/summary.json")
        summary = json.loads(summary_path.read_text())
        top = sorted(summary.items(), key=lambda x: x[1]["saturation"], reverse=True)[:3]
        images_dir = Path(f"runs/literature_test/{paper}/mineru/extracted/images")

        print(f"\n=== {paper} ===")
        for name, data in top:
            img_path = images_dir / name
            if not img_path.exists():
                print(f"Skip {name[:25]}: not found")
                continue
            img = np.array(Image.open(img_path).convert("RGB"))
            try:
                r = review_page_overview(img, [], page_idx=0, mode="auto")
                print(
                    f"{name[:22]}...  sat={data['saturation']:.3f}  "
                    f"type={r.figure_type}  conf={r.confidence:.2f}  "
                    f"panels={len(r.panels)}  colorbar={r.has_colorbar}"
                )
                results[f"{paper}/{name}"] = {
                    "figure_type": r.figure_type,
                    "confidence": r.confidence,
                    "panels": len(r.panels),
                    "has_colorbar": r.has_colorbar,
                    "saturation": data["saturation"],
                }
            except Exception as e:
                print(f"{name[:22]}...  ERROR: {type(e).__name__}: {str(e)[:60]}")
                results[f"{paper}/{name}"] = {"error": str(e)}

    out = Path("runs/literature_test/vlm_review_top3.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
