"""Review MinerU extracted images with VLM to identify concept model figures."""

import json
from pathlib import Path

from PIL import Image
import numpy as np

from geoseg.modules.vlm_client.client import review_page_overview


IMAGES_TO_REVIEW = [
    # (filename, note)
    ("a159c86ae19f9939c961e19ce34ae84f5a2cfaafcfff11e3ce8af11b48b245b9.jpg", "Figure 9 - page 8 - Surface karst fractured-vuggy reservoir model"),
    ("0e771080ea47bf2a42f4d03bb848aaa572c4d9840b7c1e1d30e09556c33e823a.jpg", "Figure 10 - page 9 - Fault-related karst models"),
    ("569a4836fe44ce05a039cf04f4a941e4303bddacfb9ca29e8bc1c53dcddf6eb8.jpg", "Figure 11 - page 9 - W3 actual seismic profile"),
    ("14ee332af244e78d8215a62bf1f34d9a0408d62f73b13150723c4aee335565e6.jpg", "Figure 14 - page 11 - W3 seismic geological model (already reviewed)"),
    ("51296a60c47cd5ae38dc5a8ea89ccaee1cd89c158b154f38265d45020c8e9d17.jpg", "Figure 16a - page 13 - W4 seismic geological model"),
    ("0d1a87dfdf3c7d4d3b96d5d1a64509d58ff34317615b327ffced7afa5fb66182.jpg", "Figure 16b - page 13 - W4 local zoom-in"),
    ("446264e98b69225a38535b68ce06f5517dd24f63ceff3ac7a42006f6ef636508.jpg", "Figure 17 - page 14 - W4 forward modeling results"),
]


def main():
    base = Path(__file__).resolve().parents[3]
    img_dir = base / "runs" / "mineru" / "extracted" / "images"
    out_dir = base / "runs" / "mineru" / "vlm_review"
    audit_dir = out_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for filename, note in IMAGES_TO_REVIEW:
        img_path = img_dir / filename
        if not img_path.exists():
            print(f"Skip: {filename} not found")
            continue

        print(f"\n=== Reviewing {note} ===")
        print(f"  File: {filename}")

        img = Image.open(img_path)
        img_np = np.array(img.convert("RGB"))
        print(f"  Size: {img_np.shape[1]}x{img_np.shape[0]}")

        try:
            overview = review_page_overview(
                img_np, text_blocks=[], page_idx=0, audit_dir=audit_dir, mode="auto"
            )
            print(f"  figure_type={overview.figure_type}")
            print(f"  panels={len(overview.panels)}")
            print(f"  target_panel_id={overview.target_panel_id}")
            print(f"  confidence={overview.confidence}")
            print(f"  has_colorbar={overview.has_colorbar}")

            results.append({
                "filename": filename,
                "note": note,
                "width": img_np.shape[1],
                "height": img_np.shape[0],
                "figure_type": overview.figure_type,
                "panels_count": len(overview.panels),
                "target_panel_id": overview.target_panel_id,
                "confidence": overview.confidence,
                "has_colorbar": overview.has_colorbar,
                "error": None,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "filename": filename,
                "note": note,
                "width": img_np.shape[1],
                "height": img_np.shape[0],
                "figure_type": None,
                "panels_count": 0,
                "target_panel_id": None,
                "confidence": 0.0,
                "has_colorbar": False,
                "error": str(e),
            })

    # Save summary
    summary_path = out_dir / "mineru_vlm_review.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n=== Summary saved to {summary_path} ===")

    # Print summary table
    print("\n=== Results ===")
    for r in results:
        status = "OK" if r["error"] is None else "ERR"
        conf = r["confidence"]
        print(f"  [{status}] {r['filename'][:20]}... | {r['figure_type'] or 'N/A'} | panels={r['panels_count']} | conf={conf:.2f} | {r['note'][:50]}")


if __name__ == "__main__":
    main()
