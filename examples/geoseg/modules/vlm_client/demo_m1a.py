"""M1a standalone demo: real VLM review on rasterized PDF pages.

Input: tests/fixtures/ph01/gxae11701.pdf
Output: runs/M1a/page_overview_*.json + audit
Verification: VLM correctly identifies which page contains the concept model
and how many panels it has, using full-page rasterize (300dpi).

WARNING: mode="auto" calls Claude CLI and consumes API tokens (~$0.15-0.30 per page).
"""

import json
from pathlib import Path

import numpy as np

from geoseg.modules.pdf_extractor.extract import rasterize_page
from .client import review_page_overview


# Pages to test (likely concept model pages based on captions)
PAGES = [
    (6, "Page 7 — Figure 7/8: seismic profile + geological models"),
    (7, "Page 8 — Figure 9: reservoir model (ph01 source page)"),
    (8, "Page 9 — Figure 10/11: fault-related karst + seismic profile"),
    (9, "Page 10 — Figure 12/13: velocity models + contact relationships"),
]


def main():
    base = Path(__file__).resolve().parents[3]
    fixture = base / "tests" / "fixtures" / "ph01"
    pdf_path = fixture / "gxae11701.pdf"
    out_dir = base / "runs" / "M1a"
    audit_dir = out_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for page_idx, note in PAGES:
        print(f"\n=== Reviewing Page {page_idx + 1} ({note}) ===")

        # Rasterize at 300dpi (full page)
        raster = rasterize_page(pdf_path, page_idx=page_idx, dpi=300)
        img = raster["image"]
        print(f"  Rasterized: {img.shape[1]}x{img.shape[0]} @ {raster['dpi']}dpi")

        # Load text blocks for context
        text_path = base / "runs" / "M0.5" / f"page_{page_idx:03d}_text.json"
        text_blocks = []
        if text_path.exists():
            page_data = json.loads(text_path.read_text())
            text_blocks = page_data.get("text_blocks", [])

        try:
            overview = review_page_overview(
                img, text_blocks, page_idx=page_idx, audit_dir=audit_dir, mode="auto"
            )
            print(f"  figure_type={overview.figure_type}")
            print(f"  panels={len(overview.panels)}")
            print(f"  target_panel_id={overview.target_panel_id}")
            print(f"  confidence={overview.confidence}")
            print(f"  has_colorbar={overview.has_colorbar}")

            results.append({
                "page_idx": page_idx,
                "page_number": page_idx + 1,
                "figure_type": overview.figure_type,
                "panels_count": len(overview.panels),
                "target_panel_id": overview.target_panel_id,
                "confidence": overview.confidence,
                "has_colorbar": overview.has_colorbar,
                "note": note,
                "error": None,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "page_idx": page_idx,
                "page_number": page_idx + 1,
                "figure_type": None,
                "panels_count": 0,
                "target_panel_id": None,
                "confidence": 0.0,
                "has_colorbar": False,
                "note": note,
                "error": str(e),
            })

    # Save summary
    summary_path = out_dir / "m1a_review_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n=== Summary saved to {summary_path} ===")

    # Find best candidate
    valid = [r for r in results if r["confidence"] >= 0.7 and r["error"] is None]
    if valid:
        best = max(valid, key=lambda r: r["confidence"])
        print(f"\nBest candidate: Page {best['page_number']} ({best['note']})")
        print(f"  figure_type={best['figure_type']}, panels={best['panels_count']}, confidence={best['confidence']}")
    else:
        print("\nNo valid candidate found (all confidence < 0.7 or errors)")

    # Verification
    checks = []
    checks.append(("at_least_one_valid_result", len(valid) > 0))
    if valid:
        checks.append(("best_confidence_ge_0.7", best["confidence"] >= 0.7))
    else:
        checks.append(("best_confidence_ge_0.7", False))

    print("\n=== Verification ===")
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nM1a PASS: All checks passed")
    else:
        print("\nM1a FAIL: Some checks failed")


if __name__ == "__main__":
    main()
