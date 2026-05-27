"""M2 standalone demo: verify vlm_client prompt assembly + schema validation + audit.

Input: fixtures/ph01/ph01_vlm_*.json (mock VLM responses)
Output: runs/M2/ audit files + validation report
Verification: all production call points produce valid prompts, parse mock JSON correctly,
confidence >= 0.7, and write audit to disk.
"""

import json
from pathlib import Path

import numpy as np

from .client import (
    review_page_overview,
    classify_figure,
    parse_response,
)
from .prompts import PageOverview, FigureClassification


def main():
    base = Path(__file__).resolve().parents[3]
    fixture_dir = base / "tests" / "fixtures" / "ph01"
    out_dir = base / "runs" / "M2"
    audit_dir = out_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Dummy images for prompt assembly (we don't need real images for this demo)
    dummy_img = np.ones((200, 300, 3), dtype=np.uint8) * 255

    checks = []

    # ── Test 1: review_page_overview ──────────────────────────────
    print("=== Test 1: review_page_overview ===")
    text_blocks = [
        {"bbox": [0, 0, 100, 20], "text": "Figure 1. Vs model cross-sections"},
        {"bbox": [0, 30, 100, 50], "text": "(a) 5 km depth"},
    ]
    result = review_page_overview(
        dummy_img, text_blocks, page_idx=7, audit_dir=audit_dir, mode="stub"
    )
    checks.append(("overview_prompt_nonempty", len(result["prompt"]) > 100))
    checks.append(("overview_audit_exists", result["audit_path"].exists()))
    checks.append(("overview_image_audit_exists", result["image_audit_path"].exists()))
    checks.append(("overview_schema_is_PageOverview", result["schema_class"] is PageOverview))
    print(f"  Prompt length: {len(result['prompt'])}")
    print(f"  Audit: {result['audit_path']}")

    # Parse mock response
    overview_raw = json.loads((fixture_dir / "ph01_vlm_overview.json").read_text())
    overview = parse_response(overview_raw, PageOverview)
    checks.append(("overview_confidence_ge_0.7", overview.confidence >= 0.7))
    checks.append(("overview_figure_type_valid", overview.figure_type == "velocity_model_cross_section"))
    checks.append(("overview_panels_ge_4", len(overview.panels) >= 4))
    print(f"  Parsed: figure_type={overview.figure_type}, panels={len(overview.panels)}, confidence={overview.confidence}")

    # ── Test 2: classify_figure ─────────────────────────────────────
    print("\n=== Test 2: classify_figure ===")
    result = classify_figure(
        dummy_img, audit_dir=audit_dir, mode="stub"
    )
    checks.append(("class_prompt_nonempty", len(result["prompt"]) > 100))
    checks.append(("class_audit_exists", result["audit_path"].exists()))
    checks.append(("class_schema_is_FigureClassification", result["schema_class"] is FigureClassification))
    print(f"  Prompt length: {len(result['prompt'])}")
    print(f"  Audit: {result['audit_path']}")

    class_raw = json.loads((fixture_dir / "ph01_vlm_figure_class.json").read_text())
    fc = parse_response(class_raw, FigureClassification)
    checks.append(("class_confidence_ge_0.7", fc.confidence >= 0.7))
    checks.append(("class_figure_type_valid", fc.figure_type == "velocity_model"))
    checks.append(("class_segmentation_recommendation", getattr(fc, "segmentation_recommendation", None) == "proceed"))
    checks.append(("class_category_checklist_length", len(getattr(fc, "category_checklist", [])) == 10))
    checks.append(("class_primary_evidence_nonempty", len(getattr(fc, "primary_evidence", "")) > 0))
    print(f"  Parsed: figure_type={fc.figure_type}, confidence={fc.confidence}, recommendation={getattr(fc, 'segmentation_recommendation', 'N/A')}")

    # ── Test 3: parse_response with low confidence should raise ─────
    print("\n=== Test 3: parse_response low confidence raises ===")
    bad_raw = {"page_idx": 7, "figure_type": "uncertain", "panels": [], "target_panel_id": 0,
               "has_colorbar": False, "noise_elements": [], "color_zones": [], "confidence": 0.5}
    try:
        parse_response(bad_raw, PageOverview)
        checks.append(("low_confidence_raises", False))
        print("  FAIL: should have raised ValueError")
    except ValueError:
        checks.append(("low_confidence_raises", True))
        print("  PASS: correctly raised ValueError")

    # ── Test 4: parse_response with invalid schema should raise ─────
    print("\n=== Test 4: parse_response invalid schema raises ===")
    invalid_raw = {"page_idx": "seven", "confidence": 0.9}
    try:
        parse_response(invalid_raw, PageOverview)
        checks.append(("invalid_schema_raises", False))
        print("  FAIL: should have raised ValueError")
    except ValueError:
        checks.append(("invalid_schema_raises", True))
        print("  PASS: correctly raised ValueError")

    # ── Summary ────────────────────────────────────────────────────
    print("\n=== Verification ===")
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nM2 PASS: All checks passed")
    else:
        print("\nM2 FAIL: Some checks failed")


if __name__ == "__main__":
    main()
