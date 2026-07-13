#!/usr/bin/env python3
"""Test VLM figure classification on a representative sample."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.vlm_client.client import classify_figure

SAMPLES = [
    # gras2019 — ok (CV 放行)
    ("gras2019", "066b806217fda3da75edc5ab38230e669482fc5dd7364234ea5b40e3547e0d6d", "cv_ok"),
    ("gras2019", "1a84e177d0fafd1fd5b116e83dbabd8356f2518d44a40f71a031bcd45e7d3040", "cv_ok"),
    ("gras2019", "7129e17309065c2482479c6bfbb2fe15e0a65d2b0be78061b3c5bf0df69c1849", "cv_ok"),
    # gras2019 — not_run (CV 排除)
    ("gras2019", "27a21d1e0afc605804e66626da796191d9cd7969ed9d625a68fdd69ad5cbc001", "cv_notrun"),
    ("gras2019", "405835dabf72f356db031ac2c5c0e5a1487100580c854d2520fec083da46ca8e", "cv_notrun"),
    # ma_2022 — ok
    ("ma_2022", "717b5cb28dbcdc1e3e1457cd17880083469cc1b51a760568e716c62eeb7e8fc6", "cv_ok"),
    ("ma_2022", "c05e29052093c6c2717b10e7d5faf413adb43dced0c8b8dce848ff01b0c932e3", "cv_ok"),
    ("ma_2022", "78d3eb2bf0e0d3ac6e899a1c20540af16992e02e53b413cbe74a2cccaaf018a2", "cv_ok"),
    # ma_2022 — not_run
    ("ma_2022", "218bbdffb566fa5e8610658a96bc82d0af22baa4c7b576be1fddfc60c83fbcaa", "cv_notrun"),
    ("ma_2022", "5905f6495f35f971cf7e25cb628466f1be5000bb7ca539b5ac43732fa5407312", "cv_notrun"),
    # zailac2023 — ok
    ("zailac2023", "68c9c363fcfaa524e81d5256efd0d55768b7d4e66bebf8c79bd6286a37d77244", "cv_ok"),
    ("zailac2023", "2fce889159fe8a23343a21dfd9626954918e700315b7f79f2c9f11ec917573f1", "cv_ok"),
    ("zailac2023", "1d677d598a552b987e023c6f1a7ad5c620a12239d7cda040f381286f3b133abe", "cv_ok"),
    # zailac2023 — not_run
    ("zailac2023", "422b739753b513ea76aa3e281fe0eb6e34f1601f1366d02bcbf61e94f23b6847", "cv_notrun"),
    ("zailac2023", "0303218396b7fd5730cfaa1ddc9ba628ffb5fbc6646478e1b39cb02b482b867d", "cv_notrun"),
    # silixa2021 — ok
    ("silixa2021", "page5_img0", "cv_ok"),
    ("silixa2021", "page6_img0", "cv_ok"),
    ("silixa2021", "page6_img1", "cv_ok"),
    # silixa2021 — not_run (key false negatives!)
    ("silixa2021", "page4_img0", "cv_notrun"),
    ("silixa2021", "page5_img1", "cv_notrun"),
    ("silixa2021", "page2_img0", "cv_notrun"),
    # ph01 — not_run
    ("ph01", "page_002_img_0", "cv_notrun"),
    ("ph01", "page_003_img_0", "cv_notrun"),
]

OUTPUT_ROOT = Path("runs/agent_review")
RESULTS: list[dict] = []


def main() -> None:
    total = len(SAMPLES)
    for i, (paper, fig_name, cv_status) in enumerate(SAMPLES, 1):
        img_path = OUTPUT_ROOT / paper / fig_name / "01_original.jpg"
        if not img_path.exists():
            print(f"[{i}/{total}] SKIP {paper}/{fig_name}: image not found")
            continue

        print(f"[{i}/{total}] VLM classifying {paper}/{fig_name} ...")
        try:
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            result = classify_figure(img_rgb, mode="auto")
            
            record = {
                "paper": paper,
                "fig_name": fig_name,
                "cv_status": cv_status,
                "vlm_type": result.figure_type,
                "vlm_confidence": result.confidence,
                "vlm_reason": result.reason,
            }
            RESULTS.append(record)
            print(f"  -> {result.figure_type} (conf={result.confidence:.2f}): {result.reason}")
        except Exception as exc:
            record = {
                "paper": paper,
                "fig_name": fig_name,
                "cv_status": cv_status,
                "vlm_type": f"ERROR: {exc}",
                "vlm_confidence": 0.0,
                "vlm_reason": str(exc),
            }
            RESULTS.append(record)
            print(f"  -> ERROR: {exc}")

    # Save results
    out_file = OUTPUT_ROOT / "vlm_classify_test_results.json"
    out_file.write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {out_file}")

    # Summary
    print("\n=== Summary ===")
    for r in RESULTS:
        marker = "✓" if r["vlm_type"] not in ("ERROR", "") else "✗"
        print(f"{marker} {r['paper']}/{r['fig_name']}: cv={r['cv_status']} -> vlm={r['vlm_type']} ({r['vlm_confidence']:.2f})")


if __name__ == "__main__":
    main()
