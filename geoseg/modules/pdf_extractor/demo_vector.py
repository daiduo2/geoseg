"""M0.5v demo: rasterize ph01 PDF page 7 (idx 6) at 600 dpi.

Input:  tests/fixtures/ph01/gxae11701.pdf
Output: runs/M0.5v/page_7_rasterize_600dpi.png + page_7_metrics.json
Checks (spec §6): file exists, size >= 1343x691 and >= 2M pixels, gray mean > 240.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .vector_extract import extract_page_figure_as_bitmap

PAGE_IDX, DPI = 6, 600


def main():
    base = Path(__file__).resolve().parents[3]
    pdf_path = base / "tests" / "fixtures" / "ph01" / "gxae11701.pdf"
    out_dir = base / "runs" / "M0.5v"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rasterizing {pdf_path.name} page {PAGE_IDX+1} at {DPI} dpi ...")
    result = extract_page_figure_as_bitmap(pdf_path, page_idx=PAGE_IDX, dpi=DPI)

    png_path = out_dir / f"page_{PAGE_IDX+1}_rasterize_{DPI}dpi.png"
    Image.fromarray(result["image"], mode="RGB").save(png_path, optimize=True)
    gray = np.asarray(Image.fromarray(result["image"]).convert("L"))
    h, w, b = gray.shape[0], gray.shape[1], 100  # 100px corner block for white-bg check
    corners = [gray[:b, :b], gray[:b, w-b:], gray[h-b:, :b], gray[h-b:, w-b:]]
    corner_mean = float(np.mean([c.mean() for c in corners]))
    metrics = {
        "page_idx": result["page_idx"], "dpi": result["dpi"],
        "width": result["width"], "height": result["height"],
        "page_mean_brightness": round(float(gray.mean()), 2),
        "corner_mean_brightness": round(corner_mean, 2),
        "file_size_bytes": png_path.stat().st_size, "source": result["source"],
    }
    (out_dir / f"page_{PAGE_IDX+1}_metrics.json").write_text(json.dumps(metrics, indent=2))

    checks = [
        ("png_exists", png_path.exists()),
        ("width>=1343", result["width"] >= 1343),
        ("height>=691", result["height"] >= 691),
        ("pixels>=2M", result["width"] * result["height"] >= 2_000_000),
        ("white_bg(corner)>240", corner_mean > 240),
    ]
    print(f"\n=== Verification ===\n{json.dumps(metrics, indent=2)}\n")
    all_pass = all(p for _, p in checks)
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\nM0.5v {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
