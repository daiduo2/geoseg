"""M1b standalone demo: verify cv_detect on ph01 concept model image.

Input: ph01.jpg (from e026 inputs, the concept model image with 4 panels)
Output: runs/M1b/ph01_candidates.png with drawn bbox
Verification: ≥4 candidates, approximate match to known panel layout.

NOTE: PDF extracted embedded images (page_007_img_0/1, page_008_img_0) are NOT
the concept model figure — they have dark backgrounds and different content.
The concept model appears to be a vector graphic or a separate image (ph01.jpg).
This will be revisited when M0.5 pdf_extractor is enhanced for vector content.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .detect import find_panel_candidates


def main():
    base = Path(__file__).resolve().parents[3]
    # Use ph01.jpg — the clearest concept model image available
    img_path = Path(
        "/Users/daiduo2/Documents/knowlege/Projects/精密院-地震逆散射/"
        "photo/experiments/e026_ph01_conceptual/inputs/ph01.jpg"
    )
    out_dir = base / "runs" / "M1b"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_path.exists():
        print(f"ERROR: {img_path} not found.")
        return

    img = np.array(Image.open(img_path).convert("RGB"))
    print(f"Loaded image: {img.shape[1]}x{img.shape[0]}")

    candidates = find_panel_candidates(
        img,
        white_threshold=220,
        gap_ratio=0.2,
        min_gap_width=10,
    )
    print(f"Found {len(candidates)} panel candidates")
    for c in candidates:
        x, y, w, h = c["bbox"]
        cx, cy = x + w / 2, y + h / 2
        print(f"  id={c['id']}: bbox=[{x},{y},{w},{h}], area={c['area']}, center=({cx:.1f},{cy:.1f})")

    # Draw candidates
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    for c in candidates:
        x, y, w, h = c["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
        draw.text((x + 2, y + 2), str(c["id"]), fill=(255, 0, 0))

    out_path = out_dir / "ph01_candidates.png"
    pil_img.save(out_path)
    print(f"Saved: {out_path}")

    # Verification
    checks = []
    checks.append(("candidate_count >= 4", len(candidates) >= 4))

    if len(candidates) >= 4:
        # Check approximate equal widths for first 4 candidates
        widths = [c["bbox"][2] for c in candidates[:4]]
        avg_w = sum(widths) / len(widths)
        max_w_diff = max(abs(w - avg_w) for w in widths)
        checks.append(("panel_widths_similar", max_w_diff <= avg_w * 0.5))

        # Check horizontal arrangement (y positions similar)
        ys = [c["bbox"][1] for c in candidates[:4]]
        y_spread = max(ys) - min(ys)
        checks.append(("panels_horizontally_aligned", y_spread <= 50))

    print("\n=== Verification ===")
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    # Save result JSON
    result = {
        "image": str(img_path),
        "candidates": candidates,
        "checks": {name: passed for name, passed in checks},
        "all_pass": all_pass,
    }
    (out_dir / "candidates.json").write_text(json.dumps(result, indent=2))

    if all_pass:
        print("\nM1b PASS: All checks passed")
    else:
        print("\nM1b FAIL: Some checks failed")


if __name__ == "__main__":
    main()
