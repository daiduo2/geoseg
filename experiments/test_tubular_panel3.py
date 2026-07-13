"""Validate tubular_structure engine on panel 3."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from geoseg.modules.segment_engines.tubular_structure import segment as tubular_segment


def main():
    enhanced_path = Path("runs/3d_schematic_correct_e2e/panel_3_front/00_enhanced.jpg")
    output_dir = Path("runs/tubular_panel3")
    output_dir.mkdir(parents=True, exist_ok=True)

    enhanced = np.array(Image.open(enhanced_path).convert("RGB"))
    print(f"Image shape: {enhanced.shape}")

    # Test 1: n_layers=2 (tube vs background)
    print("\n[1] tubular_structure with n_layers=2")
    result2 = tubular_segment(enhanced, n_layers=2)
    labels2 = result2["labels"]
    overlay2 = result2["overlay"]
    unique2 = sorted(np.unique(labels2))
    print(f"  Labels: {unique2}, n_layers={result2['meta']['n_layers']}")
    print(f"  Vesselness: min={result2['meta']['vesselness_stats']['min']:.4f}, "
          f"max={result2['meta']['vesselness_stats']['max']:.4f}, "
          f"mean={result2['meta']['vesselness_stats']['mean']:.4f}")
    Image.fromarray(overlay2).save(output_dir / "tubular_n2.jpg", quality=90)

    # Test 2: n_layers=4 (tube + 3 background layers)
    print("\n[2] tubular_structure with n_layers=4")
    result4 = tubular_segment(enhanced, n_layers=4)
    labels4 = result4["labels"]
    overlay4 = result4["overlay"]
    unique4 = sorted(np.unique(labels4))
    print(f"  Labels: {unique4}, n_layers={result4['meta']['n_layers']}")
    Image.fromarray(overlay4).save(output_dir / "tubular_n4.jpg", quality=90)

    # Test 3: Force black_ridges=False (tube brighter than background)
    print("\n[3] tubular_structure with black_ridges=False")
    result_white = tubular_segment(enhanced, n_layers=2, black_ridges=False)
    labels_white = result_white["labels"]
    overlay_white = result_white["overlay"]
    unique_white = sorted(np.unique(labels_white))
    print(f"  Labels: {unique_white}, n_layers={result_white['meta']['n_layers']}")
    Image.fromarray(overlay_white).save(output_dir / "tubular_white.jpg", quality=90)

    # Test 4: Larger sigmas for thicker tubes
    print("\n[4] tubular_structure with sigmas=(2,3,4,5)")
    result_large = tubular_segment(enhanced, n_layers=2, sigmas=(2.0, 3.0, 4.0, 5.0))
    labels_large = result_large["labels"]
    overlay_large = result_large["overlay"]
    unique_large = sorted(np.unique(labels_large))
    print(f"  Labels: {unique_large}, n_layers={result_large['meta']['n_layers']}")
    Image.fromarray(overlay_large).save(output_dir / "tubular_large_sigma.jpg", quality=90)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
