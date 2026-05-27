"""Demo / test scenario for the end-to-end controller pipeline.

Tests: run_pipeline with synthetic figure images.

Run:
    python -m geoseg.controller_demo
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from geoseg.controller import run_pipeline


def _make_synthetic_conceptual(size: tuple[int, int] = (120, 200)) -> np.ndarray:
    """Create a synthetic figure with distinct colored blocks and edges."""
    h, w = size
    rng = np.random.default_rng(42)
    img = np.ones((h, w, 3), dtype=np.uint8) * 245

    # Distinct colored regions
    img[10:h // 2 - 5, 10:w // 2 - 5] = [200, 70, 70]
    img[10:h // 2 - 5, w // 2 + 5:-10] = [70, 200, 70]
    img[h // 2 + 5:-10, 10:-10] = [70, 70, 200]

    # Fine scribbles / annotation-like lines
    for _ in range(60):
        x = rng.integers(10, w - 10)
        y = rng.integers(10, h - 10)
        angle = rng.random() * 3.14159
        length = rng.integers(8, 30)
        color = [30, 30, 30]
        for l in range(length):
            px = int(x + l * np.cos(angle))
            py = int(y + l * np.sin(angle))
            if 0 <= px < w and 0 <= py < h:
                img[py, px] = color

    return img


def main() -> int:
    print("=== test skip path (observational_data-like image) ===")
    gray_img = np.full((100, 200, 3), 180, dtype=np.uint8)
    result = run_pipeline(gray_img, n_layers=3, skip_non_velocity_model=True, use_vlm=False)
    assert result["status"] == "skipped"
    print(f"  status={result['status']} reason={result['reason']}")

    print("\n=== test ok path (synthetic conceptual image) ===")
    img = _make_synthetic_conceptual((120, 200))
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "output"
        result = run_pipeline(
            img,
            n_layers=3,
            skip_non_velocity_model=False,
            use_vlm=False,
            output_dir=out_dir,
            save_intermediates=True,
        )
        print(f"  status={result['status']}")
        print(f"  classification={result['classification']['figure_type']}")
        print(f"  n_panels={result['summary']['n_panels']}")

        for p in result["panels"]:
            print(f"  panel {p['panel_id']}: status={p['status']}")
            if p["status"] == "ok":
                print(f"    engine={p['engines_used']} layers={len(p['color_names'])}")
                print(f"    components={p['n_components']} polygons={p['n_polygons']}")

        # Check exported files exist
        files = sorted(out_dir.iterdir()) if out_dir.exists() else []
        print(f"  artifacts: {', '.join(f.name for f in files)}")
        assert any(f.name.endswith("_tomo.xyz") for f in files), "Missing tomography file"
        assert any(f.name.endswith("_polygons.geojson") for f in files), "Missing geojson"

    print("\nAll controller pipeline tests passed.")
    return 0


if __name__ == "__main__":
    exit(main())
