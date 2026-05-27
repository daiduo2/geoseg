"""Demo / test scenario for SPECFEM exporter.

Tests: labels_to_grids, write_tomography_file, write_parfile_snippet.

Run:
    python -m geoseg.modules.exporter.demo
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from geoseg.modules.exporter.specfem import (
    labels_to_grids,
    write_external_model_ascii,
    write_parfile_snippet,
    write_tomography_file,
)


def main() -> int:
    labels = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 2, 0],
        [2, 2, 2, 0],
    ], dtype=np.int32)

    props = {
        "layer_1": {"Vp": 5000.0, "Vs": 2887.0, "rho": 2600.0},
        "layer_2": {"Vp": 4000.0, "Vs": 2309.0, "rho": 2400.0},
    }

    print("=== test labels_to_grids ===")
    vp, vs, rho = labels_to_grids(labels, props)
    assert vp.shape == labels.shape
    assert vs.shape == labels.shape
    assert rho.shape == labels.shape
    # label 1 area
    mask1 = labels == 1
    assert np.all(vp[mask1] == 5000.0)
    assert np.all(vs[mask1] == 2887.0)
    assert np.all(rho[mask1] == 2600.0)
    # label 2 area
    mask2 = labels == 2
    assert np.all(vp[mask2] == 4000.0)
    # background defaults
    bg = labels == 0
    assert vp[bg][0] == 3000.0
    print(f"  vp range: [{vp.min():.1f}, {vp.max():.1f}]")
    print(f"  vs range: [{vs.min():.1f}, {vs.max():.1f}]")
    print(f"  rho range: [{rho.min():.1f}, {rho.max():.1f}]")

    print("\n=== test write_tomography_file ===")
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        x_coords = np.linspace(0, 3, 4)
        z_coords = np.linspace(0, 3, 4)
        tomo_path = tdir / "tomo.xyz"
        write_tomography_file(vp, vs, rho, x_coords, z_coords, tomo_path)
        lines = tomo_path.read_text(encoding="utf-8").strip().split("\n")
        assert lines[0].startswith("#x #z #Vp #Vs #rho")
        assert len(lines) == 17  # header + 4*4 rows
        print(f"  wrote {len(lines)} lines to {tomo_path.name}")

        print("\n=== test write_parfile_snippet ===")
        snippet_path = tdir / "parfile_snippet.txt"
        write_parfile_snippet(
            ["layer_1", "layer_2"], props, snippet_path, nx=4, nz=4, dx=1.0, dz=1.0
        )
        snippet = snippet_path.read_text(encoding="utf-8")
        assert "MODEL" in snippet
        assert "nx_tomo" in snippet
        print(f"  wrote snippet ({len(snippet)} chars)")

        print("\n=== test write_external_model_ascii ===")
        ext_path = tdir / "external_model.dat"
        write_external_model_ascii(vp, vs, rho, x_coords, z_coords, ext_path)
        ext_lines = ext_path.read_text(encoding="utf-8").strip().split("\n")
        assert ext_lines[0] == "4 4"
        print(f"  wrote {len(ext_lines)} lines")

    print("\nAll exporter tests passed.")
    return 0


if __name__ == "__main__":
    exit(main())
