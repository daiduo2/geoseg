"""Quick napari demo to verify three-layer architecture.

Loads existing geoseg labels + overlay, displays:
- Image layer: overlay (reference)
- Labels layer: labels array (data layer, readonly display)
- Shapes layer: boundary polygons extracted from labels (user interaction layer)

Usage:
    source .venv/bin/activate
    python geoseg/modules/editor/demo_napari.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure


def extract_boundary_polygons(labels: np.ndarray) -> list[np.ndarray]:
    """Extract boundary polygons from labels array.

    For each label, find contours and return as list of (N, 2) vertex arrays.
    Each polygon is a closed boundary of one region.
    """
    polygons: list[np.ndarray] = []
    unique_labels = sorted(set(labels.flatten()) - {0})

    for label_id in unique_labels:
        mask = labels == label_id
        if not mask.any():
            continue

        # find contours at level 0.5
        contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
        for cnt in contours:
            if len(cnt) < 3:
                continue
            # cnt is (N, 2) in (row, col) = (y, x) order
            # napari Shapes layer expects [y, x] coordinates — keep as-is
            polygons.append(cnt)

    return polygons


def run_demo() -> None:
    # Paths — gras2019 example with 6 layers, clearer boundaries
    runs_dir = Path("/Users/daiduo2/geoseg/runs/self_heal_v1/gras2019_c11b8db")
    labels_path = runs_dir / "final_labels.npz"
    overlay_path = runs_dir / "final_overlay.jpg"

    # Load labels
    labels = np.load(labels_path)["labels"]
    print(f"Labels shape: {labels.shape}, unique: {sorted(set(labels.flatten()))}")

    # Load overlay as reference image
    overlay = np.array(Image.open(overlay_path))
    print(f"Overlay shape: {overlay.shape}")

    # Extract boundary polygons for Shapes layer
    polygons = extract_boundary_polygons(labels)
    print(f"Extracted {len(polygons)} boundary polygons")

    # Launch napari
    import napari

    viewer = napari.Viewer(title="geoseg editor demo — three-layer architecture")

    # Layer 1: Image (reference)
    viewer.add_image(overlay, name="reference", opacity=0.4, visible=True)

    # Layer 2: Labels (data layer, readonly display)
    labels_layer = viewer.add_labels(labels, name="regions")

    # Layer 3: Shapes (user interaction — boundary polygons)
    shapes_layer = viewer.add_shapes(
        polygons,
        shape_type="polygon",
        name="boundaries",
        edge_color="white",
        edge_width=2,
        face_color="transparent",
    )

    # Make labels layer read-only in terms of direct pixel editing
    labels_layer.mode = "pan_zoom"

    # Print instructions
    print("\n=== Napari Demo ===")
    print("Layers:")
    print("  1. 'reference' — overlay image (40% opacity, toggle with 'R')")
    print("  2. 'regions' — labels array (colored regions, readonly)")
    print("  3. 'boundaries' — white polygon edges (try selecting and dragging vertices)")
    print("\nTry:")
    print("  - Click 'boundaries' layer → select a polygon → drag vertices")
    print("  - Toggle 'reference' visibility (click eye icon)")
    print("  - Zoom / pan with mouse")
    print("  - Close window to exit")

    napari.run()


if __name__ == "__main__":
    run_demo()
