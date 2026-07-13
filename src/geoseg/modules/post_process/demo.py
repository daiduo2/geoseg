"""Demo / test scenario for post_process pipeline.

Tests: extract_components, labels_to_polygons, assign_properties.

Run:
    python -m geoseg.modules.post_process.demo
"""

from __future__ import annotations

import numpy as np

from geoseg.modules.post_process.polygon import extract_components, labels_to_polygons
from geoseg.modules.post_process.properties import assign_properties, build_properties_template


def main() -> int:
    # Simple 4x4 label map with 2 regions
    labels = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 2, 0],
        [2, 2, 2, 0],
    ], dtype=np.int32)

    print("=== test extract_components ===")
    comps = extract_components(labels, min_area=1)
    assert len(comps) == 2, f"Expected 2 components, got {len(comps)}"
    assert all(k in comps[0] for k in ("id", "layer_id", "bbox", "area", "centroid"))
    assert comps[0]["id"] == 0
    assert comps[0]["layer_id"] == 1
    assert comps[0]["area"] == 4
    assert comps[1]["layer_id"] == 2
    assert comps[1]["area"] == 6
    for c in comps:
        print(f"  id={c['id']} layer={c['layer_id']} area={c['area']} bbox={c['bbox']} centroid={c['centroid']}")

    print("\n=== test labels_to_polygons ===")
    geojson = labels_to_polygons(labels, min_area=1, hole_size=1, simplify_tol=0.5)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 2
    for f in geojson["features"]:
        assert f["type"] == "Feature"
        assert f["geometry"]["type"] == "Polygon"
        coords = f["geometry"]["coordinates"][0]
        assert len(coords) >= 4
        assert coords[0] == coords[-1]
        props = f["properties"]
        print(f"  layer={props['layer_id']} color={props['color_name']} vertices={props['n_vertices']}")

    print("\n=== test assign_properties ===")
    color_names = ["red", "blue"]
    props = assign_properties(color_names)
    assert "red" in props and "blue" in props
    assert props["red"]["Vp"] > props["blue"]["Vp"]
    print(f"  red: Vp={props['red']['Vp']} Vs={props['red']['Vs']} rho={props['red']['rho']}")
    print(f"  blue: Vp={props['blue']['Vp']} Vs={props['blue']['Vs']} rho={props['blue']['rho']}")

    print("\n=== test build_properties_template ===")
    template = build_properties_template(["green"])
    assert "green" in template
    print(f"  green template: {template['green']}")

    print("\nAll post_process tests passed.")
    return 0


if __name__ == "__main__":
    exit(main())
