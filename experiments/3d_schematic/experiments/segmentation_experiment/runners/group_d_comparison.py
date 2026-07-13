"""Group D: Cross-strategy comparison and 2D visualization.

Generates:
1. Horizontal comparison grids (per panel, best of each strategy side-by-side)
2. Vertical parameter sweep strips (per panel, per strategy, parameter variation)
3. Master report with metrics summary
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
RESULTS = BASE / "results" / "segmentation_experiment"
OUT = RESULTS / "group_d"
OUT.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def make_label(text: str, width: int, height: int = 40) -> np.ndarray:
    """Create a labeled strip."""
    strip = np.full((height, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(strip, text, (10, 28), font, 0.7, (0, 0, 0), 2)
    return strip


def hstack_with_labels(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    """Stack images horizontally with text labels above each."""
    h = images[0].shape[0]
    w = images[0].shape[1]
    label_h = 40
    strips = []
    for img, lbl in zip(images, labels):
        label = make_label(lbl, w, label_h)
        strips.append(np.vstack([label, img]))
    return np.hstack(strips)


def build_horizontal_comparison(panel_idx: int) -> np.ndarray | None:
    """Build horizontal comparison: original + best of A + best of B + best of C."""
    stem = f"panel_{panel_idx}"
    original = cv2.imread(str(BASE / f"figures/panels/{stem}.png"))
    if original is None:
        return None
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Best of Group A: pick v4_kmeans n_layers=6 (middle ground)
    a_best = load_image(RESULTS / "group_a" / f"{stem}_v4_kmeans_nl6_fill.png")

    # Best of Group B: blur_sigma=3.0 (baseline / middle)
    b_best = load_image(RESULTS / "group_b" / f"{stem}_blur_sigma_3.0_fill.png")

    # Best of Group C: felz_scale=300.0 (baseline / middle)
    c_best = load_image(RESULTS / "group_c" / f"{stem}_felz_scale_300.0_fill.png")

    images = [original]
    labels = ["Original"]
    if a_best is not None:
        images.append(a_best)
        labels.append("A: v4_kmeans nl=6")
    if b_best is not None:
        images.append(b_best)
        labels.append("B: diff-overlay")
    if c_best is not None:
        images.append(c_best)
        labels.append("C: v3_pipeline")

    if len(images) < 2:
        return None

    # Resize all to same height
    target_h = original.shape[0]
    resized = []
    for img in images:
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized.append(img)

    grid = hstack_with_labels(resized, labels)
    return grid


def build_vertical_sweep(panel_idx: int, group: str, param_name: str, values: list) -> np.ndarray | None:
    """Build vertical parameter sweep strip for a single parameter."""
    stem = f"panel_{panel_idx}"
    images = []
    labels = []

    for v in values:
        if group == "b":
            img = load_image(RESULTS / f"group_{group}" / f"{stem}_{param_name}_{v}_fill.png")
        else:
            img = load_image(RESULTS / f"group_{group}" / f"{stem}_{param_name}_{v}_fill.png")
        if img is not None:
            images.append(img)
            labels.append(f"{param_name}={v}")

    if len(images) < 2:
        return None

    # Resize all to same width
    target_w = images[0].shape[1]
    resized = []
    for img in images:
        if img.shape[1] != target_w:
            scale = target_w / img.shape[1]
            new_h = int(img.shape[0] * scale)
            img = cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized.append(img)

    # Add labels and stack vertically
    labeled = []
    for img, lbl in zip(resized, labels):
        label = make_label(lbl, target_w, 30)
        labeled.append(np.vstack([label, img]))

    return np.vstack(labeled)


def read_summary_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def main():
    print("Building horizontal comparisons...")
    for idx in [1, 2, 3]:
        grid = build_horizontal_comparison(idx)
        if grid is not None:
            out_path = OUT / f"panel_{idx}_horizontal_comparison.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"  Saved {out_path}")

    print("\nBuilding vertical sweep strips...")
    # Group B sweeps
    b_params = [
        ("blur_ksize", [7, 15, 21]),
        ("blur_sigma", [1.5, 3.0, 5.0]),
        ("diff_thresh", [10.0, 20.0, 30.0]),
        ("expand_radius", [8, 15, 25]),
    ]
    for idx in [1, 2, 3]:
        for param_name, values in b_params:
            strip = build_vertical_sweep(idx, "b", param_name, values)
            if strip is not None:
                out_path = OUT / f"panel_{idx}_b_{param_name}_sweep.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
                print(f"  Saved {out_path}")

    # Group C sweeps
    c_params = [
        ("felz_scale", [300.0, 600.0]),
        ("felz_sigma", [0.3, 0.5, 0.8]),
        ("felz_min_size", [10, 30, 50]),
    ]
    for idx in [1, 2, 3]:
        for param_name, values in c_params:
            strip = build_vertical_sweep(idx, "c", param_name, values)
            if strip is not None:
                out_path = OUT / f"panel_{idx}_c_{param_name}_sweep.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
                print(f"  Saved {out_path}")

    # Build metrics summary
    print("\nBuilding metrics summary...")
    summaries = []
    for group in ["a", "b", "c"]:
        rows = read_summary_csv(RESULTS / f"group_{group}" / "summary.csv")
        for row in rows:
            row["group"] = group
        summaries.extend(rows)

    if summaries:
        # Collect all unique fieldnames across groups
        all_fields = set()
        for row in summaries:
            all_fields.update(row.keys())
        fieldnames = sorted(all_fields)
        with open(OUT / "master_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)
        print(f"  Saved master_summary.csv with {len(summaries)} rows")

    print(f"\nGroup D complete. Results in {OUT}")


if __name__ == "__main__":
    main()
