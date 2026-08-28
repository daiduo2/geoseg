"""Test e014: Edge-guided K-means segmentation on 181218 and 184140.

Usage:
    cd ~/.claude/skills/geo-segment
    python tests/test_e014_edge_guided.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Add geo-segment lib to path
sys.path.insert(0, str(Path.home() / ".claude/skills/geo-segment"))

from lib.segment_edge_guided import segment_jet_vivid_edge_guided
from lib.segment import segment_jet_vivid
from lib.segment_kmeans import segment_jet_vivid_kmeans


PROJECT_ROOT = Path("/Users/daiduo2/Documents/knowlege/Projects/精密院-地震逆散射/photo")

IMG_218 = PROJECT_ROOT / "data" / "source" / "181218.jpg"
VLM_218 = PROJECT_ROOT / "data" / "phase0" / "vlm" / "kimi_181218.json"

IMG_140 = PROJECT_ROOT / "data" / "source" / "184140.jpg"
VLM_140 = PROJECT_ROOT / "data" / "phase0" / "vlm" / "vlm_184140.json"

OUT_DIR = PROJECT_ROOT / "experiments" / "e014_edge_guided"


def load_vlm_reps(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    for key in ["T3_color_zones_top_panel", "T3_color_zones_panel_A"]:
        if key in data:
            return data[key]["zones"]
    raise ValueError(f"No recognized zones key in {path}")


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_color_overlay(panel_rgb: np.ndarray, labels: np.ndarray, palette: np.ndarray, out_path: Path):
    from skimage.color import label2rgb
    overlay = label2rgb(labels, image=panel_rgb / 255.0, colors=palette / 255.0, alpha=0.4, bg_label=-1)
    overlay = (overlay * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def save_boundary_overlay(panel_rgb: np.ndarray, labels: np.ndarray, out_path: Path):
    from skimage.segmentation import mark_boundaries
    overlay = mark_boundaries(panel_rgb / 255.0, labels, color=(1, 0, 0), mode="thick")
    overlay = (overlay * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def save_edge_overlay(panel_rgb: np.ndarray, edge_mask: np.ndarray, out_path: Path):
    """Overlay detected edges in green on the original panel."""
    overlay = panel_rgb.copy().astype(np.float32)
    overlay[edge_mask] = overlay[edge_mask] * 0.5 + np.array([0, 255, 0], dtype=np.float32) * 0.5
    overlay = overlay.clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def run_edge_guided_on_image(img_path: Path, vlm_path: Path, out_subdir: Path, img_name: str):
    print(f"\n{'='*60}")
    print(f"Processing {img_name} ({img_path.name})")
    print(f"{'='*60}")

    img = load_image(img_path)
    reps = load_vlm_reps(vlm_path)

    with open(vlm_path, "r") as f:
        data = json.load(f)
    panel = img
    for key in ["T1_panels"]:
        if key in data:
            p = data[key]["panels"][0]
            x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
            panel = img[y1:y2, x1:x2]
            break

    print(f"  Panel shape: {panel.shape}")
    print(f"  VLM reps: {len(reps)}")

    out_subdir.mkdir(parents=True, exist_ok=True)

    # --- Baseline: nearest_median ---
    print("\n  --- Baseline: nearest_median ---")
    t0 = time.perf_counter()
    result_nm = segment_jet_vivid(panel, reps, max_auto_k=3)
    t_nm = time.perf_counter() - t0
    print(f"  Time: {t_nm:.3f}s  seeds={len(result_nm.color_names)}  auto_k={result_nm.notes['auto_k_added']}")
    save_boundary_overlay(panel, result_nm.labels, out_subdir / f"{img_name}_baseline_nearest_median.png")

    # --- Baseline: K-means ---
    print("\n  --- Baseline: K-means ---")
    t0 = time.perf_counter()
    result_km = segment_jet_vivid_kmeans(panel, reps, max_auto_k=3)
    t_km = time.perf_counter() - t0
    print(f"  Time: {t_km:.3f}s  seeds={len(result_km.color_names)}  auto_k={result_km.notes['auto_k_added']}")
    save_boundary_overlay(panel, result_km.labels, out_subdir / f"{img_name}_baseline_kmeans.png")

    # --- Edge-guided K-means with varying edge_weight ---
    edge_weights = [0.1, 0.3, 0.5]
    eg_results = {}
    for ew in edge_weights:
        print(f"\n  --- Edge-guided K-means (edge_weight={ew}) ---")
        t0 = time.perf_counter()
        result_eg = segment_jet_vivid_edge_guided(panel, reps, max_auto_k=3, edge_weight=ew)
        t_eg = time.perf_counter() - t0
        print(f"  Time: {t_eg:.3f}s  seeds={len(result_eg.color_names)}  auto_k={result_eg.notes['auto_k_added']}")
        print(f"  Edge pixels: {result_eg.notes['edge_pixels_pct']:.2f}%")
        save_boundary_overlay(panel, result_eg.labels, out_subdir / f"{img_name}_edgeguided_ew{ew}.png")
        save_color_overlay(panel, result_eg.labels, result_eg.palette, out_subdir / f"{img_name}_edgeguided_ew{ew}_color.png")
        eg_results[ew] = {
            "result": result_eg,
            "time_sec": t_eg,
        }

    # Save edge overlay from the middle run (edge_weight=0.3)
    from lib.segment_edge_guided import _compute_edge_map
    from skimage.color import rgb2lab
    panel_lab = rgb2lab(panel)
    gradient, edge_mask = _compute_edge_map(
        panel_lab,
        method="canny",
        canny_sigma=1.0,
        canny_low=0.02,
        canny_high=0.1,
    )
    save_edge_overlay(panel, edge_mask, out_subdir / f"{img_name}_edges.png")

    # --- Review JSON ---
    review = {
        "image": img_path.name,
        "vlm": vlm_path.name,
        "crop_shape": list(panel.shape),
        "num_vlm_seeds": len(reps),
        "saturation_ratio": float(result_nm.saturation_ratio),
        "comparisons": {
            "nearest_median": {
                "time_sec": round(t_nm, 3),
                "num_seeds": len(result_nm.color_names),
                "auto_k_added": result_nm.notes["auto_k_added"],
                "color_names": result_nm.color_names,
            },
            "kmeans": {
                "time_sec": round(t_km, 3),
                "num_seeds": len(result_km.color_names),
                "auto_k_added": result_km.notes["auto_k_added"],
                "color_names": result_km.color_names,
            },
        },
    }

    for ew in edge_weights:
        r = eg_results[ew]["result"]
        review["comparisons"][f"edge_guided_ew{ew}"] = {
            "time_sec": round(eg_results[ew]["time_sec"], 3),
            "num_seeds": len(r.color_names),
            "auto_k_added": r.notes["auto_k_added"],
            "edge_weight": ew,
            "edge_pixels_pct": r.notes["edge_pixels_pct"],
            "sigma": r.notes["sigma"],
            "color_names": r.color_names,
            "seed_sources": [
                {
                    "name": rep["name"],
                    "source": rep["source"],
                    "xy": [rep["internal_x"], rep["internal_y"]],
                }
                for rep in r.notes["reps_refined"]
            ],
            "palette_rgb": r.palette.tolist(),
        }

    with open(out_subdir / "review.json", "w") as f:
        json.dump(review, f, indent=2)

    print(f"\n  Saved outputs to {out_subdir}")
    return review


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    review_218 = run_edge_guided_on_image(IMG_218, VLM_218, OUT_DIR / "181218", "181218")
    review_140 = run_edge_guided_on_image(IMG_140, VLM_140, OUT_DIR / "184140", "184140")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, review in [("181218", review_218), ("184140", review_140)]:
        print(f"\n{name}:")
        for method, info in review["comparisons"].items():
            print(f"  {method:25s}: {info['time_sec']:6.3f}s  seeds={info['num_seeds']}  auto_k={info.get('auto_k_added', 'N/A')}")


if __name__ == "__main__":
    main()
