"""Test e016: Edge-enhanced region growing on 181218.

Usage:
    cd ~/.claude/skills/geo-segment
    python tests/test_e016_edge_grow.py
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

from lib.segment_edge_grow import segment_jet_vivid_edge_grow
from lib.segment import segment_jet_vivid
from lib.crop import crop_panel


PROJECT_ROOT = Path("/Users/daiduo2/Documents/knowlege/Projects/精密院-地震逆散射/photo")

IMG = PROJECT_ROOT / "data" / "source" / "181218.jpg"
VLM = PROJECT_ROOT / "data" / "phase0" / "vlm" / "kimi_181218.json"
OUT_DIR = PROJECT_ROOT / "experiments" / "e016_edge_grow" / "181218"


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


def save_edge_map_visualization(panel_rgb: np.ndarray, edge_map: np.ndarray, out_path: Path):
    """Save edge map as a heatmap overlay on the panel (no matplotlib)."""
    em = edge_map / (edge_map.max() + 1e-9)
    # Simple jet-like colormap: blue -> cyan -> green -> yellow -> red
    h, w = em.shape
    heat = np.zeros((h, w, 3), dtype=np.float32)
    # Map 0-1 to RGB using piecewise linear approximation of jet
    r = np.clip(1.5 - np.abs(em * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(em * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(em * 4 - 1), 0, 1)
    heat = np.stack([r, g, b], axis=2)
    heat = (heat * 255).astype(np.uint8)
    blended = (0.5 * panel_rgb + 0.5 * heat).clip(0, 255).astype(np.uint8)
    Image.fromarray(blended).save(out_path)


def run_edge_grow(img_path: Path, vlm_path: Path, out_dir: Path):
    print(f"\n{'='*60}")
    print(f"Edge-enhanced region growing: {img_path.name}")
    print(f"{'='*60}")

    img = load_image(img_path)
    reps = load_vlm_reps(vlm_path)

    # Crop panel from VLM JSON panel bbox
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

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Baseline (nearest_median) ---
    print("\n  --- Baseline (nearest_median) ---")
    t0 = time.perf_counter()
    result_baseline = segment_jet_vivid(panel, reps, max_auto_k=3)
    t_baseline = time.perf_counter() - t0
    print(f"  Time: {t_baseline:.3f}s")
    print(f"  Seeds: {len(result_baseline.color_names)} ({result_baseline.color_names})")
    print(f"  Auto-k added: {result_baseline.notes['auto_k_added']}")

    save_boundary_overlay(panel, result_baseline.labels, out_dir / "baseline_nearest_median.png")
    save_color_overlay(panel, result_baseline.labels, result_baseline.palette, out_dir / "baseline_nearest_median_color.png")

    # --- Edge-enhanced Dijkstra with varying penalties ---
    penalties = [10.0, 50.0, 100.0, 200.0]
    results = {}

    for penalty in penalties:
        print(f"\n  --- Edge grow (penalty={penalty}) ---")
        t0 = time.perf_counter()
        result = segment_jet_vivid_edge_grow(panel, reps, max_auto_k=3, edge_penalty=penalty)
        t_edge = time.perf_counter() - t0
        print(f"  Time: {t_edge:.3f}s")
        print(f"  Seeds: {len(result.color_names)} ({result.color_names})")
        print(f"  Auto-k added: {result.notes['auto_k_added']}")
        em_stats = result.notes["edge_map_stats"]
        print(f"  Edge map: min={em_stats['min']:.4f} max={em_stats['max']:.4f} mean={em_stats['mean']:.4f} median={em_stats['median']:.4f}")

        results[penalty] = {
            "result": result,
            "time_sec": t_edge,
        }

        save_boundary_overlay(panel, result.labels, out_dir / f"edge_grow_p{int(penalty)}.png")
        save_color_overlay(panel, result.labels, result.palette, out_dir / f"edge_grow_p{int(penalty)}_color.png")

    # --- Edge map visualization (from last result) ---
    # Recompute edge map for visualization
    from skimage.color import rgb2lab
    from skimage.filters import sobel
    panel_lab = rgb2lab(panel)
    h, w = panel.shape[:2]
    gradient = np.zeros((h, w), dtype=np.float32)
    for c in range(3):
        gradient += sobel(panel_lab[..., c]) ** 2
    gradient = np.sqrt(gradient)
    edge_map = gradient / (gradient.max() + 1e-9)
    save_edge_map_visualization(panel, edge_map, out_dir / "edge_map.png")
    # Also save raw edge map as grayscale
    em_gray = (edge_map * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(em_gray).save(out_dir / "edge_map_gray.png")

    # --- Review JSON ---
    review = {
        "image": img_path.name,
        "vlm": vlm_path.name,
        "crop_shape": list(panel.shape),
        "num_vlm_seeds": len(reps),
        "saturation_ratio": float(result_baseline.saturation_ratio),
        "baseline": {
            "method": "nearest_median",
            "time_sec": round(t_baseline, 3),
            "num_seeds": len(result_baseline.color_names),
            "auto_k_added": result_baseline.notes["auto_k_added"],
            "color_names": result_baseline.color_names,
        },
        "edge_grow": {},
    }

    for penalty in penalties:
        r = results[penalty]["result"]
        review["edge_grow"][f"penalty_{int(penalty)}"] = {
            "time_sec": round(results[penalty]["time_sec"], 3),
            "num_seeds": len(r.color_names),
            "auto_k_added": r.notes["auto_k_added"],
            "edge_penalty": r.notes["edge_penalty"],
            "edge_map_stats": r.notes["edge_map_stats"],
            "color_names": r.color_names,
            "seed_sources": [
                {
                    "name": rep["name"],
                    "source": rep["source"],
                    "xy": [rep["internal_x"], rep["internal_y"]],
                }
                for rep in r.notes["reps_refined"]
            ],
        }

    with open(out_dir / "review.json", "w") as f:
        json.dump(review, f, indent=2)

    print(f"\n  Saved outputs to {out_dir}")
    return review


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    review = run_edge_grow(IMG, VLM, OUT_DIR)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n181218:")
    b = review["baseline"]
    print(f"  {'baseline':20s}: {b['time_sec']:6.3f}s  seeds={b['num_seeds']}  auto_k={b['auto_k_added']}")
    for penalty_key, info in review["edge_grow"].items():
        print(f"  {penalty_key:20s}: {info['time_sec']:6.3f}s  seeds={info['num_seeds']}  auto_k={info['auto_k_added']}  penalty={info['edge_penalty']}")


if __name__ == "__main__":
    main()
