"""Test e011: GMM probabilistic segmentation on 181218 (vivid jet) and 184140 (pastel).

Usage:
    cd ~/.claude/skills/geo-segment
    python tests/test_e011_gmm.py
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

from lib.segment_gmm import segment_jet_vivid_gmm
from lib.segment import segment_jet_vivid, segment_pastel_faded
from lib.crop import crop_panel


PROJECT_ROOT = Path("/Users/daiduo2/Documents/knowlege/Projects/精密院-地震逆散射/photo")

# 181218 (vivid jet)
IMG_218 = PROJECT_ROOT / "data" / "source" / "181218.jpg"
VLM_218 = PROJECT_ROOT / "data" / "phase0" / "vlm" / "kimi_181218.json"

# 184140 (pastel)
IMG_140 = PROJECT_ROOT / "data" / "source" / "184140.jpg"
VLM_140 = PROJECT_ROOT / "data" / "phase0" / "vlm" / "vlm_184140.json"

OUT_DIR = PROJECT_ROOT / "experiments" / "e011_gmm"


def load_vlm_reps(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    # Determine key name
    for key in ["T3_color_zones_top_panel", "T3_color_zones_panel_A"]:
        if key in data:
            return data[key]["zones"]
    raise ValueError(f"No recognized zones key in {path}")


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_color_overlay(panel_rgb: np.ndarray, labels: np.ndarray, palette: np.ndarray, out_path: Path):
    """Save a color-coded segmentation overlay."""
    from skimage.color import label2rgb
    overlay = label2rgb(labels, image=panel_rgb / 255.0, colors=palette / 255.0, alpha=0.4, bg_label=-1)
    overlay = (overlay * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def save_boundary_overlay(panel_rgb: np.ndarray, labels: np.ndarray, out_path: Path):
    """Save boundary overlay on original panel."""
    from skimage.segmentation import mark_boundaries
    overlay = mark_boundaries(panel_rgb / 255.0, labels, color=(1, 0, 0), mode="thick")
    overlay = (overlay * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def run_gmm_on_image(img_path: Path, vlm_path: Path, out_subdir: Path, img_name: str, is_pastel: bool = False):
    print(f"\n{'='*60}")
    print(f"Processing {img_name} ({img_path.name})")
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

    out_subdir.mkdir(parents=True, exist_ok=True)

    # --- GMM (full) ---
    print("\n  --- GMM (covariance_type='full') ---")
    t0 = time.perf_counter()
    result_gmm_full = segment_jet_vivid_gmm(panel, reps, max_auto_k=3, covariance_type="full")
    t_gmm_full = time.perf_counter() - t0
    print(f"  Time: {t_gmm_full:.3f}s")
    print(f"  Seeds: {len(result_gmm_full.color_names)} ({result_gmm_full.color_names})")
    print(f"  Auto-k added: {result_gmm_full.notes['auto_k_added']}")
    print(f"  GMM converged: {result_gmm_full.notes['gmm_converged']} in {result_gmm_full.notes['gmm_n_iter']} iter")

    save_boundary_overlay(panel, result_gmm_full.labels, out_subdir / f"{img_name}_gmm_full.png")
    save_color_overlay(panel, result_gmm_full.labels, result_gmm_full.palette, out_subdir / f"{img_name}_gmm_full_color.png")

    # --- GMM (tied) ---
    print("\n  --- GMM (covariance_type='tied') ---")
    t0 = time.perf_counter()
    result_gmm_tied = segment_jet_vivid_gmm(panel, reps, max_auto_k=3, covariance_type="tied")
    t_gmm_tied = time.perf_counter() - t0
    print(f"  Time: {t_gmm_tied:.3f}s")
    print(f"  Seeds: {len(result_gmm_tied.color_names)} ({result_gmm_tied.color_names})")
    print(f"  Auto-k added: {result_gmm_tied.notes['auto_k_added']}")
    print(f"  GMM converged: {result_gmm_tied.notes['gmm_converged']} in {result_gmm_tied.notes['gmm_n_iter']} iter")

    save_boundary_overlay(panel, result_gmm_tied.labels, out_subdir / f"{img_name}_gmm_tied.png")

    # --- K-means baseline (e007) ---
    print("\n  --- K-means baseline ---")
    from lib.segment_kmeans import segment_jet_vivid_kmeans
    t0 = time.perf_counter()
    result_kmeans = segment_jet_vivid_kmeans(panel, reps, max_auto_k=3)
    t_kmeans = time.perf_counter() - t0
    print(f"  Time: {t_kmeans:.3f}s")
    print(f"  Seeds: {len(result_kmeans.color_names)} ({result_kmeans.color_names})")
    print(f"  Auto-k added: {result_kmeans.notes['auto_k_added']}")

    save_boundary_overlay(panel, result_kmeans.labels, out_subdir / f"{img_name}_kmeans.png")

    # --- Nearest-median baseline (e002) ---
    print("\n  --- nearest_median baseline ---")
    t0 = time.perf_counter()
    result_nm = segment_jet_vivid(panel, reps, max_auto_k=3)
    t_nm = time.perf_counter() - t0
    print(f"  Time: {t_nm:.3f}s")
    print(f"  Seeds: {len(result_nm.color_names)} ({result_nm.color_names})")
    print(f"  Auto-k added: {result_nm.notes['auto_k_added']}")

    save_boundary_overlay(panel, result_nm.labels, out_subdir / f"{img_name}_nearest_median.png")

    # --- Review JSON ---
    review = {
        "image": img_path.name,
        "vlm": vlm_path.name,
        "crop_shape": list(panel.shape),
        "num_vlm_seeds": len(reps),
        "saturation_ratio": float(result_gmm_full.saturation_ratio),
        "comparisons": {
            "gmm_full": {
                "time_sec": round(t_gmm_full, 3),
                "num_seeds": len(result_gmm_full.color_names),
                "auto_k_added": result_gmm_full.notes["auto_k_added"],
                "covariance_type": "full",
                "gmm_converged": result_gmm_full.notes["gmm_converged"],
                "gmm_n_iter": result_gmm_full.notes["gmm_n_iter"],
                "gmm_log_likelihood": result_gmm_full.notes["gmm_log_likelihood"],
                "fit_time_sec": result_gmm_full.notes["fit_time_sec"],
                "predict_time_sec": result_gmm_full.notes["predict_time_sec"],
                "color_names": result_gmm_full.color_names,
                "seed_sources": [
                    {
                        "name": r["name"],
                        "source": r["source"],
                        "xy": [r["internal_x"], r["internal_y"]],
                    }
                    for r in result_gmm_full.notes["reps_refined"]
                ],
                "palette_rgb": result_gmm_full.palette.tolist(),
            },
            "gmm_tied": {
                "time_sec": round(t_gmm_tied, 3),
                "num_seeds": len(result_gmm_tied.color_names),
                "auto_k_added": result_gmm_tied.notes["auto_k_added"],
                "covariance_type": "tied",
                "gmm_converged": result_gmm_tied.notes["gmm_converged"],
                "gmm_n_iter": result_gmm_tied.notes["gmm_n_iter"],
                "gmm_log_likelihood": result_gmm_tied.notes["gmm_log_likelihood"],
                "fit_time_sec": result_gmm_tied.notes["fit_time_sec"],
                "predict_time_sec": result_gmm_tied.notes["predict_time_sec"],
                "color_names": result_gmm_tied.color_names,
            },
            "kmeans": {
                "time_sec": round(t_kmeans, 3),
                "num_seeds": len(result_kmeans.color_names),
                "auto_k_added": result_kmeans.notes["auto_k_added"],
                "color_names": result_kmeans.color_names,
            },
            "nearest_median": {
                "time_sec": round(t_nm, 3),
                "num_seeds": len(result_nm.color_names),
                "auto_k_added": result_nm.notes["auto_k_added"],
                "color_names": result_nm.color_names,
            },
        },
    }

    with open(out_subdir / "review.json", "w") as f:
        json.dump(review, f, indent=2)

    print(f"\n  Saved outputs to {out_subdir}")
    return review


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 181218 - vivid jet
    review_218 = run_gmm_on_image(IMG_218, VLM_218, OUT_DIR / "181218", "181218")

    # 184140 - pastel (still use GMM for comparison, though it's pastel)
    review_140 = run_gmm_on_image(IMG_140, VLM_140, OUT_DIR / "184140", "184140", is_pastel=True)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, review in [("181218", review_218), ("184140", review_140)]:
        print(f"\n{name}:")
        for method, info in review["comparisons"].items():
            print(f"  {method:20s}: {info['time_sec']:6.3f}s  seeds={info['num_seeds']}  auto_k={info['auto_k_added']}")


if __name__ == "__main__":
    main()
