"""Test e015: Bilateral filter + K-means segmentation on 181218 and 184140.

Usage:
    cd ~/.claude/skills/geo-segment
    python tests/test_e015_bilateral.py
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

from lib.segment_bilateral import segment_jet_vivid_bilateral
from lib.segment_kmeans import segment_jet_vivid_kmeans
from lib.segment import segment_jet_vivid


PROJECT_ROOT = Path("/Users/daiduo2/Documents/knowlege/Projects/精密院-地震逆散射/photo")

# 181218 (vivid jet)
IMG_218 = PROJECT_ROOT / "data" / "source" / "181218.jpg"
VLM_218 = PROJECT_ROOT / "data" / "phase0" / "vlm" / "kimi_181218.json"

# 184140 (pastel)
IMG_140 = PROJECT_ROOT / "data" / "source" / "184140.jpg"
VLM_140 = PROJECT_ROOT / "data" / "phase0" / "vlm" / "vlm_184140.json"

OUT_DIR = PROJECT_ROOT / "experiments" / "e015_bilateral"


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


def save_comparison_triptych(
    original_rgb: np.ndarray,
    smoothed_rgb: np.ndarray,
    labels: np.ndarray,
    palette: np.ndarray,
    out_path: Path,
    sigma_spatial: float,
    sigma_color: float,
):
    """Save original | smoothed | segmented side by side."""
    h, w = original_rgb.shape[:2]
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)

    # Original
    canvas[:, :w] = original_rgb

    # Smoothed
    canvas[:, w:2*w] = smoothed_rgb

    # Segmented (color-coded by palette)
    segmented = palette[labels]
    canvas[:, 2*w:] = segmented

    # Add labels
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()

    draw.text((10, 10), "Original", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    draw.text((w + 10, 10), f"Smoothed (s={sigma_spatial}, c={sigma_color})", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    draw.text((2*w + 10, 10), "Segmented", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))

    img.save(out_path)


def compute_segmentation_score(panel_rgb: np.ndarray, labels: np.ndarray, palette: np.ndarray) -> dict:
    """Compute metrics to help pick the best parameter combination."""
    from skimage.color import rgb2lab

    h, w = labels.shape
    flat_rgb = panel_rgb.reshape(-1, 3)
    flat_labels = labels.reshape(-1)

    # Intra-cluster variance (lower = tighter clusters)
    intra_var = 0.0
    n_pixels = len(flat_labels)
    for i in range(len(palette)):
        mask = flat_labels == i
        if mask.sum() == 0:
            continue
        cluster_rgb = flat_rgb[mask]
        mean_rgb = cluster_rgb.mean(axis=0)
        var = float(((cluster_rgb - mean_rgb) ** 2).mean())
        intra_var += var * (mask.sum() / n_pixels)

    # Boundary coherence: measure color gradient at label boundaries
    from skimage.filters import sobel
    gray = panel_rgb.mean(axis=2)
    grad_mag = np.sqrt(sobel(gray) ** 2 + sobel(gray.T).T ** 2)

    # Label boundary mask
    from scipy import ndimage
    boundary = np.zeros_like(labels, dtype=bool)
    for axis in (0, 1):
        diff = np.diff(labels, axis=axis, prepend=0)
        boundary |= diff != 0
    # Dilate slightly
    boundary = ndimage.binary_dilation(boundary, iterations=1)

    boundary_strength = float(grad_mag[boundary].mean()) if boundary.any() else 0.0

    # Number of connected components per label (fewer = more coherent)
    from skimage.measure import label
    total_cc = 0
    for i in range(len(palette)):
        mask = labels == i
        if mask.sum() == 0:
            continue
        cc = label(mask, connectivity=2)
        total_cc += cc.max()

    return {
        "intra_cluster_variance": round(float(intra_var), 2),
        "boundary_strength": round(float(boundary_strength), 2),
        "total_connected_components": int(total_cc),
        "num_labels": int(len(palette)),
    }


def run_grid_search(img_path: Path, vlm_path: Path, out_subdir: Path, img_name: str):
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

    # --- Baseline: K-means without bilateral ---
    print("\n  --- Baseline: K-means (no bilateral) ---")
    t0 = time.perf_counter()
    result_kmeans = segment_jet_vivid_kmeans(panel, reps, max_auto_k=3)
    t_kmeans = time.perf_counter() - t0
    print(f"  Time: {t_kmeans:.3f}s")
    print(f"  Seeds: {len(result_kmeans.color_names)} ({result_kmeans.color_names})")
    print(f"  Auto-k added: {result_kmeans.notes['auto_k_added']}")

    save_boundary_overlay(panel, result_kmeans.labels, out_subdir / f"{img_name}_baseline_kmeans.png")
    save_color_overlay(panel, result_kmeans.labels, result_kmeans.palette, out_subdir / f"{img_name}_baseline_kmeans_color.png")

    baseline_score = compute_segmentation_score(panel, result_kmeans.labels, result_kmeans.palette)
    print(f"  Score: {baseline_score}")

    # --- Baseline: nearest_median (e002) ---
    print("\n  --- Baseline: nearest_median ---")
    t0 = time.perf_counter()
    result_nm = segment_jet_vivid(panel, reps, max_auto_k=3)
    t_nm = time.perf_counter() - t0
    print(f"  Time: {t_nm:.3f}s")
    print(f"  Seeds: {len(result_nm.color_names)} ({result_nm.color_names})")

    save_boundary_overlay(panel, result_nm.labels, out_subdir / f"{img_name}_baseline_nearest_median.png")

    # --- Grid search over bilateral parameters ---
    sigma_spatial_values = [5, 10, 15, 25]
    sigma_color_values = [25, 50, 75, 100]

    print(f"\n  --- Grid search: {len(sigma_spatial_values)} x {len(sigma_color_values)} combinations ---")

    results = []
    best_result = None
    best_score = None

    for sigma_spatial in sigma_spatial_values:
        for sigma_color in sigma_color_values:
            print(f"\n    sigma_spatial={sigma_spatial}, sigma_color={sigma_color}")
            t0 = time.perf_counter()
            result = segment_jet_vivid_bilateral(
                panel, reps, max_auto_k=3,
                sigma_spatial=sigma_spatial, sigma_color=sigma_color,
            )
            t_total = time.perf_counter() - t0

            score = compute_segmentation_score(panel, result.labels, result.palette)

            print(f"      Time: {t_total:.3f}s (filter: {result.notes['bilateral']['filter_time_sec']:.3f}s, kmeans: {result.notes['bilateral']['kmeans_time_sec']:.3f}s)")
            print(f"      Seeds: {len(result.color_names)} ({result.color_names})")
            print(f"      Auto-k added: {result.notes['auto_k_added']}")
            print(f"      Score: {score}")

            # Save outputs
            suffix = f"s{sigma_spatial}_c{sigma_color}"
            save_boundary_overlay(panel, result.labels, out_subdir / f"{img_name}_bilateral_{suffix}.png")
            save_color_overlay(panel, result.labels, result.palette, out_subdir / f"{img_name}_bilateral_{suffix}_color.png")

            # Save comparison triptych
            from lib.segment_bilateral import _apply_bilateral_filter
            smoothed = _apply_bilateral_filter(panel, sigma_spatial=sigma_spatial, sigma_color=sigma_color)
            save_comparison_triptych(
                panel, smoothed, result.labels, result.palette,
                out_subdir / f"{img_name}_compare_{suffix}.png",
                sigma_spatial, sigma_color,
            )

            results.append({
                "sigma_spatial": sigma_spatial,
                "sigma_color": sigma_color,
                "time_sec": round(t_total, 3),
                "filter_time_sec": result.notes["bilateral"]["filter_time_sec"],
                "kmeans_time_sec": result.notes["bilateral"]["kmeans_time_sec"],
                "num_seeds": len(result.color_names),
                "auto_k_added": result.notes["auto_k_added"],
                "score": score,
                "color_names": result.color_names,
                "palette_rgb": result.palette.tolist(),
            })

            # Pick best by lowest intra-cluster variance + highest boundary strength
            # We want low variance (tight clusters) and high boundary strength (sharp boundaries)
            composite = score["intra_cluster_variance"] / (score["boundary_strength"] + 1e-6)
            if best_score is None or composite < best_score:
                best_score = composite
                best_result = result
                best_params = (sigma_spatial, sigma_color)

    # --- Save best result prominently ---
    if best_result is not None:
        best_ss, best_sc = best_params
        print(f"\n  --- BEST: sigma_spatial={best_ss}, sigma_color={best_sc} ---")
        save_boundary_overlay(panel, best_result.labels, out_subdir / f"{img_name}_bilateral_best.png")
        save_color_overlay(panel, best_result.labels, best_result.palette, out_subdir / f"{img_name}_bilateral_best_color.png")

        from lib.segment_bilateral import _apply_bilateral_filter
        best_smoothed = _apply_bilateral_filter(panel, sigma_spatial=best_ss, sigma_color=best_sc)
        save_comparison_triptych(
            panel, best_smoothed, best_result.labels, best_result.palette,
            out_subdir / f"{img_name}_compare_best.png",
            best_ss, best_sc,
        )

    # --- Review JSON ---
    review = {
        "image": img_path.name,
        "vlm": vlm_path.name,
        "crop_shape": list(panel.shape),
        "num_vlm_seeds": len(reps),
        "saturation_ratio": float(result_kmeans.saturation_ratio),
        "baselines": {
            "kmeans": {
                "time_sec": round(t_kmeans, 3),
                "num_seeds": len(result_kmeans.color_names),
                "auto_k_added": result_kmeans.notes["auto_k_added"],
                "color_names": result_kmeans.color_names,
                "score": baseline_score,
            },
            "nearest_median": {
                "time_sec": round(t_nm, 3),
                "num_seeds": len(result_nm.color_names),
                "auto_k_added": result_nm.notes["auto_k_added"],
                "color_names": result_nm.color_names,
            },
        },
        "grid_search": results,
        "best": {
            "sigma_spatial": best_params[0] if best_result else None,
            "sigma_color": best_params[1] if best_result else None,
            "composite_score": round(best_score, 2) if best_score else None,
        } if best_result else None,
    }

    with open(out_subdir / "review.json", "w") as f:
        json.dump(review, f, indent=2)

    print(f"\n  Saved outputs to {out_subdir}")
    return review


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 181218 - vivid jet
    review_218 = run_grid_search(IMG_218, VLM_218, OUT_DIR / "181218", "181218")

    # 184140 - pastel
    review_140 = run_grid_search(IMG_140, VLM_140, OUT_DIR / "184140", "184140")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, review in [("181218", review_218), ("184140", review_140)]:
        print(f"\n{name}:")
        print(f"  Baseline k-means:  {review['baselines']['kmeans']['score']}")
        if review.get("best"):
            b = review["best"]
            print(f"  Best bilateral:    sigma_spatial={b['sigma_spatial']}, sigma_color={b['sigma_color']}, composite={b['composite_score']}")

        print(f"\n  Grid search results (sorted by composite score):")
        sorted_results = sorted(review["grid_search"], key=lambda r: r["score"]["intra_cluster_variance"] / (r["score"]["boundary_strength"] + 1e-6))
        for r in sorted_results[:5]:
            s = r["score"]
            composite = s["intra_cluster_variance"] / (s["boundary_strength"] + 1e-6)
            print(f"    s={r['sigma_spatial']:2d} c={r['sigma_color']:3d}:  var={s['intra_cluster_variance']:6.1f}  boundary={s['boundary_strength']:6.1f}  cc={s['total_connected_components']:3d}  composite={composite:6.2f}  time={r['time_sec']:.2f}s")


if __name__ == "__main__":
    main()
