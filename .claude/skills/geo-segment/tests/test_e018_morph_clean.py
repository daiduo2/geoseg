"""Test e018: e014 + Morphology denoising on 181218, 181647, 181659.

Usage:
    cd ~/.claude/skills/geo-segment
    python tests/test_e018_morph_clean.py
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

from lib.segment_morph_clean import segment_jet_vivid_morph_clean
from lib.segment_edge_guided import segment_jet_vivid_edge_guided
from lib.segment import segment_jet_vivid


PROJECT_ROOT = Path("/Users/daiduo2/Documents/knowlege/Projects/精密院-地震逆散射/photo")

# Image paths
IMG_218 = PROJECT_ROOT / "data" / "source" / "181218.jpg"
IMG_647 = PROJECT_ROOT / "data" / "source" / "181647.jpg"
IMG_659 = PROJECT_ROOT / "data" / "source" / "181659.jpg"

# VLM paths
VLM_218 = PROJECT_ROOT / "data" / "phase0" / "vlm" / "kimi_181218.json"
VLM_659 = PROJECT_ROOT / "data" / "phase0" / "vlm" / "vlm_181659.json"

OUT_DIR = PROJECT_ROOT / "experiments" / "e018_morph_clean"


def load_vlm_reps(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    for key in ["T3_color_zones_top_panel", "T3_color_zones_panel_A", "T3_color_zones"]:
        if key in data:
            return data[key]["zones"]
    raise ValueError(f"No recognized zones key in {path}")


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_boundary_overlay(panel_rgb: np.ndarray, labels: np.ndarray, out_path: Path):
    from skimage.segmentation import mark_boundaries
    overlay = mark_boundaries(panel_rgb / 255.0, labels, color=(1, 0, 0), mode="thick")
    overlay = (overlay * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def save_color_overlay(panel_rgb: np.ndarray, labels: np.ndarray, palette: np.ndarray, out_path: Path):
    from skimage.color import label2rgb
    overlay = label2rgb(labels, image=panel_rgb / 255.0, colors=palette / 255.0, alpha=0.4, bg_label=-1)
    overlay = (overlay * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def save_label_image(labels: np.ndarray, out_path: Path):
    """Save labels as a grayscale PNG (scaled to 0-255)."""
    max_l = labels.max()
    if max_l > 0:
        img = (labels.astype(np.float32) / max_l * 255).astype(np.uint8)
    else:
        img = labels.astype(np.uint8)
    Image.fromarray(img, mode="L").save(out_path)


def compute_intra_label_variance(panel_rgb: np.ndarray, labels: np.ndarray) -> float:
    """Average per-label RGB variance (lower = more homogeneous regions)."""
    unique = np.unique(labels)
    variances = []
    for lbl in unique:
        if lbl < 0:
            continue
        mask = labels == lbl
        if mask.sum() < 2:
            continue
        pixels = panel_rgb[mask].astype(np.float32)
        var = np.var(pixels, axis=0).mean()
        variances.append(var)
    return float(np.mean(variances)) if variances else 0.0


def compute_boundary_smoothness(labels: np.ndarray) -> float:
    """Measure boundary smoothness via perimeter^2 / area of each label region.
    Lower average = smoother boundaries.
    """
    from skimage.measure import label, regionprops
    smoothness_scores = []
    for lbl in np.unique(labels):
        if lbl < 0:
            continue
        mask = labels == lbl
        cc = label(mask, connectivity=2)
        for r in regionprops(cc):
            area = max(r.area, 1e-9)
            perim = r.perimeter
            ratio = float("inf") if perim == 0 else (perim ** 2) / area
            smoothness_scores.append(ratio)
    return float(np.mean(smoothness_scores)) if smoothness_scores else 0.0


def run_morph_clean_on_image(img_path: Path, vlm_path: Path | None, out_subdir: Path, img_name: str):
    print(f"\n{'='*60}")
    print(f"Processing {img_name} ({img_path.name})")
    print(f"{'='*60}")

    img = load_image(img_path)

    if vlm_path is not None and vlm_path.exists():
        reps = load_vlm_reps(vlm_path)
        # Crop panel if VLM has panel coords
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
    else:
        # No VLM: use full image and create dummy reps from image center
        panel = img
        h, w = panel.shape[:2]
        reps = [
            {"color_name": "center", "representative_point": {"x": w // 2, "y": h // 2}}
        ]
        print(f"  Panel shape: {panel.shape}")
        print(f"  No VLM reps found, using dummy center rep")

    out_subdir.mkdir(parents=True, exist_ok=True)

    # --- Baseline: nearest_median ---
    print("\n  --- Baseline: nearest_median ---")
    t0 = time.perf_counter()
    result_nm = segment_jet_vivid(panel, reps, max_auto_k=3)
    t_nm = time.perf_counter() - t0
    print(f"  Time: {t_nm:.3f}s  seeds={len(result_nm.color_names)}  auto_k={result_nm.notes['auto_k_added']}")
    save_boundary_overlay(panel, result_nm.labels, out_subdir / f"{img_name}_baseline_nearest_median.png")
    nm_regions = len(np.unique(result_nm.labels))
    nm_var = compute_intra_label_variance(panel, result_nm.labels)
    nm_smooth = compute_boundary_smoothness(result_nm.labels)

    # --- Raw e014 ---
    print("\n  --- Raw e014 edge-guided ---")
    t0 = time.perf_counter()
    result_e014 = segment_jet_vivid_edge_guided(panel, reps, max_auto_k=3, edge_weight=0.3)
    t_e014 = time.perf_counter() - t0
    print(f"  Time: {t_e014:.3f}s  seeds={len(result_e014.color_names)}  auto_k={result_e014.notes['auto_k_added']}")
    print(f"  Edge pixels: {result_e014.notes['edge_pixels_pct']:.2f}%")
    save_boundary_overlay(panel, result_e014.labels, out_subdir / f"{img_name}_raw_e014.png")
    save_color_overlay(panel, result_e014.labels, result_e014.palette, out_subdir / f"{img_name}_raw_e014_color.png")
    e014_regions = len(np.unique(result_e014.labels))
    e014_var = compute_intra_label_variance(panel, result_e014.labels)
    e014_smooth = compute_boundary_smoothness(result_e014.labels)

    # --- e018: Morph clean ---
    print("\n  --- e018: Morph clean ---")
    t0 = time.perf_counter()
    result_clean = segment_jet_vivid_morph_clean(
        panel, reps, max_auto_k=3, edge_weight=0.3
    )
    t_clean = time.perf_counter() - t0
    print(f"  Time: {t_clean:.3f}s  regions={len(np.unique(result_clean.labels))}")
    print(f"  CC initial: {result_clean.notes['n_cc_initial']}  final: {result_clean.notes['n_cc_final']}")
    print(f"  Small merged: {result_clean.notes['n_small_merged']}  Thin merged: {result_clean.notes['n_thin_merged']}")
    save_boundary_overlay(panel, result_clean.labels, out_subdir / f"{img_name}_morph_clean.png")
    save_color_overlay(panel, result_clean.labels, result_clean.palette, out_subdir / f"{img_name}_morph_clean_color.png")
    save_label_image(result_clean.labels, out_subdir / f"{img_name}_morph_clean_labels.png")

    clean_regions = len(np.unique(result_clean.labels))
    clean_var = compute_intra_label_variance(panel, result_clean.labels)
    clean_smooth = compute_boundary_smoothness(result_clean.labels)

    # --- Review JSON ---
    review = {
        "image": img_path.name,
        "vlm": vlm_path.name if vlm_path and vlm_path.exists() else None,
        "crop_shape": list(panel.shape),
        "num_vlm_seeds": len(reps),
        "saturation_ratio": float(result_nm.saturation_ratio),
        "comparisons": {
            "nearest_median": {
                "time_sec": round(t_nm, 3),
                "num_regions": nm_regions,
                "num_seeds": len(result_nm.color_names),
                "auto_k_added": result_nm.notes["auto_k_added"],
                "intra_label_variance": round(nm_var, 2),
                "boundary_smoothness": round(nm_smooth, 2),
            },
            "raw_e014": {
                "time_sec": round(t_e014, 3),
                "num_regions": e014_regions,
                "num_seeds": len(result_e014.color_names),
                "auto_k_added": result_e014.notes["auto_k_added"],
                "edge_pixels_pct": result_e014.notes["edge_pixels_pct"],
                "intra_label_variance": round(e014_var, 2),
                "boundary_smoothness": round(e014_smooth, 2),
            },
            "morph_clean": {
                "time_sec": round(t_clean, 3),
                "num_regions": clean_regions,
                "num_seeds": len(result_clean.color_names),
                "n_cc_initial": result_clean.notes["n_cc_initial"],
                "n_cc_final": result_clean.notes["n_cc_final"],
                "n_small_merged": result_clean.notes["n_small_merged"],
                "n_thin_merged": result_clean.notes["n_thin_merged"],
                "intra_label_variance": round(clean_var, 2),
                "boundary_smoothness": round(clean_smooth, 2),
                "cleaning_overhead_sec": round(t_clean - t_e014, 3),
            },
        },
    }

    with open(out_subdir / "review.json", "w") as f:
        json.dump(review, f, indent=2)

    print(f"\n  Saved outputs to {out_subdir}")
    return review


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    review_218 = run_morph_clean_on_image(IMG_218, VLM_218, OUT_DIR / "181218", "181218")
    review_647 = run_morph_clean_on_image(IMG_647, None, OUT_DIR / "181647", "181647")
    review_659 = run_morph_clean_on_image(IMG_659, VLM_659, OUT_DIR / "181659", "181659")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, review in [("181218", review_218), ("181647", review_647), ("181659", review_659)]:
        print(f"\n{name}:")
        for method, info in review["comparisons"].items():
            print(f"  {method:20s}: time={info['time_sec']:6.3f}s  regions={info['num_regions']:3d}  var={info['intra_label_variance']:7.2f}  smooth={info['boundary_smoothness']:7.2f}")
        clean = review["comparisons"]["morph_clean"]
        print(f"    -> Merged: small={clean['n_small_merged']}  thin={clean['n_thin_merged']}  overhead={clean['cleaning_overhead_sec']:.3f}s")


if __name__ == "__main__":
    main()
