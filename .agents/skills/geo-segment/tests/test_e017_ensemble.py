"""Test e017: Multi-algorithm ensemble voting on 181218, 181659, 181210.

Usage:
    cd ~/.claude/skills/geo-segment
    python tests/test_e017_ensemble.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path.home() / ".claude/skills/geo-segment"))

from lib.segment import segment_jet_vivid
from lib.segment_edge_guided import segment_jet_vivid_edge_guided
from lib.segment_merge import segment_jet_vivid_merge
from lib.segment_ensemble import segment_jet_vivid_ensemble


PROJECT_ROOT = Path("/Users/daiduo2/Documents/knowlege/Projects/精密院-地震逆散射/photo")

IMAGES = {
    "181218": {
        "img": PROJECT_ROOT / "data" / "source" / "181218.jpg",
        "vlm": PROJECT_ROOT / "data" / "phase0" / "vlm" / "kimi_181218.json",
    },
    "181659": {
        "img": PROJECT_ROOT / "data" / "source" / "181659.jpg",
        "vlm": PROJECT_ROOT / "data" / "phase0" / "vlm" / "vlm_181659.json",
    },
    "181210": {
        "img": PROJECT_ROOT / "data" / "source" / "181210.jpg",
        "vlm": PROJECT_ROOT / "data" / "phase0" / "vlm" / "kimi_181210.json",
    },
}

OUT_DIR = PROJECT_ROOT / "experiments" / "e017_ensemble"


def load_vlm_reps(path: Path) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    for key in ["T3_color_zones_top_panel", "T3_color_zones_panel_A"]:
        if key in data:
            return data[key]["zones"]
    raise ValueError(f"No recognized zones key in {path}")


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def extract_panel(img: np.ndarray, vlm_path: Path) -> np.ndarray:
    with open(vlm_path, "r") as f:
        data = json.load(f)
    for key in ["T1_panels"]:
        if key in data:
            p = data[key]["panels"][0]
            x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
            return img[y1:y2, x1:x2]
    return img


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


def save_disagreement_overlay(panel_rgb: np.ndarray, disagreements: np.ndarray, out_path: Path):
    """Overlay disagreement pixels in magenta."""
    overlay = panel_rgb.copy().astype(np.float32)
    overlay[disagreements] = overlay[disagreements] * 0.5 + np.array([255, 0, 255], dtype=np.float32) * 0.5
    overlay = overlay.clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def compute_boundary_gradient(panel_rgb: np.ndarray, labels: np.ndarray) -> float:
    """Average Sobel gradient magnitude at label boundaries."""
    from skimage.filters import sobel
    gray = panel_rgb.mean(axis=2)
    grad = sobel(gray)
    from skimage.segmentation import find_boundaries
    boundaries = find_boundaries(labels, mode="thick")
    if boundaries.sum() == 0:
        return 0.0
    return float(grad[boundaries].mean())


def compute_intra_label_variance(panel_rgb: np.ndarray, labels: np.ndarray) -> float:
    """Average per-label RGB variance (lower = more uniform regions)."""
    unique = np.unique(labels)
    variances = []
    flat_rgb = panel_rgb.reshape(-1, 3).astype(np.float32)
    flat_labels = labels.reshape(-1)
    for lid in unique:
        mask = flat_labels == lid
        pixels = flat_rgb[mask]
        if len(pixels) > 1:
            variances.append(float(np.var(pixels, axis=0).mean()))
    return float(np.mean(variances)) if variances else 0.0


def compute_consistency_score(labels: np.ndarray) -> float:
    """Fraction of pixels whose 8-neighborhood has the same label."""
    from scipy import ndimage
    def _mode_count(x):
        x_int = x.astype(np.int64)
        return float(np.bincount(x_int).max())
    modes = ndimage.generic_filter(labels.astype(np.int32), _mode_count, size=3, mode="nearest")
    neighborhood_size = 9
    return float((modes / neighborhood_size).mean())


def run_ensemble_on_image(img_path: Path, vlm_path: Path, out_subdir: Path, img_name: str):
    print(f"\n{'='*60}")
    print(f"Processing {img_name} ({img_path.name})")
    print(f"{'='*60}")

    img = load_image(img_path)
    reps = load_vlm_reps(vlm_path)
    panel = extract_panel(img, vlm_path)

    print(f"  Panel shape: {panel.shape}")
    print(f"  VLM reps: {len(reps)}")

    out_subdir.mkdir(parents=True, exist_ok=True)

    # --- Run individual algorithms ---
    print("\n  --- Baseline: nearest_median ---")
    t0 = time.perf_counter()
    r1 = segment_jet_vivid(panel, reps, max_auto_k=3)
    t1 = time.perf_counter() - t0
    print(f"  Time: {t1:.3f}s  labels={len(np.unique(r1.labels))}  auto_k={r1.notes['auto_k_added']}")
    save_boundary_overlay(panel, r1.labels, out_subdir / f"{img_name}_baseline.png")

    print("\n  --- Edge-guided K-means ---")
    t0 = time.perf_counter()
    r2 = segment_jet_vivid_edge_guided(panel, reps, max_auto_k=3, edge_weight=0.3)
    t2 = time.perf_counter() - t0
    print(f"  Time: {t2:.3f}s  labels={len(np.unique(r2.labels))}  auto_k={r2.notes['auto_k_added']}")
    save_boundary_overlay(panel, r2.labels, out_subdir / f"{img_name}_edgeguided.png")

    print("\n  --- Mean Shift + hierarchical merge ---")
    t0 = time.perf_counter()
    r3 = segment_jet_vivid_merge(panel, reps, max_auto_k=3, use_hierarchy=True)
    t3 = time.perf_counter() - t0
    print(f"  Time: {t3:.3f}s  labels={len(np.unique(r3.labels))}  auto_k={r3.notes.get('auto_k_added', 'N/A')}")
    save_boundary_overlay(panel, r3.labels, out_subdir / f"{img_name}_merge.png")

    # --- Run ensemble ---
    print("\n  --- Ensemble voting ---")
    t0 = time.perf_counter()
    re = segment_jet_vivid_ensemble(panel, reps, max_auto_k=3)
    te = time.perf_counter() - t0
    print(f"  Time: {te:.3f}s  labels={len(np.unique(re.labels))}  disagreement={re.notes['disagreement_pct']:.2f}%")
    save_boundary_overlay(panel, re.labels, out_subdir / f"{img_name}_ensemble.png")
    save_color_overlay(panel, re.labels, re.palette, out_subdir / f"{img_name}_ensemble_color.png")

    # --- Compute metrics ---
    def _metrics(r, t_sec):
        return {
            "time_sec": round(t_sec, 3),
            "num_labels": int(len(np.unique(r.labels))),
            "boundary_gradient": round(compute_boundary_gradient(panel, r.labels), 4),
            "intra_label_variance": round(compute_intra_label_variance(panel, r.labels), 2),
            "consistency_score": round(compute_consistency_score(r.labels), 4),
        }

    metrics = {
        "baseline": _metrics(r1, t1),
        "edge_guided": _metrics(r2, t2),
        "merge": _metrics(r3, t3),
        "ensemble": _metrics(re, te),
    }

    for name, m in metrics.items():
        print(f"  {name:15s}: grad={m['boundary_gradient']:.4f}  var={m['intra_label_variance']:.2f}  "
              f"consistency={m['consistency_score']:.4f}  time={m['time_sec']:.3f}s")

    # --- Disagreement visualization ---
    from skimage.color import rgb2lab
    panel_lab = rgb2lab(panel)
    common_palette_lab = rgb2lab(r1.palette[np.newaxis, ...])[0]

    def _map_labels(src_labels, src_palette):
        src_palette_lab = rgb2lab(src_palette[np.newaxis, ...])[0]
        mapping = {}
        for i, sp in enumerate(src_palette_lab):
            d = np.linalg.norm(common_palette_lab - sp, axis=1)
            mapping[i] = int(d.argmin())
        return np.vectorize(mapping.get)(src_labels)

    l1 = r1.labels
    l2 = _map_labels(r2.labels, r2.palette)
    l3 = _map_labels(r3.labels, r3.palette)
    labels_stack = np.stack([l1, l2, l3], axis=2)
    disagreements = np.zeros(panel.shape[:2], dtype=bool)
    for y in range(panel.shape[0]):
        for x in range(panel.shape[1]):
            disagreements[y, x] = len(np.unique(labels_stack[y, x, :])) > 1

    save_disagreement_overlay(panel, disagreements, out_subdir / f"{img_name}_disagreements.png")

    # --- Review JSON ---
    review = {
        "image": img_path.name,
        "vlm": vlm_path.name,
        "crop_shape": list(panel.shape),
        "num_vlm_seeds": len(reps),
        "saturation_ratio": float(re.saturation_ratio),
        "metrics": metrics,
        "ensemble_details": {
            "disagreement_pct": re.notes["disagreement_pct"],
            "num_labels_per_algo": re.notes["num_labels_per_algo"],
            "final_num_labels": re.notes["final_num_labels"],
            "timings_sec": re.notes["timings_sec"],
        },
    }

    with open(out_subdir / "review.json", "w") as f:
        json.dump(review, f, indent=2)

    print(f"\n  Saved outputs to {out_subdir}")
    return review


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_reviews = {}
    for name, paths in IMAGES.items():
        review = run_ensemble_on_image(paths["img"], paths["vlm"], OUT_DIR / name, name)
        all_reviews[name] = review

    # --- Summary review ---
    with open(OUT_DIR / "review.json", "w") as f:
        json.dump(all_reviews, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, review in all_reviews.items():
        print(f"\n{name}:")
        for method, info in review["metrics"].items():
            print(f"  {method:15s}: grad={info['boundary_gradient']:.4f}  "
                  f"var={info['intra_label_variance']:.2f}  consistency={info['consistency_score']:.4f}  "
                  f"time={info['time_sec']:.3f}s  labels={info['num_labels']}")


if __name__ == "__main__":
    main()
