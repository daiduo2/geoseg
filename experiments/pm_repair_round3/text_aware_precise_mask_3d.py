"""Text-aware segmentation with precise text-mask extraction inside OCR ROIs.

Pipeline:
    1. Detect text boxes with PaddleOCR (rectangular ROIs).
    2. For each ROI, crop the region and run color segmentation (k-means k=2 or
       brightness threshold) to separate text pixels from background.
    3. Build a precise text mask from the text cluster.
    4. Use the precise mask for local inpaint + nearest-color relabel.
    5. Compare against baseline and against rectangular-ROI repair.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "pm_repair_round3"))

from geoseg.modules.segment_engines._shared import _create_overlay
from geoseg.modules.segment_engines.v4_kmeans import segment as v4_segment
from pm_repair import assign_label_to_background

OCR_VENV_PYTHON = ROOT / ".venv_ocr" / "bin" / "python"
OCR_ROI_SCRIPT = ROOT / "experiments" / "pm_repair_round3" / "ocr_roi.py"


def detect_text_rois_subprocess(image_path: Path) -> list[dict]:
    """Run OCR in the isolated .venv_ocr via subprocess."""
    cmd = [
        str(OCR_VENV_PYTHON),
        str(OCR_ROI_SCRIPT),
        "--all",
        str(image_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _poly_to_mask(
    poly: list[tuple[float, float]],
    shape: tuple[int, int],
) -> np.ndarray:
    """Convert a polygon to a binary mask of the given shape."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask.astype(bool)


def _choose_text_cluster_kmeans(
    roi_rgb: np.ndarray,
    text_hint: str = "light",
    poly_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Run k-means k=2 inside ROI and return mask of the text cluster.

    If poly_mask is provided, only pixels inside the polygon participate in
    clustering, and the returned mask is constrained to the polygon.

    Text pixels are typically the smaller cluster inside a tight OCR polygon.
    Use brightness as a consistency check, but area is the primary signal.
    """
    h, w = roi_rgb.shape[:2]
    if poly_mask is None:
        poly_mask = np.ones((h, w), dtype=bool)

    pixels = roi_rgb[poly_mask].astype(np.float32)
    if pixels.shape[0] < 2:
        return np.zeros((h, w), dtype=bool)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels,
        2,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    # Map cluster labels back to the polygon-masked region.
    cluster_map = np.zeros((h, w), dtype=np.uint8)
    cluster_map[poly_mask] = labels.flatten()
    cluster_mask_0 = (cluster_map == 0) & poly_mask
    cluster_mask_1 = (cluster_map == 1) & poly_mask

    brightness = centers.mean(axis=1)
    area_0 = cluster_mask_0.sum()
    area_1 = cluster_mask_1.sum()

    small_is_0 = area_0 < area_1
    small_mask = cluster_mask_0 if small_is_0 else cluster_mask_1
    small_bright = brightness[0] if small_is_0 else brightness[1]
    large_bright = brightness[1] if small_is_0 else brightness[0]

    # Text is the smaller cluster; verify brightness hint is not contradictory.
    if text_hint == "light" and small_bright >= large_bright * 0.8:
        return small_mask
    if text_hint == "dark" and small_bright <= large_bright * 1.2:
        return small_mask

    # Brightness contradicts area: fall back to brightness hint.
    if text_hint == "light":
        return cluster_mask_0 if brightness[0] > brightness[1] else cluster_mask_1
    return cluster_mask_0 if brightness[0] < brightness[1] else cluster_mask_1


def _refine_text_mask(
    text_mask: np.ndarray,
    min_area: int = 10,
    dilate_iters: int = 1,
) -> np.ndarray:
    """Clean up the text mask: remove tiny fragments, optionally dilate."""
    labeled, num = ndimage.label(text_mask)
    refined = np.zeros_like(text_mask, dtype=bool)
    for i in range(1, num + 1):
        comp = labeled == i
        if comp.sum() >= min_area:
            refined |= comp

    if dilate_iters > 0:
        struct = np.ones((3, 3), dtype=bool)
        refined = ndimage.binary_dilation(refined, structure=struct, iterations=dilate_iters)
    return refined


def extract_precise_text_mask(
    roi_rgb: np.ndarray,
    method: str = "kmeans2",
    text_hint: str = "light",
    poly: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Extract a precise text mask inside a single OCR ROI.

    Args:
        roi_rgb: Cropped RGB image of the OCR ROI.
        method: "kmeans2" (default), "threshold", or "otsu".
        text_hint: "light" or "dark" — expected text color relative to background.
        poly: Optional polygon (list of (x, y)) in original image coordinates. If
            provided, clustering is constrained to the polygon interior.

    Returns:
        Boolean mask of text pixels inside the ROI.
    """
    h, w = roi_rgb.shape[:2]
    poly_mask = None
    if poly is not None:
        poly_mask = _poly_to_mask(poly, (h, w))

    if method == "kmeans2":
        text_mask = _choose_text_cluster_kmeans(roi_rgb, text_hint=text_hint, poly_mask=poly_mask)
    elif method == "threshold":
        gray = roi_rgb.mean(axis=2)
        if text_hint == "light":
            text_mask = gray > 200
        else:
            text_mask = gray < 55
        if poly_mask is not None:
            text_mask &= poly_mask
    elif method == "otsu":
        gray = roi_rgb.mean(axis=2).astype(np.uint8)
        if poly_mask is not None:
            # Only run Otsu on pixels inside the polygon.
            poly_pixels = gray[poly_mask]
            if poly_pixels.size < 2:
                text_mask = np.zeros((h, w), dtype=bool)
            else:
                _, thresh = cv2.threshold(poly_pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                otsu_val = thresh[0, 0]
                if text_hint == "light":
                    text_mask = (gray > otsu_val) & poly_mask
                else:
                    text_mask = (gray < otsu_val) & poly_mask
        else:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_val = thresh
            if text_hint == "light":
                text_mask = gray > otsu_val
            else:
                text_mask = gray < otsu_val
    else:
        raise ValueError(f"Unknown method: {method}")

    return _refine_text_mask(text_mask)


def repair_roi_with_precise_mask(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    roi: tuple[int, int, int, int],
    poly: list[tuple[float, float]] | None = None,
    method: str = "kmeans2",
    text_hint: str = "light",
    inpaint_radius: int = 2,
) -> dict:
    """Repair labels inside an OCR ROI using a precise text mask."""
    x1, y1, x2, y2 = roi
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(panel_rgb.shape[1], int(x2)), min(panel_rgb.shape[0], int(y2))
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid ROI: {roi}")

    roi_rgb = panel_rgb[y1:y2, x1:x2].copy()
    roi_labels = labels[y1:y2, x1:x2].copy()

    poly_local = None
    if poly is not None:
        poly_local = [(px - x1, py - y1) for px, py in poly]

    text_mask = extract_precise_text_mask(roi_rgb, method=method, text_hint=text_hint, poly=poly_local)
    if not text_mask.any():
        return {
            "labels": labels.copy(),
            "roi": (x1, y1, x2, y2),
            "text_mask": np.zeros_like(labels, dtype=bool),
            "roi_text_mask": text_mask,
        }

    inpainted_roi = cv2.inpaint(
        roi_rgb,
        text_mask.astype(np.uint8) * 255,
        inpaint_radius,
        cv2.INPAINT_NS,
    )

    unique_labels = sorted({int(lbl) for lbl in np.unique(roi_labels) if lbl != 0})
    if not unique_labels:
        repaired_labels = labels.copy()
        repaired_labels[y1:y2, x1:x2] = roi_labels
        return {
            "labels": repaired_labels,
            "roi": (x1, y1, x2, y2),
            "text_mask": np.zeros_like(labels, dtype=bool),
            "roi_text_mask": text_mask,
        }

    max_label = max(unique_labels)
    palette = np.zeros((max_label + 1, 3), dtype=np.float32)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    for lbl in unique_labels:
        mask = (roi_labels == lbl) & (~text_mask)
        if mask.any():
            palette[lbl] = np.median(roi_rgb[mask], axis=0)
            counts[lbl] = mask.sum()

    valid_labels = [lbl for lbl in unique_labels if counts[lbl] > 0]
    valid_palette = palette[valid_labels]
    if valid_palette.shape[0] > 0:
        flat_pixels = inpainted_roi.reshape(-1, 3).astype(np.float32)
        d2 = ((flat_pixels[:, None, :] - valid_palette[None, :, :]) ** 2).sum(axis=2)
        roi_labels = np.array(valid_labels, dtype=np.int64)[d2.argmin(axis=1)].reshape(
            inpainted_roi.shape[:2]
        )

    repaired_labels = labels.copy()
    repaired_labels[y1:y2, x1:x2] = roi_labels

    full_text_mask = np.zeros_like(labels, dtype=bool)
    full_text_mask[y1:y2, x1:x2] = text_mask

    return {
        "labels": repaired_labels,
        "roi": (x1, y1, x2, y2),
        "text_mask": full_text_mask,
        "roi_text_mask": text_mask,
    }


def visualize_text_masks(
    panel_rgb: np.ndarray,
    matches: list[dict],
    method: str = "kmeans2",
    text_hint: str = "light",
) -> np.ndarray:
    """Draw OCR boxes + precise text masks on the original image for debugging."""
    vis = panel_rgb.copy()
    for match in matches:
        x1, y1, x2, y2 = match["roi"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(panel_rgb.shape[1], x2), min(panel_rgb.shape[0], y2)
        roi_rgb = panel_rgb[y1:y2, x1:x2]

        # Convert polygon to ROI-local coordinates.
        poly = match.get("poly")
        if poly is not None:
            poly_local = [(px - x1, py - y1) for px, py in poly]
        else:
            poly_local = None

        text_mask = extract_precise_text_mask(roi_rgb, method=method, text_hint=text_hint, poly=poly_local)

        # Draw ROI rectangle.
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Overlay text mask in green.
        mask_vis = np.zeros_like(roi_rgb)
        mask_vis[text_mask] = [0, 255, 0]
        vis[y1:y2, x1:x2] = cv2.addWeighted(vis[y1:y2, x1:x2], 0.6, mask_vis, 0.4, 0)

        # Label text.
        cv2.putText(
            vis,
            match["text"][:10],
            (x1, max(y1 - 3, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
        )
    return vis


def process_panel(
    image_path: Path,
    output_dir: Path,
    n_layers: int = 5,
    method: str = "kmeans2",
    text_hint: str = "light",
) -> dict:
    """Run baseline segmentation + precise text-mask repair."""
    panel_id = image_path.stem
    out = output_dir / panel_id
    out.mkdir(parents=True, exist_ok=True)

    panel_rgb = np.array(Image.open(image_path).convert("RGB"))
    h, w = panel_rgb.shape[:2]

    # --- 1. OCR text detection ---
    matches = detect_text_rois_subprocess(image_path)

    # --- 2. Visualize precise text masks on original ---
    mask_vis = visualize_text_masks(panel_rgb, matches, method=method, text_hint=text_hint)
    Image.fromarray(mask_vis).save(out / "00_ocr_precise_masks.jpg", quality=95)

    # --- 3. Baseline segmentation ---
    baseline_result = v4_segment(panel_rgb, n_layers=n_layers)
    baseline_labels = baseline_result["labels"]
    baseline_overlay = baseline_result["overlay"]
    baseline_labels = assign_label_to_background(baseline_labels)

    np.savez_compressed(out / "labels_baseline.npz", labels=baseline_labels)
    Image.fromarray(baseline_overlay).save(out / "01_baseline_overlay.jpg", quality=95)

    # --- 4. Precise-mask repair per ROI ---
    repaired_labels = baseline_labels.copy()
    total_text_pixels = 0
    for match in matches:
        roi = tuple(match["roi"])
        poly = match.get("poly")
        try:
            result = repair_roi_with_precise_mask(
                panel_rgb,
                repaired_labels,
                roi,
                poly=poly,
                method=method,
                text_hint=text_hint,
                inpaint_radius=2,
            )
            repaired_labels = result["labels"]
            total_text_pixels += int(result["text_mask"].sum())
        except Exception as e:
            print(f"  WARNING: failed to repair ROI {roi} ({match['text']}): {e}")

    repaired_overlay = _create_overlay(
        panel_rgb,
        repaired_labels,
        np.zeros((1, 3), dtype=np.uint8),
        alpha=0.65,
        boundary_mode="thin",
        skip_background=True,
        min_area_frac=0.001,
        fill_mode="blend",
    )
    np.savez_compressed(out / "labels_precise_repaired.npz", labels=repaired_labels)
    Image.fromarray(repaired_overlay).save(out / "02_precise_repaired_overlay.jpg", quality=95)

    # --- 5. Difference map ---
    diff_mask = baseline_labels != repaired_labels
    diff_vis = np.full((h, w, 3), 32, dtype=np.uint8)
    diff_vis[diff_mask] = [255, 0, 0]
    Image.fromarray(diff_vis).save(out / "03_label_difference.jpg", quality=95)

    diff_frac = float(diff_mask.sum() / diff_mask.size)

    summary = {
        "panel_id": panel_id,
        "image_path": str(image_path),
        "method": method,
        "text_hint": text_hint,
        "n_text_boxes": len(matches),
        "total_text_pixels": total_text_pixels,
        "diff_fraction": round(diff_frac, 4),
        "outputs": {
            "ocr_precise_masks": str(out / "00_ocr_precise_masks.jpg"),
            "baseline_overlay": str(out / "01_baseline_overlay.jpg"),
            "precise_repaired_overlay": str(out / "02_precise_repaired_overlay.jpg"),
            "label_difference": str(out / "03_label_difference.jpg"),
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n{panel_id}: diff_fraction={diff_frac:.4f}, text_pixels={total_text_pixels}")
    print(f"  outputs: {out}")
    return summary


def main() -> int:
    base = Path("/Users/daiduo2/geoseg/docs/best_overlays_3d_schematic")
    output_dir = Path("/Users/daiduo2/geoseg/runs/pm_repair_ocr_experiment/text_aware_precise_mask")
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        (base / "01_panel_1_original.png", 5, "light"),
        (base / "02_panel_2_original.png", 5, "light"),
        (base / "03_panel_3_original.png", 6, "light"),
    ]

    summaries = []
    for path, n_layers, text_hint in panels:
        if not path.exists():
            print(f"SKIP: {path} not found")
            continue
        summary = process_panel(
            path,
            output_dir,
            n_layers=n_layers,
            method="otsu",
            text_hint=text_hint,
        )
        summaries.append(summary)

    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nAll outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
