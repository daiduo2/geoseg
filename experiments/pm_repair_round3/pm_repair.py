"""Targeted PM-like text artifact repair via ROI inpaint + nearest-color relabel.

Experimental. Use when a small text annotation (e.g. "PM") breaks an otherwise
clean segmentation label. Not intended for production generalization.
"""
from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from scipy import ndimage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from geoseg.modules.segment_engines._shared import _create_overlay


def detect_text_roi(
    panel_rgb: np.ndarray,
    dark_threshold: int = 45,
    min_area: int = 20,
    margin: int = 5,
) -> tuple[int, int, int, int] | None:
    """Detect a tight bounding box around dark text-like components."""
    gray = panel_rgb.mean(axis=2)
    dark = gray < dark_threshold
    if not dark.any():
        return None

    labeled, num = ndimage.label(dark)
    if num == 0:
        return None

    h, w = panel_rgb.shape[:2]
    best_bbox: tuple[int, int, int, int] | None = None
    best_score = -1.0

    for i in range(1, num + 1):
        mask = labeled == i
        area = int(mask.sum())
        if area < min_area:
            continue
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        bbox_area = (y1 - y0 + 1) * (x1 - x0 + 1)
        fill_ratio = area / bbox_area
        score = area * fill_ratio
        if score > best_score:
            best_score = score
            best_bbox = (
                max(0, x0 - margin),
                max(0, y0 - margin),
                min(w, x1 + margin + 1),
                min(h, y1 + margin + 1),
            )

    return best_bbox


def merge_small_roi_fragments(
    labels: np.ndarray,
    roi: Sequence[int],
    target_label: int,
    min_area: int = 200,
) -> np.ndarray:
    """Merge small non-background, non-target fragments inside a ROI to *target_label*."""
    x1, y1, x2, y2 = roi
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(labels.shape[1], int(x2)), min(labels.shape[0], int(y2))
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid ROI: {roi}")

    result = labels.copy()
    roi_labels = labels[y1:y2, x1:x2].copy()

    for lbl in [target_label, *sorted({int(l) for l in np.unique(roi_labels) if l not in (0, target_label)})]:
        labeled, num = ndimage.label(roi_labels == lbl)
        for i in range(1, num + 1):
            comp = labeled == i
            if comp.sum() >= min_area:
                continue
            if lbl == target_label:
                continue
            ys, xs = np.where(comp)
            touches_boundary = (
                xs.min() == 0
                or ys.min() == 0
                or xs.max() == x2 - x1 - 1
                or ys.max() == y2 - y1 - 1
            )
            if touches_boundary:
                continue
            roi_labels[comp] = target_label

    result[y1:y2, x1:x2] = roi_labels
    return result


def assign_label_to_background(
    labels: np.ndarray,
    background_label: int = 0,
) -> np.ndarray:
    """Assign a new unique label ID to each connected background region."""
    result = labels.copy()
    bg_mask = labels == background_label
    if not bg_mask.any():
        return result

    bg_labeled, num = ndimage.label(bg_mask)
    next_label = int(labels.max()) + 1
    for i in range(1, num + 1):
        result[bg_labeled == i] = next_label
        next_label += 1

    return result


def _build_text_mask(
    panel_rgb: np.ndarray,
    dark_threshold: int = 55,
    median_diff: int = 25,
    median_size: int = 7,
    dilate_iters: int = 2,
) -> np.ndarray:
    """Build a boolean mask of likely text pixels."""
    gray = panel_rgb.mean(axis=2).astype(np.float32)
    dark_mask = gray < dark_threshold

    smoothed = np.stack(
        [ndimage.median_filter(panel_rgb[:, :, c], size=median_size) for c in range(3)],
        axis=2,
    ).astype(np.float32)
    diff = np.linalg.norm(panel_rgb.astype(np.float32) - smoothed, axis=2)
    outlier_mask = diff > median_diff

    text_mask = dark_mask | outlier_mask
    if dilate_iters > 0:
        struct = np.ones((3, 3), dtype=bool)
        text_mask = ndimage.binary_dilation(text_mask, structure=struct, iterations=dilate_iters)
    return text_mask


def repair_pm_artifact(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    roi: Sequence[int] | None = None,
    dark_threshold: int = 55,
    median_diff: int = 25,
    inpaint_radius: int = 3,
) -> dict:
    """Repair a PM-like text artifact inside a small ROI.

    Pipeline: detect ROI -> build text mask -> inpaint -> nearest-color relabel -> fuse back.
    """
    if panel_rgb.shape[:2] != labels.shape:
        raise ValueError("panel_rgb and labels must have the same spatial shape")

    if roi is None:
        detected = detect_text_roi(panel_rgb)
        if detected is None:
            raise ValueError("Could not detect text ROI automatically")
        roi = detected

    x1, y1, x2, y2 = roi
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(panel_rgb.shape[1], int(x2)), min(panel_rgb.shape[0], int(y2))
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid ROI: {roi}")

    roi_rgb = panel_rgb[y1:y2, x1:x2].copy()
    roi_labels = labels[y1:y2, x1:x2].copy()

    text_mask = _build_text_mask(roi_rgb, dark_threshold, median_diff)

    inpainted_roi = cv2.inpaint(
        roi_rgb,
        text_mask.astype(np.uint8) * 255,
        inpaint_radius,
        cv2.INPAINT_NS,
    )

    unique_labels = sorted({int(lbl) for lbl in np.unique(roi_labels) if lbl != 0})
    if not unique_labels:
        cleaned_rgb = panel_rgb.copy()
        cleaned_rgb[y1:y2, x1:x2] = inpainted_roi
        return {
            "cleaned_rgb": cleaned_rgb,
            "labels": labels.copy(),
            "overlay": _create_overlay(
                cleaned_rgb, labels, np.zeros((1, 3), dtype=np.uint8)
            ),
            "roi": (x1, y1, x2, y2),
            "text_mask": np.zeros_like(labels, dtype=bool),
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

    cleaned_rgb = panel_rgb.copy()
    cleaned_rgb[y1:y2, x1:x2] = inpainted_roi

    overlay = _create_overlay(
        cleaned_rgb,
        repaired_labels,
        np.zeros((1, 3), dtype=np.uint8),
        alpha=0.65,
        boundary_mode="thin",
        skip_background=True,
        min_area_frac=0.001,
        fill_mode="blend",
    )

    full_text_mask = np.zeros_like(labels, dtype=bool)
    full_text_mask[y1:y2, x1:x2] = text_mask

    return {
        "cleaned_rgb": cleaned_rgb,
        "labels": repaired_labels,
        "overlay": overlay,
        "roi": (x1, y1, x2, y2),
        "text_mask": full_text_mask,
    }


def repair_pm_artifact_no_merge(
    labels: np.ndarray,
    rois: Sequence[Sequence[int]],
    artifact_labels: Sequence[int] | None = None,
    per_roi_artifact_labels: Sequence[Sequence[int]] | None = None,
    margin: int = 20,
    fill_mode: str = "nearest",
    row_margin: int = 40,
) -> np.ndarray:
    """Remove artifact labels from small annotation ROIs in no-merge mode.

    Text inpainting frequently pulls in lower colorbar labels (cyan/blue) that
    do not belong to the surrounding geological layer. This function detects
    those artifact labels inside each ROI and reassigns them to a valid label.

    ``fill_mode`` controls how the replacement label is chosen:

    - ``nearest`` (default): nearest valid (non-artifact) label in the ROI plus
      a small surrounding margin. Good for compact, isolated artifacts.
    - ``row_horizontal``: for each row inside the ROI, look left/right outside
      the ROI (plus ``row_margin``) for the nearest valid label. This respects
      horizontal stratification and avoids pulling labels from vertically
      adjacent layers. Falls back to ``nearest`` if a row has no valid reference.

    ``per_roi_artifact_labels`` allows a different artifact set per ROI. When
    provided it must have the same length as ``rois`` and overrides
    ``artifact_labels`` for that ROI.
    """
    if fill_mode not in {"nearest", "row_horizontal"}:
        raise ValueError(f"Unknown fill_mode: {fill_mode!r}")

    if per_roi_artifact_labels is not None and len(per_roi_artifact_labels) != len(rois):
        raise ValueError("per_roi_artifact_labels must match rois length")

    default_artifacts = list(range(11, 16)) if artifact_labels is None else list(artifact_labels)

    result = labels.copy()
    h, w = labels.shape
    for idx, roi in enumerate(rois):
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x1 >= x2 or y1 >= y2:
            continue

        artifact_set = {
            int(l)
            for l in (
                per_roi_artifact_labels[idx]
                if per_roi_artifact_labels is not None
                else default_artifacts
            )
        }
        if not artifact_set:
            continue

        roi_mask = np.zeros((h, w), dtype=bool)
        roi_mask[y1:y2, x1:x2] = True
        artifact_mask = roi_mask & np.isin(labels, list(artifact_set))
        if not artifact_mask.any():
            continue

        if fill_mode == "row_horizontal":
            rr, cc = np.where(artifact_mask)
            for y in range(y1, y2):
                row_cols = cc[rr == y]
                if row_cols.size == 0:
                    continue

                ref = np.zeros(w, dtype=bool)
                ref[: max(0, x1 - row_margin)] = True
                ref[min(w, x2 + row_margin) :] = True
                row_labels = labels[y, :]
                ref &= ~np.isin(row_labels, list(artifact_set))

                if not ref.any():
                    # Fall back to nearest valid pixel anywhere in the margin.
                    x0 = max(0, x1 - margin)
                    x3 = min(w, x2 + margin)
                    y0 = max(0, y1 - margin)
                    y3 = min(h, y2 + margin)
                    valid_crop = (
                        ~np.isin(labels[y0:y3, x0:x3], list(artifact_set))
                    )
                    if not valid_crop.any():
                        continue
                    _, indices = ndimage.distance_transform_edt(
                        ~valid_crop, return_indices=True
                    )
                    rel_r = y - y0
                    rel_c = row_cols - x0
                    result[y, row_cols] = labels[
                        y0 + indices[0][rel_r, rel_c],
                        x0 + indices[1][rel_r, rel_c],
                    ]
                    continue

                ref_2d = ref[np.newaxis, :]
                _, indices = ndimage.distance_transform_edt(
                    ~ref_2d, return_indices=True
                )
                result[y, row_cols] = row_labels[indices[1][0, row_cols]]
            continue

        # fill_mode == "nearest"
        x0 = max(0, x1 - margin)
        x3 = min(w, x2 + margin)
        y0 = max(0, y1 - margin)
        y3 = min(h, y2 + margin)
        context_mask = np.zeros((h, w), dtype=bool)
        context_mask[y0:y3, x0:x3] = True
        context_mask[y1:y2, x1:x2] = False

        valid = (context_mask | roi_mask) & ~np.isin(labels, list(artifact_set))
        if not valid.any():
            continue

        _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
        rr, cc = np.where(artifact_mask)
        result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]

    return result


if __name__ == "__main__":
    import json
    from PIL import Image

    ROOT = Path("/Users/daiduo2/geoseg")
    ORIG_DIR = ROOT / "runs/feng_fig6_final_v4/crop_tests"
    LABELS_DIR = ROOT / "runs/feng_fig6_comparisons_v7"
    OUT_DIR = ROOT / "runs/pm_repair_experiment"

    AGENT_ROIS = {
        "fig6_profile_04": (124, 17, 162, 41),
        "fig6_profile_05": (95, 35, 165, 80),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for panel_id, roi in AGENT_ROIS.items():
        img = np.array(Image.open(ORIG_DIR / f"{panel_id}_cropped.jpg").convert("RGB"))
        labels = np.load(LABELS_DIR / panel_id / "labels.npz")["labels"]

        labels = assign_label_to_background(labels)
        result = repair_pm_artifact(img, labels, roi=roi)

        panel_dir = OUT_DIR / panel_id
        panel_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(result["cleaned_rgb"]).save(panel_dir / "cleaned_rgb.jpg", quality=95)
        Image.fromarray(result["overlay"]).save(panel_dir / "overlay_repaired.jpg", quality=95)
        np.savez_compressed(panel_dir / "labels_repaired.npz", labels=result["labels"])

        from geoseg.modules.segment_engines._shared import _create_overlay
        orig_overlay = _create_overlay(
            img, labels, np.zeros((1, 3), dtype=np.uint8),
            alpha=0.65, boundary_mode="thin", skip_background=True,
            min_area_frac=0.001, fill_mode="blend",
        )
        mask_overlay = _create_overlay(
            result["cleaned_rgb"], result["labels"], np.zeros((1, 3), dtype=np.uint8),
            alpha=1.0, boundary_mode="thin", skip_background=True,
            min_area_frac=0.001, fill_mode="mask",
        )

        h, w = img.shape[:2]
        gap = 10
        comparison = np.full((h, w * 3 + gap * 2, 3), 32, dtype=np.uint8)
        comparison[:, :w] = img
        comparison[:, w + gap : 2 * w + gap] = orig_overlay
        comparison[:, 2 * w + gap * 2 :] = mask_overlay
        Image.fromarray(comparison).save(panel_dir / "comparison.jpg", quality=95)

        summary[panel_id] = {
            "roi": result["roi"],
            "text_mask_pixels": int(result["text_mask"].sum()),
            "comparison_path": str(panel_dir / "comparison.jpg"),
        }
        print(f"{panel_id}: {panel_dir / 'comparison.jpg'}")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAll outputs: {OUT_DIR}")
