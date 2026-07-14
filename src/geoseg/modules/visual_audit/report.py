"""Visual audit report generation.

Combines views, crops, and diagnostic signals into a report image + JSON.
This module does NOT produce a PASS/FAIL verdict. The report contains:
- paths to problem-exposing views and crops
- diagnostic signals (objective metrics, no thresholds)
- label-color mapping for the overlay legend

The agent reads these materials and produces a RegionalAudit with frozen/retry
labels and repair directions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geoseg.modules.segment_engines.metrics import compute_all
from geoseg.modules.visual_audit.crops import create_audit_crops, save_crops
from geoseg.modules.visual_audit.rendering import create_overlay_with_legend
from geoseg.modules.visual_audit.semantic import (
    _find_manual_gt_mask,
    compute_semantic_fidelity,
)
from geoseg.modules.visual_audit.views import create_audit_views, save_views


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types and keys to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {
            int(k) if isinstance(k, np.integer) else
            float(k) if isinstance(k, np.floating) else
            bool(k) if isinstance(k, np.bool_) else k:
            _sanitize_for_json(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """Resize an image to a target height, preserving aspect ratio."""
    h, w = image.shape[:2]
    if h == 0:
        return image
    scale = target_height / h
    new_w = max(1, int(w * scale))
    return np.array(
        Image.fromarray(image).resize((new_w, target_height), Image.LANCZOS)
    )


def _tile_images(images: list[np.ndarray], labels: list[str], cols: int = 2) -> np.ndarray:
    """Tile images into a grid with text labels."""
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    target_h = 400
    resized = [_resize_to_height(img, target_h) for img in images]

    rows = (len(resized) + cols - 1) // cols
    pad = 10
    font = _load_font(max(14, target_h // 25))

    row_heights = []
    for r in range(rows):
        start = r * cols
        end = min(start + cols, len(resized))
        row_imgs = resized[start:end]
        row_heights.append(max(img.shape[0] for img in row_imgs) + pad * 2)

    total_w = max(
        sum(img.shape[1] for img in resized[r * cols : (r + 1) * cols]) + pad * (cols + 1)
        for r in range(rows)
    ) if rows > 0 else 100
    total_h = sum(row_heights) + pad

    canvas = np.full((total_h, total_w, 3), 32, dtype=np.uint8)
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)

    y = pad
    for r in range(rows):
        start = r * cols
        end = min(start + cols, len(resized))
        row_imgs = resized[start:end]
        row_labels = labels[start:end]
        row_h = max(img.shape[0] for img in row_imgs) + pad * 2

        x = pad
        for img, label_text in zip(row_imgs, row_labels):
            canvas[y + pad : y + pad + img.shape[0], x : x + img.shape[1]] = img
            draw.text((x, y), label_text, fill=(255, 255, 255), font=font)
            x += img.shape[1] + pad
        y += row_h

    return np.array(pil)


def create_audit_report(
    labels: np.ndarray,
    panel_rgb: np.ndarray,
    output_dir: str,
    no_text_rgb: np.ndarray | None = None,
    text_mask: np.ndarray | None = None,
    panel3_mode: bool = False,
    labels_path: str | None = None,
    gt_mask_path: str | None = None,
) -> dict:
    """Generate a complete visual audit report.

    Saves views, crops, a tiled summary image, and a JSON report.

    Returns:
        dict with keys:
        - view_paths: dict[str, str]
        - crop_paths: dict
        - diagnostic_signals: dict
        - summary_image_path: str
        - report_path: str
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    views_dir = out / "views"
    crops_dir = out / "crops"

    gt_mask = None
    if gt_mask_path:
        gt_img = np.array(Image.open(gt_mask_path).convert("L"))
        gt_mask = gt_img > 0
    if gt_mask is None:
        gt_mask = _find_manual_gt_mask(labels_path)
    views = create_audit_views(labels, panel_rgb, no_text_rgb, text_mask, gt_mask)
    view_paths = save_views(views, str(views_dir))

    crops = create_audit_crops(panel_rgb if no_text_rgb is None else no_text_rgb)
    crop_paths = save_crops(crops, str(crops_dir))

    metrics = compute_all(labels, panel_rgb)
    semantic = compute_semantic_fidelity(labels, panel_rgb, gt_mask)

    overlay_legend = create_overlay_with_legend(panel_rgb, labels)
    label_color_map = {}
    for lbl in sorted(set(labels.flatten()) - {0}):
        mask = labels == lbl
        ys, xs = np.where(mask)
        label_color_map[str(int(lbl))] = {
            "area_frac": round(float(mask.sum() / labels.size), 4),
            "median_y": round(float(np.median(ys)), 1) if len(ys) > 0 else None,
            "color": overlay_legend[ys[0], xs[0]].tolist() if len(ys) > 0 else [128, 128, 128],
        }

    report = {
        "diagnostic_signals": {**metrics, **semantic},
        "label_color_map": label_color_map,
        "view_paths": view_paths,
        "crop_paths": crop_paths,
    }

    # Summary image: views on top, crops below
    view_items = [
        (views.get("side_by_side"), "side_by_side"),
        (views.get("plume_comparison"), "plume_comparison"),
        (views.get("pure_mask"), "pure_mask"),
        (views.get("fragment_highlight"), "fragment_highlight"),
        (views.get("boundary_on_no_text") if views.get("boundary_on_no_text") is not None else views.get("boundary_on_original"), "boundary"),
        (views.get("text_residual_map"), "text_residual"),
        (views.get("topology_map"), "topology"),
        (views.get("difference_heatmap"), "diff_heatmap"),
        (views.get("color_residual"), "color_residual"),
    ]
    view_images = [img for img, _ in view_items if img is not None]
    view_labels = [name for img, name in view_items if img is not None]

    crop_images = []
    crop_labels = []
    for name, value in crops.items():
        if isinstance(value, list):
            for i, img in enumerate(value):
                crop_images.append(img)
                crop_labels.append(f"{name}_{i}")
        else:
            crop_images.append(value)
            crop_labels.append(name)

    top_tile = _tile_images(view_images, view_labels, cols=2)
    bottom_tile = _tile_images(crop_images, crop_labels, cols=2)

    h_top, w_top = top_tile.shape[:2]
    h_bottom, w_bottom = bottom_tile.shape[:2]
    total_w = max(w_top, w_bottom)
    total_h = h_top + h_bottom + 20

    summary = np.full((total_h, total_w, 3), 32, dtype=np.uint8)
    summary[:h_top, :w_top] = top_tile
    summary[h_top + 20 : h_top + 20 + h_bottom, :w_bottom] = bottom_tile

    # Add audit banner (no verdict)
    pil = Image.fromarray(summary)
    draw = ImageDraw.Draw(pil)
    font = _load_font(max(18, total_h // 30))
    draw.text((20, h_top + 5), "VISUAL AUDIT — review by agent", fill=(200, 200, 200), font=font)
    summary = np.array(pil)

    summary_path = out / "summary.jpg"
    Image.fromarray(summary).save(summary_path, quality=90)

    report["summary_image_path"] = str(summary_path)

    report = _sanitize_for_json(report)

    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)

    return report
