"""Prepare per-panel baseline segmentation and overlay-with-legend for visual audit.

This script is intentionally self-contained and lives in experiments/ so it does
not modify any source files under src/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure src/ is on the path when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geoseg.modules.segment_engines._shared import _detect_background_label
from geoseg.modules.segment_engines.v4_kmeans import segment as v4_segment
from geoseg.preprocessing.absorption import absorb_artifacts
from geoseg.preprocessing.detectors import detect_black_crosses, detect_red_lines, detect_text
from geoseg.preprocessing.panel_split import split_panels_colored_components


INPUT_IMAGE = Path("/Users/daiduo2/geoseg/newimage.jpg")
OUTPUT_DIR = Path("/Users/daiduo2/geoseg/runs/visual_audit_merge_experiment")
N_LAYERS = 5


def _color_name(rgb: tuple[int, int, int]) -> str:
    """Return a simple descriptive name for an RGB color."""
    r, g, b = rgb
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    if max_val - min_val < 30:
        if max_val < 80:
            return "black"
        if max_val > 200:
            return "white"
        return "gray"
    channel_names = []
    if r > 150 and r >= g and r >= b:
        channel_names.append("red")
    if g > 150 and g >= r and g >= b:
        channel_names.append("green")
    if b > 150 and b >= r and b >= g:
        channel_names.append("blue")
    if r > 180 and g > 120 and b < 100:
        return "orange"
    if r > 180 and g > 180 and b < 120:
        return "yellow"
    if r > 120 and g < 120 and b > 120:
        return "purple"
    if r < 120 and g > 150 and b > 150:
        return "cyan"
    return "-".join(channel_names) if channel_names else "mixed"


def _build_legend(
    overlay: np.ndarray,
    labels: np.ndarray,
    label_ids: list[int],
    bg_label: int | None,
    target_height: int,
) -> np.ndarray:
    """Create a legend image mapping each non-background label to its overlay color."""
    legend = np.full((target_height, 320, 3), 255, dtype=np.uint8)
    pil_legend = Image.fromarray(legend)
    draw = ImageDraw.Draw(pil_legend)

    try:
        font_id = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
        font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
    except OSError:
        font_id = ImageFont.load_default()
        font_title = font_id

    draw.text((20, 20), "Label Legend", fill=(0, 0, 0), font=font_title)
    y = 70
    row_height = 50

    for lbl in label_ids:
        if bg_label is not None and lbl == bg_label:
            continue
        mask = labels == lbl
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        sample_y, sample_x = int(np.median(ys)), int(np.median(xs))
        color = tuple(int(c) for c in overlay[sample_y, sample_x])
        name = _color_name(color)

        swatch_size = 32
        draw.rectangle(
            [20, y, 20 + swatch_size, y + swatch_size],
            fill=tuple(color),
            outline=(0, 0, 0),
            width=2,
        )
        draw.text(
            (20 + swatch_size + 15, y),
            f"ID {lbl}: {name}",
            fill=(0, 0, 0),
            font=font_id,
        )
        y += row_height
        if y + row_height > target_height - 20:
            break

    return np.array(pil_legend)


def _create_overlay_legend(
    overlay: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Composite overlay (left) and legend (right)."""
    bg_label = _detect_background_label(labels)
    unique_labels = sorted(int(l) for l in np.unique(labels) if l >= 0)

    target_height = max(overlay.shape[0], 300)
    scale = target_height / overlay.shape[0]
    new_width = int(overlay.shape[1] * scale)
    resized_overlay = cv2.resize(overlay, (new_width, target_height), interpolation=cv2.INTER_AREA)

    legend = _build_legend(overlay, labels, unique_labels, bg_label, target_height)

    composite = np.full(
        (target_height, resized_overlay.shape[1] + legend.shape[1], 3),
        255,
        dtype=np.uint8,
    )
    composite[:, :resized_overlay.shape[1]] = resized_overlay
    composite[:, resized_overlay.shape[1]:] = legend
    return composite


def main() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_rgb = np.array(Image.open(INPUT_IMAGE).convert("RGB"))
    panel_bboxes = split_panels_colored_components(img_rgb)

    panels_meta = []
    for idx, (x, y, w, h) in enumerate(panel_bboxes):
        panel_dir = OUTPUT_DIR / f"panel_{idx}"
        panel_dir.mkdir(parents=True, exist_ok=True)

        panel = img_rgb[y : y + h, x : x + w]
        Image.fromarray(panel).save(panel_dir / "original_panel.jpg", quality=95)

        text_mask = detect_text(panel)
        red_mask = detect_red_lines(panel) & ~text_mask
        cross_mask = detect_black_crosses(panel)
        combined = red_mask | cross_mask

        cleaned = absorb_artifacts(
            panel,
            combined,
            inpaint_radius=7,
            dilate_iters=2,
            dilate_kernel_size=3,
            method="NS",
        )
        Image.fromarray(cleaned).save(panel_dir / "cleaned.jpg", quality=95)

        seg = v4_segment(cleaned, n_layers=N_LAYERS)
        labels = seg["labels"]
        overlay = seg["overlay"]

        np.savez(panel_dir / "labels.npz", labels=labels)
        Image.fromarray(overlay).save(panel_dir / "overlay.jpg", quality=95)

        overlay_legend = _create_overlay_legend(overlay, labels)
        Image.fromarray(overlay_legend).save(panel_dir / "overlay_legend.jpg", quality=95)

        panels_meta.append(
            {
                "panel_id": idx,
                "bbox": [int(x), int(y), int(w), int(h)],
                "original_panel_path": str(panel_dir / "original_panel.jpg"),
                "cleaned_path": str(panel_dir / "cleaned.jpg"),
                "labels_path": str(panel_dir / "labels.npz"),
                "overlay_path": str(panel_dir / "overlay.jpg"),
                "overlay_legend_path": str(panel_dir / "overlay_legend.jpg"),
                "n_labels": int(len(np.unique(labels[labels >= 0]))),
                "background_label": int(_detect_background_label(labels)) if _detect_background_label(labels) is not None else None,
            }
        )

    metadata = {
        "image": str(INPUT_IMAGE),
        "output_dir": str(OUTPUT_DIR),
        "n_layers": N_LAYERS,
        "panels": panels_meta,
    }

    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, ensure_ascii=False))
