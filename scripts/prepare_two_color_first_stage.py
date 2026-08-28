#!/usr/bin/env python3
"""Prepare connected two-color material labels from a schematic cross-section."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    component_id = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return cc == component_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed-distance", type=float, default=28.0)
    args = parser.parse_args()

    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    red, green, blue = np.moveaxis(rgb, -1, 0)
    yellow_seed = (
        (red > 175) & (green > 170) & (blue < green - 25) & (red - blue > 30)
    )
    blue_seed = (
        (red > 70)
        & (red < 205)
        & (green > red + 12)
        & (blue > green + 5)
    )
    if yellow_seed.sum() < 1000 or blue_seed.sum() < 1000:
        raise ValueError("could not find enough yellow/blue color seeds")

    yellow_rgb = np.median(rgb[yellow_seed], axis=0).astype(np.uint8)
    blue_rgb = np.median(rgb[blue_seed], axis=0).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    palette_rgb = np.stack([yellow_rgb, blue_rgb])
    palette_lab = cv2.cvtColor(
        palette_rgb.reshape(1, 2, 3), cv2.COLOR_RGB2LAB
    ).reshape(2, 3).astype(np.float32)
    distances = np.linalg.norm(lab[:, :, None, :] - palette_lab[None, None, :, :], axis=3)

    confident_fill = distances.min(axis=2) <= args.seed_distance
    confident_fill = cv2.morphologyEx(
        confident_fill.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 11)),
    ) > 0
    body = largest_component(confident_fill)
    body = ndimage.binary_fill_holes(body)
    body = cv2.morphologyEx(
        body.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 7)),
    ) > 0

    labels = (np.argmin(distances, axis=2) + 1).astype(np.uint8)
    labels = ndimage.median_filter(labels, size=(3, 9)).astype(np.uint8)
    labels[~body] = 0

    reassigned = {"1": 0, "2": 0}
    for _ in range(3):
        changed = False
        for material_id in (1, 2):
            material = labels == material_id
            main = largest_component(material)
            islands = material & ~main
            count = int(np.count_nonzero(islands))
            if count:
                labels[islands] = 3 - material_id
                reassigned[str(material_id)] += count
                changed = True
        if not changed:
            break
    labels[~body] = 0

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red_lines = (
        ((hsv[..., 0] <= 10) | (hsv[..., 0] >= 170))
        & (hsv[..., 1] >= 110)
        & (hsv[..., 2] >= 130)
        & body
    )

    fill = np.full_like(rgb, 255)
    fill[labels == 1] = yellow_rgb
    fill[labels == 2] = blue_rgb
    overlay = cv2.addWeighted(rgb, 0.45, fill, 0.55, 0)
    overlay[~body] = rgb[~body]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(fill).save(args.output_dir / "first_stage_regions.png")
    Image.fromarray(overlay).save(args.output_dir / "first_stage_overlay.png")
    np.savez_compressed(
        args.output_dir / "first_stage_labels.npz",
        labels=labels,
        body_mask=body,
        red_lines=red_lines,
        palette_rgb=palette_rgb,
    )
    report = {
        "image": str(args.image),
        "shape_hw": list(labels.shape),
        "palette_rgb": {
            "yellow": yellow_rgb.tolist(),
            "blue": blue_rgb.tolist(),
        },
        "areas": {
            "body": int(body.sum()),
            "yellow": int(np.count_nonzero(labels == 1)),
            "blue": int(np.count_nonzero(labels == 2)),
            "red_lines": int(red_lines.sum()),
        },
        "reassigned_island_pixels": reassigned,
    }
    (args.output_dir / "first_stage_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
