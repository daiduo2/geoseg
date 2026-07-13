"""E026 segmentation pipeline (thin wrapper over core + components).

Assembles the full segmentation flow: seeds → NN segmentation → overlay → components.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .core import auto_extract_seeds, segment_fixed_nn, create_vivid_overlay
from .components import build_segmentation_result


def run_panel(
    content_crop: np.ndarray,
    n_layers: int = 7,
    min_area: int = 200,
    alpha: float = 0.5,
) -> dict:
    """Run the full e026 segmentation pipeline on a single panel content crop.

    Args:
        content_crop: RGB image array (H, W, 3) of the panel content zone.
        n_layers: Number of layers to extract (default 7).
        min_area: Minimum component area in pixels (default 200).
        alpha: Overlay blending factor (default 0.5).

    Returns:
        segmentation_result dict with keys:
            "components": [...],
            "layers": [...],
            "overlay": np.ndarray,  # RGB overlay image
            "labels": np.ndarray,   # label array
    """
    seeds = auto_extract_seeds(content_crop, n_layers=n_layers)
    labels = segment_fixed_nn(content_crop, seeds)
    overlay, vivid_colors = create_vivid_overlay(content_crop, labels, alpha=alpha)
    result = build_segmentation_result(labels, content_crop, min_area=min_area)

    result["overlay"] = overlay
    result["labels"] = labels
    result["seeds"] = seeds
    result["vivid_colors"] = [c.tolist() for c in vivid_colors]

    return result


def save_result(result: dict, out_dir: Path, name: str) -> dict:
    """Save overlay and JSON to disk, returning result with overlay_path set.

    Args:
        result: Output from run_panel().
        out_dir: Directory to save files.
        name: Base filename (e.g. "pattern1").

    Returns:
        Result dict with "overlay_path" key added.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = out_dir / f"{name}_overlay.png"
    Image.fromarray(result["overlay"]).save(overlay_path)

    # Labels visualization
    max_label = int(result["labels"].max())
    scale = 255 // max_label if max_label > 0 else 1
    labels_img = (result["labels"].astype(np.uint32) * scale).clip(0, 255).astype(np.uint8)
    Image.fromarray(labels_img, mode="L").save(out_dir / f"{name}_labels.png")

    # Save JSON (exclude numpy arrays)
    json_result = {
        "components": result["components"],
        "layers": result["layers"],
        "seeds": result["seeds"],
        "vivid_colors": result["vivid_colors"],
        "overlay_path": str(overlay_path),
    }
    (out_dir / f"{name}_segmentation_result.json").write_text(
        json.dumps(json_result, ensure_ascii=False, indent=2)
    )

    result["overlay_path"] = str(overlay_path)
    return result
