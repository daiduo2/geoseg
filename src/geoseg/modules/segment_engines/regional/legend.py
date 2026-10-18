"""Overlay and legend rendering for regional repair."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geoseg.modules.segment_engines.internal.color import _distinct_colors
from geoseg.modules.segment_engines.internal.overlay import _create_overlay


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try common system fonts, fallback to default."""
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


def _draw_legend(
    overlay_rgb: np.ndarray,
    labels: np.ndarray,
    label_colors: dict[int, np.ndarray] | None = None,
    box_size: int = 12,
    font_size: int = 10,
) -> np.ndarray:
    """Draw label legend on overlay bottom-right corner.

    Returns RGB array with semi-transparent legend overlay.
    """
    h, w = overlay_rgb.shape[:2]
    unique = sorted(set(labels.flatten()) - {0})
    n = len(unique)
    if n == 0:
        return overlay_rgb

    rgba = np.dstack([overlay_rgb, np.full((h, w), 255, dtype=np.uint8)])
    img = Image.fromarray(rgba, mode="RGBA")

    item_h = box_size + 4
    pad = 6
    leg_h = n * item_h + pad * 2
    leg_w = box_size + 28 + pad * 2

    lx = max(0, w - leg_w - 8)
    ly = max(0, h - leg_h - 8)

    bg = Image.new("RGBA", (leg_w, leg_h), (0, 0, 0, 180))
    img.paste(bg, (lx, ly), bg)

    draw = ImageDraw.Draw(img)
    base_colors = _distinct_colors(max(unique) + 1)
    font = _load_font(font_size)

    for i, lbl in enumerate(unique):
        y = ly + pad + i * item_h
        if label_colors is not None and lbl in label_colors:
            color = tuple(int(c) for c in label_colors[lbl]) + (255,)
        else:
            color = tuple(int(c) for c in base_colors[lbl]) + (255,)
        draw.rectangle(
            [lx + pad, y, lx + pad + box_size, y + box_size],
            fill=color,
        )
        draw.text(
            (lx + pad + box_size + 4, y - 1),
            str(lbl),
            fill=(255, 255, 255, 255),
            font=font,
        )

    return np.array(img.convert("RGB"))


def draw_legend(
    overlay_rgb: np.ndarray,
    labels: np.ndarray,
    label_colors: dict[int, np.ndarray] | None = None,
    box_size: int = 12,
    font_size: int = 10,
) -> np.ndarray:
    """Draw label legend on overlay bottom-right corner."""
    return _draw_legend(
        overlay_rgb,
        labels,
        label_colors=label_colors,
        box_size=box_size,
        font_size=font_size,
    )


def generate_overlay_with_legend(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    seeds_rgb: np.ndarray | None = None,
    alpha: float = 0.65,
) -> np.ndarray:
    """Create overlay with bottom-right label legend for agent audit."""
    overlay = _create_overlay(panel_rgb, labels, seeds_rgb, alpha=alpha)
    return draw_legend(overlay, labels)


__all__ = [
    "_draw_legend",
    "draw_legend",
    "generate_overlay_with_legend",
]
