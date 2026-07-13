"""Image generation and loading utilities for FH/SLIC experiment."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _synthetic_panel_with_text(
    h: int = 400,
    w: int = 600,
    n_layers: int = 5,
    text_density: str = "medium",
) -> np.ndarray:
    """Generate a synthetic jet-colormap panel with overlaid text."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    band_h = h // n_layers
    colors = [
        [0, 0, 255],
        [0, 255, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 0],
        [128, 0, 0],
        [255, 0, 255],
    ]
    for i in range(n_layers):
        y0 = i * band_h
        y1 = (i + 1) * band_h if i < n_layers - 1 else h
        img[y0:y1] = colors[i % len(colors)]

    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    font = None
    for font_path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, 16)
                break
            except Exception:
                pass

    labels = ["Vp=1.5", "Vs=2.8", "Rho=2.1", "Depth(km)", "Moho", "Sed.", "Basement"]
    densities = {"low": 8, "medium": 20, "high": 40}
    n_texts = densities.get(text_density, 20)

    rng = np.random.default_rng(42)
    for _ in range(n_texts):
        x = int(rng.integers(20, w - 80))
        y = int(rng.integers(20, h - 20))
        text = rng.choice(labels)
        color = (0, 0, 0) if rng.random() > 0.3 else (255, 255, 255)
        draw.text((x, y), text, fill=color, font=font)

    return np.array(pil_img)


def _load_real_image(project_root: Path) -> np.ndarray | None:
    """Load a real test image if available."""
    candidates = [
        Path("tests/fixtures/ph01/ph01_page8_300dpi.png"),
        Path("docs/assets/example1_original.png"),
        Path("docs/assets/example2_original.png"),
        Path("docs/assets/example3_original.png"),
    ]
    for c in candidates:
        full = project_root / c
        if full.exists():
            return np.array(Image.open(full).convert("RGB"))
    return None
