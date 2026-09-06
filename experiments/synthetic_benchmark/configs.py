"""Configuration dataclasses for the synthetic benchmark.

All parameters are explicit so an agent (or human) can read the config
back and know exactly what was generated. Nothing is sampled from a
hidden RNG state — the RNG is created per-call so results are reproducible
given the same seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PanelConfig:
    """Per-panel generation parameters.

    Inspired by Wu et al. (Geophysics 2020) BSM. We collapse the
    3D generator to 2D and keep the four components geoseg cares about:
    layers, faults, channels, unconformities.
    """

    # Geometry
    height: int = 512
    width: int = 768
    n_layers: int = 5  # 3-8 reasonable for conceptual models

    # Layer shape
    dip_deg: float = 0.0  # 0 = horizontal, positive = slope down-right
    layer_thickness_jitter: float = 0.3  # ±30% relative thickness variation
    warp_amplitude: float = 0.0  # gentle undulation amplitude (pixels)

    # Optional structures
    has_fault: bool = False
    fault_x_frac: float = 0.6  # fault vertical line at x_frac * width
    fault_offset: int = 30  # vertical jump in pixels at the fault

    has_channel: bool = False
    channel_amplitude: float = 60.0  # how much the channel meanders vertically
    channel_thickness: int = 18
    channel_y_frac: float = 0.45  # channel centre y as fraction of height

    has_unconformity: bool = False
    unconformity_y_frac: float = 0.6  # horizontal truncation line
    unconformity_dip_below: float = 20.0  # layers below the unconformity dip
    unconformity_dip_above: float = 0.0  # layers above are flat

    # Rendering
    colormap: str = "jet"  # one of: jet, viridis, plasma, gray
    seed: int = 0  # dedicated seed for this panel (no shared RNG)

    def rng(self) -> np.random.Generator:
        """Fresh RNG so generation is deterministic per seed."""
        return np.random.default_rng(self.seed)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Whole-benchmark configuration."""

    panels: tuple[PanelConfig, ...] = field(default_factory=tuple)
    engines: tuple[str, ...] = (
        "v4_kmeans",
        "kmeans_full",
        "slic_kmeans",
        "edge_guided",
        "edge_grow",
    )
    apply_realistic_noise: bool = True
    output_dir: str = ""  # empty = caller provides tmp dir

    @property
    def n_panels(self) -> int:
        return len(self.panels)
