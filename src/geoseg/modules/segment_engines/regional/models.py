"""Data models for regional segmentation repair."""

from __future__ import annotations

from dataclasses import dataclass, field

from geoseg.core.models import RegionalAudit


@dataclass
class FusionConfig:
    """Regional fusion configuration."""

    primary_engine: str = "v4_kmeans"
    secondary_engines: list[str] = field(
        default_factory=lambda: ["edge_guided", "kmeans_full"]
    )
    max_iterations: int = 3
    seam_smooth_width: int = 3
    enable_legend: bool = True


__all__ = ["FusionConfig", "RegionalAudit"]
