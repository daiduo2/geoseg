"""Data models for regional segmentation repair."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RegionalAudit:
    """Agent regional audit result."""

    frozen_labels: list[int] = field(default_factory=list)
    retry_labels: list[int] = field(default_factory=list)
    notes: str = ""
    iteration: int = 0
    repair_strategy: str = "regional_fusion"
    secondary_engine: str = ""
    local_fixes: list[dict] = field(default_factory=list)


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
