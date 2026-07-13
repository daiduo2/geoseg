"""Segmentation engine family for geoseg v2.

Engines:
- e027_slic_graphcut: SLIC + Graph Cut (ICM) for conceptual model panels
- v4_kmeans: Dual-path K-means (jet_vivid / colorbar_guided / pastel_faded)
- slic_kmeans: SLIC superpixel + K-Means (text-robust, low fragments)
- kmeans_full: Global K-means in LAB space with VLM seeds
- edge_guided: Canny edge + selective snap K-means
- edge_grow: Dijkstra region growing with edge barrier
- ensemble: Consistency-weighted voting across multiple algorithms
"""

from geoseg.modules.segment_engines.e027_slic_graphcut import segment as e027_segment
from geoseg.modules.segment_engines.v4_kmeans import segment as v4_kmeans_segment
from geoseg.modules.segment_engines.slic_kmeans import segment as slic_kmeans_segment
from geoseg.modules.segment_engines.kmeans_full import segment as kmeans_full_segment
from geoseg.modules.segment_engines.edge_guided import segment as edge_guided_segment
from geoseg.modules.segment_engines.edge_grow import segment as edge_grow_segment
from geoseg.modules.segment_engines.ensemble import segment as ensemble_segment
from geoseg.modules.segment_engines.grayscale import segment as grayscale_segment
from geoseg.modules.segment_engines.horizon_refinement import segment as horizon_refinement_segment
from geoseg.modules.segment_engines.tubular_structure import segment as tubular_segment
from geoseg.modules.segment_engines.full_pipeline import process_figure
from geoseg.modules.segment_engines.regional_fusion import (
    regional_segment,
    fuse_with_freeze,
    generate_overlay_with_legend,
    RegionalAudit,
    FusionConfig,
)

__all__ = [
    "e027_segment",
    "v4_kmeans_segment",
    "slic_kmeans_segment",
    "kmeans_full_segment",
    "edge_guided_segment",
    "edge_grow_segment",
    "ensemble_segment",
    "grayscale_segment",
    "horizon_refinement_segment",
    "tubular_segment",
    "process_figure",
    "regional_segment",
    "fuse_with_freeze",
    "generate_overlay_with_legend",
    "RegionalAudit",
    "FusionConfig",
]
