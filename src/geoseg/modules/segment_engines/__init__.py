"""Stable entry points for the geoseg segmentation engine family.

Concrete engine modules remain importable as submodules, for example
``geoseg.modules.segment_engines.v4_kmeans``. Package-level exports are kept
small so product code depends on routing and execution contracts instead of
individual implementation details.
"""

from geoseg.modules.segment_engines.policy import select_engine
from geoseg.modules.segment_engines.registry import (
    ENGINE_REGISTRY,
    EngineSpec,
    get_engine_spec,
    list_engines,
)
from geoseg.modules.segment_engines.retry import (
    RETRY_CHAIN,
    count_foreground_labels,
    retry_undersegmentation,
)
from geoseg.modules.segment_engines.router import route_and_segment
from geoseg.modules.segment_engines.runner import run_engine

__all__ = [
    "ENGINE_REGISTRY",
    "RETRY_CHAIN",
    "EngineSpec",
    "count_foreground_labels",
    "get_engine_spec",
    "list_engines",
    "retry_undersegmentation",
    "route_and_segment",
    "run_engine",
    "select_engine",
]
