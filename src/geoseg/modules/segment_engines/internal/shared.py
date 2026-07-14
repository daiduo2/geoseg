"""Backward-compatible aggregate imports for segmentation engine internals."""

from geoseg.modules.segment_engines.internal import (
    color as _color,
    overlay as _overlay,
    preprocess as _preprocess,
    regions as _regions,
    seeds as _seeds,
)
from geoseg.modules.segment_engines.internal.color import *  # noqa: F401,F403
from geoseg.modules.segment_engines.internal.overlay import *  # noqa: F401,F403
from geoseg.modules.segment_engines.internal.preprocess import *  # noqa: F401,F403
from geoseg.modules.segment_engines.internal.regions import *  # noqa: F401,F403
from geoseg.modules.segment_engines.internal.seeds import *  # noqa: F401,F403

__all__ = [
    *_color.__all__,
    *_overlay.__all__,
    *_preprocess.__all__,
    *_regions.__all__,
    *_seeds.__all__,
]
