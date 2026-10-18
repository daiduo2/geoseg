"""Backward-compatible aggregate imports for segmentation engine internals.

The preprocessing filters live in :mod:`geoseg.core.image_ops`; this module
re-exports them so existing engine imports keep working without maintaining a
second copy.
"""

from geoseg.core.image_ops import (
    adaptive_blur,
    estimate_noise_level,
    row_median_filter,
)
from geoseg.modules.segment_engines.internal import (
    color as _color,
    overlay as _overlay,
    regions as _regions,
    seeds as _seeds,
)
from geoseg.modules.segment_engines.internal.color import *  # noqa: F401,F403
from geoseg.modules.segment_engines.internal.overlay import *  # noqa: F401,F403
from geoseg.modules.segment_engines.internal.regions import *  # noqa: F401,F403
from geoseg.modules.segment_engines.internal.seeds import *  # noqa: F401,F403

__all__ = [
    *_color.__all__,
    *_overlay.__all__,
    "adaptive_blur",
    "estimate_noise_level",
    "row_median_filter",
    *_regions.__all__,
    *_seeds.__all__,
]
