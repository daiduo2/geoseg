"""v4 K-Means segmentation engine paths."""

from geoseg.modules.segment_engines.v4.colorbar_guided import segment_colorbar_guided
from geoseg.modules.segment_engines.v4.jet_vivid import segment_jet_vivid
from geoseg.modules.segment_engines.v4.pastel import segment_pastel_faded
from geoseg.modules.segment_engines.v4.paths import JET_VIVID_RATIO, segment

__all__ = [
    "JET_VIVID_RATIO",
    "segment",
    "segment_colorbar_guided",
    "segment_jet_vivid",
    "segment_pastel_faded",
]
