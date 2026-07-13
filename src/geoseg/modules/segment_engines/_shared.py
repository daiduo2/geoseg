"""Backward-compatible shim for segmentation engine internals.

New code inside the engine family should import from
``geoseg.modules.segment_engines.internal.shared``. Product code outside the
engine family should use stable facades such as ``geoseg.core.image_ops``.
"""

from geoseg.modules.segment_engines.internal.shared import *  # noqa: F401,F403

