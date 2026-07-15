"""Legacy import path; implementation lives in ``segment_engines.compat``.

New code inside the engine family should import from
``geoseg.modules.segment_engines.internal.shared``. Product code outside the
engine family should use stable facades such as ``geoseg.core.image_ops``.
"""

from geoseg.modules.segment_engines.compat.shared import *  # noqa: F401,F403
