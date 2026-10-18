"""Backward-compatible controller entry points.

Deprecated: new code should import ``run_pipeline`` from
:mod:`geoseg.pipeline.run` (or :mod:`geoseg.api`) and
``run_post_process_and_export`` from :mod:`geoseg.pipeline.export`.
"""

from __future__ import annotations

from geoseg.pipeline.export import run_post_process_and_export
from geoseg.pipeline.run import run_pipeline

__all__ = ["run_pipeline", "run_post_process_and_export"]
