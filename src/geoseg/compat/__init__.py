"""Historical compatibility entry points.

These modules preserve the import and CLI paths that existed before the v2
layout: the Flask/FastAPI server, the PySide6 GUI, the controller facade, the
batch processor facade, and the old pipeline contracts module. Product code
must not depend on them; new orchestration lives in ``geoseg.pipeline`` and new
contracts in ``geoseg.core.models``.
"""
