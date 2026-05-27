"""geo-segment library.

Submodules:
    crop        — panel bbox refinement
    clean       — VLM + CV noise mask composition
    segment     — dual-path color segmentation (jet-vivid / pastel-faded)
    polygon     — label map → GeoJSON polygons
    properties  — color zone → Vp/Vs/Rho mapping
    mesh        — polygon rasterization → regular grid
    specfem     — SPECFEM2D model file writers
    vlm         — Prompt templates + JSON loader (vision done in-session)
    interactive — human-in-the-loop review hook (Phase 1 stub)
    cli         — orchestration entry point
"""

from . import clean, crop, interactive, literature, mesh, polygon, properties, segment, specfem, vlm

__all__ = ["clean", "crop", "interactive", "literature", "mesh", "polygon", "properties", "segment", "specfem", "vlm"]
