"""Literature-text association stub.

Advisor guidance (2026-05-14):
    Future feature: agent reads PDF figure captions to extract physical
    parameters (Vp values, depth ranges, coordinate extents) and auto-fills
    the properties template so the user does not have to look up numbers
    manually.

Phase 2 scope: stub only.  Implementation needs:
    - PDF text extraction (PyMuPDF or pdfplumber)
    - Figure-caption pairing heuristic (search for "Figure N" near caption)
    - Regex-based value extraction (Vp, Vs, depth, distance, lat/lon)
    - Mapping extracted values to the ``properties.py`` template fields
"""

from __future__ import annotations

from pathlib import Path


def extract_caption_text(pdf_path: str | Path, figure_number: str | int) -> str:
    """Return the caption text for a given figure number.

    Stub: always returns empty string.  Future implementation will scan
    the PDF for "Figure {n}" and extract the following paragraph.
    """
    return ""


def parse_physical_params(caption_text: str) -> dict:
    """Extract Vp, Vs, depth, distance ranges from caption text.

    Stub: returns empty dict.  Future implementation will use regexes like:
        - Vp?\\s*=\\s*(\\d+(?:\\.\\d+)?)\\s*km/s
        - depth\\s+range\\s+(\\d+)-(\\d+)\\s*km
    """
    return {}


def suggest_properties(caption_text: str, color_names: list[str]) -> dict[str, dict]:
    """Suggest a ``color_name -> {vp, vs, rho}`` map from caption text.

    Stub: returns empty dict (falls back to DEFAULT_PROPERTIES).
    """
    return {}


__all__ = ["extract_caption_text", "parse_physical_params", "suggest_properties"]
