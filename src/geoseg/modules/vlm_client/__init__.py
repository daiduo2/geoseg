"""VLM client module — prompt assembly, JSON validation, and audit."""

from .client import (
    review_page_overview,
    classify_figure,
    quality_review,
    parse_response,
)
from .prompts import (
    PageOverviewSchema,
    FigureClassificationSchema,
    VERSION,
)

__all__ = [
    "review_page_overview",
    "classify_figure",
    "quality_review",
    "parse_response",
    "PageOverviewSchema",
    "FigureClassificationSchema",
    "VERSION",
]
