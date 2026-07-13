"""CV detection module: panel bbox detection and figure classification."""

from .detect import find_panel_candidates
from .panel_detector import detect_panels
from .figure_classifier import classify
from .quality_filter import check_image_quality, filter_directory
from .colorbar_extractor import extract_colorbar

__all__ = [
    "find_panel_candidates",
    "detect_panels",
    "classify",
    "check_image_quality",
    "filter_directory",
    "extract_colorbar",
]
