"""Backward-compatible imports for pipeline panel-segmentation stages."""

from geoseg.pipeline.stages import crop_panel_for_segmentation, segment_panel_stage

__all__ = ["crop_panel_for_segmentation", "segment_panel_stage"]
