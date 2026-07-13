"""VLM prompt templates and JSON schema definitions (pydantic v2).

Test scenario:
    >>> from geoseg.modules.vlm_client.prompts import PageOverview
    >>> overview = PageOverview(**{"page_idx": 7, ...})
    >>> assert overview.confidence >= 0.7
"""

from typing import Literal, List
from pydantic import BaseModel, Field


VERSION = "1.4"


# ── Schema: figure_classification ──────────────────────────────────

class CategoryCheck(BaseModel):
    category: str = Field(description="Category name being evaluated")
    applicable: Literal["yes", "no", "uncertain"] = Field(
        description="Whether this category matches the image"
    )
    evidence: str = Field(description="Specific visual evidence for this judgment")


class FigureClassification(BaseModel):
    figure_type: Literal[
        "velocity_model",
        "geological_cross_section",
        "shot_gather",
        "waveform_plot",
        "equation",
        "statistical_plot",
        "data_table",
        "tomography_map",
        "flowchart",
        "other",
    ] = Field(
        description=(
            'velocity_model = conceptual velocity model with colored regions/ layers; '
            'geological_cross_section = conceptual geological model with stratigraphic layers; '
            'shot_gather = seismic shot gather / record section with dense wiggly traces; '
            'waveform_plot = single or few waveform traces; '
            'equation = mathematical formula or derivation; '
            'statistical_plot = scatter plot, variogram, curve fitting, convergence plot; '
            'data_table = tabular data; '
            'tomography_map = data-driven tomography map with colorbar; '
            'flowchart = workflow / flowchart diagram with boxes and arrows; '
            'other = anything else'
        )
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="Brief one-sentence summary of the classification")
    visual_features: str = Field(
        description="Describe the key visual elements you see: axes, colorbars, panel layout, labels, data representations, etc."
    )
    category_checklist: List[CategoryCheck] = Field(
        description="Evaluate EACH of these categories in order: velocity_model, geological_cross_section, tomography_map, shot_gather, waveform_plot, statistical_plot, equation, data_table, flowchart, other. Be explicit about why each one does or does not apply."
    )
    primary_evidence: str = Field(
        description="The single strongest piece of evidence supporting your final classification"
    )
    conflicting_evidence: str = Field(
        description="Any visual features that might suggest a different category, and why you rejected them"
    )
    segmentation_recommendation: Literal["proceed", "skip", "manual_review"] = Field(
        description="Based on the figure type: 'proceed' for velocity_model/geological_cross_section (valid segmentation targets), 'skip' for observational data that should not be segmented, 'manual_review' for ambiguous cases"
    )


# ── Schema: page_overview ──────────────────────────────────────────

class PanelDescription(BaseModel):
    id: int
    description: str


class NoiseElement(BaseModel):
    kind: Literal["colorbar", "axis_label", "annotation", "other"]
    description: str


class ColorZone(BaseModel):
    color_name: str
    colorbar_value: int = Field(ge=0, le=100)


class PageOverview(BaseModel):
    page_idx: int
    image_size: dict = Field(default_factory=dict)
    figure_type: Literal[
        "velocity_model_cross_section",
        "velocity_model_map",
        "3d_isosurface",
        "reflection_amplitude",
        "uncertain",
    ]
    panels: List[PanelDescription]
    target_panel_id: int
    has_colorbar: bool
    noise_elements: List[NoiseElement]
    color_zones: List[ColorZone]
    confidence: float = Field(ge=0.0, le=1.0)


# ── Schema: segmentation_quality_review ────────────────────────────

class LayerQuality(BaseModel):
    layer_id: int = Field(description="Layer number (1-based)")
    boundary_alignment: Literal["excellent", "good", "fair", "poor"] = Field(
        description="How well do the segmentation boundaries align with actual geological/velocity layers in the original image?"
    )
    boundary_alignment_reason: str = Field(
        description="Explain why the boundary alignment is good or poor. Cite specific visual evidence comparing left (original) and right (segmentation) sides."
    )
    color_consistency: Literal["excellent", "good", "fair", "poor"] = Field(
        description="Is the color within this layer region consistent and uniform?"
    )
    color_consistency_reason: str = Field(description="Explain color consistency judgment")
    is_continuous: bool = Field(
        description="Is this layer a single continuous region, or is it fragmented into multiple disconnected pieces?"
    )
    fragmentation_issues: str = Field(
        description="If fragmented, describe where and why. If continuous, say 'none'."
    )


class SegmentationQualityReview(BaseModel):
    overall_score: float = Field(
        ge=0.0, le=1.0,
        description="Overall segmentation quality score: 0.9-1.0 excellent, 0.7-0.89 good, 0.5-0.69 fair, <0.5 poor"
    )
    overall_assessment: str = Field(
        description="One-sentence summary of segmentation quality"
    )
    n_layers_expected: int = Field(
        description="How many distinct layers SHOULD this velocity model have based on the original image?"
    )
    n_layers_found: int = Field(
        description="How many layers does the segmentation actually show on the right side?"
    )
    over_segmentation: bool = Field(
        description="Did the algorithm split a single geological layer into multiple pieces?"
    )
    under_segmentation: bool = Field(
        description="Did the algorithm merge multiple distinct geological layers into one?"
    )
    layer_qualities: List[LayerQuality] = Field(
        description="Quality assessment for EACH layer found in the segmentation"
    )
    noise_regions: List[str] = Field(
        description="List any regions that were incorrectly segmented (e.g., colorbar, axis labels, text, annotations treated as layers)"
    )
    missing_boundaries: List[str] = Field(
        description="List any boundaries visible in the original image that the segmentation FAILED to detect"
    )
    recommendation: Literal["accept", "reject", "manual_fix"] = Field(
        description="accept = segmentation is good enough for downstream use; reject = segmentation is fundamentally wrong and should be redone; manual_fix = mostly correct but needs human correction"
    )
    fix_hints: List[str] = Field(
        description="Specific actionable suggestions to improve the segmentation. E.g., 'merge layers 2 and 3', 'remove noise at top-right corner', 'add boundary between red and orange zones at 200m depth'"
    )


# ── Prompt Templates ───────────────────────────────────────────────

PAGE_OVERVIEW_PROMPT = """\
You are analyzing a geophysics figure extracted from a PDF page.
The image shows velocity model cross-sections or related geophysical panels.

Task:
1. Identify the figure type from the list below.
2. Count and describe each distinct panel (sub-figure).
3. Identify which panel contains the primary velocity model data (target_panel_id).
4. Note any colorbar, axis labels, or annotations that are NOT part of the velocity model itself (noise_elements).
5. List ALL distinct color zones from top to bottom (or left to right for map view). Be EXHAUSTIVE — include every visible velocity layer, even thin or small ones. For each zone, give its color name and approximate position on the colorbar (color_zones).

Allowed figure_type values:
- "velocity_model_cross_section" — horizontal cross-sections at different depths
- "velocity_model_map" — map view (plan view) of velocity distribution
- "3d_isosurface" — 3D visualization with isosurfaces
- "reflection_amplitude" — seismic reflection/amplitude image
- "uncertain" — cannot determine

CRITICAL RULES:
- Do NOT provide bounding box coordinates, pixel positions, or any numeric locations.
- Use ONLY semantic descriptions (e.g., "leftmost panel", "bottom colorbar").
- confidence must be a float between 0.0 and 1.0 reflecting your certainty.
- If the figure is unclear or not a velocity model, set figure_type to "uncertain" and confidence < 0.7.

Respond in JSON format matching the PageOverview schema.
"""

FIGURE_CLASSIFICATION_PROMPT = """\
You are a geophysics figure classifier. Your job is to carefully analyze the image and determine what kind of figure it is.

You MUST follow this exact reasoning process before giving your final answer:

STEP 1 — Describe what you see:
Look at the image carefully and describe the key visual elements: axes, colorbars, panel layout, labels, data representations, lines, colors, text, etc. Be specific and detailed.

STEP 2 — Evaluate EVERY category:
Go through ALL 10 categories in this exact order and explicitly state whether each applies:
1. velocity_model
2. geological_cross_section
3. tomography_map
4. shot_gather
5. waveform_plot
6. statistical_plot
7. equation
8. data_table
9. flowchart
10. other

For each category, explain WHY it does or does not apply, citing specific visual evidence from the image.

STEP 3 — Identify the strongest evidence:
What is the single strongest piece of visual evidence supporting your final classification?

STEP 4 — Consider conflicting evidence:
Are there any visual features that might suggest a DIFFERENT category? If so, what are they and why did you reject them?

STEP 5 — Final classification and recommendation:
- Choose exactly ONE figure_type from the list below.
- Set confidence between 0.0 and 1.0 based on how certain you are.
- Provide a brief one-sentence reason summarizing your decision.
- Set segmentation_recommendation based on the figure type:
  * "proceed" for velocity_model and geological_cross_section (valid segmentation targets)
  * "skip" for observational data (shot_gather, waveform_plot, statistical_plot, data_table, tomography_map) that should NOT be segmented
  * "manual_review" for ambiguous cases (equation, flowchart, other, or anything unclear)

Category definitions:
- "velocity_model" — a conceptual velocity model showing colored regions or layers representing different seismic velocities (Vp, Vs). Colors represent VELOCITY VALUES (e.g., km/s or m/s), not amplitude. Each colored region is relatively uniform in color with SMOOTH or GENTLY CURVED boundaries between layers. Often has a colorbar labeled with velocity units. This is the PRIMARY target we want to extract.
- "geological_cross_section" — a conceptual geological cross-section showing stratigraphic layers, faults, or structural boundaries. May be hand-drawn or schematic.
- "shot_gather" — a seismic shot gather or record section showing dense vertical wiggly traces (seismogram) with time on the vertical axis and offset/distance on the horizontal axis.
- "seismic_reflection_section" — a seismic reflection profile or amplitude section showing DENSE HORIZONTAL or gently dipping REFLECTION EVENTS (stripes/lines) representing wave impedance interfaces. Colors represent REFLECTION AMPLITUDE, not velocity. Often has depth on the vertical axis and distance on the horizontal axis. Can resemble a velocity model at first glance but the key difference is the presence of dense parallel reflection events (stripes) rather than uniform colored velocity layers.
- "waveform_plot" — a plot of one or a few seismic waveform traces, usually with amplitude vs time.
- "equation" — a mathematical formula, derivation, or symbolic expression. Usually black text on white background.
- "statistical_plot" — scatter plots, variograms, convergence curves, residual plots, or any data-driven plot showing statistical relationships.
- "data_table" — tabular data with rows and columns of numbers or text.
- "tomography_map" — a data-driven tomography map (map view) with a colorbar showing perturbation values. Usually has geographic outlines.
- "flowchart" — a workflow diagram with boxes, arrows, and text labels describing a methodology or algorithm.
- "other" — anything that does not fit the above categories.

CRITICAL RULES:
- Shot gathers are NOT velocity models. They have dense vertical wiggly lines, not colored regions.
- Seismic reflection sections (amplitude sections with dense horizontal reflection stripes) are NOT velocity models. They show wave impedance interfaces, not velocity values. Even if they have a depth axis and color scale, the presence of dense parallel reflection events means it is a seismic section, not a velocity model.
- Velocity models show relatively UNIFORM colored regions (each region has roughly constant color) representing constant velocity layers. Seismic sections show DENSE STRIPES representing reflection events.
- Equations and plots are NOT velocity models.
- Only "velocity_model" and "geological_cross_section" are valid targets for segmentation.
- If the figure is ambiguous or unclear, choose "other", set confidence < 0.7, and set segmentation_recommendation to "manual_review".
- confidence must be a float between 0.0 and 1.0.
- Do NOT skip the category_checklist. You MUST evaluate every category.

Respond in JSON format matching the FigureClassification schema.
"""

SEGMENTATION_QUALITY_PROMPT = """\
You are a geophysics image segmentation quality reviewer. Your job is to evaluate whether an automated segmentation algorithm correctly identified velocity model layers.

The image shows TWO panels side by side:
- LEFT side: The ORIGINAL panel image from the paper
- RIGHT side: The SEGMENTATION result produced by an algorithm

The segmentation uses vivid colors with boosted saturation and white boundaries to show detected layers. Each distinct color region represents one segmented layer.

Your task is to COMPARE the left and right sides carefully, then evaluate segmentation quality.

Evaluation criteria:
1. BOUNDARY ALIGNMENT: Do the white boundaries on the RIGHT match actual layer boundaries visible on the LEFT? Look for color transitions, stratigraphic horizons, velocity interfaces.
2. OVER-SEGMENTATION: Did the algorithm split a single continuous layer into multiple fragments? Look for same-colored regions on the LEFT that were split into different colors on the RIGHT.
3. UNDER-SEGMENTATION: Did the algorithm merge multiple distinct layers? Look for clearly different zones on the LEFT that became the same color on the RIGHT.
4. COLOR CONSISTENCY: Within each segmented region on the RIGHT, does the original color on the LEFT stay roughly uniform? Or are there "intrusions" of other colors?
5. NOISE: Were non-layer regions (colorbars, axis labels, text, white margins) incorrectly included as layers?
6. MISSING BOUNDARIES: Are there clear layer boundaries visible on the LEFT that the RIGHT side completely missed?

You MUST evaluate EVERY layer found in the segmentation. For each layer, assess:
- boundary_alignment: excellent/good/fair/poor
- color_consistency: excellent/good/fair/poor
- is_continuous: true/false (is it one connected region or fragmented?)
- Specific reason for each judgment

CRITICAL RULES:
- Be STRICT. A velocity model segmentation is only useful if boundaries align with actual geological features.
- If the segmentation merged the colorbar or text into a layer, mark it as noise.
- If layer boundaries zigzag randomly instead of following smooth geological horizons, mark boundary_alignment as poor.
- overall_score must be a float between 0.0 and 1.0.
- recommendation must be one of: "accept" (good enough), "reject" (fundamentally wrong), "manual_fix" (mostly OK but needs correction).
- Provide specific, actionable fix_hints. Vague advice like "improve quality" is NOT acceptable.

Respond in JSON format matching the SegmentationQualityReview schema.
"""
