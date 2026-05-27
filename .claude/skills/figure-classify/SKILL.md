---
name: figure-classify
description: >
  Classify a geophysics figure image to determine if it is a valid
  velocity_model target for segmentation. Use when the user uploads or
  references a figure image and you need to decide whether it should proceed
  to segmentation or be skipped. Triggers: "classify this figure",
  "is this a velocity model", "should we segment this", "figure type"
argument-hint: <image_path>
allowed-tools: Bash, Read, Write
---

# figure-classify

Semantic classification of geophysics figure images. Determines whether a
figure is a valid velocity_model segmentation target.

## Classification Criteria

Evaluate the image against these categories. ONLY `velocity_model` and
`geological_cross_section` are valid segmentation targets.

### Valid Targets (proceed)

- **velocity_model**: Conceptual velocity model with colored regions/layers
  representing seismic velocities (Vp, Vs). Colors represent VELOCITY VALUES.
  Each region is relatively uniform with smooth or gently curved boundaries.
  Often has a colorbar labeled with velocity units.
- **geological_cross_section**: Stratigraphic layers, faults, or structural
  boundaries. Hand-drawn or schematic.

### Must Skip (skip)

- **shot_gather**: Dense vertical wiggly traces (seismogram). NOT colored regions.
- **seismic_reflection_section**: Dense horizontal reflection stripes/events.
  Colors represent AMPLITUDE, not velocity. Even with depth axis and color scale,
  the presence of dense parallel reflection events means it is a seismic section.
- **waveform_plot**: One or a few waveform traces.
- **statistical_plot**: Scatter plots, variograms, convergence curves.
- **equation**: Mathematical formulas.
- **data_table**: Tabular data.
- **tomography_map**: Data-driven map with geographic outlines.
- **flowchart**: Workflow diagrams.

### Critical Distinctions

- **Velocity models** show UNIFORM colored regions (roughly constant color per
  region) with SMOOTH boundaries. Seismic sections show DENSE STRIPES.
- Shot gathers have dense vertical wiggly lines, not colored regions.
- Even if a figure has a depth axis and color scale, dense parallel reflection
  events mean it is a seismic section, NOT a velocity model.

## Workflow

1. **Read** the image file at the given path.
2. Analyze visual elements: axes, colorbars, panel layout, labels, data
   representations, lines, colors, text.
3. Evaluate EVERY category from the list above, explaining why each does or
   does not apply.
4. Choose the single strongest piece of evidence supporting your final
   classification.
5. Identify any conflicting evidence and explain why you rejected it.
6. Set `segmentation_recommendation`:
   - `"proceed"` for velocity_model / geological_cross_section
   - `"skip"` for observational data (shot_gather, seismic_reflection_section,
     waveform_plot, statistical_plot, data_table, tomography_map)
   - `"manual_review"` for ambiguous cases (equation, flowchart, other)

## Output Format

Write the classification result as JSON to the audit path:

```
runs/audit/{timestamp}_figure_classification.json
```

JSON structure (matching `geoseg/modules/vlm_client/prompts.py:FigureClassification`):

```json
{
  "figure_type": "velocity_model",
  "confidence": 0.92,
  "reason": "Colored layers with smooth boundaries and velocity colorbar",
  "visual_features": "...",
  "category_checklist": [...],
  "primary_evidence": "...",
  "conflicting_evidence": "...",
  "segmentation_recommendation": "proceed"
}
```

Also print a brief summary to stdout:
```
Figure: {path}
Type: {figure_type}
Confidence: {confidence}
Recommendation: {segmentation_recommendation}
```

## Constraints

- Confidence must be a float between 0.0 and 1.0.
- If unclear or not a velocity model, choose "other", set confidence < 0.7,
  and set recommendation to "manual_review".
- Do NOT provide bounding box coordinates or pixel positions.
- When in doubt, be conservative: prefer "skip" over false positive.
