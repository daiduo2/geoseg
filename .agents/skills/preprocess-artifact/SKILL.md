# preprocess-artifact

Absorb annotation artifacts (red fault lines, black earthquake crosses, text) into the panel background before segmentation, then run a colorbar-guided segmentation comparison.

## When to Use

- The input figure contains red fault traces (YBF, GJF, WMF), black earthquake location crosses, or other non-geological marks that would otherwise become spurious segmentation labels.
- You need a clean baseline for `sandbox-segment` or `geo-segment`.
- You want a reproducible, parameter-driven preprocessing step instead of one-off experimental scripts.

## When NOT to Use

- Do NOT use this skill as a replacement for visual review. It performs artifact detection and inpainting; segmentation quality still requires agent visual inspection.
- Do NOT use it for figures that have no annotation artifacts — run `sandbox-segment` directly instead.

## Workflow

```
Agent visually inspects the original figure
    ↓
Run geoseg.cli.preprocess_artifact to detect/absorb artifacts
    ↓
Inspect 03_cleaned.jpg; if residuals remain, iterate detection parameters
    ↓
Run per-panel segmentation with --per-panel (generates overlay_legend per panel)
    ↓
Agent visually reviews each panel overlay and decides artifact labels
    ↓
Re-run with --artifact-labels '{"panel_id": [label_id, ...]}' to merge them
    ↓
Hand off cleaned image / labels to sandbox-segment or geo-segment
```

For figures where the whole-image colorbar-guided segmentation collapses (common
with stacked low-contrast panels), use `--per-panel` with the `v4_kmeans`
engine.  The merge step is driven by the agent reading the per-panel
`overlay_legend.jpg`, not by hard-coded thresholds.

## CLI Entry Point

```bash
# Whole-image comparison (legacy colorbar-guided mode)
PYTHONPATH=src python -m geoseg.cli.preprocess_artifact \
  --image <path> \
  --output-dir <dir> \
  --colorbar-roi x,y,w,h \
  --n-layers 5 \
  --merge-max-brightness 80

# Per-panel artifact absorption + visual-audit driven merge
PYTHONPATH=src python -m geoseg.cli.preprocess_artifact \
  --image <path> \
  --output-dir <dir> \
  --per-panel \
  --artifact-labels '{"0": [4], "1": [4], "3": [4]}' \
  --red-params '{"frangi_threshold": 0.015, "angle_ranges": [[15,75],[70,110],[105,165]]}' \
  --cross-params '{"max_gray": 160, "min_diff": 2}'
```

All tunables can also be provided via `--config <json>`; CLI flags override the config file.

## Key Parameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--colorbar-roi` | None | **Required from visual review.** Bounding box of the colorbar in the original image. No auto-detection scoring is used. |
| `--n-layers` | 5 | Number of colorbar seed colors / target layers. |
| `--merge-max-brightness` | 80 | Merge small/dark artifact labels whose mean brightness is below this value into their most common neighbor. |
| `--merge-min-area-frac` | 0.001 | Merge labels smaller than this fraction of the image. |
| `--inpaint-radius` | 7 | Radius passed to `cv2.inpaint` for artifact absorption. |
| `--inpaint-dilate-iters` | 2 | Dilation iterations applied to the artifact mask before inpainting. |
| `--skip-red` | False | Disable red-line detection. |
| `--skip-crosses` | False | Disable black-cross detection. |
| `--per-panel` | False | Run segmentation per panel and assemble a full-image overlay. |
| `--artifact-labels` | `{}` | JSON dict mapping `panel_id` → list of artifact label IDs to merge. Used after visual audit of per-panel overlays. |
| `--red-params` | `{}` | JSON dict passed to `detect_red_lines` (e.g., `{"frangi_threshold": 0.02}`). |
| `--cross-params` | `{}` | JSON dict passed to `detect_black_crosses` (e.g., `{"max_gray": 120}`). |
| `--text-params` | `{}` | JSON dict passed to `detect_text`. |

## Visual Review Rules

1. **Colorbar ROI must come from visual understanding.** Never use hard-coded scoring rules or heuristics to "find" the colorbar. Read the image, locate the colorbar visually, and pass its bbox as `--colorbar-roi`.
2. **No binary PASS/FAIL.** Inspect the outputs and describe what still needs repair (e.g., "red streak remains in panel 3", "black cross cluster at (x, y) not merged").
3. **Iterate parameters, not scripts.** If residuals remain, adjust `--cross-params`, `--red-params`, `--merge-max-brightness`, `--merge-min-area-frac`, or `--inpaint-dilate-iters` and re-run the same CLI. Do not create a new one-off script.
4. **Agent-native only.** All decisions (ROI, quality, parameter tuning) are made by the agent reading the images. Do not spawn Python code that calls a VLM service or CLI subprocess to "score" regions.

## Limitations

- `colorbar_guided` segmentation is only meaningful when the panel has visible color variation along the colorbar. For low-contrast or single-dominant-color panels it may collapse to one effective label; in that case `merge_artifact_labels` has nothing to merge and `sandbox-segment` should be used instead.
- Black earthquake crosses are small and can be missed when they sit on dark geological layers. Increase detection coverage by raising `max_gray` / lowering `min_diff` in `--cross-params`, then visually verify that no real layers are absorbed.

## Output Files

| File | Purpose |
|------|---------|
| `01_original.jpg` | Input image, re-saved. |
| `02_combined_mask_full.jpg` | Detected artifact mask overlay. |
| `03_cleaned.jpg` | Image after inpainting. |
| `04_seg_original_overlay.jpg` | Baseline segmentation on the original image. |
| `05_seg_cleaned_overlay.jpg` | Baseline segmentation on the cleaned image. |
| `06_seg_difference.jpg` | Pixels where baseline segmentation changed. |
| `07_seg_colorbar_guided_cleaned.jpg` | Colorbar-guided segmentation on the cleaned image. |
| `08_seg_colorbar_guided_merged.jpg` | Same, after merging small/dark artifact labels. |
| `09_per_panel_overlay.jpg` | Assembled per-panel segmentation overlay (cleaned background). |
| `09_per_panel_labels.npz` | Assembled per-panel label map. |
| `panels/panel_{i}/overlay_legend.jpg` | Per-panel overlay with label legend for visual audit. |
| `panels/panel_{i}/labels.npz` | Per-panel label map. |

## Integration

- Use `03_cleaned.jpg` as the input to `sandbox-segment` when the figure needs engine selection.
- Use `panels/panel_{i}/labels.npz` and `09_per_panel_labels.npz` as a per-panel zoning reference.
- Persist the final config alongside the outputs so the run is reproducible.
