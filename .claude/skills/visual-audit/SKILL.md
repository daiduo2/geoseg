---
name: visual-audit
description: >
  Agent-driven visual critic for geophysics segmentation results.
  Reads the overlay-with-legend and original panel side-by-side, identifies
  problematic regions by label/color, and outputs a structured RegionalAudit
  with frozen labels, retry labels, and repair suggestions.
  Triggers: "audit this segmentation", "visual audit", "check segmentation quality",
  "review this overlay", "这个结果质量怎么样", "指出需要修改的区域"
argument-hint: <labels.npz> <panel_image_path> [--output-dir=path]
allowed-tools: Bash, Read, Write, Edit
---

# visual-audit

Audit a segmentation result for quality. This skill does **NOT** enforce hard
rejection gates. Instead, it treats the agent as a visual critic: you inspect
the overlay-with-legend against the original panel, decide which regions are
good, which need repair, and output a structured audit that an executor agent
can act upon.

## Quick Start

```
User: /visual-audit runs/sandbox/panel_0/labels.npz runs/sandbox/panel_0/panel.png
Agent: [generates overlay_legend.jpg + views + label_color_map]
       "Audit result: label 2 (green, upper-middle) is fragmented; label 3 (blue,
        right side) covers text. Frozen=[1,4], Retry=[2,3]."
```

## When to Use

- After any segmentation engine run, before deciding whether to accept or repair.
- When `sandbox-segment` produces a result that looks suspicious.
- When the user asks which regions need modification.
- Before final export, as one of the HITL review inputs (but the agent decides, not a gate).

## When NOT to Use

- Do NOT use for classification (`figure-classify`) — this is for segmentation only.
- Do NOT output a binary PASS/FAIL. Output structured findings instead.

## Audit Workflow

### Step 0: Generate Audit Materials

Run the visual audit module to create the overlay-with-legend, auxiliary views,
crops, and diagnostic signals:

```bash
uv run python -c "
import json
import numpy as np
from PIL import Image
from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend
from geoseg.modules.visual_audit import create_audit_views, save_views, create_audit_crops, save_crops
from geoseg.modules.visual_audit.semantic import compute_semantic_fidelity

labels = np.load('runs/sandbox/{panel_id}/labels.npz')['labels']
img = np.array(Image.open('{panel_path}').convert('RGB'))
out_dir = 'runs/sandbox/{panel_id}/visual_audit'

# Main audit input: overlay with legend
overlay = generate_overlay_with_legend(img, labels)
Image.fromarray(overlay).save('runs/sandbox/{panel_id}/overlay_legend.jpg', quality=90)

# Auxiliary views
views = create_audit_views(labels, img)
view_paths = save_views(views, f'{out_dir}/views')

# Crops
crops = create_audit_crops(img)
crop_paths = save_crops(crops, f'{out_dir}/crops')

# Diagnostic signals (objective facts, not verdicts)
semantic = compute_semantic_fidelity(labels, img)
label_color_map = {}
for lbl in sorted(set(labels.flatten()) - {0}):
    mask = labels == lbl
    ys, xs = np.where(mask)
    label_color_map[str(int(lbl))] = {
        'area_frac': round(float(mask.sum() / labels.size), 4),
        'median_y': round(float(np.median(ys)), 1),
        'color': overlay[ys[0], xs[0]].tolist() if len(ys) > 0 else [128, 128, 128],
    }

report = {
    'views': view_paths,
    'crops': crop_paths,
    'diagnostic_signals': semantic,
    'label_color_map': label_color_map,
}

import pathlib
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(f'{out_dir}/report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('overlay_legend.jpg and audit materials ready.')
"
```

### Step 1: Read the Overlay-with-Legend

Read `runs/sandbox/{panel_id}/overlay_legend.jpg` first. It is the primary audit
input. The bottom-right legend maps each color to a `label_id`. Use the label
IDs and colors to point at specific regions in your audit output.

Also read the original panel (`panel.png`) and the auxiliary views in
`visual_audit/views/` as needed:

- `side_by_side.jpg`: original vs pure mask — start here for semantic check.
- `pure_mask.jpg`: discrete colors, no blending — shows every tiny label.
- `fragment_highlight.jpg`: tiny islands in red.
- `text_residual_map.jpg`: text mask + boundaries.
- `difference_heatmap.jpg`: boundary vs color-edge alignment (green=aligned, red=misaligned).
- `crops/*.jpg`: zoomed regions around text, plume, top/bottom boundaries.

### Step 2: Inspect by Color / Label ID

For each colored region in the overlay, ask:

- Does this color correspond to ONE visible geological layer?
- Is any visible layer missing from the overlay?
- Is any layer split into multiple colors without a geological reason?
- Are multiple distinct layers merged into one color?
- Does any color cover text, colorbar, axes, or legend?
- Are boundaries aligned with real color/texture transitions?
- For plume/uplift/dome structures: is the shape and extent correct?

When you describe a problem, always reference the **label_id** (and optionally
color) from the legend. For example:

- "label 2 (green, upper-middle) is fragmented into 3 disconnected pieces"
- "label 5 (purple, bottom-left) covers the axis text"
- "labels 3 and 4 (blue and yellow) should be merged — they are the same layer"

### Step 3: Decide Frozen vs Retry

Classify each non-background label into one of:

- **Frozen**: region is geologically correct and should not change.
- **Retry**: region has problems and should be re-segmented or merged.
- **Uncertain**: mark as retry if you cannot tell from the views.

### Step 4: Choose Repair Strategy

| Strategy | When to use | Implementation hint |
|----------|-------------|---------------------|
| `regional_fusion` | Some labels good, some bad | `regional_segment(..., audit=RegionalAudit(frozen_labels=..., retry_labels=...))` |
| `merge_labels` | Two+ labels are actually the same layer | `merge_labels_by_ids(labels, label_ids, target_id)` |
| `switch_engine` | Whole image is wrong | Re-run with a different engine or parameters |
| `post_process` | Boundaries are rough but topology is right | `horizon_refinement.refine_boundaries` or morphological cleanup |
| `accept` | No significant problems | Empty `retry_labels` |

### Step 5: Output Structured RegionalAudit

Write the audit result to `runs/sandbox/{panel_id}/regional_audit.json`:

```json
{
  "frozen_labels": [1, 4],
  "retry_labels": [2, 3],
  "notes": "label 2 (green, upper-middle) is fragmented; label 3 (blue, right) covers text.",
  "repair_strategy": "regional_fusion",
  "secondary_engine": "edge_guided",
  "local_fixes": [
    {"action": "merge_labels", "label_ids": [5, 6], "target_id": 5, "rationale": "same layer"}
  ],
  "iteration": 1
}
```

## Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `frozen_labels` | list[int] | Labels that are good and should be preserved |
| `retry_labels` | list[int] | Labels that need re-segmentation or repair |
| `notes` | str | Human-readable diagnosis |
| `repair_strategy` | str | One of: `regional_fusion`, `merge_labels`, `switch_engine`, `post_process`, `accept` |
| `secondary_engine` | str | Recommended engine for regional_fusion (e.g., `edge_guided`, `kmeans_full`) |
| `local_fixes` | list[dict] | Immediate fixes like label merges |
| `iteration` | int | Current audit-execution loop iteration |

## What NOT to Do

- **Do NOT output PASS/FAIL.** Output findings and repair directions.
- **Do NOT use hard thresholds** like "fragment_ratio > 0.30" to decide. Use your visual judgment.
- **Do NOT ignore the legend.** Always reference label IDs so the executor knows exactly which region to modify.
- **Do NOT recommend whole-image re-segmentation if regional_fusion can fix it.** Preserve good regions.

## Integration with Other Skills

- `sandbox-segment`: Call `visual-audit` after each candidate engine result. Feed the audit JSON into the executor loop.
- `batch-segment`: Run `visual-audit` per figure, then present audit notes (not binary PASS/FAIL) during the HITL review stage.
- `geo-segment`: Use `visual-audit` as a review aid before presenting the overlay to the user for final confirmation.

## Rules

1. **Agent judgment is primary.** Objective metrics are only diagnostic signals.
2. **Always use label IDs from the legend** when describing regions.
3. **Prefer regional repair over whole-image re-run.** Freeze good regions first.
4. **When in doubt, mark as retry.** The executor can decide how to handle it.
5. **No binary verdicts.** There are only "frozen", "retry", and "uncertain" labels.
