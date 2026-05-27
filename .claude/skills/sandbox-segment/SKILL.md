---
name: sandbox-segment
description: >
  Autonomous segmentation of a geophysics panel image into velocity layers.
  Given a panel crop (with colorbar already removed), the agent selects
  segmentation engines, evaluates results, and chooses or fuses the best
  output. Use when the user asks to segment a velocity model panel, extract
  layers from a cross-section, or when a previous segmentation needs retry.
  Triggers: "segment this panel", "extract layers", "velocity zones",
  "segmentation failed, try again"
argument-hint: <panel_image_path> [--n-layers=N] [--reps-json=path] [--colorbar=path]
allowed-tools: Bash, Read, Write, Edit
---

# sandbox-segment

Autonomous velocity model segmentation. You are given a panel image and must
produce the best possible layer segmentation through iterative exploration.

## Available Engines

You can run any of these engines via inline Python scripts (Bash `python -c "..."`).
Each engine is a Python function; you construct the call inline:

| Engine | Best For | Example Call |
|--------|----------|--------------|
| `v4_kmeans` | General purpose, vivid colors | `python -c "from geoseg.modules.segment_engines.v4_kmeans import segment; ..."` |
| `kmeans_full` | Vivid colors with rep seeds | `python -c "from geoseg.modules.segment_engines.kmeans_full import segment; ..."` |
| `edge_guided` | Smooth geological boundaries | `python -c "from geoseg.modules.segment_engines.edge_guided import segment; ..."` |
| `edge_grow` | Region growing from edges | `python -c "from geoseg.modules.segment_engines.edge_grow import segment; ..."` |
| `ensemble` | Best quality (slow) | `python -c "from geoseg.modules.segment_engines.ensemble import segment; ..."` |
| `grayscale` | Near-zero saturation | `python -c "from geoseg.modules.segment_engines.grayscale import segment; ..."` |

Each `segment()` call returns a dict with:
- `labels`: int32 numpy array (0 = background/unassigned)
- `overlay`: RGB overlay for visual inspection
- `meta`: dict with engine name, color_names, n_layers

Save outputs to `runs/sandbox/{panel_id}/`:
```bash
python -c "
import numpy as np
from PIL import Image
# ... run engine, get result ...
np.save('runs/sandbox/panel_0/labels.npy', result['labels'])
Image.fromarray(result['overlay']).save('runs/sandbox/panel_0/overlay.png')
"
```

## Autonomous Workflow (Closed-Loop)

### Step 0: Read Strategy Memory (Pre-Flight)

Before making any decisions, check if similar panels have been processed before:

```bash
python -c "
import json
from geoseg.modules.segment_engines.strategy_memory import query_similar, load_templates
import numpy as np
from PIL import Image

img = np.array(Image.open('{panel_path}').convert('RGB'))
similar = query_similar(img, top_k=3)
templates = load_templates()

print('=== Similar History ===')
for rec in similar:
    print(f\"Engine: {rec['engine']}, Outcome: {rec['outcome']}, Score: {rec['scores'].get('overall_score', 0)}\")

print('=== Strategy Templates ===')
for p in templates.get('patterns', [])[:5]:
    print(f\"Pattern: {p['feature_pattern']} -> {p['recommended_engine']} (rate={p['success_rate']}, conf={p['confidence']})\")
"
```

Use this information to inform your initial strategy choice. If history strongly
recommends a particular engine for this image type, start with that engine.

### Step 1: Analyze the Panel Image

- Saturation level (vivid / pastel / grayscale)
- Presence of clear layer boundaries
- Color uniformity within regions
- Presence of noise (text, axis labels, annotations)

### Step 2: Select Initial Strategy

Combine visual analysis + memory recommendations:
- Vivid (rich colors, sat > 0.5): start with `kmeans_full` or `edge_guided`
- Pastel / faded (sat < 0.1): start with `v4_kmeans` or `grayscale`
- Mixed: start with `v4_kmeans`
- If `reps-json` provided: prefer `kmeans_full`, `edge_guided`, `edge_grow`
- If history strongly recommends a specific engine for this feature pattern, prioritize it

### Step 3: Run Engine (Bash)

### Step 4: VLM Visual Evaluation (PRIMARY) + Objective Metrics (AUXILIARY)

**IMPORTANT: Your visual judgment is the PRIMARY evaluation.** Quantitative
metrics are only objective facts to help you quickly spot obvious problems.
They do NOT replace your semantic understanding of geophysics images.

**VLM Visual Evaluation** (Read overlay.png side-by-side with original panel):
Ask yourself these questions:
- Does the segmentation capture ALL layers visible in the original image?
- Are there MISSING boundaries (color transitions in original not reflected in segmentation)?
- Are there EXTRA boundaries (segmentation invented boundaries where original has none)?
- Does any layer include non-layer regions (text, colorbar, axis labels)?
- Are boundaries REASONABLE for this geology — even if rough or irregular?
  - Faults, unconformities, and erosion surfaces are SUPPOSED to be irregular.
  - Do NOT penalize rough boundaries unless they clearly misalign with the original.
- Is any legitimate layer MERGED with another (under-segmentation)?
- Is any single layer SPLIT into fragments without geological reason (over-segmentation)?

**Objective Metrics** (Bash — facts only, no "scores"):
```bash
python -c "
import json, numpy as np
from PIL import Image
from geoseg.modules.segment_engines.metrics import compute_all

labels = np.load('runs/sandbox/{panel_id}/labels.npy')
img = np.array(Image.open('{panel_path}').convert('RGB'))
metrics = compute_all(labels, img)
print(json.dumps(metrics, indent=2))
"
```

These metrics report OBJECTIVE FACTS:
- `n_layers`: pure count
- `boundary_alignment`: fraction of seg boundaries that align with image edges
- `tiny_fragments`: list of very small regions (may be thin layers or over-segmentation)
- `noise_warnings`: regions that look like text/colorbar/axes (verify visually!)
- `region_stats`: per-layer area and component count (n_components > 1 may indicate断层)

**How to use metrics**:
- `n_layers < 2` → almost certainly under-segmented, retry mandatory
- `boundary_alignment < 0.3` → boundaries may not align with image, verify visually
- `noise_warnings` → check if these regions are actually noise (VLM decides)
- `tiny_fragments` → check if they are legitimate thin layers or over-segmentation

**How NOT to use metrics**:
- Do NOT reject a result just because boundaries are "not smooth"
- Do NOT reject a result because a layer has multiple components (may be断层)
- Do NOT reject a result because colors within a layer vary (may be depth gradient)

### Step 5: Iterate if Needed

Decision hierarchy (VLM judgment overrides everything):

1. **n_layers < 2** (objective) → MUST retry, regardless of visual appearance
2. **VLM sees clear missing boundaries** → retry with different engine or more layers
3. **VLM sees clear extra/noise regions** → retry or post-process to remove
4. **VLM sees clear over/under-segmentation** → retry
5. **metrics suggest problems but VLM thinks it's fine** → trust VLM
6. **metrics look good but VLM sees problems** → trust VLM, retry

You may **fuse results**: e.g., use A's coarse segmentation + B's edge refinement.

### Step 6: Final Selection

Choose the result where:
1. **VLM visual assessment**: best alignment with original image (PRIMARY)
2. **Layer count**: matches visible geological layers (must-have)
3. **Objective metrics**: no red flags (no massive noise inclusion, no total misalignment)

If no result is satisfactory, pick the least-bad and note specific issues in `strategy.log`.

### Step 7: Save Results + Update Memory

Save to `runs/sandbox/{panel_id}/`:
```bash
python -c "
import numpy as np
from PIL import Image
import json
from geoseg.modules.segment_engines.strategy_memory import record_attempt

labels = np.load('runs/sandbox/{panel_id}/labels.npy')
img = np.array(Image.open('{panel_path}').convert('RGB'))

# Record this attempt in strategy memory
record_attempt(
    panel_rgb=img,
    engine='{best_engine}',
    params={'n_layers': {n_found}, 'reps': ...},
    scores={scores_dict},
    outcome='success',  # or 'retry' if this was not the first attempt
    notes='{strategy_notes}',
)
print('Memory updated.')
"
```

Files to save:
- `labels.npy` — best label map
- `overlay.png` — colored overlay
- `meta.json` — engine name, color_names, n_layers
- `strategy.log` — which engines were tried, scores, why this one was chosen

Also write the objective metrics to `metrics.json` for audit:
```json
{"n_layers": 5, "boundary_alignment": 0.91, "tiny_fragments": [], "noise_warnings": {...}}
```

## Evaluation Criteria (VLM Visual Judgment PRIMARY)

1. **Fidelity to original**: Does the segmentation match the original image?
   - All visible layers captured?
   - No missing boundaries?
   - No invented boundaries?
2. **Layer count correctness**: n_layers should match visible geological layers
3. **Boundary reasonableness**: Boundaries align with actual color transitions
   - Rough/irregular boundaries are FINE if they match the original (断层, unconformities)
   - Do NOT penalize roughness per se
4. **Noise exclusion**: text, colorbars, axis labels should NOT be segmented as layers
5. **Component count is NOT a criterion**: A layer may legitimately have multiple
   components due to断层 or erosion. Judge by geological sense, not connectivity.

## Constraints

- Try at least 2 different engines before settling on a result.
- Do not exceed 5 engine runs per panel (cost control).
- If `n-layers` is specified, use it as a target, but trust visual evidence if the
  image clearly has a different number of layers.
- Always save the overlay image for visual verification.
- Write `strategy.log` documenting your decisions for audit.
