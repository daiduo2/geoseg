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
| `horizon_refinement` | Post-process: smooth boundaries | `python -c "from geoseg.modules.segment_engines.horizon_refinement import refine_boundaries; ..."` |

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

### Step 5b: Repair Playbook (Self-Healing)

When you detect a problem, apply ONE specific repair strategy and re-run.
Do NOT randomly tweak — diagnose first, then pick the matching strategy.
Maximum 2 repair iterations per panel.

**CRITICAL: All engines now default to `max_auto_k=0`.** Do NOT enable auto_k unless
you have strong evidence (e.g., VLM clearly sees a layer that ALL engines miss).
Auto_k tends to over-segment smooth jet-colormap gradients.

| Problem Detected | Root Cause | Repair Strategy | Implementation |
|-----------------|------------|-----------------|----------------|
| **Under-segmented: too few layers** | n_layers too low or background swallowed a layer | **Strategy A**: Increase `n_layers` by 1-2. Try `kmeans_full` or `v4_colorbar_guided` (if colorbar present). Do NOT enable auto_k. | `segment(panel, n_layers=n+1, max_auto_k=0)` |
| **Top layer missing / merged with background** | Background color similar to top sediment color | **Strategy B**: Try `edge_guided` with lower `edge_weight=0.2` to capture subtle gradients. Or use `v4_colorbar_guided` with explicit colorbar seeds. | `segment(panel, n_layers=n, max_auto_k=0, edge_weight=0.2)` |
| **Bottom layer truncated** | Background label occupies bottom edge | **Strategy C**: Increase `n_layers` by 1. Try `kmeans_full` (more sensitive to bottom transitions). Post-process: check if background covers bottom row — if so, expand nearest layer downward. | `segment(panel, n_layers=n+1, max_auto_k=0)` or post-process |
| **Over-segmentation / "broken glass" fragmentation** | Jet colormap smooth gradients + vertical noise | **Strategy D**: Pre-process with Gaussian blur before segmentation. Start with `sigma=1.0`, increase to `sigma=2.0` or `sigma=3.5` if still fragmented. Then use `kmeans_full` with `n_layers` visually estimated count and `max_auto_k=0`. Alternatively, accept the best coarse result and apply `horizon_refinement` post-processing. | `cv2.GaussianBlur(panel, (0,0), sigmaX=sigma)` then `segment(...)` or `refine_boundaries(img, coarse_labels=labels)` |
| **Wellbore / annotation splits layers** | Vertical bright artifact cuts through layers | **Strategy E**: Mask out the artifact column before segmentation. Detect bright orange/red vertical strip, set to excluded label. Or post-process: merge left/right components of the same layer across the wellbore. | Create mask, exclude column, then segment |
| **Boundaries zigzag / not smooth** | Edge weight too high or noise | **Strategy F**: Use `edge_guided` with higher `sigma=4.0`. Or pre-process with Gaussian blur `sigma=1.5`. | `segment(panel, n_layers=n, max_auto_k=0, sigma=4.0)` |

**Repair iteration protocol:**
1. Run initial segmentation with 2+ engines
2. Evaluate all results
3. If best result quality < 0.90 → pick ONE strategy from table above
4. Re-run with fix applied
5. Re-evaluate. If still < 0.90 → pick a DIFFERENT strategy (do not repeat same fix)
6. Report final result with repair log documenting each iteration's {problem, strategy, outcome}

### Step 6: Final Selection

Choose the result where:
1. **VLM visual assessment**: best alignment with original image (PRIMARY)
2. **Layer count**: matches visible geological layers (must-have)
3. **Objective metrics**: no red flags (no massive noise inclusion, no total misalignment)

If no result is satisfactory, pick the least-bad and note specific issues in `strategy.log`.

### Step 6b: Horizon Refinement (Optional Post-Process)

After selecting the best engine result, check if boundaries need smoothing:

```bash
python -c "
import numpy as np
from PIL import Image
from geoseg.modules.segment_engines.metrics import compute_all

labels = np.load('runs/sandbox/{panel_id}/labels.npy')
img = np.array(Image.open('{panel_path}').convert('RGB'))
metrics = compute_all(labels, img)
frag = metrics.get('total_fragment_area_fraction', 0)
print(f'Fragmentation: {frag:.4f}')
if frag > 0.02:
    print('TRIGGER_REFINEMENT')
else:
    print('SKIP_REFINEMENT')
"
```

**Trigger condition**: `total_fragment_area_fraction > 0.02`
- Based on batch test analysis: 90% of clean results have frag < 0.01; fragmented results cluster at 0.03+
- This is a **programmatic** trigger, not a VLM judgment

**If triggered**, run horizon refinement:
```bash
python -c "
import numpy as np
from PIL import Image
from geoseg.modules.segment_engines.horizon_refinement import refine_boundaries

labels = np.load('runs/sandbox/{panel_id}/labels.npy')
img = np.array(Image.open('{panel_path}').convert('RGB'))

refined, boundaries = refine_boundaries(img, coarse_labels=labels, method='savgol')
np.save('runs/sandbox/{panel_id}/labels_refined.npy', refined)
print(f'Refined: {len(boundaries)} boundaries fitted')
print(f'Same as coarse: {np.array_equal(labels, refined)}')
"
```

**VLM Re-evaluation after refinement** (MANDATORY):
Read both overlays side-by-side:
- `runs/sandbox/{panel_id}/overlay.png` (coarse)
- Create refined overlay and compare

Judge:
- Are boundaries SMOOTHER without losing geological accuracy?
- Did any thin layer disappear?
- Did any fault/unconformity get incorrectly smoothed?

**Acceptance rule**: Accept refinement ONLY if VLM judges it visually better or equal. Otherwise keep coarse.

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
- `labels.npy` — best label map (refined if accepted, otherwise coarse)
- `overlay.png` — colored overlay for visual verification
- `meta.json` — engine name, color_names, n_layers, refinement_applied (bool)
- `strategy.log` — which engines were tried, scores, why this one was chosen, whether horizon refinement was triggered/accepted

If horizon refinement was applied and accepted, also save:
- `labels_coarse.npy` — pre-refinement label map (for audit/comparison)
- `overlay_coarse.png` — pre-refinement overlay

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
