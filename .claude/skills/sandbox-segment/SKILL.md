---
name: sandbox-segment
description: >
  Autonomous segmentation of a geophysics panel image into velocity layers.
  Given a panel crop (with colorbar already removed), the agent selects
  segmentation engines, evaluates results, and chooses or fuses the best
  output. Text and annotations are handled by the audit agent: they are
  segmented as ordinary labels first, then identified by color/position and
  merged into the surrounding geology. Use when the user asks to segment a
  velocity model panel, extract layers from a cross-section, or when a
  previous segmentation needs retry.
  Triggers: "segment this panel", "extract layers", "velocity zones",
  "segmentation failed, try again"
argument-hint: <panel_image_path> [--n-layers=N] [--oversegment] [--reps-json=path] [--colorbar=path]
allowed-tools: Bash, Read, Write, Edit
---

# sandbox-segment

Autonomous velocity model segmentation. You are given a panel image and must
produce the best possible layer segmentation through iterative exploration.

**Important: do NOT run a text-removal preprocessor.** Segment the original
panel directly. The audit agent will later identify text/annotation labels by
color and merge them into the surrounding layers using
`remove_labels_by_ids(..., fill='nearest')`.

## Available Engines

You can run any of these engines via inline Python scripts (Bash `uv run python -c "..."`).
Each engine is a Python function; you construct the call inline:

| Engine | Best For | Example Call |
|--------|----------|--------------|
| `v4_kmeans` | General purpose, vivid colors | `uv run python -c "from geoseg.modules.segment_engines.v4_kmeans import segment; ..."` |
| `kmeans_full` | Vivid colors with rep seeds | `uv run python -c "from geoseg.modules.segment_engines.kmeans_full import segment; ..."` |
| `edge_guided` | Smooth geological boundaries | `uv run python -c "from geoseg.modules.segment_engines.edge_guided import segment; ..."` |
| `edge_grow` | Region growing from edges | `uv run python -c "from geoseg.modules.segment_engines.edge_grow import segment; ..."` |
| `ensemble` | Best quality (slow) | `uv run python -c "from geoseg.modules.segment_engines.ensemble import segment; ..."` |
| `grayscale` | Near-zero saturation | `uv run python -c "from geoseg.modules.segment_engines.grayscale import segment; ..."` |
| `lab_l_kmeans` | Low-contrast / gradient / funnel on similar hue background | `uv run python -c "from geoseg.modules.segment_engines.lab_l_kmeans import segment; ..."` |
| `seeded_region_grow` | Funnel / plume where agent can place seed points | `uv run python -c "from geoseg.modules.segment_engines.seeded_region_grow import segment; ..."` |
| `horizon_refinement` | Post-process: smooth boundaries | `uv run python -c "from geoseg.modules.segment_engines.horizon_refinement import refine_boundaries; ..."` |

Each `segment()` call returns a dict with:
- `labels`: int32 numpy array (0 = background/boundary)
- `overlay`: RGB overlay for visual inspection
- `meta`: dict with engine name, color_names, n_layers

Save outputs to `runs/sandbox/{panel_id}/`:
```bash
uv run python -c "
import numpy as np
from PIL import Image
# ... run engine, get result ...
np.savez_compressed('runs/sandbox/panel_0/labels.npz', labels=result['labels'])
Image.fromarray(result['overlay']).save('runs/sandbox/panel_0/overlay.jpg', quality=90)
"
```

## Autonomous Workflow (Closed-Loop)

### Step 0: Read Strategy Memory (Pre-Flight)

Before making any decisions, check if similar panels have been processed before:

```bash
uv run python -c "
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

Read the original panel (`panel.png`).

- Saturation level (vivid / pastel / grayscale)
- Presence of clear layer boundaries
- Color uniformity within regions
- Presence and color of annotation text, leader lines, arrows, and labels

Remember: text will likely become one or more labels. That is fine. You will
identify and remove them in the audit step.

### Step 2: Select Initial Strategy

Combine visual analysis + memory recommendations (historical hints, not a lookup table):
- Vivid (rich colors, sat > 0.5): consider `kmeans_full` or `edge_guided`
- Pastel / faded (sat < 0.1): consider `v4_kmeans` or `grayscale`
- Mixed: consider `v4_kmeans`
- Low-contrast / gradient where a feature shares hue with background (e.g. panel_3 funnel): consider `lab_l_kmeans` or `seeded_region_grow`
- If `reps-json` provided: consider `kmeans_full`, `edge_guided`, `edge_grow`
- If history strongly recommends a specific engine for this feature pattern, prioritize it

### Step 3: Run Engine (Bash)

Run the chosen engine on the **original panel** (not a text-removed version).
Save the resulting labels, overlay, and engine metadata.

```bash
uv run python -c "
import json, pathlib, numpy as np
from PIL import Image
from geoseg.modules.segment_engines.v4_kmeans import segment as v4_segment
from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend

panel_id = '{panel_id}'
out = pathlib.Path(f'runs/sandbox/{panel_id}')
out.mkdir(parents=True, exist_ok=True)

img = np.array(Image.open('{panel_path}').convert('RGB'))
result = v4_segment(img, n_layers=5)

np.savez_compressed(out / 'labels.npz', labels=result['labels'])
overlay = generate_overlay_with_legend(img, result['labels'])
Image.fromarray(overlay).save(out / 'overlay_legend.jpg', quality=90)
print('Segmented on original panel.')
"
```

#### Over-segment then merge (soft gradients / low contrast)

If the panel has soft gradients or a low-contrast feature (e.g. panel_3's funnel
plume on an orange gradient), deliberately request **more** layers than the
target so the engine splits color variations apart. The audit agent then merges
labels that belong to the same geological layer.

```bash
# Example: target 5 layers, oversegment with 10 using v4_kmeans
result = v4_segment(img, n_layers=10)
```

Prefer `v4_kmeans` or `kmeans_full` for oversegment; `ensemble` voting can
collapse gradient fragments back together.

#### Seeded region grow (funnel / plume)

`seeded_region_grow` runs marker-controlled watershed on a smoothed color-gradient
cost map. It is ideal when a feature is separated from the background mainly by
spatial position and subtle lightness differences, even if the hues are similar.

**Seed placement rules:**
- Get the ACTUAL full-resolution dimensions first (`Image.open(path).size`).
- Place one seed in the visual center of EACH target layer, not just feature vs.
  background.
- For panel_3 (1740x3480 px) the target layers are: top dark surface/weak zone,
  blue weak zone, funnel-shaped refractory peridotite residues, orange mantle,
  yellow mantle base.
- Keep seeds away from white/black text annotations and leader lines.
- Use label IDs 1..N; they will be compacted later.

```bash
uv run python -c "
import json, pathlib, numpy as np
from PIL import Image
from geoseg.modules.segment_engines.seeded_region_grow import segment
from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend

panel_id = '{panel_id}'
out = pathlib.Path(f'runs/sandbox/{panel_id}')
img = np.array(Image.open('{panel_path}').convert('RGB'))
h, w = img.shape[:2]

# Example for a 5-layer panel like panel_3 — adjust y coordinates to the image
seeds = [
  {'y': int(h*0.06), 'x': w//2, 'label': 1},  # top dark surface / weak zone
  {'y': int(h*0.22), 'x': w//2, 'label': 2},  # blue weak zone
  {'y': int(h*0.36), 'x': w//2, 'label': 3},  # funnel / refractory residues
  {'y': int(h*0.58), 'x': w//2, 'label': 4},  # orange mantle
  {'y': int(h*0.86), 'x': w//2, 'label': 5},  # yellow mantle base
]
(out / 'seeds.json').write_text(json.dumps(seeds), encoding='utf-8')

result = segment(img, seeds=seeds, color_space='LAB')
np.savez_compressed(out / 'labels.npz', labels=result['labels'])
overlay = generate_overlay_with_legend(img, result['labels'])
Image.fromarray(overlay).save(out / 'overlay_legend.jpg', quality=90)
print('Seeded region grow done.')
"
```

The agent should pick seed coordinates by visually inspecting the original panel
and using the full-resolution coordinate system.

### Step 4: Visual Critic Audit (Agent Judgment PRIMARY)

**Your visual judgment is the PRIMARY evaluation.** Objective metrics are only
facts to help you quickly spot problems; they do NOT replace your geological
understanding.

#### Step 4a: Generate Audit Materials

For each candidate result, generate the overlay-with-legend and auxiliary views
using the **original panel** as the background:

```bash
uv run python -c "
import json, pathlib
import numpy as np
from PIL import Image
from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend
from geoseg.modules.visual_audit import create_audit_views, save_views, create_audit_crops, save_crops
from geoseg.modules.visual_audit.semantic import compute_semantic_fidelity

labels = np.load('runs/sandbox/{panel_id}/labels.npz')['labels']
img = np.array(Image.open('{panel_path}').convert('RGB'))
out_dir = pathlib.Path('runs/sandbox/{panel_id}/visual_audit')
out_dir.mkdir(parents=True, exist_ok=True)

overlay = generate_overlay_with_legend(img, labels)
Image.fromarray(overlay).save('runs/sandbox/{panel_id}/overlay_legend.jpg', quality=90)

views = create_audit_views(labels, img)
view_paths = save_views(views, str(out_dir / 'views'))
crops = create_audit_crops(img)
crop_paths = save_crops(crops, str(out_dir / 'crops'))

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
(out_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Audit materials ready.')
"
```

#### Step 4b: Read and Audit

Read these in order:

1. **`overlay_legend.jpg`**: primary input. Use the bottom-right legend to map
colors to `label_id`. Always reference regions by label ID in your audit output.
2. **`panel.png`**: original panel for side-by-side comparison.
3. **`visual_audit/views/side_by_side.jpg`**: original vs pure mask.
4. **`visual_audit/views/fragment_highlight.jpg`**: tiny islands in red.
5. **`visual_audit/crops/*.jpg`**: zoomed regions as needed.

Ask yourself:
- Does each colored region correspond to ONE visible geological layer?
- Are there MISSING boundaries?
- Are there EXTRA boundaries invented by the segmentation?
- Does any region cover text, colorbar, axes, or legend?
  - If yes, note its label ID as a `text_label`.
- Are boundaries geologically reasonable (even if rough)?
- Is any layer merged with another (under-segmentation)?
- Is any layer split without geological reason (over-segmentation)?

#### Step 4c: Output RegionalAudit

Write `runs/sandbox/{panel_id}/regional_audit.json`:

```json
{
  "frozen_labels": [1, 4],
  "retry_labels": [2, 3],
  "text_labels": [5],
  "notes": "label 2 (green, upper-middle) fragmented; label 3 (blue, right) covers text; label 5 is the white text blob.",
  "repair_strategy": "remove_text_labels",
  "secondary_engine": "edge_guided",
  "local_fixes": [
    {"label": 2, "action": "merge_labels", "label_ids": [2, 6], "target_id": 2}
  ],
  "iteration": 1
}
```

Rules:
- `frozen_labels` + `retry_labels` should cover all non-background labels you have an opinion on.
- `text_labels`: label IDs that are clearly text/annotation. Use `repair_strategy: "remove_text_labels"` to delete them (vacated pixels fill from nearest remaining label).
- Reference every region by **label ID** (with color/position as backup).
- Do NOT output PASS/FAIL. Output findings and repair directions only.
- `regional_fusion` is the preferred strategy when only some labels are wrong.
- `merge_labels` is appropriate when two or more labels are actually the same layer.
- `remove_text_labels` is appropriate when the only/main issue is text/annotation labels.
- `switch_engine` is appropriate when the whole image is poor.

### Step 5: Executor Loop (Audit → Repair → Re-audit)

Read `runs/sandbox/{panel_id}/regional_audit.json`.

- If `retry_labels` is empty and `text_labels` is empty → proceed to Step 6.
- If `retry_labels` is non-empty or `text_labels` is non-empty → execute the repair strategy, regenerate overlay,
  and re-run the visual audit. Maximum 3 iterations.

#### Step 5a: Choose Repair Strategy

Use the `repair_strategy` from the audit output. If missing, choose based on the
audit notes:

| `repair_strategy` | When to use | Executor action |
|-------------------|-------------|-----------------|
| `regional_fusion` | Some labels good, some bad | Freeze `frozen_labels`, re-segment `retry_labels` with `secondary_engine` |
| `merge_labels` | Two+ labels are the same layer | Call `merge_labels_by_ids` |
| `remove_text_labels` | Labels are text/annotation | Call `remove_labels_by_ids(..., fill='nearest')` |
| `switch_engine` | Whole image is poor | Re-run a different engine on the full panel |
| `post_process` | Topology right, boundaries rough OR many tiny fragments | Call `horizon_refinement.refine_boundaries` or `merge.filter_small_components(min_area_ratio=0.001)` |
| `accept` | No significant problems | Proceed to Step 6 |

#### Step 5b: Execute Regional Fusion

When `repair_strategy == "regional_fusion"`:

```bash
uv run python -c "
import json, numpy as np
from PIL import Image
from geoseg.modules.segment_engines.regional_fusion import regional_segment, RegionalAudit, FusionConfig

labels = np.load('runs/sandbox/{panel_id}/labels.npz')['labels']
img = np.array(Image.open('{panel_path}').convert('RGB'))
audit_json = json.load(open('runs/sandbox/{panel_id}/regional_audit.json'))

audit = RegionalAudit(
    frozen_labels=audit_json['frozen_labels'],
    retry_labels=audit_json['retry_labels'],
    notes=audit_json.get('notes', ''),
    iteration=audit_json.get('iteration', 1),
)

result = regional_segment(
    img,
    n_layers={n},
    primary_result={'labels': labels},
    audit=audit,
    config=FusionConfig(
        primary_engine='{best_engine}',
        secondary_engines=[audit_json.get('secondary_engine', 'edge_guided'), 'kmeans_full', 'slic_kmeans'],
    ),
)

np.savez_compressed('runs/sandbox/{panel_id}/labels.npz', labels=result['labels'])
Image.fromarray(result['overlay']).save('runs/sandbox/{panel_id}/overlay.jpg', quality=90)
print(json.dumps(result['meta'], indent=2))
"
```

Then regenerate `overlay_legend.jpg` on the original panel and return to Step 4 for re-audit.

#### Step 5c: Execute Label Merge / Text-Label Removal

When `repair_strategy == "merge_labels"`, `repair_strategy == "remove_text_labels"`,
or `local_fixes` contains merge/remove actions:

```bash
uv run python -c "
import json, numpy as np
from PIL import Image
from geoseg.modules.post_process.merge import merge_labels_by_ids, remove_labels_by_ids
from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend

labels = np.load('runs/sandbox/{panel_id}/labels.npz')['labels']
img = np.array(Image.open('{panel_path}').convert('RGB'))
audit_json = json.load(open('runs/sandbox/{panel_id}/regional_audit.json'))

# Merge over-fragmented gradient pieces
for fix in audit_json.get('local_fixes', []):
    if fix['action'] == 'merge_labels':
        labels = merge_labels_by_ids(labels, fix['label_ids'], target_id=fix.get('target_id', fix['label_ids'][0]))

# Remove text/annotation labels (vacated pixels fill from nearest remaining label)
for lbl in audit_json.get('text_labels', []):
    labels = remove_labels_by_ids(labels, [lbl], fill='nearest')

np.savez_compressed('runs/sandbox/{panel_id}/labels.npz', labels=labels)
overlay = generate_overlay_with_legend(img, labels)
Image.fromarray(overlay).save('runs/sandbox/{panel_id}/overlay_legend.jpg', quality=90)
print('merge/remove applied')
"
```

Then return to Step 4 for re-audit.

#### Step 5d: Switch Engine or Post-Process

For `switch_engine`, re-run a different engine with adjusted parameters and
return to Step 4.

For `post_process`, call the appropriate post-processing function
(`horizon_refinement.refine_boundaries`, morphological cleanup, or
`filter_small_components`) and return to Step 4.

Example hard fragment filter:

```bash
uv run python -c "
import json, numpy as np
from PIL import Image
from geoseg.modules.post_process.merge import filter_small_components
from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend

labels = np.load('runs/sandbox/{panel_id}/labels.npz')['labels']
img = np.array(Image.open('{panel_path}').convert('RGB'))

filtered = filter_small_components(labels, min_area_ratio=0.001)
np.savez_compressed('runs/sandbox/{panel_id}/labels.npz', labels=filtered)
overlay = generate_overlay_with_legend(img, filtered)
Image.fromarray(overlay).save('runs/sandbox/{panel_id}/overlay_legend.jpg', quality=90)
print('small fragments filtered')
"
```

#### Step 5e: Iteration Cap

Maximum 3 audit-repair iterations per panel. If problems remain after 3
iterations, pick the least-bad result and document remaining issues in
`strategy.log`. Escalate to human review if the result is clearly unusable.

### Step 6: Final Selection

Choose the result where:
1. **Agent visual judgment**: best alignment with original image (PRIMARY)
2. **Layer count**: matches visible geological layers
3. **No significant unresolved issues**: all retry_labels and text_labels from the last audit are acceptable

If no result is satisfactory, pick the least-bad and note specific issues in `strategy.log`.

### Step 6b: Horizon Refinement (Optional Post-Process)

After selecting the best engine result, you MAY apply horizon refinement if the
audit notes indicate boundaries are rough or zigzag but the topology is correct.

Run refinement:

```bash
uv run python -c "
import numpy as np
from PIL import Image
from geoseg.modules.segment_engines.horizon_refinement import refine_boundaries

labels = np.load('runs/sandbox/{panel_id}/labels.npz')['labels']
img = np.array(Image.open('{panel_path}').convert('RGB'))

refined, boundaries = refine_boundaries(img, coarse_labels=labels, method='savgol')
np.save('runs/sandbox/{panel_id}/labels_refined.npy', refined)
print(f'Refined: {len(boundaries)} boundaries fitted')
print(f'Same as coarse: {np.array_equal(labels, refined)}')
"
```

**Agent visual re-evaluation after refinement** (MANDATORY):
Read both overlays side-by-side:
- `runs/sandbox/{panel_id}/overlay.jpg` (coarse)
- Create refined overlay and compare

Judge:
- Are boundaries SMOOTHER without losing geological accuracy?
- Did any thin layer disappear?
- Did any fault/unconformity get incorrectly smoothed?

**Acceptance rule**: Accept refinement ONLY if agent visual judgment finds it visually better or equal. Otherwise keep coarse.

### Step 7: Save Results + Update Memory

Save to `runs/sandbox/{panel_id}/`:
```bash
uv run python -c "
import numpy as np
from PIL import Image
import json
from geoseg.modules.segment_engines.strategy_memory import record_attempt

labels = np.load('runs/sandbox/{panel_id}/labels.npz')['labels']
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
- `labels.npz` — best label map (refined if accepted, otherwise coarse)
- `overlay.jpg` — colored overlay for visual verification
- `meta.json` — engine name, color_names, n_layers, refinement_applied (bool)
- `strategy.log` — which engines were tried, scores, why this one was chosen, whether horizon refinement was triggered/accepted

If horizon refinement was applied and accepted, also save:
- `labels_coarse.npz` — pre-refinement label map (for audit/comparison)
- `overlay_coarse.jpg` — pre-refinement overlay

Also write the objective metrics to `metrics.json` for audit:
```json
{"n_layers": 5, "boundary_alignment": 0.91, "tiny_fragments": [], "noise_warnings": {...}}
```

## Evaluation Criteria (Agent Visual Judgment PRIMARY)

1. **Fidelity to original**: Does the segmentation match the original image?
   - All visible layers captured?
   - No missing boundaries?
   - No invented boundaries?
2. **Layer count correctness**: n_layers should match visible geological layers
3. **Boundary reasonableness**: Boundaries align with actual color transitions
   - Rough/irregular boundaries are FINE if they match the original (断层, unconformities)
   - Do NOT penalize roughness per se
4. **Noise exclusion**: text, colorbars, axis labels should NOT be segmented as layers
   - Identify text labels and remove them in the audit/repair loop.
5. **Component count is NOT a criterion**: A layer may legitimately have multiple
components due to断层 or erosion. Judge by geological sense, not connectivity.

## Constraints

- You are encouraged to try multiple engines for comparison, but use your judgment
  on how many runs are worthwhile for a given panel.
- If `n-layers` is specified, use it as a target, but trust visual evidence if the
  image clearly has a different number of layers.
- Always save the overlay image for visual verification.
- Write `strategy.log` documenting your decisions for audit.
- **Do not use a text-removal preprocessor.** Text is handled by the audit agent.
