---
name: geo-segment
description: >
  [WORKFLOW] Convert a published geophysics interpretation figure into a SPECFEM-ready
  velocity zone model. End-to-end pipeline orchestrated via Dynamic Workflows.
  Stages: classify → detect → segment → present → (HITL) → export.
  Triggers: "速度分区", "解释图", "SPEFEM", "SEM 输入", "断面图分区",
  "segment this figure", "geo-segment", "process this figure"
argument-hint: <image_path> [--n-layers=N] [--output-dir=path] [--session=path]
allowed-tools: Bash, Read, Write, Edit
---

# geo-segment

End-to-end conversion with CLI human-in-the-loop. One figure = one round-trip
of (auto-run → present → feedback → optionally re-run).

## Quick Start

```
User: /geo-segment runs/M0.5/fig1.png --n-layers=5
Agent: [auto-runs classify → detect → segment → presents overlay]
       "Accept / Modify / Skip / Backtrack ?"
User: 修改。右下角颜色条误分了，去掉。中间断层应拆成两层。
Agent: [re-runs sandbox with mask + n_layers+1 → presents new overlay]
       "Accept / Modify / Skip / Backtrack ?"
User: 接受
Agent: [exports SPECFEM]
```

## Workflow Orchestration

This skill is designed for **Dynamic Workflows**. Codex generates a JS
orchestration script with the following stage graph:

```
STAGE 0: init_session
  ↓
STAGE 1: classify (agent — Read image → JSON)
  ↓ [if proceed]
STAGE 2: detect_panels (Bash — cv_detect)
  ↓
STAGE 3: select_panel + crop (agent — Read + Bash)
  ↓
STAGE 4: segment (agent — sandbox-segment behavior)
  ↓
STAGE 5: present (agent — Read overlay → summary)
  ↓ [HITL pause]
STAGE 6: napari_review (Bash — launch editor, block until close)
  ↓ [on Accept]
STAGE 7: export (Bash — post_process)
  ↓ [on Modify]
STAGE 4b: re-segment (agent — sandbox with mask/layer adjustments)
  ↓
STAGE 5b: re-present
  ↓
STAGE 6b: napari_review (re-open editor)
```

**Parallelizable**: None within single-figure pipeline (stages are sequential
with data dependencies). Parallelism happens at the **batch** level
(`batch-segment` skill).

**Backtrack edges**: Stage 6 (Modify) → Stage 4b; Stage 6 (Backtrack) → any upstream.

## Stage Definitions

### STAGE 0: Initialize Session State

If `--session` provided, load existing session; else create a new one.
Save path defaults to `runs/sessions/{timestamp}.json`.

```python
from geoseg.session_state import create_session, save_session

state = create_session([image_path])
save_session(state, session_path)
```

### STAGE 1: Classify

**Type**: agent (Read → reasoning → JSON)
**Tool**: Read, Write
**Output**: `classification.json`, state update

Read image, decide if velocity_model / geological_cross_section.
- If skip: update state → `SKIPPED`, report reason, STOP.

### STAGE 2: Detect Panels

**Type**: Bash (Python tool)
**Tool**: Bash
**Output**: panel bbox list

Bash inline `cv_detect.panel_detector`.

### STAGE 3: Select Target Panel + Crop

**Type**: agent (Read → reasoning) + Bash
**Tool**: Read, Bash
**Output**: cropped panel path

Read image, pick primary panel (e.g. "inverted model").
Bash inline crop + `colorbar_extractor`.

### STAGE 4: Segment

**Type**: agent (Bash → Read → evaluate → iterative)
**Tool**: Bash, Read
**Output**: `labels.npz`, `overlay.jpg`, `meta.json`

Activate `sandbox-segment` behavior.
- Try ≥2 engines, evaluate visually, pick best.
- Save to `runs/sandbox/{figure_id}/`.

Update state → `update_figure(status=SEGMENTED, segmentation=...)`.

### STAGE 5: Present Result

Show a concise summary + overlay image:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 fig1.png  分割完成
   类型: velocity_model (置信度 0.92)
   Panels: 3 → 目标: #1 "(b) Inverted model"
   引擎: kmeans_full → 5 层
   质量: 0.85

[Read 展示 overlay.jpg]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Overlay Generation (`_create_overlay`)**

The overlay uses vivid, perceptually distinct colors (golden-ratio HSV
palette) so both VLM and human reviewers can clearly see every segmented
region. Three fill modes are available:

| Mode | Alpha | Description | Best for |
|------|-------|-------------|----------|
| `blend` | 0.65 | Distinct colors blended over original | Human review (default) |
| `solid` | 0.85 | Near-opaque color fill | Complex figures where blend is too subtle |
| `mask`  | 1.00 | Pure segmentation map, no original | VLM-only audit / README showcase |

Default is `blend`. The agent may switch to `solid` or `mask` when:
- The original figure has low contrast between layers
- VLM review reports "cannot distinguish regions"
- The user requests a clearer mask view

Background label is auto-detected and skipped. Tiny fragments (<0.2% area)
are merged before boundary drawing. Boundaries are drawn thin (`mode="thin"`).

### STAGE 6: Review via Napari Editor (HITL)

**Type**: HITL — agent launches napari, blocks until user closes window
**Tool**: Bash (launch napari)
**Output**: edited labels (via `--output-labels`)

Launch napari editor with auto-save on exit:

```bash
uv run python -m geoseg.modules.editor.napari_app \
  --session {session_path} \
  --figure {figure_id}
```

`napari.run()` blocks until the user closes the window. The editor auto-saves
shapes to `--output-shapes` and recomputed labels to `--output-labels` on exit.

**User workflow in napari**:
1. **Inspect** — view overlay, zoom/pan to verify boundaries
2. **Edit** — use native napari tools:
   - `L` Add Line — draw open boundary to split a region
   - `P` Add Polygon — draw closed boundary to create isolated region
   - `S` Select + `Delete` — remove boundary to merge regions
   - `D` Direct — drag vertices to reshape boundaries
3. **Save** — shapes auto-save on window close (no manual action needed)
4. **Close** — close napari window to return to agent

**After napari closes**, agent:
1. Read `labels_edited.npz` (or fallback to original if user made no edits)
2. Generate new overlay from edited labels
3. Present updated result in conversation:
   ```
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   📊 fig1.png  编辑完成
      变化: 新增 1 条边界，合并 0 个区域
      [展示编辑后的 overlay]
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ```
4. Ask for confirmation:
   ```
   请选择：
   [1] ✅ 接受 → 导出 SPECFEM
   [2] ✏️  重新编辑 → 再次启动 napari（加载已保存的 shapes）
   [3] ⏭️  跳过 → 标记为 SKIPPED
   [4] 🔙 回溯 → 重新 classify / panel / segment
   ```

**Choice 1 — Accept** → STAGE 7: Export. Use `labels_edited.npz` as source.
Update state → `EXPORTED`.

**Choice 2 — Re-open napari** → Re-launch napari with edited labels:
```bash
uv run python -m geoseg.modules.editor.napari_app \
  --session {session_path} \
  --figure {figure_id}
```
Napari re-extracts boundary shapes from the edited labels automatically.

**Choice 3 — Skip** → Update state → `SKIPPED`.

**Choice 4 — Backtrack** → Ask which stage to backtrack to:
```
回溯到：
[a] classify — 重新判断 figure 类型
[p] panel    — 重新检测/选择 panel
[s] segment  — 重新分割（保留 panel）
```
Use `backtrack(state, figure_id, to_stage=...)` to clear downstream data,
then re-run from that stage.

### STAGE 7: Export (on Accept)

**Type**: Bash (Python tool)
**Tool**: Bash
**Output**: SPECFEM files

Use `labels_edited.npz` if it exists (user edited in napari), otherwise fall back
to original `labels.npz`:

```bash
uv run python -c "
from pathlib import Path
from geoseg.session_state import load_session, update_figure, FigureStatus, ExportRecord
from geoseg.controller import run_post_process_and_export

state = load_session('{session_path}')
entry = ...  # find figure

# Prefer edited labels if napari was used
labels_path = entry.segmentation.labels_path
edited = labels_path.parent / 'labels_edited.npz'
if edited.exists():
    labels_path = edited

labels = np.load(str(labels_path))["labels"]
result = run_post_process_and_export(
    labels=labels,
    output_dir='{output_dir}',
)
state = update_figure(state, '{figure_id}',
    status=FigureStatus.EXPORTED,
    export=ExportRecord(tomo_xyz=result['tomo'], parfile_snippet=result['parfile'])
)
save_session(state, '{session_path}')
"
```

Produces:
- `runs/M4/{figure_id}_tomo.xyz`
- `runs/M4/{figure_id}_Par_file_snippet.txt`

## Session State Integration

Always update session state after each significant step:

```python
from geoseg.session_state import (
    load_session, save_session, update_figure,
    ClassificationRecord, PanelSelection, SegmentationRecord,
    FigureStatus,
)

# After classify
state = update_figure(state, figure_id,
    status=FigureStatus.CLASSIFIED,
    classification=ClassificationRecord(...))

# After segment
state = update_figure(state, figure_id,
    status=FigureStatus.SEGMENTED,
    segmentation=SegmentationRecord(...))

# After user accepts
state = update_figure(state, figure_id,
    status=FigureStatus.EXPORTED,
    export=ExportRecord(...))

save_session(state, session_path)
```

## Output Structure

```
{output_dir}/
  {figure_id}/
    panel0/
      labels.npz
      overlay.jpg
      meta.json
      strategy.log
      polygons.geojson
      properties.json
      tomo.xyz
      parfile_snippet.txt
runs/sessions/
  {timestamp}.json   # persistent session state
```

## Constraints

- One figure = one round-trip. Do not proceed to export without explicit user accept.
- Skip non-velocity-model figures immediately (prefer false negative).
- Panel images <300x200 px should be skipped.
- Every significant step writes to session state (recoverable on crash).
- Natural language modify: always map to concrete sandbox action, never guess silently.
