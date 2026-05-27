---
name: geo-segment
description: >
  Convert a published geophysics interpretation figure into a SPECFEM-ready
  velocity zone model. CLI-native end-to-end entry point with human-in-the-loop
  at overlay review. Agent auto-runs pipeline, presents result, waits for user
  feedback in natural language. Supports backtracking to upstream stages.
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

## Workflow

### Step 0: Initialize Session State

If `--session` provided, load existing session; else create a new one.
Save path defaults to `runs/sessions/{timestamp}.json`.

```python
from geoseg.session_state import create_session, save_session

state = create_session([image_path])
save_session(state, session_path)
```

### Step 1: Auto-Run Pipeline (Silent)

Execute without user interaction:

1. **Classify** — Read image, decide if velocity_model / geological_cross_section.
   - If skip: update state → `SKIPPED`, report reason, STOP.
2. **Detect Panels** — Bash inline `cv_detect.panel_detector`.
3. **Identify Target Panel** — Read image, pick primary panel (e.g. "inverted model").
4. **Crop + Remove Colorbar** — Bash inline crop + `colorbar_extractor`.
5. **Autonomous Segmentation** — Activate `sandbox-segment` behavior.
   - Try ≥2 engines, evaluate visually, pick best.
   - Save to `runs/sandbox/{figure_id}/`.
6. **Update State** — `update_figure(status=SEGMENTED, segmentation=...)`.

### Step 2: Present Result

Show a concise summary + overlay image:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 fig1.png  分割完成
   类型: velocity_model (置信度 0.92)
   Panels: 3 → 目标: #1 "(b) Inverted model"
   引擎: kmeans_full → 5 层
   质量: 0.85

[Read 展示 overlay.png]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 3: Ask for Feedback

Present choices:

```
请选择：
[1] ✅ 接受 → 导出 SPECFEM
[2] ✏️  修改 → 描述问题（自然语言）
[3] ⏭️  跳过 → 标记为 SKIPPED
[4] 🔙 回溯 → 重新 classify / panel / segment
```

**Choice 1 — Accept**: Run post-process + export. Update state → `EXPORTED`.

**Choice 2 — Modify**: Parse natural language into actionable changes.

| User says | Agent parses | Sandbox action |
|-----------|-------------|----------------|
| "去掉右上角颜色条" | Exclude colorbar region | Mask colorbar bbox, re-segment |
| "底层应分两层" | Increase layer count | `n_layers += 1`, retry engine |
| "中间断层不要拆开" | Merge two layers | `merge_labels(a, b)` |
| "边界太粗糙" | Prefer smooth boundaries | Switch to `edge_guided` |
| "用灰度分割试试" | Different engine strategy | Retry with `grayscale` |
| "左边panel才是目标" | Wrong target panel | `target_panel_id -= 1`, re-crop, re-segment |
| "红色区域Vp值不对" | Wrong property mapping | Fix color→Vp/Vs mapping |

After modify: re-run sandbox, present new overlay, ask again.

**Choice 3 — Skip**: Update state → `SKIPPED`. Ask for skip reason (optional).

**Choice 4 — Backtrack**: Ask which stage to backtrack to.

```
回溯到：
[a] classify — 重新判断 figure 类型
[p] panel    — 重新检测/选择 panel
[s] segment  — 重新分割（保留 panel）
```

Use `backtrack(state, figure_id, to_stage=...)` to clear downstream data,
then re-run from that stage.

### Step 4: Export (on Accept)

```bash
python -c "
from geoseg.session_state import load_session, update_figure, FigureStatus, ExportRecord
from geoseg.controller import run_post_process_and_export

state = load_session('{session_path}')
entry = ...  # find figure
result = run_post_process_and_export(
    labels_path=entry.segmentation.labels_path,
    panel_path=entry.source_path,
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
    report.json
    panel/
      labels.npy
      overlay.png
      meta.json
      strategy.log
      polygons.geojson
      properties.json
      tomo.xyz
      Par_file_snippet.txt
runs/sessions/
  {timestamp}.json   # persistent session state
```

## Constraints

- One figure = one round-trip. Do not proceed to export without explicit user accept.
- Skip non-velocity-model figures immediately (prefer false negative).
- Panel images <300x200 px should be skipped.
- Every significant step writes to session state (recoverable on crash).
- Natural language modify: always map to concrete sandbox action, never guess silently.
