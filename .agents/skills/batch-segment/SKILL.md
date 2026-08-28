---
name: batch-segment
description: >
  [WORKFLOW] Batch process a directory of geophysics figures into SPECFEM velocity models.
  Orchestrated via Dynamic Workflows with parallel stages.
  Pipeline: ingest → classify (parallel) → segment (parallel, ≤5 agents)
  → 集中 review → export. All figures are processed first, then reviewed together.
  Triggers: "batch process", "批量处理", "process directory", "segment all figures"
argument-hint: <directory> [--n-layers=N] [--output-dir=path] [--session=path]
allowed-tools: Bash, Read, Write, Agent
---

# batch-segment

Batch figure → SPECFEM pipeline with 集中 review. Process everything first,
then let the user review all results in one pass.

## Quick Start

```
User: /batch-segment runs/M0.5/ --n-layers=5
Agent: [Stage 1-3: scan → classify all → segment all]
       "5 张目标图已处理完毕，请 review。"
       [展示 5 张 overlay 缩略图 + 质量评分]

User: 1,3,4 接受；2 修改：底层应分两层；5 跳过
Agent: [导出 1,3,4；重跑 2；跳过 5]
       "全部完成。"
```

## Workflow Orchestration

This skill is designed for **Dynamic Workflows**. Codex generates a JS
orchestration script with the following stage graph:

```
STAGE 1: ingest (Bash — scan directory, create session)
  ↓
STAGE 2: classify_all
  └── PARALLEL: for each figure
        └── agent("figure-classify", {image: fig})
  ↓ [barrier: all classify done]
STAGE 3: segment_all
  └── PARALLEL: for each non-skipped figure (max 5 concurrent)
        └── agent("sandbox-segment", {panel: ..., n_layers: N})
  ↓ [barrier: all segment done]
STAGE 4: 集中 review (HITL)
  ↓ [per-figure decisions]
STAGE 5: export (Bash — for all REVIEWED figures)
```

**Key constraints for parallel blocks**:
- `classify_all`: no concurrency limit (lightweight, Read-only)
- `segment_all`: max 5 concurrent agents (memory: ~1.5GB per agent on Mac mini M4)
- Each parallel agent updates session state independently → write contention safe
  because state updates append to `state.workset[i]` (per-figure entry), not shared data.

## Stage Definitions

### STAGE 1: Ingest

**Type**: Bash (Python)
**Tool**: Bash
**Output**: session state JSON

Scan directory for image files (`.png`, `.jpg`, `.jpeg`, `.tiff`).
Create session state:

```python
from geoseg.session_state import create_session, save_session
from pathlib import Path

paths = [str(p) for p in Path(directory).glob("**/*") if p.suffix.lower() in image_exts]
state = create_session(paths)
save_session(state, session_path)
```

Report: `Found {N} images in {directory}. Starting batch processing...`

### STAGE 2: Classify All (PARALLEL)

**Type**: parallel agents
**Tool**: Agent, Read, Write
**Concurrency**: unlimited
**Barrier**: wait for all classify agents to complete

For each figure, spawn a `figure-classify` agent in parallel.
Each agent:
1. Read image → classify
2. Write result to `runs/audit/{fig_id}_classification.json`
3. Update session state → `update_figure(status=CLASSIFIED, classification=...)`

If NOT velocity_model / geological_cross_section:
- `update_figure(status=SKIPPED, skip_reason="...")`

After barrier, report summary:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 Stage 2 完成 — Figure 分类结果
   总计: 12 张
   ✅ 目标图 (velocity_model):     5 张
   ⏭️  已跳过:                     7 张
      - fig2.png: shot_gather
      - fig6.png: waveform_plot
      - ...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### STAGE 3: Segment All (PARALLEL, ≤5 Agents)

**Type**: parallel agents
**Tool**: Agent, Bash, Read, Write
**Concurrency**: max 5
**Barrier**: wait for all segment agents to complete

For each non-skipped figure, spawn a `sandbox-segment` agent.
Each agent performs the full single-figure segment pipeline:
1. Detect panels (Bash — cv_detect)
2. Identify target panel (agent — Read)
3. Crop + remove colorbar (Bash)
4. Autonomous segmentation (≥2 engines, evaluate visually with `visual-audit`, pick best)
5. Generate overlay with **vivid distinct colors** (`_create_overlay`)
   - Default `fill_mode="blend"` (α=0.65, distinct HSV palette over original)
   - Auto-detect and skip background label
   - Pre-merge tiny fragments; thin white boundaries
   - Agent may override to `"solid"` (α=0.85) or `"mask"` (pure map) if figure has low-contrast layers
6. Generate `overlay_legend.jpg` and write `regional_audit.json` (agent-driven audit, no hard gates)
7. Save to `runs/sandbox/{figure_id}/`
8. Update state → `SEGMENTED`

Save session state after each completion.

Progress report every N figures:

```
📦 分割进度: 3/5 完成
   fig1.png ✅ 质量 0.85, 5层, kmeans_full
   fig3.png ✅ 质量 0.91, 4层, ensemble
   fig4.png ⚠️  质量 0.62, 3层, v4_kmeans  [建议 review 时关注]
```

### STAGE 4: 集中 Review (HITL)

**Type**: HITL — two-phase: conversation filtering → napari editing
**Tool**: None (conversation) → Bash (napari)
**Barrier**: all segment agents must complete before this stage

**Design rationale**: Napari is a heavy GUI ( launches per figure, ~1-2s startup).
For batch review, we first filter in conversation, then only launch napari for
figures that actually need editing.

#### Step 4a: 快速浏览（对话内，不启动 napari）

Present all results for quick batch decisions. Agent reads each
`overlay_legend.jpg` and `regional_audit.json` to summarize the key issues for
the user. There is no hard-reject state; the user always has the final say.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Stage 4 — 请 review 以下结果

[1] fig1.png  ✅ 5层 kmeans_full  [overlay]
    └─ audit: 无明显问题
[2] fig3.png  ✅ 4层 edge_guided  [overlay]
    └─ audit: 无明显问题
[3] fig4.png  ⚠️ 5层 v4_kmeans  [overlay]
    └─ audit: label 2 (绿色，中上部) 碎片化；label 3 (蓝色，右侧) 覆盖文字
[4] fig7.png  ✅ 6层 ensemble  [overlay]
    └─ audit: 边界略粗糙，可接受
[5] fig9.png  ⚠️ 2层 v4_kmeans  [overlay]
    └─ audit: 层数不足，底层缺失

输入指令（可多选，逗号分隔）：
- "1,2,4 接受"       → 标记为 REVIEWED，不进 napari
- "3 重跑 segment"   → backtrack 到 segment，重新分割（保留 classify/panel）
- "3 跳过"           → 标记为 SKIPPED
- "5 修改"           → 标记为 NEEDS_EDIT，进入 Step 4b（逐个 napari）
- "5 跳过"           → 标记为 SKIPPED
- "3 回溯到 classify" → 重新从 classify 开始
- "全部接受"          → 所有 ✅ 标记为 REVIEWED

或输入编号查看大图 / audit report: "view 3"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

| 用户指令 | Agent 动作 |
|----------|-----------|
| "N 接受" | `update_figure(status=REVIEWED)`，不启动 napari |
| "N 修改" | `update_figure(status=NEEDS_EDIT)`，进入 Step 4b |
| "N 重跑 segment" | `backtrack(..., to_stage="segment")`，重新分割 |
| "N 跳过" | `update_figure(status=SKIPPED)` |
| "N 回溯到 X" | `backtrack(..., to_stage="X")`，重新跑该 figure 的对应 stage |

**处理建议**:
- 如果 `regional_audit.json` 显示 `retry_labels` 非空，默认建议 "重跑 segment" 或 "修改"
- 没有硬性拒审状态；用户始终拥有最终决定权

#### Step 4b: Napari 逐个编辑（仅 NEEDS_EDIT）

For each figure with `status=NEEDS_EDIT`, launch napari sequentially:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🖱️  Napari 编辑器 [1/2] — fig4.png
   质量 0.62，建议关注：边界碎片化、层数偏少
   正在启动 napari...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Bash launch (blocking until window closes):
```bash
uv run python -m geoseg.modules.editor.napari_app \
  --session {session_path} \
  --figure {figure_id}
```

**User workflow in napari**:
1. Inspect → zoom/pan to verify boundaries
2. Edit → `L`画线分割、`P`画多边形、`S`+`Delete`删线合并、`D`拖拽顶点
3. Close window → auto-saves shapes + recomputed labels

**After napari closes**, agent:
1. Read `labels_edited.npz`
2. Generate new overlay
3. Present in conversation:
   ```
   fig4.png 编辑完成
   变化: 新增 1 条边界，合并 0 个区域
   [展示编辑后的 overlay]
   ```
4. Ask: "接受 / 重新编辑 / 回溯到 segment ?"
   - 接受 → `update_figure(status=REVIEWED)`
   - 重新编辑 → re-launch napari with `labels_edited.npz`
   - 回溯 → `backtrack(...)`

Proceed to next NEEDS_EDIT figure. When all done:

#### Step 4c: 最终确认

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Review 完成
   REVIEWED (待导出): 3 张
   SKIPPED: 1 张
   
   是否进入导出阶段？
   [1] 是 → 导出 SPECFEM
   [2] 返回修改某张图
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### STAGE 5: Export

**Type**: Bash (Python)
**Tool**: Bash
**Input**: all `REVIEWED` figures from session state
**Output**: SPECFEM files per figure

For all `REVIEWED` figures, run post-process + SPECFEM export.
Prefer `labels_edited.npz` if it exists (figure was edited in napari), else
fallback to original `labels.npz`:

```bash
uv run python -c "
from pathlib import Path
from geoseg.session_state import load_session, update_figure, FigureStatus, ExportRecord
from geoseg.controller import run_post_process_and_export

state = load_session('{session_path}')
for entry in state.workset:
    if entry.status != FigureStatus.REVIEWED:
        continue
    labels_path = Path(entry.segmentation.labels_path)
    edited = labels_path.parent / 'labels_edited.npz'
    if edited.exists():
        labels_path = edited
    labels = np.load(str(labels_path))["labels"]
    result = run_post_process_and_export(
        labels=labels,
        output_dir='{output_dir}',
    )
    state = update_figure(state, entry.figure_id,
        status=FigureStatus.EXPORTED,
        export=ExportRecord(tomo_xyz=result['tomo'], parfile_snippet=result['parfile'])
    )
save_session(state, '{session_path}')
"
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📤 Stage 5 — 批量导出
   导出 3 张图:
   - fig1.png → runs/M4/fig1_tomo.xyz
   - fig3.png → runs/M4/fig3_tomo.xyz
   - fig7.png → runs/M4/fig7_tomo.xyz

   跳过 2 张:
   - fig4.png (用户修改后仍未接受)
   - fig9.png (用户跳过)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Update state → `EXPORTED`. Save final session state.

## Session State Path

Default: `runs/sessions/batch_{timestamp}.json`

User can specify `--session` to resume an interrupted batch:

```
User: /batch-segment runs/M0.5/ --session=runs/sessions/batch_20260527.json
Agent: [load existing session, check which stages are incomplete, resume]
```

## Resume Logic

When loading an existing session:
1. Check `get_summary(state)` to see which stages are incomplete.
2. Skip already-classified figures (unless user explicitly requests re-classify).
3. Skip already-segmented figures.
4. Resume from the first uncompleted stage.

```python
from geoseg.session_state import get_summary, list_ready_for_review

summary = get_summary(state)
if summary["pending"] > 0:
    # Resume Stage 2 (classify)
if summary["classified"] > 0:
    # Resume Stage 3 (segment)
if summary["segmented"] > 0:
    # Resume Stage 4 (review)
```

## Constraints

- Max 5 concurrent sandbox-segment agents (Mac mini M4 16GB).
- Save session state after EVERY figure completes (crash recovery).
- Batch size: if >20 figures, warn user and suggest splitting into sub-batches.
- Never auto-export without explicit user review (even for "good" results).
- Natural language modify: same mapping table as `geo-segment` skill.
