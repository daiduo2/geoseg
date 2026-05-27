---
name: batch-segment
description: >
  Batch process a directory of geophysics figures into SPECFEM velocity models.
  5-stage pipeline: ingest → classify (parallel) → segment (parallel, ≤5 agents)
  →集中 review → export. All figures are processed first, then reviewed together.
  Triggers: "batch process", "批量处理", "process directory", "segment all figures"
argument-hint: <directory> [--n-layers=N] [--output-dir=path] [--session=path]
allowed-tools: Bash, Read, Write, Agent
---

# batch-segment

Batch figure → SPECFEM pipeline with集中 review. Process everything first,
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

## 5-Stage Pipeline

### Stage 1: Ingest

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

### Stage 2: Classify (Parallel via Agent Spawning)

Spawn figure-classify agents for each image (or process sequentially if N is small).

```python
# For each figure in state.workset:
#   Agent reads image → classify → update state
#   If NOT velocity_model / geological_cross_section:
#       update_figure(status=SKIPPED, skip_reason="...")
```

After all classified, report summary:

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

### Stage 3: Segment (Parallel, ≤5 Agents)

For each non-skipped figure, spawn a `sandbox-segment` agent (max 5 concurrent).
Each agent:
1. Detect panels
2. Identify target panel
3. Crop + remove colorbar
4. Run autonomous segmentation (≥2 engines, pick best)
5. Generate overlay with **vivid distinct colors** (`_create_overlay`)
   - Default `fill_mode="blend"` (α=0.65, distinct HSV palette over original)
   - Auto-detect and skip background label
   - Pre-merge tiny fragments; thin white boundaries
   - Agent may override to `"solid"` (α=0.85) or `"mask"` (pure map) if figure has low-contrast layers
6. Save to `runs/sandbox/{figure_id}/`
7. Update state → `SEGMENTED`

Save session state after each completion.

Progress report every N figures:

```
📦 分割进度: 3/5 完成
   fig1.png ✅ 质量 0.85, 5层, kmeans_full
   fig3.png ✅ 质量 0.91, 4层, ensemble
   fig4.png ⚠️  质量 0.62, 3层, v4_kmeans  [建议 review 时关注]
```

### Stage 4: 集中 Review

**All figures are segmented. Present results for review.**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Stage 4 — 请 review 以下结果

[1] fig1.png  ✅ 质量 0.85  5层  [overlay]
[2] fig3.png  ✅ 质量 0.91  4层  [overlay]
[3] fig4.png  ⚠️  质量 0.62  3层  [overlay]  ← 建议修改
[4] fig7.png  ✅ 质量 0.78  6层  [overlay]
[5] fig9.png  ⚠️  质量 0.58  2层  [overlay]  ← 建议修改

输入指令（可多选，逗号分隔）：
- "1,2,3 接受"       → 标记为 REVIEWED，待导出
- "4 修改：..."      → 自然语言修改，agent 重新 segment
- "5 跳过"           → 标记为 SKIPPED
- "4 回溯到 classify" → 重新从 classify 开始
- "全部接受"          → 所有 ✅ 标记为 REVIEWED

或输入编号查看大图: "view 3"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Modify workflow** (same as geo-segment single-figure):
- Parse natural language → concrete sandbox action
- Re-run segmentation for that figure only
- Present new overlay
- Ask again for that figure

**Backtrack workflow**:
- "N 回溯到 [classify/panel/segment]"
- Use `backtrack(state, figure_id, to_stage=...)`
- Re-run from that stage for that figure only
- Other figures unaffected

After review, save state. All figures should be either `REVIEWED`, `SKIPPED`, or `SEGMENTED` (if modify/backtrack created new results awaiting review).

### Stage 5: Export

For all `REVIEWED` figures, run post-process + SPECFEM export.

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
