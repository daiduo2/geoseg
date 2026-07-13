# geoseg 组装层

> 本目录下的 `controller.py`、`batch_processor.py` 为后端组装层。
> `session_state.py` 为 CLI 交互模型的持久化状态层。
> `gui/` 和 `server.py` 已废弃（Tauri/FastAPI 前端路线已终止）。
> 启动 cc 时：`cd geoseg && cc` 会自动加载本文件 + 根 CLAUDE.md。

## Build / Test

```bash
# 单图端到端流水线
python -m geoseg.controller_demo

# 批量目录处理
python -m geoseg.batch_processor_demo

# 会话状态 demo
python -m geoseg.session_state_demo

# 接口契约 demo
python -m geoseg.pipeline_interfaces_demo
```

## 架构要点

- `pipeline_interfaces.py`：模块间接口契约（TypedDict + Protocol）
- `session_state.py`：**CLI HITL 会话状态**。持久化每张 figure 的生命周期（pending → classified → segmented → reviewed → exported），支持回溯到上游阶段（classify / panel / segment）
- `controller.py`：`run_pipeline(img_rgb, config) → dict` 是组装层的核心 API，串联 classify → segment → post-process → export
- `batch_processor.py`：基于 `controller.run_pipeline` 的批量封装，支持 resume、单图错误隔离、JSON summary
- ~~`server.py`~~：FastAPI HTTP 后端（v0.7）**已废弃**。前后端解耦不再必要，交互完全在 Claude Code 对话内完成
- ~~`gui/`~~：PySide6 交互视图 **已废弃**。`geoseg-gui/`（Tauri）同步废弃

## CLI Human-in-the-Loop 交互模型（v0.8）

**核心原则**：Agent 全自动跑 pipeline → 展示 overlay → 对话内等用户反馈 → 自然语言修改 → 现场 sandbox 重跑。

### 单图流程（`geo-segment` skill）

```
用户: /geo-segment fig1.png
Agent: [classify → detect → segment] → 展示 overlay
       "Accept / Modify / Skip / Backtrack ?"
用户: 修改。去掉颜色条，底层分两层。
Agent: [re-segment with mask + n_layers+1] → 展示新 overlay
       "Accept / Modify / Skip / Backtrack ?"
用户: 接受
Agent: [export SPECFEM]
```

### 批量流程（`batch-segment` skill）

5-stage: **ingest → classify → segment → 集中 review → export**

- Stage 1-3 全自动，Stage 4 全部完成后逐个/批量 review
- 用户可批量操作："1,3,4 接受；2 修改：...；5 跳过"
- 每张图独立回溯，不影响其他图

### 自然语言 → 操作映射

| 用户说 | Agent 解析 | Sandbox 操作 |
|--------|-----------|-------------|
| "去掉右上角颜色条" | Exclude colorbar | Mask colorbar bbox, re-segment |
| "底层应分两层" | Increase layer count | `n_layers += 1`, retry |
| "中间断层不要拆开" | Merge layers | `merge_labels(a, b)` |
| "边界太粗糙" | Prefer smooth | Switch to `edge_guided` |
| "用灰度分割试试" | Different engine | Retry with `grayscale` |
| "左边panel才是目标" | Wrong panel | Re-crop target, re-segment |
| "回溯到 classify" | Backtrack upstream | `backtrack(state, fig, classify)` |

### 会话状态数据结构

```python
SessionState
├── session_id, created_at
└── workset: List[FigureEntry]
    ├── figure_id, source_path, status: FigureStatus
    ├── classification: ClassificationRecord  (figure_type, confidence)
    ├── panels: PanelSelection  (detected[], target_panel_id)
    ├── segmentation: SegmentationRecord  (engine, n_layers, quality_score, attempts[])
    └── export: ExportRecord  (tomo_xyz, parfile_snippet)
```

状态流转：`PENDING → CLASSIFIED → SEGMENTED → REVIEWED → EXPORTED`
回溯时清除下游数据：`backtrack(to_stage=classify)` 清空 classification 及以下所有字段。

## 近期进展

### CLI HITL 交互模型（v0.8，已完成）

- **废弃 Tauri + FastAPI 前端**：v0.7 的 6 周前端工作量全部终止
- **新增 `session_state.py`**：pydantic 持久化状态，支持 CRUD + 回溯
- **更新 Skills**：`geo-segment` 和 `batch-segment` SKILL.md 重写为 CLI-native 交互流程
- **关键决策**：human-in-the-loop 只在 overlay 确认环节；自然语言修改可回溯到 classify/panel/segment 任意上游阶段

### Agent-Native 闭环反馈（已完成）

- **Skills**：`geo-segment`、`figure-classify`、`sandbox-segment`、`batch-segment` 已部署
- **Rules**：`architecture.md` 定义模块边界与 Agent-Driven Pipeline 约束
- **Metrics**：`metrics.py` 移除物理先验偏见，仅报告客观事实（n_layers、boundary_alignment、fragments、noise）
- **Strategy Memory**：`strategy_memory.py` 记录引擎选择历史，`analyze_batch()` 自动提取策略模板
- **Meta-Learning**：每次 batch 后自动更新 `runs/sandbox/strategy_templates.json`

### 大规模 Batch 测试（已完成）

- 5 agents × 20 张图像并行处理，共积累 33 条 strategy memory 记录
- 提取 2 个策略模式：
  - `mixed + high edge` → ensemble（100%，conf=0.55）
  - `vivid + high edge` → ensemble（100%，conf=1.0）
- v4_kmeans 成功率 90%，ensemble 100%

### 高斯模糊预处理（已完成）

- `_shared.py` 新增 `adaptive_blur()`（sigma 自适应 0.5-2.0）+ `estimate_noise_level()`（基于 edge_density）
- `parallel_segment.py` 集成预处理决策：噪声高时自动对比原图/blur 版本，overlay 始终基于原图
- blur 对 v4_kmeans 碎片化有显著改善（page_011：frag 25.5% → 7.8%，score +10.8%）
- blur 对 ensemble 帮助有限（ensemble 本身已很稳健）
