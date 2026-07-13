# geoseg v2

> 只做概念模型提取。figure_classifier 宁可误拒也不要误放 observational_data。
> 交互模型：**CLI Human-in-the-Loop**（v0.8）。废弃 Tauri/FastAPI 前端路线。
> 设计稿见 docs/DESIGN.md（**v0.7 已签字，v0.8 CLI 交互模型未入 DESIGN.md**），代码库地图见 docs/CODEBASE.md。

## 架构（Agent-Native + CLI HITL）

Pipeline 由 Claude Code skill 驱动，**agent 直接 Read 看图** + Bash 调用 Python 工具。
交互完全在 Claude Code 对话内完成：agent 自动跑 pipeline → 展示 overlay → **启动 napari 编辑器** → 用户鼠标交互修改边界 → 保存关闭 → agent 重新计算 labels → 确认 → 导出。

```
用户对话触发 skill
    ↓
Agent 自主执行（Bash/Read/Write/Edit 工具）
    ├── figure-classify: agent Read 看图 → 分类 JSON
    ├── cv_detect: Bash 运行 Python 工具函数
    ├── sandbox-segment: agent 自选引擎、视觉评估、融合
    │   ├── strategy_memory: 读历史 → 选引擎 → 跑 segmentation
    │   ├── metrics: 客观指标辅助（VLM 视觉判断为主）
    │   └── 迭代 2+ 引擎，save best result
    └── post_process / exporter: Bash 运行 Python 导出
    ↓
展示 overlay → 启动 napari 编辑器（鼠标画线/删线/拖拽顶点修改边界）
    ↓
用户关闭 napari → agent 重新计算 labels → 展示更新结果
    ↓
确认接受 → 导出 SPECFEM
```

- **Skill 入口**（CLI-native）：
  - `geo-segment`（单图端到端，对话内 HITL）
  - `batch-segment`（批量，先全部跑完再集中 review）
  - `figure-classify`（分类 agent）
  - `sandbox-segment`（自主分割 agent）
- **会话状态**：`geoseg/session_state.py` 持久化每张 figure 的生命周期（pending → classified → segmented → reviewed → exported），支持回溯到 classify/panel/segment 任意上游阶段
- **`vlm_client/` 角色**：schema + prompt 定义库（pydantic）。VLM 调用已迁移至 agent skill，client.py 已 DEPRECATED
- **`controller.py` 角色**：后端组装层工具函数
- ~~`server.py`~~：FastAPI HTTP 后端（v0.7）**已废弃**
- ~~`geoseg-gui/`~~：Tauri 前端（v0.7）**已废弃**
- ~~`gui/`~~：PySide6 视图 **已废弃**
- **并行上限**：≤5 个 segmenter agent（Mac mini M4 16GB）
- **Skill 索引**：`.claude/skills/README.md`

## Dynamic Workflows 编排

Pipeline skills（`geo-segment`、`batch-segment`）已重构为 **Dynamic Workflows** 模式。
Claude Code v2.1.154+ 会自动生成 JS 编排脚本，取代原来的 agent 手动判断下一步。

### Stage Graph

**单图（`geo-segment`）—— 全顺序，无并行**：
```
init → classify → detect → select_panel → segment → present → napari_review → export
```
- `napari_review`: agent Bash 启动 napari（阻塞），用户鼠标编辑边界 → 关闭窗口 → 自动保存 `labels_edited.npz`
- 每个 stage 的输出是下一个 stage 的输入，数据依赖不允许并行。

**批量（`batch-segment`）—— 两阶段并行 + 对话筛选**：
```
ingest
  ↓
PARALLEL classify_all  (无并发限制，Read-only 轻量)
  ↓ [barrier]
PARALLEL segment_all   (max 5 并发，内存瓶颈)
  ↓ [barrier]
对话筛选 (HITL) — 用户批量决定：接受 / 需修改 / 跳过
  ↓
napari 逐个编辑 — 仅对标记为 NEEDS_EDIT 的图启动 napari
  ↓
export (顺序 Bash，优先使用 labels_edited.npz)
```
- 批量 review 先对话筛选，只对需要修改的图启动 napari（避免每张图都开 GUI）

### Workflow 约束

- `batch-segment` 的 `segment_all` **最多 5 并发**（Mac mini M4 16GB ≈ 1.5GB/agent）
- 每个并行 agent 独立更新 session state 的 `workset[i]`，无写冲突
- HITL 阶段（present / review）**不能并行**，必须等用户反馈
- Napari review 是**阻塞式 GUI**：agent Bash 启动 napari → `napari.run()` 阻塞直到窗口关闭 → agent 继续
- Backtrack / Modify 会生成子 workflow：单图重跑 segment → present → napari_review

## Napari 交互模型

Review 阶段使用 **napari-based Shapes-primary 编辑器**（`geoseg/modules/editor/`）。

### 为什么用 napari 而不是自然语言

| 修改类型 | 自然语言（旧） | Napari 编辑器（新） |
|---------|--------------|-------------------|
| "去掉右上角颜色条" | agent 猜 bbox → 重跑 segment | 用户直接画线分割/删除 |
| "底层应分两层" | agent 调 `n_layers` → 重跑引擎 | 用户直接画分割线 |
| "边界太粗糙" | agent 换引擎 → 重跑 | 用户拖拽顶点微调 |
| "中间断层不要拆开" | agent 调用 `merge_labels` | 用户删除分割线 |

**核心优势**：边界级别的精确控制，无需语言描述 → agent 猜测 → 重跑的循环。

### 启动方式

```bash
python -m geoseg.modules.editor.napari_app \
  --labels runs/sandbox/{figure_id}/labels.npz \
  --image runs/sandbox/{figure_id}/panel.png \
  --output-shapes runs/sandbox/{figure_id}/shapes.json \
  --output-labels runs/sandbox/{figure_id}/labels_edited.npz
```

### 用户操作（原生 napari 工具）

| 快捷键 | 工具 | 效果 |
|--------|------|------|
| `L` | Add Line | 画开放线分割区域（两端自动吸附到边界） |
| `P` | Add Polygon | 画闭合多边形创建独立区域 |
| `S` + `Delete` | Select + Delete | 删除边界线，两侧区域合并 |
| `D` | Direct | 拖拽顶点微调边界形状 |
| `Ctrl+S` | 保存 | shapes 自动保存到 `--output-shapes` |
| 关闭窗口 | 退出 | labels 自动重新计算并保存到 `--output-labels` |

### 批量 review 策略

**不对每张图都启动 napari**（启动慢、窗口多）。流程：

1. **对话筛选**：展示所有 overlay 缩略图，用户批量指令 "1,3 接受；2,5 修改；4 跳过"
2. **仅 NEEDS_EDIT 进 napari**：逐个打开，改完一张自动打开下一张
3. **Export 优先使用 `labels_edited.npz`**：自动检测编辑后的文件

## Agent 视觉能力

**所有 agent（主 agent 与 background agent）均具备原生视觉输入能力。** 这不是通过独立 VLM 服务实现的，而是 Claude Code Agent 的底座 LLM 本身已具备视觉理解（image → text）。因此：

- Agent 可直接 `Read` 图像文件进行视觉评估
- Background agent 可并行查看 overlay 结果并返回视觉判断
- 无需额外的 "VLM client" 或 "vision API" 封装
- 视觉评估是 agent 工作流的一等公民，与代码生成、文件编辑同等重要

## 视觉审阅原则

**任何视觉审阅的过程拒绝任何的评分规则和硬编码，必须经过视觉理解。**

- Agent 直接 `Read` 图像文件进行视觉评估，不依赖硬编码阈值、启发式评分或子进程调用外部 VLM/CLI。
- 颜色条 ROI、panel bbox、质量判断、参数调整一律由视觉理解驱动；不允许写 Python 脚本给区域“打分”再取最高分的做法。
- 客观指标（如面积、边界对齐度、碎片数）只能作为诊断信号，不能替代 agent 的视觉判断。
- 若需要迭代参数，复用同一 CLI / skill，通过配置调整；禁止为单次实验新建硬编码脚本。

## 全局约束

- **Schema 改动 = 一次 PR 内更新所有 consumer + 跑 `tests/test_integration_ph01.py`**
- **双管线接口**：通过 `geoseg/pipeline_interfaces.py` 通信，模块不感知上游来源
- **Schema 定义**：`vlm_client/prompts.py` 是 schema + prompt 模板的唯一来源。所有 VLM 语义推理走 `.claude/skills/`（agent-native），禁止写新的 Python client 调 CLI（`client.py` 中 `_call_claude_cli` 已 DEPRECATED）。
- **Agent-Native Pipeline 铁律**：pipeline 必须由 Claude Code Agent 工具纯 agent 驱动（Read 看图 → Bash 调用工具 → 自主决策），**绝不写 Python 批量脚本代劳**。Background agents 是执行单元，脚本是反模式。
- **不开 swarm / agent 团队**（Phase 0 单人，但用 Claude Code 原生 Agent 工具）
- **不创建** `docs/DESIGN.md` / `docs/CODEBASE.md` / 模块 CLAUDE.md **之外的新 markdown**

## Context Window Management

- **自动压缩阈值**：项目 `.claude/settings.json` 已配置 `autoCompactEnabled: true` + `autoCompactWindow: 256000`；上下文接近 256k tokens 时自动压缩。
- **长任务分片**：跨多文件重构或长流程任务优先使用子 agent / Dynamic Workflow，避免单会话题累计过长。
- **上下文归档**：阶段确认后（如 classify → detect 完成），将中间推理归档到 `memory/` 或 session state，不在主对话中保留详细过程。

## 输出规范

每轮回复末尾输出一段**中文 summary**：

```
---
本轮小结：做了什么 + 当前状态 + 下一步建议/等待用户决策。
```
