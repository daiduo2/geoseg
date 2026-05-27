# geoseg v2

> 只做概念模型提取。figure_classifier 宁可误拒也不要误放 observational_data。
> 交互模型：**CLI Human-in-the-Loop**（v0.8）。废弃 Tauri/FastAPI 前端路线。
> 设计稿见 docs/DESIGN.md（**v0.7 已签字，v0.8 CLI 交互模型未入 DESIGN.md**），代码库地图见 docs/CODEBASE.md。

## 架构（Agent-Native + CLI HITL）

Pipeline 由 Claude Code skill 驱动，**agent 直接 Read 看图** + Bash 调用 Python 工具。
交互完全在 Claude Code 对话内完成：agent 自动跑 pipeline → 展示 overlay → 停下来等用户自然语言反馈 → 现场 sandbox 修改 → 循环直到接受 → 导出。

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
展示 overlay → 用户自然语言反馈（Accept / Modify / Skip / Backtrack）
    ↓
循环直到接受 → 导出 SPECFEM
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

## 全局约束

- **Schema 改动 = 一次 PR 内更新所有 consumer + 跑 `tests/test_integration_ph01.py`**
- **双管线接口**：通过 `geoseg/pipeline_interfaces.py` 通信，模块不感知上游来源
- **Schema 定义**：`vlm_client/prompts.py` 是 schema + prompt 模板的唯一来源。所有 VLM 语义推理走 `.claude/skills/`（agent-native），禁止写新的 Python client 调 CLI（`client.py` 中 `_call_claude_cli` 已 DEPRECATED）。
- **Agent-Native Pipeline 铁律**：pipeline 必须由 Claude Code Agent 工具纯 agent 驱动（Read 看图 → Bash 调用工具 → 自主决策），**绝不写 Python 批量脚本代劳**。Background agents 是执行单元，脚本是反模式。
- **不开 swarm / agent 团队**（Phase 0 单人，但用 Claude Code 原生 Agent 工具）
- **不创建** `docs/DESIGN.md` / `docs/CODEBASE.md` / 模块 CLAUDE.md **之外的新 markdown**

## 输出规范

每轮回复末尾输出一段**中文 summary**：

```
---
本轮小结：做了什么 + 当前状态 + 下一步建议/等待用户决策。
```
