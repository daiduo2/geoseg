# geoseg v2

> 只做地球物理概念模型提取。`figure_classifier` 宁可误拒，也不要误放 `observational_data`。
> 当前主路线是 **CLI Human-in-the-Loop + Agent-Native workflow**。Tauri/FastAPI 前端路线已废弃。
> 代码库地图见 `docs/CODEBASE.md`，skill 索引见 `.Codex/skills/README.md`。

## Current Architecture

- 源码位于 `src/geoseg/`，测试位于 `tests/`，示例位于 `examples/`，实验位于 `experiments/`。
- Pipeline 由 Codex skill 驱动：agent 直接读取图像做语义/视觉判断，Python 只提供确定性工具函数。
- 核心交互路径：agent 跑 pipeline -> 展示 overlay -> 必要时启动 napari -> 用户编辑 labels -> agent 复算/确认 -> 导出。
- 会话状态由 `src/geoseg/session_state.py` 管理。
- 稳定数据契约优先放在 `src/geoseg/core/models.py`；`src/geoseg/pipeline_interfaces.py` 仅作为兼容导出层保留。
- `src/geoseg/controller.py` 是兼容入口；新编排逻辑应优先放在 `src/geoseg/pipeline/`。

## Skill Boundaries

- `geo-segment`：单图端到端，对话内 HITL。
- `batch-segment`：批量处理，segment 阶段最多 5 并发。
- `figure-classify`：agent 直接看图分类。
- `sandbox-segment`：agent 自选分割引擎、评估结果、必要时融合。
- `visual-audit`：agent 视觉审阅 overlay-with-legend，输出结构化审阅结果。
- `segment-export`：导出已接受的 segmentation。
- `preprocess-artifact`：红色断层线/黑色十字等 artifact 预处理工具。

具体 workflow graph、输入输出和失败处理写在对应 skill 内；根指令不维护重复流程细节。

## Visual Reasoning Rules

- 所有语义判断和视觉审阅必须由 Codex agent 直接读图完成。
- 禁止新增 Python/CLI VLM client 代替 agent 视觉判断；`src/geoseg/modules/vlm_client/` 只保留 schema/prompt 相关内容。
- 客观指标可以作为诊断信号，但不能替代视觉判断。
- 颜色条 ROI、panel bbox、质量判断、是否接受结果等决策不能靠一次性硬编码评分脚本拍板。
- 确定性的图像处理、导出、测试、批处理 glue code 可以写成 Python 工具；语义判断不能写脚本代劳。

## Module Boundaries

- `src/geoseg/core/`：稳定数据模型、路径和配置契约。
- `src/geoseg/pipeline/`：pipeline stage 编排。可以组合模块，但不应内嵌算法细节。
- `src/geoseg/modules/segment_engines/`：分割引擎族。新增引擎应走 registry/runner/policy/retry 边界。
- `src/geoseg/modules/post_process/` 和 `src/geoseg/modules/exporter/`：确定性后处理与导出。
- `src/geoseg/modules/editor/`：napari 编辑器 adapter。
- `src/geoseg/cli/`：可打包命令行入口。
- `scripts/`：开发/审计/批处理辅助脚本，不作为稳定 API。
- `experiments/`：实验代码。实验可以调用内部细节，但进入 `src/geoseg/` 前必须完成晋升门槛。

## Experiment Promotion Rules

实验代码进入 `src/geoseg/` 前必须同时满足：

- 有稳定函数入口和清晰输入/输出类型。
- 有最小测试覆盖核心行为。
- 不直接写死 `runs/`、论文私有路径或本机路径。
- 产品代码不得依赖 `_run_engine`、`_shared` 等私有符号；需要能力时先暴露稳定 API。
- 新 segmentation engine 必须在 `registry.py` 或相应 pipeline config 中注册。
- 更新 `docs/CODEBASE.md` 或现有模块 `AGENTS.md`，不要散落新的长期说明文件。

## Global Constraints

- Schema/模型改动必须同步所有 consumer，并至少运行 `tests/test_integration_ph01.py`；较大改动运行完整 `uv run pytest`。
- `batch-segment` 的 segmenter agent 并发上限是 5。
- HITL 阶段不能并行，必须等待用户反馈。
- Napari review 是阻塞式 GUI：agent 启动 napari，用户保存/关闭后 agent 继续。
- 不新增 Tauri/FastAPI 产品前端；`src/geoseg/server.py` 仅作为历史兼容/本地调试代码。
- 不新增长期 markdown，除非归入 `docs/CODEBASE.md`、`docs/DESIGN.md`、现有 skill 文档或模块 `AGENTS.md`。

## Response Convention

每轮回复末尾输出中文 summary：

```text
---
本轮小结：做了什么 + 当前状态 + 下一步建议/等待用户决策。
```
