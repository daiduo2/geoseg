# geoseg

> 只做地球物理概念模型提取。`figure_classifier` 宁可误拒，也不要误放 `observational_data`。
> 当前主路线是 **CLI Human-in-the-Loop + Agent-Native workflow**。Tauri/FastAPI 前端路线已废弃。
> 代码库地图见 `docs/CODEBASE.md`，skill 索引见 `.claude/skills/README.md`。

## Architecture

- `src/geoseg/`：产品代码。`core/` 稳定契约、`pipeline/` 新编排、`modules/segment_engines/` 引擎族、`modules/post_process/`、`modules/exporter/`、`cli/`、`api/` 兼容入口。
- `experiments/`：实验代码（脚本、一次性实验、benchmark）。**没有**晋升到 `src/geoseg/` 之前保持在这里。
- `tests/`、`examples/`、`scripts/`、`docs/`：分别承载测试、示例、辅助脚本、文档。
- `controller.py` 是兼容入口；**新编排逻辑写到 `src/geoseg/pipeline/`**。
- `pipeline_interfaces.py` 是兼容导出层；新代码用 `src/geoseg/core/models.py`。

### 跨模块访问边界

- 产品代码禁止直接 import `geoseg.modules.segment_engines._shared`、`segment_engines.internal`、`router._run_engine` 等私有符号。
- 按名称运行分割引擎：`from geoseg.api import run_engine`（不要手写 engine dispatch）。
- 端到端单图处理：`from geoseg.api import run_pipeline`；完整编排只在 `pipeline/run.py`。
- scripts / examples / experiments 常用 helper（CV、VLM、engine）走 `from geoseg.api import ...`，而不是从 `internal/` 拉。
- 需要 overlay / distinct colors / saturation ratio / preprocessing 等跨模块能力，从 `geoseg.core.image_ops` 导入。

## Segmentation Engines

引擎族在 `src/geoseg/modules/segment_engines/`，**dispatch 单一来源是 `registry.py`**：每个引擎是一条 `EngineSpec`，记录 `name`、callable path、`requires_reps`、`fallback_to` 等。新增引擎必须在 registry 注册。

| 引擎名 | 路径 | `requires_reps` | 用途 |
|--------|------|-----------------|------|
| `v4_kmeans` | `v4_kmeans.py` / `v4/` | False | 默认主路径（row_median 预处理） |
| `v4_kmeans_colorbar` | `v4_kmeans.py` | False | 用 colorbar 反推 color zones |
| `v4_kmeans_pastel` | `v4_kmeans.py` | False | pastel 退色图的回退 |
| `grayscale_agglomerative` | `grayscale.py` | False | 灰度层次聚类 |
| `horizon_refinement` | `horizon_refinement.py` / `horizon/` | False | 后处理细化 |
| `kmeans_full` | `kmeans_full.py` | **True** | 全图 K-Means（接受 VLM 种子） |
| `edge_guided` | `edge_guided.py` | **True** | 边缘引导分割（接受 VLM 种子） |
| `edge_grow` | `edge_grow.py` | **True** | 边缘区域生长（接受 VLM 种子） |
| `ensemble` | `ensemble.py` | **True** | 多引擎融合 |
| `tubular` | `tubular_structure.py` | False | 管状结构 |

`requires_reps=True` 的引擎：调用方传 `reps: list[{color_name, representative_point: {x,y}}]`；不传则 runner 退到 fallback（默认 `v4_kmeans`）。这是 silent fallback，会让"需要 VLM 的引擎"在 benchmark 里假性跟 `v4_kmeans` 跑出相同结果。**任何 benchmark 都要给这些引擎喂 reps，否则对比无意义**（见下文 synthetic_benchmark）。

## Skills

Skill 索引见 `.claude/skills/README.md`。常用：

- `geo-segment`：单图端到端，对话内 HITL。
- `batch-segment`：批量处理，**segment 阶段最多 5 并发**（Dynamic Workflow 在 stage 3 的 orchestration 脚本里强制，约束写在 `.claude/skills/batch-segment/SKILL.md`）。
- `figure-classify`：agent 直接看图分类。
- `sandbox-segment`：agent 自选引擎、评估、必要时融合。
- `visual-audit`：agent 视觉审阅 overlay-with-legend，输出结构化结果。
- `segment-export`：导出已接受的 segmentation。
- `preprocess-artifact`：红色断层线 / 黑色十字等 artifact 预处理工具。
- `module-demo`：模块级 demo 入口。
- `schema-bump`：pydantic schema 跨版本迁移辅助。

具体 workflow graph、输入输出、失败处理写在对应 SKILL.md；根 CLAUDE.md 不维护重复流程细节。

## Visual Reasoning Rules

- 所有语义判断和视觉审阅由 Claude Code agent 直接读图完成。
- 禁止新增 Python/CLI VLM client 代替 agent 视觉判断；`src/geoseg/modules/vlm_client/` 只保留 schema/prompt 相关内容。
- 客观指标可以作为诊断信号，但不能替代视觉判断——**mIoU/Boundary-F1 高不等于视觉正确**（见 synthetic_benchmark 教训：Hungarian mIoU 在"几何对、cluster 数不匹配"时低估；fragment 计数在"被欠分割"时假性优秀）。
- 颜色条 ROI、panel bbox、质量判断、是否接受结果等决策不能靠一次性硬编码评分脚本拍板。
- 确定性的图像处理、导出、测试、批处理 glue code 可以写成 Python 工具；语义判断不能写脚本代劳。
- **视觉对比走图而不是图床**：引擎对比、benchmark 评估、overlay 验收都先生成 PNG，agent 直接读 PNG（`overview_grid.png`、`per_engine/p<i>_<engine>.png`）。

## Engine Comparison: synthetic_benchmark 模式

`experiments/synthetic_benchmark/` 是当前引擎对比的范式落地。任何新引擎 / 现有引擎调参前先在这里跑一遍。

| 模块 | 职责 |
|------|------|
| `configs.py` | `BenchmarkConfig` + `PanelConfig`（12 个 panel 覆盖 colormap / dip / fault / channel / unconformity 组合） |
| `generator.py` | BSM-inspired 概念模型合成 + GT label + noise 应用（共用 `geoseg.core.image_ops`） |
| `noise.py` | JPEG 压缩 / text blocks / contour lines / 颜色条等真实图像瑕疵 |
| `evaluate.py` | mIoU / pixel_acc / Boundary-F1（Hungarian mapping 解 cluster 标号置换） |
| `vlm_seeds.py` | `synthesize_reps` / `synthesize_noisy_reps`：因为 benchmark 拿不到真实 VLM 输出，用 GT 标签的 centroid + 调色板名字合成"完美 VLM 种子" |
| `runner.py` | 一次合成 reps 在所有引擎之间共享，公平 head-to-head |
| `visualize.py` | 无 matplotlib 的可视化：per-engine pair、per-input、overview grid |

### 已知视觉事实（截至 2026-09）

- `v4_kmeans` 在 viridis 简单 6 层上视觉崩盘成白噪声点（mIoU 0.83 但图是散的）—— v4 对 viridis 唯一脆弱。
- `grayscale_agglomerative` 在灰度 3 层上视觉最干净（mIoU 0.99）。
- `edge_grow` 在 busy + unconformity 板上视觉完全找不到结构（mIoU 0.32-0.46）。
- `v4_kmeans_colorbar` 在 unconformity 板视觉对、几何对，但 mIoU 仅 0.70——**mIoU 数字会低估几何正确性**。

## Module Boundaries

- `src/geoseg/core/`：稳定数据模型、路径和配置契约。
- `src/geoseg/pipeline/`：pipeline stage 编排。可组合模块，但不内嵌算法细节。
- `src/geoseg/modules/segment_engines/`：分割引擎族。新增引擎走 registry/runner/policy/retry 边界。
- `src/geoseg/modules/post_process/`、`modules/exporter/`：确定性后处理与导出。
- `src/geoseg/modules/editor/`：napari 编辑器 adapter。
- `src/geoseg/cli/`：可打包命令行入口。
- `src/geoseg/compat/api/`：FastAPI HTTP app（兼容入口，不作为产品前端）。
- `scripts/`：开发/审计/批处理辅助脚本，不作为稳定 API。
- `experiments/`：实验代码。实验可以调用内部细节，但进入 `src/geoseg/` 前必须满足下面的 Promotion Rules。

## Experiment Promotion Rules

实验代码进入 `src/geoseg/` 前必须同时满足：

1. **稳定函数入口**：清晰的输入/输出类型，调用方式文档化。
2. **最小测试**：覆盖核心行为，`pytest` 全绿。
3. **无硬编码路径**：不写死 `runs/`、论文私有路径、本机绝对路径。
4. **不依赖私有符号**：产品代码不得依赖 `_run_engine`、`_shared`、`internal/` 等；要能力先暴露稳定 API。
5. **新 segmentation engine 必须注册**：在 `src/geoseg/modules/segment_engines/registry.py` 加 `EngineSpec`。
6. **更新 CODEBASE / 模块 CLAUDE.md**：不散落新的长期说明文件，只更新 `docs/CODEBASE.md`、`docs/DESIGN.md` 或现有模块 `CLAUDE.md`。

## Global Constraints

- Schema/模型改动必须同步所有 consumer，并至少运行 `tests/test_integration_ph01.py`；较大改动运行完整 `uv run pytest`。
- `batch-segment` 的 segmenter agent **并发上限是 5**，写在 `.claude/skills/batch-segment/SKILL.md` 的 Dynamic Workflow stage 3。
- HITL 阶段不能并行——必须等待用户反馈。
- Napari review 是阻塞式 GUI：agent 启动 napari，用户保存/关闭后 agent 继续。
- 不新增 Tauri / FastAPI / PySide6 产品前端；`src/geoseg/compat/server.py`、`src/geoseg/compat/gui/` 只保留历史兼容/本地调试代码。
- 不新增长期 markdown——归入 `docs/CODEBASE.md`、`docs/DESIGN.md`、现有 skill 文档或模块 `CLAUDE.md`。
- 日常任务用 `make`：`make test` / `make check` / `make map` / `make segment IMAGE=...` / `make bench`；不要再靠记忆翻 `scripts/`。
- git 提交粒度按功能，不按文件；commit message 解释 *why* 和 *what visual fact* 而不只是 *what changed*。

## Response Convention

每轮回复末尾输出中文 summary：

```text
---
本轮小结：做了什么 + 当前状态 + 下一步建议/等待用户决策。
```
