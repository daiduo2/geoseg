# Architecture Constraints

## Scope

- **只做概念模型提取**。不做反射振幅图正演、不做波形反演、不做全波形反演参数调优。
- **figure_classifier 宁可误拒也不要误放 observational_data**。假阴性可接受，假阳性不可接受。

## Interface Contracts

- **双管线接口**：所有模块通过 `geoseg/pipeline_interfaces.py` 通信，模块内部不感知上游来源（Pipeline A 人工 / Pipeline B Agent）。
- **Schema 改动 = 一次 PR 内更新所有 consumer + 跑 `tests/test_integration_ph01.py`**。不允许只改 schema 不改 consumer。
- **Schema 定义**：`vlm_client/prompts.py` 是 schema + prompt 模板的唯一来源。所有 VLM 语义推理走 `.claude/skills/`（agent-native），禁止写新的 Python client 调 CLI（`client.py` 中 `_call_claude_cli` 已 DEPRECATED）。

## Module Boundaries

| Module | Responsibility | Protocol |
|--------|---------------|----------|
| `cv_detect` | Panel detection, colorbar extraction | `PanelDetector` |
| `segment_engines` | Segmentation algorithm routing & execution | `Segmenter` |
| `vlm_client` | Schema + prompt definitions (pydantic). LLM calls migrated to agent skills | `QualityReviewer` (legacy) |
| `post_process` | Polygon extraction, property assignment, SPECFEM export | `PostProcessor` |

- `cv_detect` 不依赖 VLM 给的坐标（VLM 协议禁止给坐标）。
- `segment_engines` 不做 panel 检测。
- `vlm_client` 不做分割、不做 panel 检测。

## Agent-Driven Pipeline

- **废弃硬编码路由**：`router.py` 的 `select_engine` 和 `_RETRY_CHAIN` 已废弃。分割策略由 `sandbox-segment` skill 驱动的 agent 自主决定。
- **废弃 CLI 调用**：`vlm_client/client.py` 中的 `_call_claude_cli`（`claude -p` subprocess）已废弃。所有 VLM 能力走 skill + agent。
- **并行上限**：批处理时最多 spawn 5 个 segmenter agent（Mac mini M4 16GB ≈ 1.5GB agent 内存预算）。

## Data Flow

```
PDF / Image
    ↓
[Agent: figure-classify] → velocity_model / skip
    ↓
[cv_detect] → panels + colorbar
    ↓
[Agent: sandbox-segment] → best labels (agent 自主选择引擎、评估、融合)
    ↓
[post_process] → polygons + properties + SPECFEM export
```

## File Size

- 单个文件不超过 500 行。超过则拆分。
