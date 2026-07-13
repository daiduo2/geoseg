# vlm_client — 模块契约

> **Schema + Prompt 定义库**。pydantic v2 schema 与 prompt 模板的唯一来源。  
> **VLM 调用已全面迁移至 agent skill**（`.claude/skills/figure-classify`、`sandbox-segment`）。  
> **QualityReviewer Protocol 实现者**（legacy / stub mode 使用）。

## 角色转变

| 时期 | 角色 | 说明 |
|------|------|------|
| v1 | 唯一 LLM 出口 | Python 代码通过 `_call_claude_cli` 调 `claude -p` subprocess |
| **v2（当前）** | **Schema 定义库** | agent skill 直接 Read 看图 + Bash 调用工具；本模块保留 schema 供 skill 与 legacy code 引用 |

## 3 个 Schema / 调用点

| 函数 | 输入 | 输出 schema | 用途 | Protocol |
|------|------|-------------|------|----------|
| `classify_figure` | 图片 | `FigureClassification` | 语义分类 | `QualityReviewer` |
| `review_page_overview` | 嵌入图 + caption | `PageOverview` | 颜色分区提示，辅助 colorbar | `QualityReviewer` |
| `review_segmentation_quality` | 原图 + 叠加对比图 | `SegmentationQualityReview` | 分割质量审查（VLM-SQ） | — |

**注意**：上述函数仍可在 `mode="stub"` 或 legacy code 中调用。新代码（skill）应直接使用 `prompts.py` 中的 schema 定义，由 agent 自主完成语义推理。

Schema 详见 `docs/DESIGN.md` §4 和 `geoseg/pipeline_interfaces.py`。改 schema = 一次 PR 内更新所有 consumer + 跑 `tests/test_integration_ph01.py`。

## 已废弃（代码中已标记 DEPRECATED）

| 函数 | 状态 | 替代方案 |
|------|------|----------|
| `_call_claude_cli` | DEPRECATED | agent skill 直接 Read 看图 |
| `_call_with_retry` | DEPRECATED | agent skill 自主决策 + 策略记忆 |

## 输入输出约束

- **VLM 不给坐标**：只返回语义描述（`description` / `repair_hints` / `fix_hints`），精确 bbox / 像素级定位由 `cv_detect` 负责
- **不黑盒决策**：每次调用必须落 audit（路径：`runs/audit/`），包含 prompt 版本号 + 完整 request/response JSON
- **Prompt 版本管理**：`prompts.py` 中维护，修改递增版本号，版本号写入 audit

## 重试预算（硬上限）

| 范围 | 上限 | 超限行为 |
|------|------|----------|
| 单个 review 点 | 5 次 | 标记失败 |
| 全页累计 | 10 次 | **强制人工 review**，不要自动继续 |

不允许写「重试到成功」的循环。

## 不做

- 不让 VLM 输出 bbox / 坐标（任何形式）
- 不在本模块外创建新的 LLM subprocess 调用点（若需要新 schema，先扩 `prompts.py`）
- 不省略 audit 落盘（即使本地调试也要落，保证可追溯）
- 不做 panel 检测（交给 `cv_detect`，遵循 `PanelDetector` Protocol）
- 不做分割（交给 `segment_engines`，遵循 `Segmenter` Protocol）

## 测试

```bash
python -m geoseg.modules.vlm_client.demo
python -m geoseg.modules.vlm_client.demo_m1a
```
