# geoseg 后续重构路线图

> 目标：给后续 session 直接执行使用。本文只覆盖当前架构收口主线，不记录算法实验结论。

## 当前状态

已完成的主线：

1. `full_pipeline.py` 已压成兼容 facade，真正 figure 级编排走 `geoseg.pipeline.segment.run_segmentation_stage`。
2. `pipeline/stages/` 已拆分为 classify / detect / review / panel / summary stage helper 包。
3. `server.py`、`batch_processor.py` 已变成薄兼容入口，实际实现分别在 `api/` 和 `batch/`。
4. `segment_engines/compat/` 已集中 legacy shim，旧平铺导入路径仍保留 re-export。
5. `horizon_refinement.py` 已拆到 `segment_engines/horizon/`，原文件是公共 facade。
6. `internal/seeds.py` 已拆到 `segment_engines/internal/seeds/` 包。
7. `v4_kmeans.py` 已拆到 `segment_engines/v4/`，原文件是公共 facade。
8. `scripts/` 中常见 CV/VLM/engine helper 已迁移到 `geoseg.experiments` facade。
9. `segment_engines` legacy facade 文件头已统一说明实现位于 `compat/`，并有边界测试禁止产品代码直连 `segment_engines.compat.*`。
10. `runner.py` 已改为 registry-driven dispatch；`EngineSpec` 维护 callable path 和 adapter，`runner` 不再直接导入具体 engine 模块。
11. `regional_fusion.py` 已拆到 `segment_engines/regional/`，原文件保留 compatibility facade。
12. `edge_guided.py` / `edge_grow.py` 已抽出 `segment_engines/edge/` 共享 seed、gradient、postprocess helpers。
13. `strategy_memory.py` 已拆到 `segment_engines/strategy/`，原文件保留 compatibility facade。
14. `batch/service.py` 已拆出 `batch/audit.py`、`batch/entry.py`、`batch/export.py`、`batch/session.py`，`service.py` 保留目录级编排。
15. `api/app.py` 已拆成 app assembly，routes 分到 `routes_agent.py`、`routes_manual.py`、`routes_export.py`、`routes_pdf.py`。
16. import boundary 测试已覆盖主要边界，当前全量测试通过：`uv run pytest`。

## 重构原则

- 保持旧导入路径兼容，先迁移实现，再缩小 facade。
- 产品编排放在 `pipeline/`、`api/`、`batch/`，不要塞回 `modules/segment_engines/`。
- `segment_engines/` 主目录保留 engine、registry、policy、runner、retry 和必要 facade。
- engine 内部共享工具只在 `segment_engines/internal/` 内使用；跨模块能力优先走 `core/` 或模块级 facade。
- `scripts/` 可以直接调用具体算法做实验，但常见 CV/VLM/engine helper 应优先走 `geoseg.experiments`。
- 每一批重构都至少跑 `uv run pytest tests/test_import_boundaries.py`，涉及 pipeline 或 engine 时跑全量 `uv run pytest`。

## 已完成：收紧 `segment_engines` 主目录

### 状态

`segment_engines/` 主目录仍保留旧导入路径 facade，但已统一标记为 legacy import path；实现集中到 `compat/`。边界测试已禁止产品代码直接导入 `segment_engines.compat.*`。

### 后续保持

1. 主目录文件保持四类：
   - engine：`edge_*`、`ensemble.py`、`grayscale.py`、`kmeans_full.py`、`v4_kmeans.py` 等
   - routing：`registry.py`、`policy.py`、`runner.py`、`retry.py`、`router.py`
   - public facade：`regions.py`、`metrics.py`
   - compat：`classify.py`、`detect.py`、`panel_segment.py`、`review.py`、`summary.py`、`full_pipeline.py`、`pipeline_stages.py`、`_shared.py`
2. 新产品代码不得导入 `segment_engines.compat.*`。
3. 新 legacy facade 必须只做 re-export，并写明实现所在目录。

### 验收

```bash
uv run pytest tests/test_import_boundaries.py
uv run pytest
```

## 已完成：拆 `runner.py` 的 engine dispatch

### 状态

`runner.py` 已从 registry 加载 engine callable。`EngineSpec` 当前字段包括：

- `callable_path`
- `adapter`
- `requires_reps`
- `fallback_engine`
- `is_post_processor`

`tests/test_import_boundaries.py` 已验证每个注册 engine callable 可加载，并验证 `runner.py` 不直接导入具体 engine 模块。

### 后续保持

1. 新增 engine 时先登记 `EngineSpec`，不要在 `runner.py` 增加按 engine 名称导入的分支。
2. 只有参数形态确实不同的时候才新增 adapter。
3. 新 adapter 必须补 smoke test 或覆盖已有 registry callable 加载测试。

### 验收

```bash
uv run pytest tests/test_import_boundaries.py tests/test_pipeline_segment.py tests/test_integration_ph01.py
uv run pytest
```

## 已完成：继续拆 engine 大文件

### 已完成：`regional_fusion.py`

状态：融合策略、overlay legend、split/merge 逻辑已拆到：

```text
segment_engines/regional/
  __init__.py
  fusion.py
  models.py
  overlay.py
  split_merge.py
```

旧 `regional_fusion.py` 保留 facade。当前没有独立 scoring 逻辑，因此未创建空的 `scoring.py`。

### 已完成：`edge_guided.py` / `edge_grow.py`

状态：两者共享的 seed refinement、gradient/edge-map、postprocess 已拆到：

```text
segment_engines/edge/
  __init__.py
  seeds.py
  gradients.py
  postprocess.py
```

旧 engine 文件继续保留 `segment()`，并只保留各自核心算法：edge-guided K-means 与 Dijkstra region grow。

### 已完成：`strategy_memory.py`

状态：策略记录、统计、持久化、评分已拆到：

```text
segment_engines/strategy/
  __init__.py
  store.py
  scoring.py
  records.py
```

旧 `strategy_memory.py` 保留 facade。`tests/test_strategy_memory.py` 覆盖 JSONL 记录、相似查询、批量分析和模板存取。

### 验收

每拆一个 engine，至少跑：

```bash
uv run pytest tests/test_regional_fusion.py tests/test_regional_refinement.py
uv run pytest tests/test_integration_ph01.py
uv run pytest
```

## 已完成：收紧 `scripts/` 实验入口

### 状态

批量处理、筛选、审计类脚本已迁移到 `geoseg.experiments` facade。`run_panel3_best.py` 也已改走 facade 导出的 engine callable。剩余直接导入具体算法的脚本均为 relabel/closing/debug 专项实验，并在文件头标注：

```text
Algorithm-specific experiment: imports concrete engine internals intentionally.
```

### 后续保持

1. 扫描 `scripts/`：

```bash
rg "from geoseg\\.modules\\.(segment_engines|vlm_client|cv_detect)" scripts -n
```

2. 对批量/审计/筛选脚本改走 `geoseg.experiments`。
3. 对确实是在比较具体算法的脚本，保留直接导入，并在文件顶部注释说明 algorithm-specific experiment。
4. `tests/test_import_boundaries.py` 只禁止公共 helper 直连，不禁止算法专项实验。

### 验收

```bash
uv run pytest tests/test_import_boundaries.py
uv run pytest
```

## 已完成：整理 API / batch 的后续边界

### 状态

`api/app.py` 已经收敛为 app assembly only；route handlers 分散到专门模块。`batch/service.py` 已完成二次拆分，保留目录级编排。

`api/` 已完成：

```text
api/
  app.py              # app assembly only
  routes_agent.py
  routes_manual.py
  routes_export.py
  routes_pdf.py
  schemas.py
  serialization.py
```

`batch/` 已完成：

```text
batch/
  service.py          # process_directory orchestration
  entry.py            # process one SessionEntry
  audit.py
  export.py
  session.py
  cli.py              # existing CLI
```

### 验收

```bash
uv run pytest tests/test_integration_ph01.py tests/test_import_boundaries.py
uv run pytest
```

## 已完成：文档同步与旧设计收敛

### 状态

`docs/DESIGN.md` 顶部已标记为历史设计文档，明确当前可执行架构地图以 `docs/CODEBASE.md`、本 roadmap 和模块内 `CLAUDE.md` 为准。旧关键词仍保留在历史内容中，用于追溯当时设计，不再作为实时架构说明。

### 后续保持

1. 以 `docs/CODEBASE.md` 为当前真实地图。
2. 将实时架构说明集中到：
   - `docs/CODEBASE.md`
   - `docs/refactor_roadmap_2026-07-14.md`
   - 模块内 `CLAUDE.md`
3. 搜索过时关键词时，区分历史文档和当前架构文档：

```bash
rg "full_pipeline|pipeline_interfaces|server.py|_shared.py" docs -n
```

### 验收

```bash
uv run pytest tests/test_import_boundaries.py
```

## 建议执行顺序

当前 roadmap 主线已完成。后续新重构应先更新本文件，再按同样的“facade 兼容 + 边界测试 + 全量测试”节奏执行。

## 每轮完成标准

每轮重构结束前执行：

```bash
uv run pytest
git status --short
```

如果需要提交，提交信息建议使用：

```text
Refactor <area> boundaries
```
