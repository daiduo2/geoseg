# geoseg 代码库地图

> 只读导航。模块特定约束在各 `CLAUDE.md`，沿目录树自动加载（在子目录启动 cc 时生效）。
> **Phase 0：单人开发**，不开 swarm。
> 复盘见 `~/Documents/knowlege/Projects/精密院-地震逆散射/photo/geo-segment-gui/docs/MISTAKE_LOG.md`。

## 核心层（`src/geoseg/core/`, `src/geoseg/pipeline/`）

| 路径 | 职责 | 备注 |
|------|------|------|
| `core/models.py` | 稳定数据契约：`PanelInput`、`SegmentationResult`、`FigureClassification` 等 | 新代码优先从这里导入 |
| `core/image_ops.py` | 跨模块图像工具 facade：overlay、distinct colors、saturation ratio | 产品代码不要直接依赖 `segment_engines/internal/shared.py` 或旧 `_shared.py` |
| `pipeline/segment.py` | segmentation stage wrapper | 调用 engine family，不内嵌算法细节 |
| `pipeline/export.py` | post-process + SPECFEM export stage | `controller.py` 通过它导出 panel |
| `controller.py` | 兼容入口：`run_pipeline`、`run_post_process_and_export` | 新 stage 编排优先放 `pipeline/` |
| `pipeline_interfaces.py` | 兼容导出层 | 新代码改用 `core/models.py` |

## 模块（`src/geoseg/modules/`）

| 模块 | 职责 | 关键文件 | 模块契约 |
|------|------|----------|----------|
| `mineru_client/` | **M0.5-MinerU**：MinerU v4 API 客户端。上传 PDF → 轮询提取 → 下载 figure 图片 + caption markdown + content_list.json | `client.py`, `review_extracted.py` | — |
| `pdf_extractor/` | **M0.5-Fallback**：PyMuPDF 提取 `{XObject(Image) + 页面文字块}`；`rasterize_page()` 整页/区域 rasterize。MinerU 拆分 figure 或提取尺寸过小时 fallback | `extract.py`, `vector_extract.py` | [pdf_extractor/CLAUDE.md](../src/geoseg/modules/pdf_extractor/CLAUDE.md) |
| `cv_detect/` | **M1b**：CV 检测 panel 候选 bbox。子模块：figure 分类器、panel 检测器（含 e026 版）、colorbar 提取器、质量过滤器 | `detect.py`, `figure_classifier.py`, `panel_detector.py`, `panel_detector_e026.py`, `colorbar_extractor.py`, `quality_filter.py` | [cv_detect/CLAUDE.md](../src/geoseg/modules/cv_detect/CLAUDE.md) |
| `vlm_client/` | **Schema + Prompt 定义库**。VLM 调用已全面迁移至 agent skill（`figure-classify` / `sandbox-segment`），本模块不再作为 LLM 调用出口。保留 schema（pydantic）和 prompt 模板供 skill 与 legacy code 引用 | `client.py`（`_call_claude_cli` 已 DEPRECATED）, `prompts.py` | [vlm_client/CLAUDE.md](../src/geoseg/modules/vlm_client/CLAUDE.md) |
| `segment_engines/` | **M3-Engine Family**：多算法分割引擎族。核心引擎 + registry / policy / runner / retry 边界 | `router.py`（兼容 facade）, `registry.py`, `policy.py`, `runner.py`, `retry.py`, `ensemble.py`, `v4_kmeans.py`, `edge_guided.py`, `edge_grow.py`, `e027_slic_graphcut.py`, `kmeans_full.py`, `grayscale.py`, `full_pipeline.py`, `vlm_reps.py`, `strategy_memory.py` | [segment_engines/CLAUDE.md](../src/geoseg/modules/segment_engines/CLAUDE.md) |
| `segment_engines/internal/` | engine family 内部工具 | `shared.py`；旧 `_shared.py` 仅兼容 re-export | — |
| `segment_engines/diagnostics/` | 诊断、评估、批量对比工具 | `metrics.py`, `batch_test.py`, `compare_results.py`；旧同名文件仅兼容 re-export/entrypoint | — |
| `post_process/` | **M3.5→M4 桥梁**：从分割 labels 提取多边形 + 连通域属性 + 物理属性分配（Vp/Vs/rho） | `polygon.py`, `properties.py` | [post_process/CLAUDE.md](../src/geoseg/modules/post_process/CLAUDE.md) |
| `exporter/` | **M4**：SPECFEM2D/3D 模型导出。`tomography_file.xyz` + `Par_file` snippet | `specfem.py` | [exporter/CLAUDE.md](../src/geoseg/modules/exporter/CLAUDE.md) |
| `e026_algo/` | ~~已弃用。`segment_engines/` 已完全替代~~ | ~~`core.py`, `components.py`~~ | [e026_algo/CLAUDE.md](../src/geoseg/modules/e026_algo/CLAUDE.md) |
| `editor/` | **Napari-based Shapes-primary 编辑器**（v0.8）。用户鼠标交互修改边界 | `napari_app.py` | [editor/CLAUDE.md](../src/geoseg/modules/editor/CLAUDE.md) |
| `visual_audit/` | **视觉审计**：生成 overlay-with-legend 与辅助视图，为 agent 视觉批评提供输入；不输出 PASS/FAIL | `views.py`, `crops.py`, `semantic.py`, `report.py` | — |

## 私有依赖边界

- 产品代码不得直接导入 `geoseg.modules.segment_engines._shared`、`segment_engines.internal` 或 `router._run_engine`。
- 需要 overlay/colors/saturation 等跨模块能力时，从 `geoseg.core.image_ops` 导入。
- 需要按名称运行分割引擎时，从 `geoseg.modules.segment_engines.runner import run_engine` 导入。
- `segment_engines/` 内部可以使用 `internal/shared.py`，旧 `_shared.py` 只保留给历史代码兼容。
- `experiments/`、`scripts/`、`examples/` 中仍有历史私有依赖，允许暂留；若要晋升进 `src/geoseg/`，必须先改为稳定 API。

## 组装层（`src/geoseg/` 根级）

| 文件 | 职责 |
|------|------|
| `core/` | 稳定数据模型和跨模块 facade |
| `pipeline/` | Stage 编排：segment/export 等 |
| `pipeline_interfaces.py` | 旧导入兼容层；新代码使用 `core/models.py` |
| `controller.py` | 端到端兼容 facade：`figure image → classify → segment → post-process → export SPECFEM` |
| `batch_processor.py` | 批量目录处理。支持 resume、单图错误隔离、结构化 JSON summary |
| `server.py` | FastAPI HTTP 后端（v0.7）。暴露 `/api/agent/*` 和 `/api/manual/*` endpoint |
| `gui/` | PySide6 GUI 包（v0.7 已废弃） |

## Skills（`.claude/skills/`）

| Skill | 职责 | 调用方式 |
|-------|------|----------|
| `geo-segment` | 端到端 orchestrator：figure → SPECFEM | `/geo-segment` 或语境匹配 |
| `figure-classify` | 图像分类：判断是否 velocity model | `/figure-classify` |
| `sandbox-segment` | Agent 自主分割：自选引擎、评估、融合 | `/sandbox-segment` |
| `batch-segment` | 批量处理目录（≤5 并行 agent） | `/batch-segment` |
| `visual-audit` | Agent 视觉批评：读 overlay 输出 RegionalAudit | `/visual-audit` |
| `module-demo` | 运行 `examples/geoseg/` 下的示例验证模块 workflow | `/module-demo` |
| `schema-bump` | Schema 变更协议 | `/schema-bump` |

**Skill 索引**：`.claude/skills/README.md`

## Build / Test

本项目使用 **uv** 管理 Python 环境，**pnpm** 管理 TS/JS 依赖。

```bash
# 环境检查
uv run scripts/env_check.py

# 示例
uv run python examples/geoseg/modules/segment_engines/demo.py
uv run python examples/geoseg/controller_demo.py
uv run python examples/geoseg/batch_processor_demo.py
uv run python examples/geoseg/pipeline_interfaces_demo.py

# 集成测试
uv run pytest tests/test_integration_ph01.py
```

禁止混用 `pip`/`python3 -m venv` 或 `npm`/`yarn`；统一使用 `uv run` 调用 Python 工具。

## 测试与产物

| 路径 | 内容 |
|------|------|
| `tests/fixtures/ph01/` | 集成测试数据（PDF + 3 个 VLM mock JSON） |
| `tests/test_integration_ph01.py` | 组装阶段集成测试（PR 必跑，暂未创建） |
| `runs/M*/` | 各模块 demo 产物（gitignore） |
| `runs/mineru/` | MinerU 提取产物（zip / images / markdown / VLM review） |
| `runs/audit/` | VLM 调用审计轨迹 |
| `runs/literature_test/` | 文献数据集 e2e 测试输入/输出（各子目录：gras2019, zailac2023, ma_2022） |
| `runs/sandbox/` | sandbox-segment agent 约定路径 |

## 设计文档

| 文档 | 用途 |
|------|------|
| [`DESIGN.md`](./DESIGN.md) | 一页设计稿 **v0.7 已签字**（2026-05-23）。§4 = JSON schema 契约、§5 = 模块行预算、§8 = 开工顺序 |
| [`PDF_VECTOR_EXTRACTION_SPEC.md`](./PDF_VECTOR_EXTRACTION_SPEC.md) | M0.5v 矢量提取规格（并行 session 开发，状态：待开发） |
| [`ALGORITHM_FAMILY.md`](./ALGORITHM_FAMILY.md) | e001-e027 实验全景 + 算法路由设计 |

## 已废弃（不维护）

- `src/geoseg/modules/vlm_client/client.py` 中的 `_call_claude_cli` / `_call_with_retry`（`claude -p` subprocess）— **已在代码中标记 DEPRECATED**。语义推理全面迁移至 `.claude/skills/`（`figure-classify`、`sandbox-segment`），agent 直接 Read 看图 + Bash 调用工具
- `src/geoseg/modules/vlm_client/client.py` 中的 `classify_figure` / `review_page_overview` / `review_segmentation_quality` — 保留供 legacy code / stub mode 使用，新代码应走 skill
- `src/geoseg/modules/segment_engines/router.py` 是兼容 facade。新代码应优先导入 `policy.select_engine`、`runner.run_engine`、`retry.retry_undersegmentation`
- `src/geoseg/gui/` — v0.7 起全面废弃
- `geoseg-gui/` — Tauri 前端（v0.7）已废弃
- `src/geoseg/modules/e026_algo/` — `segment_engines/` 已完全替代

## 不要碰

- `tests/fixtures/**/*.{pdf,png,jpg}` — 二进制数据，无源信息
- `~/.claude/skills/geo-segment/` — e026 算法已复制进项目（`src/geoseg/modules/e026_algo/`），不再依赖外部 skill

## cc 启动建议

| 任务类型 | 启动目录 |
|----------|----------|
| 改模块内部 | `cd src/geoseg/modules/<module> && cc` — 加载本模块 CLAUDE.md + 根 CLAUDE.md |
| 改组装层 / server / 接口 | `cd src/geoseg && cc` — 加载 `src/geoseg/CLAUDE.md` + 根 CLAUDE.md |
| 跨模块改动（schema / 接口） | 项目根目录 |
| 阅读设计 / 跑集成测试 | 项目根目录 |
