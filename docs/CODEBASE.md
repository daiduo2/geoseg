# geoseg 代码库地图

> 只读导航。模块特定约束在各 `CLAUDE.md`，沿目录树自动加载（在子目录启动 cc 时生效）。
> **Phase 0：单人开发**，不开 swarm。
> 复盘见 `~/Documents/knowlege/Projects/精密院-地震逆散射/photo/geo-segment-gui/docs/MISTAKE_LOG.md`。

## 模块（`geoseg/modules/`）

| 模块 | 职责 | 关键文件 | 模块契约 |
|------|------|----------|----------|
| `mineru_client/` | **M0.5-MinerU**：MinerU v4 API 客户端。上传 PDF → 轮询提取 → 下载 figure 图片 + caption markdown + content_list.json | `client.py`, `review_extracted.py` | — |
| `pdf_extractor/` | **M0.5-Fallback**：PyMuPDF 提取 `{XObject(Image) + 页面文字块}`；`rasterize_page()` 整页/区域 rasterize。MinerU 拆分 figure 或提取尺寸过小时 fallback | `extract.py`, `vector_extract.py` | [pdf_extractor/CLAUDE.md](../geoseg/modules/pdf_extractor/CLAUDE.md) |
| `cv_detect/` | **M1b**：CV 检测 panel 候选 bbox。子模块：figure 分类器、panel 检测器（含 e026 版）、colorbar 提取器、质量过滤器 | `detect.py`, `figure_classifier.py`, `panel_detector.py`, `panel_detector_e026.py`, `colorbar_extractor.py`, `quality_filter.py` | [cv_detect/CLAUDE.md](../geoseg/modules/cv_detect/CLAUDE.md) |
| `vlm_client/` | **Schema + Prompt 定义库**。VLM 调用已全面迁移至 agent skill（`figure-classify` / `sandbox-segment`），本模块不再作为 LLM 调用出口。保留 schema（pydantic）和 prompt 模板供 skill 与 legacy code 引用 | `client.py`（`_call_claude_cli` 已 DEPRECATED）, `prompts.py` | [vlm_client/CLAUDE.md](../geoseg/modules/vlm_client/CLAUDE.md) |
| `segment_engines/` | **M3-Engine Family**：多算法分割引擎族。含 metrics 评估 + strategy_memory agent 学习 + batch 测试工具 | `router.py`（硬编码路由已废弃）, `ensemble.py`, `v4_kmeans.py`, `edge_guided.py`, `edge_grow.py`, `e027_slic_graphcut.py`, `kmeans_full.py`, `grayscale.py`, `full_pipeline.py`, `vlm_reps.py`, `metrics.py`, `strategy_memory.py`, `_shared.py`, `batch_test.py`, `compare_results.py` | — |
| `post_process/` | **M3.5→M4 桥梁**：从分割 labels 提取多边形 + 连通域属性 + 物理属性分配（Vp/Vs/rho） | `polygon.py`, `properties.py` | [post_process/CLAUDE.md](../geoseg/modules/post_process/CLAUDE.md) |
| `exporter/` | **M4**：SPECFEM2D/3D 模型导出。`tomography_file.xyz` + `Par_file` snippet | `specfem.py` | [exporter/CLAUDE.md](../geoseg/modules/exporter/CLAUDE.md) |
| `e026_algo/` | ~~已弃用。`segment_engines/` 已完全替代~~ | ~~`core.py`, `components.py`~~ | [e026_algo/CLAUDE.md](../geoseg/modules/e026_algo/CLAUDE.md) |

## 组装层（`geoseg/` 根级）

| 文件 | 职责 |
|------|------|
| `pipeline_interfaces.py` | **双管线接口契约**：TypedDict + Protocol 定义（`PanelInput`、`SegmentationResult`、`FigureClassification` 等） |
| `controller.py` | 端到端流水线编排：`figure image → classify → segment → post-process → export SPECFEM` |
| `batch_processor.py` | 批量目录处理。支持 resume、单图错误隔离、结构化 JSON summary |
| `server.py` | FastAPI HTTP 后端（v0.7）。暴露 `/api/agent/*` 和 `/api/manual/*` endpoint |
| `gui/` | PySide6 GUI 包（v0.7 将废弃，迁移至 Tauri）：`main_window.py` + `segmentation_view.py` + `figure_selector.py` + `panel_selector.py` + `pdf_import_worker.py` + `pdf_page_review_worker.py` + `demo.py` |

## Skills（`.claude/skills/`）

| Skill | 职责 | 调用方式 |
|-------|------|----------|
| `geo-segment` | 端到端 orchestrator：figure → SPECFEM | `/geo-segment` 或语境匹配 |
| `figure-classify` | 图像分类：判断是否 velocity model | `/figure-classify` |
| `sandbox-segment` | Agent 自主分割：自选引擎、评估、融合 | `/sandbox-segment` |
| `batch-segment` | 批量处理目录（≤5 并行 agent） | `/batch-segment` |
| `module-demo` | 运行模块 demo.py 验证 | `/module-demo` |
| `schema-bump` | Schema 变更协议 | `/schema-bump` |

**Skill 索引**：`.claude/skills/README.md`

## Build / Test

```bash
# 模块阶段
python -m geoseg.modules.<module>.demo

# 组装层
cd geoseg && python -m geoseg.controller_demo            # 单图端到端流水线
cd geoseg && python -m geoseg.batch_processor_demo        # 批量目录处理
cd geoseg && python -m geoseg.gui.demo                   # GUI 交互式分割视图
cd geoseg && python -m geoseg.server_demo                # FastAPI 后端 demo
cd geoseg && python -m geoseg.pipeline_interfaces_demo    # 接口契约 demo

# 集成测试
pytest tests/test_integration_ph01.py
```

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

- `geoseg/modules/vlm_client/client.py` 中的 `_call_claude_cli` / `_call_with_retry`（`claude -p` subprocess）— **已在代码中标记 DEPRECATED**。语义推理全面迁移至 `.claude/skills/`（`figure-classify`、`sandbox-segment`），agent 直接 Read 看图 + Bash 调用工具
- `geoseg/modules/vlm_client/client.py` 中的 `classify_figure` / `review_page_overview` / `review_segmentation_quality` — 保留供 legacy code / stub mode 使用，新代码应走 skill
- `geoseg/modules/segment_engines/router.py` 的 `select_engine` + `_RETRY_CHAIN`（硬编码路由）— agent 自主决策（`sandbox-segment` skill）
- `geoseg/gui/` — v0.7 起全面废弃，迁移至 Tauri 前端
- `geoseg/modules/e026_algo/` — `segment_engines/` 已完全替代

## 不要碰

- `tests/fixtures/**/*.{pdf,png,jpg}` — 二进制数据，无源信息
- `~/.claude/skills/geo-segment/` — e026 算法已复制进项目（`geoseg/modules/e026_algo/`），不再依赖外部 skill

## cc 启动建议

| 任务类型 | 启动目录 |
|----------|----------|
| 改模块内部 | `cd geoseg/modules/<module> && cc` — 加载本模块 CLAUDE.md + 根 CLAUDE.md |
| 改组装层 / server / 接口 | `cd geoseg && cc` — 加载 `geoseg/CLAUDE.md` + 根 CLAUDE.md |
| 跨模块改动（schema / 接口） | 项目根目录 |
| 阅读设计 / 跑集成测试 | 项目根目录 |
