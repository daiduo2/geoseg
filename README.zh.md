# geoseg v2

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 智能体驱动的地球物理速度分区提取工具。

**geoseg** 把已发表的地球物理解释图（彩色速度剖面、层析成像图、MATLAB 渲染的地震剖面）转成 [SPECFEM2D/3D](https://github.com/SPECFEM/specfem2d) 可直接使用的速度分区模型。整条流水线由 Claude Code 智能体驱动——不是 GUI：智能体直接看图，自主决策，只在 overlay 审阅阶段才向用户确认。

## Scope

只做**概念模型提取**。不做反射振幅图正演、不做波形反演、不做全波形反演参数调优。`figure_classifier` 取**保守**策略——假阴性可以接受，假阳性不可接受。一维测井/曲线图应当直接被拒掉，而不是被强行塞进分层管线。

## 为什么是智能体原生

传统分割工具要靠人工描线、调参数、堆领域经验。v2 用对话代替点击——自然语言就是界面，Claude Code skills 编排整条流水线。智能体通过 `Read` 工具**看到**图，再决定用哪个引擎、哪张 mask、哪一步精修。

## 核心能力

- **智能体原生编排**——figure-classify → cv_detect → sandbox-segment → visual-audit → export，全部由 Claude Code skills 驱动。
- **CLI 人机协同（HITL）**——智能体自动跑完流水线，展示 overlay，然后等待自然语言反馈。"去掉颜色条"或"底层分两层"会触发立即重分割。
- **多引擎 sandbox**——`sandbox-segment` 会试 `colorbar_guided`、`regional_fusion`、`edge_guided`、`ensemble`、`kmeans_full`、`grayscale` 等引擎，由 VLM 视觉判断 + 客观指标联合评估，挑出或融合出最优结果。
- **artifact 感知预处理**——红色断层线、黑色十字、白色空隙、label 合并、小连通域清理，都在做分区提取之前完成。
- **视觉审计**——`visual-audit` 读 overlay-with-legend，输出结构化的 `RegionalAudit`。不打 PASS/FAIL 分，由智能体自己定夺。
- **策略记忆**——历史分割经验会影响后续相似图像的引擎选择。
- **会话状态 + 回溯**——完整生命周期 `pending → classified → segmented → reviewed → exported`，可回溯到任意上游阶段。
- **批处理**——目录模式，最多 5 个 segmenter agent 并发（M 系列 Mac 单 agent 约 1.5 GB 内存预算）。
- **Napari Shapes 编辑器**——自然语言反馈不够用时，用阻塞式 GUI 做精细 label 编辑。

## 架构

```
PDF / Image
    ↓
[智能体: figure-classify] → velocity_model / skip
    ↓
[cv_detect] → panels + colorbar
    ↓
[智能体: sandbox-segment] → best labels（挑引擎 + 融合）
    ↓
[智能体: visual-audit] → RegionalAudit（语义摘要）
    ↓
[napari editor]  ←── 可选，仅当自然语言反馈不够用时
    ↓
[post_process + exporter] → polygons + properties + SPECFEM2D/3D
```

所有 VLM 推理都在 Claude Code agent 会话里通过 `Read` 工具完成。不再走 `claude -p` 子进程。

## Skills

| Skill | 作用 |
|-------|------|
| `geo-segment` | 端到端：figure → SPECFEM，对话式 HITL。 |
| `figure-classify` | 看图判断是 velocity model 还是跳过。 |
| `sandbox-segment` | 试多个引擎、评估、融合。 |
| `visual-audit` | 读 overlay-with-legend，输出结构化审计。 |
| `batch-segment` | 目录模式，最多 5 个 agent 并发。 |
| `segment-export` | 把已接受的 segmentation 导出为 SPECFEM。 |
| `preprocess-artifact` | 红色断层线、黑色十字、白色空隙修复。 |
| `module-demo` | 跑 `examples/geoseg/` 下的示例，验证模块流程。 |
| `schema-bump` | Schema 变更协议。 |

每个 skill 的 workflow 图、输入输出、失败处理写在 `.claude/skills/<skill>/SKILL.md`。

## 快速上手

### 依赖

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — Python 包管理
- [Claude Code](https://claude.ai/code) CLI
- 可选：MinerU API key（PDF 抽取）
- 可选：[rmux](https://github.com/joshmedeski/rmux)（HTML report ↔ CLI 实时反馈桥）

### 包管理器约定

| 生态 | 工具 | 用法 |
|------|------|------|
| Python | **uv** | `uv sync`、`uv run python -m geoseg…` |
| TS/JS | **pnpm** | `pnpm install`、`pnpm exec …` |

**禁止**在项目里混用 `pip` / `python3 -m venv`、`npm`、`yarn`。虚拟环境统一由 `uv` 管。

### 安装

```bash
git clone https://github.com/daiduo2/geoseg.git
cd geoseg
uv sync
uv run scripts/env_check.py
```

### 单图

```
User: /geo-segment runs/M0.5/fig1.png --n-layers=5

Agent: [自动跑 classify → detect → segment → audit]
       fig1.png  分割完成
         类型: velocity_model (0.92)
         引擎: colorbar_guided → 5 层
         audit: ok
       [展示 overlay]

       Accept / Modify / Skip / Backtrack ?

User: 修改。去掉右上角颜色条，底层分两层。

Agent: [用 mask + n_layers+1 重分割]
       [展示新 overlay]
       Accept / Modify / Skip / Backtrack ?

User: 接受

Agent: [导出 SPECFEM]
       ✅ tomo.xyz + Par_file_snippet.txt
```

### 批处理

```
User: /batch-segment runs/M0.5/ --n-layers=5

Agent: [Stage 1-3: 扫描 → 全部分类 → 全部分割]
       📦 5 张目标图已处理完毕，请 review。

       [1] fig1.png  ✅  0.85  5层
       [2] fig3.png  ✅  0.91  4层
       [3] fig4.png  ⚠️   0.62  3层  ← 建议修改
       [4] fig7.png  ✅  0.78  6层
       [5] fig9.png  ⚠️   0.58  2层  ← 建议修改

User: 1,2,4 接受；3 修改：底层应分两层；5 跳过

Agent: [导出 1,2,4；重分割 3；跳过 5]
```

## PDF 抽取（可选）

已发表的速度模型通常嵌在论文里，而不是以原始数据表的形式给出。geoseg 提供两层 PDF 抽取：

- **MinerU**（`modules/mineru_client/`）——结构化抽取（figures + caption markdown + content_list.json）。需要 `MINERU_API_KEY`。
- **PyMuPDF fallback**（`modules/pdf_extractor/`）——`{XObject + text block}` 抽取 + `rasterize_page()`。MinerU 拆分过细或尺寸过小时使用。

```bash
export MINERU_API_KEY="your-api-key"
```

## 实时反馈（可选）

HTML 报告里的 chatbox 可以通过 `rmux` 反向驱动 CLI 会话：

```bash
# 终端 1 —— Claude Code 跑在命名 rmux session 里
rmux new-session -s geoseg
# (在 rmux 内) cd /path/to/geoseg && cc

# 终端 2 —— feedback bridge
uv run python -m geoseg.feedback_bridge --rmux-session=geoseg

# 生成并打开 dashboard
uv run python -m geoseg.generate_report runs/sessions/batch_xxx.json
open runs/reports/batch_xxx.html
```

## 模块结构

```
src/geoseg/
├── core/                  # 稳定数据契约 + 跨模块 facade
│   ├── models.py
│   └── image_ops.py
├── pipeline/              # Stage 编排
│   ├── segment.py
│   ├── export.py
│   └── stages/
├── modules/
│   ├── cv_detect/         # Panel 检测 + colorbar 抽取
│   ├── segment_engines/   # 引擎族（registry / runner / policy / retry）
│   │   ├── v4/            # colorbar_guided, palette
│   │   ├── regional/      # regional_fusion
│   │   ├── edge/          # edge_guided helpers
│   │   ├── horizon/       # horizon refinement 内部实现
│   │   ├── strategy/      # 策略记忆
│   │   ├── diagnostics/   # metrics, batch_test, compare_results
│   │   └── internal/      # 共享 helpers
│   ├── post_process/      # 多边形 + 物理属性
│   ├── exporter/          # SPECFEM2D/3D 输出
│   ├── editor/            # Napari Shapes-primary 编辑器
│   ├── visual_audit/      # overlay-with-legend 视图
│   ├── mineru_client/     # MinerU PDF 抽取
│   ├── pdf_extractor/     # PyMuPDF fallback
│   └── vlm_client/        # Schema + prompt 模板（不再做 LLM 调用）
├── cli/                   # 可打包 CLI 入口
├── batch/                 # 目录批处理
├── api/                   # FastAPI schema / routes（历史兼容）
├── experiments.py         # scripts 侧的 CV / VLM / engine helper facade
├── session_state.py       # 带回溯的会话状态
├── controller.py          # 端到端兼容入口
├── pipeline_interfaces.py # 旧 import 兼容层
└── server.py              # FastAPI 兼容入口
```

完整代码地图和模块契约见 [`docs/CODEBASE.md`](docs/CODEBASE.md)。

## 设计哲学

1. **智能体原生优于 GUI**——对话就是界面。
2. **HITL 只在 review 阶段**——自动跑完一切，只在 overlay 确认处停下。
3. **支持上游回溯**——用户可回到 `classify` / `panel` / `segment` 任一阶段。
4. **保守分类**——假阴性优于假阳性。
5. **VLM 判断优先**——视觉判断 > 量化指标。
6. **状态不可变**——session state 更新返回新对象；每一步都落盘。
7. **不再新增 Tauri/FastAPI 产品前端**——`api/` 与 `server.py` 仅作为历史兼容保留。

## 测试

```bash
uv run pytest                                # 完整测试
uv run pytest tests/test_integration_ph01.py # schema 改动后的冒烟测试
```

macOS runner 上的 GUI 测试在 CI 中被排除。详见 [`tests/`](tests/)。

## 许可证

[MIT](LICENSE)。
