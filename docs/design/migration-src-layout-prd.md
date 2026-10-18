# src Layout 迁移 PRD

## 1. 目标

将 `geoseg` 从仓库根目录的 flat layout 迁移为标准 **src layout**，并同步更新文档、Skill 和构建配置。

## 2. 决策结论

| # | 决策 | 方案 |
|---|------|------|
| 1 | `geoseg-gui/` 处理 | 直接删除（Tauri 前端已废弃，git 历史保留） |
| 2 | `src/3d_schematic/` 处理 | 迁入 `experiments/3d_schematic/`，不再占用 `src/` |
| 3 | 根目录实验脚本处理 | 全部迁入 `experiments/` |
| 4 | `requirements.txt` 处理 | 合并进 `pyproject.toml`，保留 `requirements.txt` 作为运行时简明清单 |
| 5 | 执行顺序 | 写 PRD → 保存 git 快照 → 用 Workflow 执行迁移 |

## 3. 当前问题

- `geoseg/` 位于仓库根，导致源码、脚本、运行时产物混排。
- 无 `pyproject.toml` / `setup.py`，无法 `pip install -e .`。
- `geoseg/` 缺少 `__init__.py`，不是合法 Python package。
- 根目录有 8 个实验/测试 `.py` 未入 `experiments/`。
- `src/3d_schematic/` 是独立项目，与主包目标冲突。
- `geoseg-gui/` 已废弃仍占根目录。
- `.claude/skills/` 中 Bash 命令写死模块路径，需同步检查。

## 4. 目标目录结构

```text
geoseg/                              # 仓库根
├── src/
│   └── geoseg/                      # 唯一 Python 包根
│       ├── __init__.py              # 包入口 + __version__
│       ├── pipeline_interfaces.py
│       ├── controller.py
│       ├── batch_processor.py
│       ├── session_state.py
│       ├── feedback_bridge.py
│       ├── generate_report.py
│       ├── modules/
│       │   ├── cv_detect/
│       │   ├── segment_engines/
│       │   ├── post_process/
│       │   ├── exporter/
│       │   ├── pdf_extractor/
│       │   ├── mineru_client/
│       │   ├── vlm_client/
│       │   ├── editor/
│       │   └── visual_audit/
│       └── gui/                     # 已废弃，保留并加 DEPRECATED 标记
├── tests/
├── scripts/
├── docs/
│   ├── migration-src-layout-prd.md  # 本文件
│   └── CODEBASE.md                  # 路径同步更新
├── experiments/                     # 新增
│   ├── exp1_warm_merge.py
│   ├── exp4_lab_lchannel.py
│   ├── test_3d_schematic_e2e.py
│   ├── test_3d_schematic_correct_e2e.py
│   ├── test_3d_schematic_edge_guided.py
│   ├── test_regional_fusion_e2e.py
│   ├── test_tubular_panel3.py
│   ├── test_visual_comparison.py
│   └── 3d_schematic/                # 从 src/ 迁入
├── runs/                            # 运行时产物（已 gitignore）
├── .claude/
├── pyproject.toml
├── README.md
└── requirements.txt
```

## 5. 需要同步修改的路径清单

### 5.1 文档

- `docs/CODEBASE.md`
  - 更新模块路径：`geoseg/modules/...` → `src/geoseg/modules/...`
  - 更新启动建议：`cd geoseg && cc` → `cd src/geoseg && cc`
  - 更新 demo 命令：`python -m geoseg.controller_demo` 保持不变（模块名不变）
  - 更新废弃说明：`geoseg-gui/` 删除
  - 新增 `experiments/` 说明
- `geoseg/CLAUDE.md` → 迁到 `src/geoseg/CLAUDE.md`
- `geoseg/modules/*/CLAUDE.md` → 迁到 `src/geoseg/modules/*/CLAUDE.md`
- `README.md`
  - 更新安装说明，新增 `pip install -e .`
  - 更新目录结构描述

### 5.2 Skill

- `.claude/skills/README.md`
  - 检查并更新所有 `python -m geoseg.xxx` 命令（模块名通常不变，但相对路径需确认）
  - 检查 `cd geoseg && python ...` 类命令
- `.claude/skills/geo-segment/SKILL.md`
- `.claude/skills/batch-segment/SKILL.md`
- `.claude/skills/sandbox-segment/SKILL.md`
- `.claude/skills/figure-classify/SKILL.md`
- `.claude/skills/module-demo/SKILL.md`
- `.claude/skills/schema-bump/SKILL.md`

### 5.3 构建配置

- 新增 `pyproject.toml`：
  - `[build-system]`：`setuptools >= 61`
  - `[project]`：name = "geoseg"，dynamic = ["version"]
  - `[tool.setuptools.packages.find]`：where = ["src"]
  - `[tool.setuptools.dynamic]`：version = {attr = "geoseg.__version__"}
  - 将 `requirements.txt` 内容合并到 `[project] dependencies`
- 保留 `requirements.txt` 作为运行时清单（可选，与 pyproject 同步）

### 5.4 Python 包入口

- 新增 `src/geoseg/__init__.py`：
  - `__version__ = "0.8.0"`
  - 可导出核心 API（controller.run_pipeline 等）

## 6. 迁移步骤

### Phase 1: 快照

1. 创建备份分支：`git branch backup/pre-src-layout-migration`
2. 确认当前分支 `feat/regional-fusion` 未切换。

### Phase 2: 物理迁移

1. 创建目录 `src/geoseg/`。
2. `git mv geoseg/* src/geoseg/`。
3. 新增 `src/geoseg/__init__.py`。
4. `git mv src/3d_schematic experiments/3d_schematic/`。
5. 创建 `experiments/` 并迁入根目录实验脚本：
   - `exp1_warm_merge.py`
   - `exp4_lab_lchannel.py`
   - `test_3d_schematic_e2e.py`
   - `test_3d_schematic_correct_e2e.py`
   - `test_3d_schematic_edge_guided.py`
   - `test_regional_fusion_e2e.py`
   - `test_tubular_panel3.py`
   - `test_visual_comparison.py`
6. 删除 `geoseg-gui/`。
7. 新增 `pyproject.toml`。

### Phase 3: 路径同步

1. 按 5.1 更新 `docs/CODEBASE.md`。
2. 按 5.2 检查并更新 `.claude/skills/` 中的路径和命令。
3. 按 5.3 配置 `pyproject.toml`。
4. 更新 `README.md` 安装说明。

### Phase 4: 验证

1. 创建干净 venv：`python -m venv .venv-test && source .venv-test/bin/activate`
2. 安装：`pip install -e .`
3. 运行测试：`pytest tests/`
4. 验证模块导入：`python -c "import geoseg; print(geoseg.__version__)"`
5. 验证 demo：`python -m geoseg.controller_demo`（轻量验证）

### Phase 5: 清理

1. 删除临时 venv `.venv-test`。
2. 检查 `.gitignore` 是否需要更新（如 `src/geoseg.egg-info/`）。
3. 最终 `git status` 确认无遗漏。

## 7. 回滚方案

- 若迁移失败需回滚：
  1. `git checkout backup/pre-src-layout-migration`
  2. `git reset --hard backup/pre-src-layout-migration`
  3. 对于未跟踪但被移动的文件，从备份分支或 reflog 恢复。

## 8. 验收标准

- [ ] `pip install -e .` 成功
- [ ] `python -c "import geoseg"` 成功
- [ ] `pytest tests/` 通过
- [ ] `docs/CODEBASE.md` 路径已更新
- [ ] `.claude/skills/` 无失效路径
- [ ] 根目录无 `geoseg/`、`geoseg-gui/`、`src/3d_schematic/`
- [ ] 根目录无实验 `.py`
- [ ] `pyproject.toml` 包含 dependencies 和 setuptools 配置

## 9. 风险与注意事项

- **import 路径**：src layout 下 `from geoseg.modules...` 保持不变，因为包名仍是 `geoseg`。
- **Skill 路径**：需重点检查 `cd geoseg && python ...` 类命令，可能需要改为 `cd src/geoseg && python ...`。
- **CLAUDE.md 加载**：子目录启动 `cc` 时，需确认 `src/geoseg/` 和 `src/geoseg/modules/<module>/` 下的 CLAUDE.md 能被正确加载。
- **大文件**：`docs/all_overlays.zip` 等未跟踪大文件不要误提交。
- **3d_schematic**：其内部 import 可能依赖原 `src/3d_schematic/` 路径，迁入 `experiments/` 后若内部有相对 import 需单独调整。
