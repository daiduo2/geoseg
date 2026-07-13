# Visual Audit PRD v2

> 版本：v2.0
> 日期：2026-06-19
> 目标：把视觉审计从“硬编码通过/拒绝门控”改造为“agent 视觉批评 + 执行闭环”。

---

## 1. 背景与问题

v1 视觉审计（本文件前一版本）引入了硬拒绝红线：`n_labels > 25`、`tiny_island_count > 30`、`fragment_ratio > 0.30` 等。运行后发现这些红线存在根本缺陷：

1. **泛化能力差**：地质图的好坏无法用 universal threshold 刻画。同样的碎片率在一张图中是过度分割，在另一张图中可能是 legitimate 断层。
2. **与 agent-native 路线冲突**：项目引入多模态 agent 的核心原因，就是让 agent 通过对比原图和 overlay 发现硬编码规则无法描述的问题。硬红线反而把 agent 降级为阈值触发器。
3. **误杀/漏杀并存**：硬红线会拒绝一些地质上合理的结果，同时放过一些没触发红线但明显错误的结果（如两层被错误合并）。
4. **阻碍修复迭代**：v1 的“REJECTED → 停止”模式没有给出结构化修复方向，agent 只能凭经验重跑整图。

因此 v2 完全废弃硬拒绝，改为**agent 驱动的视觉批评 + 冻结好区域/重分坏区域**的闭环。

---

## 2. 设计目标

1. **Agent 是唯一裁判**：所有“这个区域好不好、需不需要修”的判断由多模态 agent 做出，模块只提供视图和客观信号。
2. **颜色即区域**：通过带 label 图例的高对比度 overlay，agent 可以用“label 2（蓝色区域）”精确指代问题区域。
3. **冻结 + 局部分割**：把好区域固定，只对问题区域重新分割，避免整图重跑把好区域也改坏。
4. **结构化输出**：audit agent 输出 `RegionalAudit` 风格 JSON，直接驱动 executor agent 执行修复。
5. **最小新增逻辑**：复用现有 `regional_fusion`、`_create_overlay`、merge utilities、`visual_audit/views.py` 等组件，不新建核心分割算法。

---

## 3. 核心哲学

> **没有硬红线。没有 PASS/FAIL。只有“哪些区域看起来对”和“哪些区域需要怎么修”。**

客观指标（`n_layers`、`boundary_alignment`、`fragment_ratio` 等）仍然存在，但只是**诊断信号**，不是门控。agent 结合原图、overlay、crops 和自己的地质常识决定这些信号是否重要。

---

## 4. 审计输入

视觉审计 agent 必须同时看到以下输入：

### 4.1 主输入（必需）

| 输入 | 用途 | 生成方式 |
|------|------|----------|
| `panel.png` | 原始 panel 图像 | `cv_detect` 输出 |
| `overlay_legend.jpg` | 带 label 图例的高对比度 overlay | `regional_fusion.generate_overlay_with_legend(panel_rgb, labels)` |

`overlay_legend.jpg` 是核心：右下角图例把颜色映射到 `label_id`，agent 可以通过颜色精确指出区域。

### 4.2 辅助输入（建议生成，可选查看）

| 输入 | 用途 | 生成方式 |
|------|------|----------|
| `side_by_side.jpg` | 原图 vs 纯 mask 并排 | `visual_audit.views.create_side_by_side` |
| `pure_mask.jpg` | 无 alpha 混合的离散色块 | `visual_audit.views.create_pure_mask` |
| `fragment_highlight.jpg` | 小区域标红 | `visual_audit.views.create_fragment_highlight` |
| `text_residual_map.jpg` | 文字区 + 边界 | `visual_audit.views.create_text_residual_map` |
| `difference_heatmap.jpg` | 边界与颜色边缘对齐情况 | `visual_audit.views.create_difference_heatmap` |
| `crops/*.jpg` | 关键区域放大 | `visual_audit.crops.create_audit_crops` |

### 4.3 结构化上下文（JSON）

```json
{
  "label_color_map": {
    "1": {"color": [255, 0, 0], "area_frac": 0.12, "median_y": 120},
    "2": {"color": [0, 255, 0], "area_frac": 0.08, "median_y": 250}
  },
  "diagnostic_signals": {
    "n_labels": 5,
    "boundary_alignment": 0.73,
    "layer_order_monotonic": true,
    "plume_fidelity": {"iou": 0.0, "split": false}
  }
}
```

---

## 5. 审计输出（RegionalAudit 风格）

visual-audit agent 的输出必须是一个结构化 JSON，可直接被 `regional_segment` 消费：

```json
{
  "frozen_labels": [1, 4],
  "retry_labels": [2, 3],
  "notes": "label 2（绿色，中上部）被拆成多个碎片，应与相邻绿色碎片合并；label 3（蓝色，右侧）覆盖了右下角文字标注，需要在该区域用 edge_guided 重分。",
  "repair_strategy": "regional_fusion",
  "secondary_engine": "edge_guided",
  "local_fixes": [
    {
      "action": "merge_labels",
      "label_ids": [2, 5],
      "target_id": 2,
      "rationale": "同属绿色纹理区"
    }
  ],
  "iteration": 1
}
```

字段说明：

| 字段 | 含义 |
|------|------|
| `frozen_labels` | 质量合格、保持不动的 label ID 列表 |
| `retry_labels` | 需要重新分割的 label ID 列表 |
| `notes` | 自然语言诊断，供 executor agent 和用户理解 |
| `repair_strategy` | 推荐修复策略：`regional_fusion`、`merge_labels`、`switch_engine`、`post_process`、`accept` |
| `secondary_engine` | 如果走 regional_fusion，推荐用什么引擎处理 retry 区域 |
| `local_fixes` | 可直接执行的局部修改（如 merge） |
| `iteration` | 当前是第几轮审计-修复循环 |

当 `frozen_labels` 包含所有非背景 label 且 `retry_labels` 为空时，表示 segmentation 可被接受。

---

## 6. 审计-执行闭环

```
引擎输出 labels.npz + overlay_legend.jpg
        ↓
visual-audit agent 读 panel + overlay_legend + 辅助视图
        ↓
输出 RegionalAudit JSON
        ↓
[retry_labels 非空?]
   ├─ 是 → executor agent 选择并执行修复
   │        ├─ regional_fusion: 冻结 frozen_labels，对 retry_labels 区域用 secondary_engine 重分
   │        ├─ merge_labels: 调用 post_process.merge.merge_labels_by_ids
   │        ├─ switch_engine: 整图换引擎重跑
   │        └─ post_process: 调用 horizon_refinement / cleanup
   │        ↓
   │     保存新 labels.npz + overlay_legend.jpg
   │        ↓
   │     回到 visual-audit agent（iteration + 1）
   │
   └─ 否 → 审计通过，进入下一 stage
```

最大迭代次数：3 轮（soft cap，防止无限循环）。

---

## 7. 模块边界

```
geoseg/modules/visual_audit/
├── __init__.py
├── views.py          # 生成审计辅助视图
├── crops.py          # 关键区域裁剪
├── semantic.py       # 诊断信号（客观指标，无阈值）
└── report.py         # 汇总报告图 + JSON（无 verdict）
```

### 7.1 职责

- **输入**：`labels`（int32 array）、`panel_rgb`（uint8 RGB）、可选 `text_mask`、`gt_mask`
- **输出**：
  - `overlay_legend.jpg`（主审计输入）
  - 辅助视图集合
  - `diagnostic_signals` JSON
  - `label_color_map` JSON
- **不做**：
  - 不输出 `rejected: true/false`
  - 不做最终 accept/reject 决策
  - 不调用 LLM/VLM
  - 不直接修改 labels

### 7.2 删除/废弃

- `reject.py`：完全删除。`compute_hard_reject` 不再被任何 skill 调用。
- `REJECTED` / `PASSED HARD GATES` verdict：从 `report.py` 和相关 skill 中移除。

---

## 8. 复用现有组件

| 能力 | 已有实现 | 位置 |
|------|---------|------|
| 高对比度 overlay | `_create_overlay` | `segment_engines/_shared.py` |
| overlay + label 图例 | `generate_overlay_with_legend` | `segment_engines/regional_fusion.py` |
| 冻结 + 重分融合 | `fuse_with_freeze` / `regional_segment` | `segment_engines/regional_fusion.py` |
| RegionalAudit 数据结构 | `RegionalAudit` dataclass | `segment_engines/regional_fusion.py` |
| label 合并 | `merge_labels_by_ids` / `merge_warm_labels` | `post_process/merge.py` |
| 辅助视图 | `create_audit_views` / `create_audit_crops` | `visual_audit/views.py` / `crops.py` |
| 诊断信号 | `compute_semantic_fidelity` | `visual_audit/semantic.py` |

**不需要新建核心分割算法或新的 VLM client。**

---

## 9. Skill 更新清单

| Skill | 更新内容 |
|-------|---------|
| `visual-audit` | 重写为“视觉批评家”：输入 overlay_legend + origin，输出 RegionalAudit JSON；删除硬拒绝相关所有描述 |
| `sandbox-segment` | 删除 hard-reject gate、删除 Repair Playbook lookup table、改为 audit → executor 闭环 |
| `batch-segment` | 删除 `AUDIT_FAILED` 状态、删除“必须先了解拒审原因”等规则、改为展示 audit notes 供用户批量决策 |
| `geo-segment` | 同步删除 visual-audit 作为“export gate”的描述 |

---

## 10. 数据流与约定路径

单图 sandbox-segment 产物路径：

```
runs/sandbox/{panel_id}/
├── panel.png                 # 原始 panel
├── labels.npz                # 当前 label map
├── overlay.jpg               # 无图例 overlay（供快速浏览）
├── overlay_legend.jpg        # 带图例 overlay（视觉审计主输入）
├── label_color_map.json      # label_id → color / area / median_y
├── visual_audit/
│   ├── views/                # 辅助视图
│   ├── crops/                # 关键区域裁剪
│   └── report.json           # 诊断信号 + view_paths + crop_paths
└── regional_audit.json       # 当前轮次 audit agent 输出
```

---

## 11. 与 Horizon Refinement 的关系

v1 中 horizon refinement 由 `frag > 0.02` 硬阈值触发。v2 中改为：

- audit agent 在 review 阶段对比 coarse overlay 和 refined overlay；
- 如果 refinement 使边界更平滑且没有丢失/改变拓扑，则接受；
- 否则保持 coarse。

即 refinement 的触发权从阈值交给 agent。

---

## 12. 验收标准

- [x] `visual_audit/reject.py` 已删除，`__init__.py` 不再导出 `compute_hard_reject`
- [x] `report.py` 不再输出 `rejected` / `reasons` / `verdict`，只输出视图、诊断信号、label_color_map
- [x] `visual-audit` skill 重写完成，示例输出为 RegionalAudit JSON
- [x] `sandbox-segment` skill 完成 critic-executor 闭环描述
- [x] `batch-segment` skill 移除 `AUDIT_FAILED` 状态和硬拒绝相关规则
- [x] `.claude/skills/README.md`、 `docs/CODEBASE.md`、 `docs/project_status_report.md` 中 visual audit 描述同步更新
- [ ] 至少一个真实 panel 跑通新闭环：初始分割 → audit → regional_fusion → re-audit → accept

---

## 13. 时间计划

| 阶段 | 预计时间 |
|------|----------|
| 删除 `reject.py` + 改造 `report.py` | 1h |
| 重写 `visual-audit` skill | 1h |
| 重写 `sandbox-segment` / `batch-segment` skill 相关章节 | 1.5h |
| 更新文档索引与状态报告 | 0.5h |
| 真实 panel 闭环验证 | 1.5h |
| **总计** | **5.5h** |
