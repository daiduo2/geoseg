# experiments/3d_schematic 最优 Overlay 清单

> 从 `experiments/3d_schematic/` 子项目（含 `experiments/`、`results/`、`figures/`）中挑选出的当前最佳结果。
> 文件同时复制到 `docs/best_overlays_3d_schematic/` 方便集中查看。

## 一、原始输入

| 编号 | 类别 | 文件 | 原始路径 | 说明 |
|------|------|------|----------|------|
| 01 | Panel 1 原图 | `best_overlays_3d_schematic/01_panel_1_original.png` | `experiments/3d_schematic/figures/panels/panel_1.png` | 地幔柱概念模型，含文字标注 |
| 02 | Panel 2 原图 | `best_overlays_3d_schematic/02_panel_2_original.png` | `experiments/3d_schematic/figures/panels/panel_2.png` |  uplift/removed lithosphere 变体 |
| 03 | Panel 3 原图 | `best_overlays_3d_schematic/03_panel_3_original.png` | `experiments/3d_schematic/figures/panels/panel_3.png` | 复杂 weak zone / residues 场景 |

## 二、文字移除（Text Removal）

| 编号 | 类别 | 文件 | 原始路径 | 说明 |
|------|------|------|----------|------|
| 04 | Panel 1 去文字 | `best_overlays_3d_schematic/04_panel_1_text_removed.png` | `experiments/3d_schematic/experiments/text_removal_v2/final_pipeline/panel_1_final.png` | 最终 pipeline 输出，文字几乎完全消除 |
| 05 | Panel 2 去文字 | `best_overlays_3d_schematic/05_panel_2_text_removed.png` | `experiments/3d_schematic/experiments/text_removal_v2/final_pipeline/panel_2_final.png` | 同上，箭头/标签残留较少 |
| 06 | Panel 3 去文字 | `best_overlays_3d_schematic/06_panel_3_text_removed.png` | `experiments/3d_schematic/experiments/text_removal_v2/final_pipeline/panel_3_final.png` | Panel 3 仍有少量文字残留 |

**文字移除关键结论**（来自 `results/experiment_plan_repair/` 对比）：
- **Telea r=3** 为全局最优：残留、边缘、颜色、伪影、结构均 5/5，总分 25。
- NS r=3 次之（25 分），Biharmonic 第三（24 分但慢 10 倍）。
- Median / Bilateral / Gaussian 都会破坏薄层边界或留下可见伪影。

## 三、分割（Segmentation）

基于历史 `experiments/segmentation_experiment/visual_audit_scores.md` 的加权评分（满分 20，目标 ≥16）。**注意**：v2 视觉审计已废弃硬评分，改为 agent 视觉批评 + RegionalAudit 闭环；下表分数仅反映历史实验结果，不代表当前通过标准。

| 编号 | 面板 | 最优引擎 | 文件 | 原始路径 | 加权分 | 状态 |
|------|------|---------|------|----------|--------|------|
| 07 | Panel 1 | v4_kmeans n=8 | `best_overlays_3d_schematic/07_panel_1_best_segmentation_v4_kmeans_n8.png` | `results/segmentation_experiment/group_a/panel_1_v4_kmeans_nl8_fill.png` | **19.3/20** | PASS |
| 08 | Panel 2 | slic_kmeans n=8 | `best_overlays_3d_schematic/08_panel_2_best_segmentation_slic_kmeans_n8.png` | `results/segmentation_experiment/group_a/panel_2_slic_kmeans_nl8_fill.png` | **17.4/20** | PASS |
| 09 | Panel 3 | v4_kmeans n=10 | `best_overlays_3d_schematic/09_panel_3_best_segmentation_v4_kmeans_n10.png` | `results/segmentation_experiment/group_a/panel_3_v4_kmeans_nl10_fill.png` | **15.5/20** | FAIL |

**说明：**
- Panel 1：v4_kmeans n=8 正确分离 crust / lithosphere / plume / mantle，仅在顶部有轻微碎片。
- Panel 2：slic_kmeans n=8 最能区分 uplift / plume / mantle 结构。
- Panel 3：当前所有引擎均未达标；v4_kmeans n=10 是相对最优，但 uplift/weak zone 仍有丢失。

## 四、端到端合并输出

| 编号 | 类别 | 文件 | 原始路径 | 说明 |
|------|------|------|----------|------|
| 10 | 三面板合并 v3 | `best_overlays_3d_schematic/10_final_merged_v3.png` | `experiments/3d_schematic/results/merged/result_final_v3.png` | Original / Text Removed / Label Fill / Boundaries 四列对比总览 |

## 五、审计/对比图

| 编号 | 类别 | 文件 | 原始路径 | 说明 |
|------|------|------|----------|------|
| 11 | Panel 1 全方法审计网格 | `best_overlays_3d_schematic/11_audit_grid_panel_1.png` | `experiments/3d_schematic/results/experiment_plan_repair/audit_grids/panel_1_audit_grid.png` | 分组展示 reference、repair replacement、detect repair、post-smooth、full pipeline 结果 |
| 12 | Panel 1 最终 audit | `best_overlays_3d_schematic/12_final_lap_audit_panel_1.png` | `experiments/3d_schematic/results/experiment_plan_repair/final_lap_audit/panel_1_full_compare.png` | 三图对比：原始 / 修复后 / 差值 |
| 13 | Panel 2 最终 audit | `best_overlays_3d_schematic/13_final_lap_audit_panel_2.png` | `experiments/3d_schematic/results/experiment_plan_repair/final_lap_audit/panel_2_full_compare.png` | 同上 |
| 14 | Panel 3 最终 audit | `best_overlays_3d_schematic/14_final_lap_audit_panel_3.png` | `experiments/3d_schematic/results/experiment_plan_repair/final_lap_audit/panel_3_full_compare.png` | 同上 |

## 六、失败/不适合的方法

| 方法 | 代表文件 | 结论 |
|------|----------|------|
| Diff-overlay | `results/segmentation_experiment/group_b/*` | 极端碎片化（2400-3900 labels，frag≈0.996），不适合 |
| Felzenszwalb-only | `results/segmentation_experiment/group_c/*` | 4200-16800 labels，frag=1.00，地质结构完全丢失 |
| Grayscale engine | `results/segmentation_experiment/group_a/panel_*_grayscale_*` | Panel 3 完全失败（5.2/20） |

## 七、关键结论

1. **文字移除**：Telea r=3 是当前最优修复策略，panel 1/2 效果干净，panel 3 仍有少量残留。
2. **分割**：geoseg 引擎族是唯一可行策略；Panel 1/2 已达标，Panel 3 仍是瓶颈。
3. **Panel 3 失败根因**：复杂结构（weak zone、refractory residues、uplift/plume 边界弱）+ 顶部文字干扰，超出当前 k-means 系列能力。
4. **建议下一步**：
   - 对 Panel 3 引入 region-aware 后处理（合并/拆分 weak zone）
   - 或尝试基于扩散模型 / SAM 的交互式精修
   - 收集 Panel 3 人工标注 ground truth，建立客观评估基准

## 使用方式

```bash
# 查看最佳 overlay
open docs/best_overlays_3d_schematic

# 查看评分来源
cat experiments/3d_schematic/experiments/segmentation_experiment/visual_audit_scores.md
```
