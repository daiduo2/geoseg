# 各项最优 Overlay 清单

> 从 `docs/all_overlays/`（5,189 文件）中按类别挑选出的当前最佳结果。
> 文件同时复制到 `docs/best_overlays/` 方便集中查看。

| 编号 | 类别 | 最优文件（相对 `docs/`） | 原始路径 | 入选理由 |
|------|------|------------------------|----------|----------|
| 01 | 2D Velocity Model (jet 鲜艳) | `best_overlays/01_2d_velocity_silixa_page5_v4_pastel.png` | `all_overlays/readme_examples_v2/silixa_page5/overlay_v4_pastel_blend.png` | 4 层清晰分离，边界平滑，井孔异常处理干净，无文字碎片 |
| 02 | 2D Velocity Model (柔和/褪色) | `best_overlays/02_2d_velocity_c11b8db_edge_guided.png` | `all_overlays/readme_examples_v2/gras2019_c11b8db/overlay_edge_guided_blend.png` | 水平层位边界自然，edge_guided 对渐变边界改善明显 |
| 03 | Horizon Refinement | `best_overlays/03_horizon_refine_c11b8db.png` | `all_overlays/horizon_refine/final_c11b8db_refined_overlay.png` | 平滑地质合理边界，几乎无可见伪影 |
| 04 | Regional Fusion | `best_overlays/04_regional_fusion.jpg` | `all_overlays/regional_fusion_e2e/03_overlay_fused.jpg` | 双面板融合成功，上层地质层清晰，下层构造边界明确 |
| 05 | 3D Schematic | `best_overlays/05_3d_schematic_panel1_fused.jpg` | `all_overlays/3d_schematic_correct_e2e/panel_1_front/visual_audit/labels_fused/views/plume_comparison.jpg` | 三图对比中融合版羽流形状最佳，primary 过度分割问题被修复 |
| 06 | Tubular / Plume 结构 | `best_overlays/06_tubular_plume_warm_merge.jpg` | `all_overlays/tubular_panel3/exp1_warm_merge.jpg` | 羽流结构连贯性优于 single/frangi，边界相对完整 |
| 07 | Engine Compare (自动最佳) | `best_overlays/07_engine_compare_best_auto.jpg` | `all_overlays/engine_compare_panel3/visuals/grayscale_color_mask.jpg` | 各自动引擎中最接近 ground truth，羽流形状完整 |
| 08 | Engine Compare (人工 GT) | `best_overlays/08_engine_compare_manual_gt.jpg` | `all_overlays/engine_compare_panel3/visuals/manual_gt_mask.jpg` | Panel 3 羽流人工标注 ground truth，用于对比基准 |
| 09 | 文字鲁棒预处理 | `best_overlays/09_text_robust_row_median_post.png` | `all_overlays/visual_comparison/page_004_img_0_panels_row_median_post.png` | row_median + post 后文字标注几乎完全消除，层位保持完整 |
| 10 | Self-Heal 最终分割 | `best_overlays/10_self_heal_silixa_final.jpg` | `all_overlays/self_heal_v1/silixa_page5/99_final_overlay.jpg` | 自修复流程最终输出，5 层干净分离，质量稳定 |
| 11 | Self-Heal 前后对比 | `best_overlays/11_self_heal_silixa_side_by_side.jpg` | `all_overlays/self_heal_v1/silixa_page5/99_side_by_side.jpg` | 左右对比展示修复效果，层数与边界显著改善 |
| 12 | Literature Test (端到端) | `best_overlays/12_literature_silixa_page4.jpg` | `all_overlays/literature_test/silixa2021/e2e_export_v4/page4_img0/panel0_overlay.jpg` | 2 层简单面板端到端成功，无噪声，边界清晰 |

## 关键观察

- **2D velocity model** 整体成熟度最高，`v4_kmeans` + `row_median` 已能稳定产出可用结果。
- **Horizon refinement** 对 `c11b8db` 效果显著，但 `16b0cf` 仍有轻微波纹伪影。
- **3D schematic / tubular** 仍是硬案例：Panel 3 羽流漏斗在自动方法中均无法完美复现，目前最优是 `exp1_warm_merge` 的近似结构。
- **Regional fusion** 在异质双面板场景下展现出修复局部失败区域的能力。
- **Engine comparison** 显示 `grayscale` 在 Panel 3 上最接近 GT，但所有自动引擎距离理想仍有差距。

## 使用方式

```bash
# 查看最佳 overlay 目录
open docs/best_overlays

# 批量 review
ls docs/best_overlays/*.png docs/best_overlays/*.jpg
```
