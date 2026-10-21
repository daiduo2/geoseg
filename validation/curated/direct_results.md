# Direct results index (recovered from artifacts)

> 2026-10-21. 目的：在重跑任何算法前，先把 curated panel 对应的历史直接结果、去噪/去文字产物和中间版本找回来。
>
> **结论：数据没有真正丢失，丢的是索引。** 结果仍在 `~/geoseg-artifacts/2026-10/runs`（repo 的 `runs` symlink），但 `validation/curated`、`build_corpus.py`、`panel_history.py` 只记录了最终 labels，没有把 OCR/text mask、clean image、geometry、refinement variants 和 audit 产物连回 panel。

## 1. 每个 panel 的直接结果

### newimage_p0..p4

- 当前 curated 来源：`runs/preprocess_newimage_iter5/panels/panel_N/{labels.npz,overlay_legend.jpg}`
- 输入/去噪：`runs/preprocess_newimage_iter5/01_original.jpg`、`02_combined_mask_full.jpg`、`03_cleaned.jpg`
- 全部历史版本：
  - `runs/preprocess_newimage_v1`（cleaned + colorbar-guided labels/overlays + diff）
  - `runs/preprocess_newimage_iter2`
  - `runs/preprocess_newimage_iter3`
  - `runs/preprocess_newimage_iter4`
  - `runs/preprocess_newimage_iter5`（curated 采用）
  - `runs/preprocess_newimage_perpanel`
  - `runs/preprocess_newimage_perpanel2`
  - `runs/preprocess_newimage_colorbar16`
  - `runs/preprocess_newimage_merged`（包含 best-split summary、comparison、txt export）
  - `runs/preprocess_newimage_final`
- 关键丢失项：`v1` 的 colorbar-guided 结果、`iter2-4` 的中间 cleaned/mask、`merged` 的 best-split audit、`colorbar16` 的 16-zone 结果，均未进入 curated manifest。

### paper22_fig5a

- 当前 curated：`runs/paper22/final/fig5a/{labels.npz,overlay.png,meta.json}`
- 直接 final 产物（都在 `runs/paper22/final/fig5a/`）：
  - `mask.png`
  - `ocr_text_mask.png`、`ocr_rois.json`（PaddleOCR ROI）
  - `boundary_audit.png`、`boundary_trace_2.91_3.3.json`
  - `residual_audit.png`、`rgb_vector_residual_abs.png`、`rgb_vector_residual_summary.json`
  - `comparison_original_mask_rgb_vector_residual.png`
  - `audit/views/`、`audit/crops/`、`audit/report.json`
  - `previous/`、`previous_ocr_residual/`（被 superseded 的结果）
- 去噪/refinement：
  - `runs/paper22/refinement/fig5a/labels_8material.npz`
  - `ocr_mask.png`、`ocr_inpainted.png`、`ocr_inpaint_median{3,5,7}.png`、`ocr_poly_inpaint_d{0,1,2}.png`
  - `palette_lab_t{8,12,16}.npz/png`、`palette_summary.json`
  - `overlay_8material.png`
- baselines：`runs/paper22/baselines/fig5a/{v4,slic,e027}_labels.npz` + overlays
- 关键丢失项：OCR mask/inpaint、palette、boundary trace、residual audit、audit views 都没有被 curated/corpus 索引。

### paper22_fig6

- 当前 curated：`runs/paper22/final/fig6/{labels.npz,overlay_rectified.png,meta.json}`
- 直接 final 产物：
  - `mask_rectified.png`、`mask_original_geometry.png`
  - `labels_original_crop.npz`、`overlay_original_geometry.png`
  - `comparison_rectified_boundary_mask.png`、`comparison_original_geometry_boundary_mask.png`
  - `mask_rectified_residual_palette.png`、`rgb_vector_residual_abs.png`、`rgb_vector_residual_summary.json`
  - `audit/views/`、`audit/crops/`、`audit/report.json`
- 几何/去噪：
  - `runs/paper22/geometry/fig6_full_crop.png`
  - `fig6_rectified_mask.png`
  - `transform.json`、`transform_candidates.json`
  - `fig6_affine_candidate.png`、`fig6_perspective_candidate.png`
- refinement：
  - `runs/paper22/refinement/fig6/barrier6_c_*`、`barrier_watershed_c_*`
  - `semantic5_c_*`
  - `watershed_sigma_*`
  - 各自 `.npz` + overlay + summary
- baselines：`runs/paper22/baselines/fig6/{v4,slic,e027}_labels.npz`
- 相关但**不是同一个 curated front panel** 的 Figure-6 profile 迭代：
  - `runs/feng_fig6_*`（多轮 text remove / palette / crop / label repair）
  - `runs/fig6_colorbar_16zone_*`（colorbar 16-zone / palette calibration / text clean / smooth）
  - `runs/pm_repair_*`（profile 04/05 的 OCR repair）
  - 这些目录里有 `fig6_profile_03..07` 的结果、text mask、cleaned_rgb、comparison；使用前必须先确认 profile，不能和 `final/fig6` 混用。
- 关键丢失项：geometry transform、rectified/normalized masks、refinement variants、feng profile 系列均不在 curated manifest。

### ph01_page8_panel3

- 当前 curated：`runs/ph01_page8/{panel3_labels.npz,panel3_overlay.jpg,panel3_properties.json,panel3_polygons.geojson,panel3_tomo.xyz}`
- 相关历史：
  - `runs/agent_review/ph01`、`runs/agent_review_vlm/ph01` 是页面级 review，不包含这个 335×258 的 panel3 labels。
  - `runs/readme_examples_v2/ph01_page011`、`ph01_page004` 是其他 page/panel，不是本 panel。
- **没有找到 ph01 的去文字/去噪变体。**
- 当前 curated labels 本身缺层：label 2 合并了浅蓝 `(183,186,217)` + 灰紫 `(167,160,176)`；label 4 合并了棕色 `(167,110,83)` + 灰褐 `(148,121,102)`；label 3 仍含大量白色背景/文字。
- 因此 ph01 不能拿 curated labels 当“已完成的分区”；需要从既有引擎族重新取更高层数结果，而不是继续在当前 labels 上合并。

### paper23_fig7a

- 当前 curated：`runs/sandbox/paper23_fig7a_clear/reviewed/secondary_labels.npz`、`secondary_overlay.png`
- 同一直接结果链：
  - `runs/sandbox/paper23_fig7a_clear/fig7a_crop.png`
  - `first_stage/first_stage_labels.npz`
  - `autonomous/secondary_labels.npz`
  - `reviewed/secondary_labels.npz`（curated 采用）
  - `reviewed/secondary_overlay_boundaries.png`、`reviewed/secondary_regions.png`
  - `reviewed/completed_fault_barrier.png`、`reviewed/splitting_barrier_mask.png`
  - `reviewed/report.json`（method、reviewed fault curves、completion history）
  - `uncertain_fault_hints.json`
  - F1/F2/F3 correction comparisons：`f3_*_before_after.png`
- 其他直接结果：
  - `runs/sandbox/paper23/{labels.npz,labels_tight_v4.npz,labels_v4.npz,edge_guided/labels.npz,ensemble/labels.npz,custom_pipeline/labels.npz}`
  - `runs/sandbox/paper23_connected/{connected_labels.npz,boundary_and_fault_overlay.png,residual_lines.png}`
  - `runs/sandbox/paper23_existing_pipeline/panel0/*`（labels/tomo/geojson/properties）
  - `runs/sandbox/paper23_existing_pipeline_red_boundary/panel0/*`（red boundary mask）
- 关键丢失项：first-stage/autonomous、fault barrier/masks、intervention history、connected labels、red-boundary pipeline 都没进 curated。

### paper23_fault_guided

- 当前 curated：`runs/sandbox/paper23_fault_guided/secondary_labels.npz`
- 同目录直接产物：
  - `secondary_overlay.png`、`secondary_overlay_boundaries.png`、`secondary_regions.png`
  - `completed_fault_barrier.png`、`splitting_barrier_mask.png`
  - `intervention_history_comparison.png`、`intervention_history_masks.png`、`intervention_history_overlays.png`、`intervention_history_comparison_before_current_final.png`
  - `uncertain_fault_hints.json`、`report.json`
- 上游：
  - `runs/sandbox/paper23_connected/{connected_labels.npz,cropped_profile.png,boundary_and_fault_overlay.png}`
  - `runs/sandbox/paper23_panel/labels.npz`
  - `runs/sandbox/paper23_existing_pipeline(_red_boundary)/panel0/*`
- 关键丢失项：fault masks/history/upstream connected labels 未进 curated。

### schematic_panel2

- 当前 curated：`runs/3d_schematic_retune/best/panel_2_best_labels.npy`（curated 转存为 npz）、`panel_2/text_removed_best.png`
- 直接结果链：
  - `runs/3d_schematic_retune/best/panel_2_best_overlay.jpg`
  - `best/panel_2_best_audit_summary.jpg`
  - `panel_2/text_removed_best.png`、`text_removed_best.json`
  - `panel_2/text_grid_results.json`、`panel_2/text/`（81 个 text-removal 网格结果）
  - `panel_2/candidates.json`、`audit_scores.json`、`panel_report.json`
  - `runs/sandbox_text/panel_2/{panel_no_text.png,text_mask.png,labels.npz,text_removal.json,overlay_legend.jpg}`
  - `runs/sandbox_notext/panel_2/{labels_cleaned.npz,overlay_cleaned.jpg}`
  - `runs/3d_schematic_correct_e2e`、`3d_schematic_e2e`、`3d_schematic_edge_guided`
- 关键丢失项：81 个 text grid、candidates/audit、sandbox_text/notext 变体均未进入 curated；curated 只保留最终 best panel2。

## 2. 索引为什么漏

1. `scripts/dev/build_corpus.py` 只扫少数固定根目录：
   - `fig6_profile_all_best_summary`、`self_heal_v1`、`j_tecto_*`、`agent_review*`、`readme_examples_v2`、`sandbox`、`preprocess_newimage_merged`。
   - 没有扫 `paper22/` dedicated tree、`preprocess_newimage_iter*`、`feng_fig6_*`、`fig6_colorbar_*`、`3d_schematic_*`、`sandbox_text/notext`。
2. 只按 `*labels*.npz` 建 panel key：
   - 3D 的 `.npy` best labels 被漏掉；
   - 只有 text mask/cleaned image 的目录没有 key；
   - OCR/residual/geometry/audit 文件没有 labels，直接被忽略。
3. `panel_history.py` 同样只统计 `*labels*.npz`，`export_view_history.py` 通过 labels 反查目录，所以没有 labels 的去噪目录不进索引。
4. `validation/curated` 是手工挑的最终标签；manifest 里没有 `variants` / `artifacts` / `denoise` 链路，只记录 final labels/overlay/export。
5. `docs/all_overlays` 是 5k+ 文件的扁平归档，没有 panel/provenance 映射；目录整理把数据收进 artifacts 后，没有重建索引。

## 3. 恢复建议

- **不要重跑** paper22、paper23、schematic、newimage 的去噪/分割；直接采用上面的直接结果作为 baseline。
- 每个 panel 恢复三条链：
  1. `source image / geometry / crop`
  2. `denoise / text mask / cleaned image`
  3. `final labels / overlay / audit / export`
- 优先恢复：
  - paper22 fig5a/fig6：final + OCR/text mask + geometry + residual/audit；
  - paper23 fig7a/fault：reviewed/secondary + fault masks + intervention history；
  - schematic panel2：best + text grid + sandbox_text/notext；
  - newimage：iter5 + merged best-split + colorbar16；
  - ph01：直接结果存在但缺层，且没有去噪变体；需单独重新取更高层数，不能冒充已完成 baseline。
- 下一步应先重建机器可读索引（panel → all artifacts → provenance），再决定哪些结果直接进入 curated，哪些需要人工视觉复核。

## 4. 直接结果的视觉审计（不含 ph01）

审计对象：上面的 direct final overlay + denoise/audit 辅助图；没有重跑任何算法。

对照图：[`runs/direct_results_review/contact.png`](../../runs/direct_results_review/contact.png)

| panel | direct 结果 | 视觉结论 | 仍需注意 |
|---|---|---|---|
| newimage_p0 | `preprocess_newimage_iter5` | 可用；3 个连续色区，边界与原图一致，远好于近期 batch | 黑色十字/文字仍留在蓝色区内 |
| newimage_p1 | iter5 | 可用；4 个连续层，n=4 合理 | HVB 文字残留 |
| newimage_p2 | iter5 | 可用；3 个层，绿色主体和顶部薄层连续 | 文字/刻度残留 |
| newimage_p3 | iter5 | 可用；3 个层，低对比区域保持连续 | 黑色文字碎片 |
| newimage_p4 | iter5 | 可用；3 个层，中央绿色透镜完整 | 文字/刻度残留 |
| paper22_fig5a | `paper22/final/fig5a` | 可用；8 个密度区完整，2.91/3.3 边界经过专门追踪 | 密度文字仍显示在 overlay；OCR mask 已保留 |
| paper22_fig6 | `paper22/final/fig6` | 可用；5 个构造单元，几何校正和边界驱动结果正确 | 文字/图例仍显示；不能再用 RGB residual 当质量指标 |
| paper23_fig7a | `sandbox/paper23_fig7a_clear/reviewed` | 可用；reviewed material/fault 分区，F1/F2/F3 修正完整 | 这是 fault-guided material map，不是纯速度层；文字/图例仍在 |
| paper23_fault_guided | `sandbox/paper23_fault_guided` | 可用；fault barrier + completion 历史完整 | 同上，不是干净的速度层 |
| schematic_panel2 | `3d_schematic_retune/best` | 可用；panel2 最佳结果，text_removed_best 已处理文字 | 3D schematic，不进入 2D SPECFEM 导出 |

总体判断：

1. 这些 direct results 明显优于近期 batch/iterative 结果，应该作为 baseline，而不是继续重跑现用算法。
2. direct results 仍不是“无噪声”：文字、图例、黑色标记仍有残留，只是它们已经被历史流程处理到可接受程度。
3. paper22 的 OCR/text mask、geometry transform、residual/audit 图是解释结果的关键证据，必须和 final labels 一起保存；不能只留 labels。
4. paper23 的 reviewed/secondary + fault masks/intervention history 是一个完整链，不能只取 secondary labels。
5. ph01 不在本表内：它的 direct labels 缺层且没有去噪链，不能作为完成 baseline。
