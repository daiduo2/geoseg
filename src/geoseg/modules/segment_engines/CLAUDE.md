# segment_engines — 模块契约

> **Segmenter Protocol 实现者**。路由 + 执行分割算法，输入 PanelInput，输出 SegmentationResult。

## 职责

- 提供多算法分割引擎族，供 agent / controller 按需调用
- 支持 `n_layers`、`reps`（种子点）、`colorbar_rgb` 等参数
- 返回标准 `SegmentationResult`（`labels` + `overlay` + `meta`）
- `registry.py` / `policy.py` / `runner.py` / `retry.py`：稳定调度边界。`registry.py` 维护 engine metadata、callable path 和 adapter；`runner.py` 只按 registry 加载并执行
- `diagnostics/metrics.py`：多引擎评估与对比
- `strategy/` + `strategy_memory.py` facade：agent 策略学习（引擎选择历史与效果追踪）
- `internal/shared.py`：引擎间内部共享工具函数
- `diagnostics/batch_test.py`：批量测试 runner
- `diagnostics/compare_results.py`：多结果对比可视化

## 实现 Protocol

| Protocol | 函数 | 说明 |
|----------|------|------|
| `Segmenter` | `route_and_segment(img_rgb, **kwargs)` | 路由到可用引擎（agent 自主决策优先） |

## 引擎族

| 文件 | 引擎 |
|------|------|
| `v4/` | v4 K-Means 路径实现：jet vivid / colorbar-guided / pastel fallback |
| `v4_kmeans.py` | v4 K-Means 兼容 facade |
| `edge_guided.py` | Edge-guided 分割 |
| `edge_grow.py` | Edge-grow 区域生长 |
| `edge/` | edge engine 共享 seed、gradient、postprocess helpers |
| `e027_slic_graphcut.py` | SLIC + GraphCut（e027） |
| `slic_kmeans.py` | SLIC 超像素 + K-Means（e028，文字鲁棒） |
| `kmeans_full.py` | K-Means 全图版 |
| `grayscale.py` | 灰度 agglomerative |
| `ensemble.py` | 多引擎融合 |
| `regional/` | regional fusion 的配置、overlay、split/merge、融合编排实现 |
| `regional_fusion.py` | regional fusion 旧公共路径 facade |
| `registry.py` | 引擎注册表，包含 callable path 与 runner adapter |
| `policy.py` | 路由策略 |
| `runner.py` | 按 registry spec 执行引擎，处理 fallback 和结果归一化 |
| `retry.py` | retry 策略 |
| `compat/` | legacy stage/full-pipeline/shared 兼容 re-export |
| `pipeline_stages.py` | legacy stage helper 旧路径；实现集中在 `compat/` |
| `full_pipeline.py` | `process_figure` 旧路径；实现集中在 `compat/` |
| `horizon/` | horizon refinement 的 coarse/boundary/fitting/refine 实现 |
| `vlm_reps.py` | VLM 种子点辅助 |
| `strategy/` | strategy memory records/scoring/template persistence 实现 |
| `strategy_memory.py` | strategy memory 旧公共路径 facade |
| `internal/shared.py` | engine family 内部工具兼容 aggregate |
| `internal/seeds/` | seed search/CV/refine/scan/auto-k helpers |
| `diagnostics/` | metrics、batch、comparison 诊断工具 |

## 预处理策略（2026-06-01 更新）

### 问题

文字标注（速度值、地层名、轴标签）覆盖在彩色地层上时：
- `adaptive_blur`（各向同性高斯模糊）把文字变成"脏迹"，仍干扰聚类
- k-means 在 LAB 空间把黑色/白色文字当作独立颜色簇
- edge-guided 把文字边缘检测为假边界

### 实验结论（e028-e030，6 张真实图像 + 视觉验证）

| 方案 | 碎片减少 | 边界保留 | 欠分割 | 结论 |
|------|---------|---------|--------|------|
| `adaptive_blur`（当前） | 基准 | 中等 | 无 | 各向同性，文字变脏迹 |
| **row_median(size=5)** | **-12%~-62%** | **优秀** | 无 | **替换 baseline** |
| row_median + median_post | **-50%~-70%** | 优秀 | 无 | **v4_kmeans 全路径启用** |
| SLIC + K-Means | ~-95% | 中等 | **严重** | 辅助引擎，非主引擎 |

### 集成决策

1. **`internal/shared.py` 预处理**：`adaptive_blur` → `row_median_filter(size=5)`
   - 利用地球物理图像**水平分层**先验：行内中值滤波去除水平文字脉冲，保留垂直层边界
   - 对垂直文字（y 轴标签）效果弱，但不破坏其周围地层

2. **`v4_kmeans.py` 后处理**：`colorbar_guided` 和 `pastel_faded` 路径补充 `median_filter(size=5)`
   - `jet_vivid` 路径已有（`_nearest_median` 中），其余两路径缺失

3. **`slic_kmeans.py` 新引擎**：SLIC 超像素 + 超像素级 K-Means
   - `n_segments=500, compactness=10`
   - 文字区域通常小于单个超像素，被自然吸收
   - **限制**：欠分割（合并相邻相似色层），仅作为 `sandbox-segment` 的可选引擎

## 与 cv_detect 的边界

- `cv_detect` 输出 `PanelInput` 列表（bbox）
- `segment_engines` 接收 `PanelInput`，对 crop 后的图像执行分割
- 分割结果通过 `SegmentationResult` 统一接口返回

## 不做

- 不做 panel 检测（交给 `cv_detect`）
- 不做 VLM 语义分析（交给 `vlm_client`）
- 不直接操作 GUI（纯后端模块）

## 测试

```bash
# 示例
python examples/geoseg/modules/segment_engines/demo.py

# 视觉对比（baseline vs improved）
python experiments/test_visual_comparison.py
```

## 实验目录

- `experiments/segment_engines/fh_slic_experiment.py` — e028：FH + SLIC 分割实验
- `experiments/segment_engines/spatial_regularized_experiment.py` — e029：空间正则化聚类实验
- `experiments/segment_engines/anisotropic_preprocessing_experiment.py` — e030：各向异性预处理实验
- `runs/visual_comparison/` — 视觉对比 overlay 输出
