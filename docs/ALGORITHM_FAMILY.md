# 分割算法家族对比与路由设计

> 基于 e001-e027 实验历史整理。
> 目标：设计一族候选算法，根据图像特征 + VLM 语义自动路由。

---

## 1. 实验全景图（e001-e027）

| 实验 | 算法 | 状态 | 关键结论 |
|------|------|------|----------|
| e001 | Watershed from seeds | 放弃 | 层间过渡渐变，无清晰梯度脊线，边界错位 |
| e002 | Nearest median | 保留 | 基线，尊重颜色分布，median_size=5 最佳 |
| e003 | Shape filter | 保留 | 后处理去噪（perimeter²/area>35），吞文字/等值线 |
| e004 | SLIC superpixel | 放弃 | 超像素与地质层位无关，层厚薄不均 |
| e005 | Auto-k | 保留 | 自动补充种子，但 tolerance 敏感 |
| e006 | Baseline | 参考 | 基准参考图，非算法 |
| e007 | K-means full panel | 保留 | LAB 空间全局 K-means，效果与 nearest_median 相当 |
| e008 | Mean Shift | 部分 | 自动发现模式，但容易 oversegmentation（18-86 modes） |
| e009 | Graph Cut | 未明确 | 多标签图割 |
| e010 | Active Contour | 未明确 | 水平集/蛇形模型 |
| e011 | GMM | 放弃 | 不比 K-means 好，pastel 更差，7× slower |
| e012 | Region Merging | 保留 | Mean Shift overseg + Ward 合并，vivid 图效果好 |
| e013 | SLIC + Graph Merge | 未明确 | |
| e014 | Edge-guided K-means | 保留 | Canny 边缘 + 选择性 snap，边界更清晰 |
| e015 | Bilateral + K-means | 放弃 | 不改善分割，8-11× slower |
| e016 | Edge-enhanced Region Grow | 保留 | Dijkstra + edge barrier，penalty=100 最佳 |
| e017 | Ensemble Voting | 保留 | 3 算法投票，一致性最高（0.98），饱和度门控 |
| e018 | Morph Clean | 未明确 | 形态学清理 |
| e019 | Prior Constrained | 未明确 | 地质先验约束 |
| e020 | Unmap Baseline | 参考 | |
| e021 | VTracer | 参考 | 矢量化 |
| e022 | Perceptual Pipeline | 保留 | v4 generic：detect_noise_mask + inpaint + kmeans_lab |
| e023 | Correction WebApp | 工具 | 人工修正（Brush/Eraser） |
| e024 | Routed Pipeline | 保留 | 像素特征自动路由（sat/edge/line） |
| e025 | Multimodal vs Pixel | 保留 | 像素 + caption/colorbar OCR，4 篇文献验证成功 |
| e026 | PH01 Conceptual | 当前 | 对 ph01 有效，但泛化性差 |
| e027 | SLIC + Graph Cut | 保留 | 边界更平滑，conceptual model panels 首选 |

---

## 2. 候选算法引擎（按适用场景分类）

### A. 颜色聚类引擎（无 VLM seeds）

| 引擎 | 核心 | 适用 | 速度 | 质量 |
|------|------|------|------|------|
| **e026** | KMeans RGB + NN | 简单 jet-colormap，已知 n_layers | ~0.1s | 中 |
| **e027** | SLIC + Graph Cut (ICM) | Conceptual model panels，需平滑边界 | ~0.3s | 高 |
| **v4_kmeans** | KMeans LAB + shape filter | Vivid jet/rainbow，有/无 VLM seeds | ~0.2s | 高 |
| **region_merge** | Mean Shift overseg + Ward | Vivid 图，需自动确定 k | ~3s | 高 |

### B. VLM-seed 驱动引擎

| 引擎 | 核心 | 输入 | 优势 |
|------|------|------|------|
| **nearest_median** | 最近邻 + median 滤波 | VLM reps | 快速，尊重颜色分布 |
| **edge_guided** | Canny + 选择性 snap | VLM reps | 渐变边界更清晰 |
| **edge_grow** | Dijkstra + edge barrier | VLM reps | 区域生长不越界 |
| **ensemble** | 多算法一致性投票 | VLM reps | 综合精度最高，一致性 0.98 |

### C. 特殊场景引擎

| 引擎 | 场景 | 关键参数 |
|------|------|----------|
| **grayscale_agglomerative** | 灰度 well log / 截面 | 行均值层次聚类 |
| **brightness_gradient** | 白底 + 线条图 | 亮度梯度阈值 |
| **pastel_faded** | 低饱和 faded 图 | colorbar seeds 引导 |

---

## 3. 两层路由架构

```
VLM Multimodal Review (M1a)
  → figure_type: velocity_model / reflection_amplitude / uncertain
  → panel_structure: 单 panel / 多 panel / 3D isosurface
  → has_colorbar: true / false
  → physical_hint: Vp/Vs 范围、depth 范围、formation 名称

Pixel Router (M1b-CV)
  → saturation_ratio > 0.5? → vivid path
  → saturation_ratio 0.1-0.5? → mixed path
  → saturation_ratio < 0.1? → pastel/grayscale path
  → edge_density + line_count → 是否 multi-panel

Algorithm Selector
  ├─ VLM 判定 non-velocity → 跳过分割
  ├─ 多 panel → detect_panels → 每个 sub-panel 递归路由
  ├─ 单 panel + vivid + 需最高精度 → ensemble (slow)
  ├─ 单 panel + vivid + 需平滑边界 → SLIC+Graph Cut (e027)
  ├─ 单 panel + vivid + 平衡速度/质量 → v4_kmeans
  ├─ 单 panel + vivid + 渐变边界 → edge_guided
  ├─ 单 panel + pastel + 有 colorbar → pastel_faded
  ├─ 单 panel + grayscale → grayscale_agglomerative
  └─ 单 panel + uncertain → fallback v4_kmeans
```

---

## 4. 关键发现：什么决定了分割质量？

从 e001-e027 的实验历史中，**种子质量 > 聚类算法 > 后处理**：

1. **种子质量是决定性因素**（e005, e011）：
   - Auto-k 能发现 VLM 遗漏的层
   - 所有聚类算法（K-means, GMM, nearest_median）在好种子下表现接近
   - GMM 的 7× 速度提升没有带来可见质量改善

2. **颜色空间选择影响边界平滑度**（e007, e014, e027）：
   - LAB 空间优于 RGB（人眼感知均匀）
   - Edge-guided 对渐变边界有改善
   - SLIC+Graph Cut 对 conceptual model 边界最平滑

3. **后处理是必需的**（e003, e012）：
   - Shape filter（perimeter²/area > 35）自动去除文字/线条
   - 小区域合并（<0.1% area）防止 oversegmentation

4. **图像类型路由比单一算法更可靠**（e024, e025）：
   - 灰度图、pastel 图、vivid 图需要完全不同的策略
   - 纯像素方法无法区分 velocity model vs 反射振幅
   - Multimodal（caption + colorbar）提供关键排除信号

---

## 5. 与 geoseg v2 的对应

| v2 模块 | 承载的算法/功能 |
|---------|----------------|
| `mineru_client` (M0.5) | 提取 figure + caption → e025 multimodal 的文本输入 |
| `vlm_client.review_page_overview` (M1a) | e025 的 VLM 语义判断：figure_type, panel 数, is_velocity, 物理量 |
| `cv_detect` (M1b) | e024 router 的像素特征提取 + panel detect |
| `segment_engines/` (M3) | 候选算法引擎池：e026 / e027 / v4_kmeans / edge_guided / ensemble |
| `e026_algo/components.py` | 统一后处理：shape filter + component extract |
| `vlm_client.review_panel_detection` (M1c) | 检查 CV panel detect 是否漏/错 |
| `vlm_client.review_segmentation` (M3.5) | 评估分割质量，决定是否换引擎重试 |

---

## 6. 待在新流程上重新验证的算法

以下算法在旧流程（孤立 RGB + 人工 crop）上测试，值得在新流程（MinerU 提取 + VLM 语义 + 物理先验）上重新实验：

| 优先级 | 算法 | 为什么值得重试 |
|--------|------|---------------|
| P0 | e027 SLIC+Graph Cut | 已在 ph01 上验证，新流程下测试其他文献 |
| P0 | v4_kmeans (segment.py) | 核心引擎，多文献基准 |
| P0 | ensemble (e017) | 最高一致性，新流程下验证速度/质量 tradeoff |
| P1 | edge_guided (e014) | 渐变边界改善，测试非 ph01 场景 |
| P1 | region_merge (e012) | 自动确定 k，测试是否需要 |
| P1 | multimodal (e025) | 已在 4 篇文献验证，整合进 v2 pipeline |
| P2 | edge_grow (e016) | Dijkstra 替代 K-means，测试边界保持 |
| P2 | e026 (当前) | 保留为轻量 fallback，对比 v4_kmeans |
| P3 | GMM (e011) | 虽然之前结论负面，但 multimodal 提供更好种子后可能改善 |
| P3 | Mean Shift (e008) | 自动 k 发现，新种子质量下可能改善 |
