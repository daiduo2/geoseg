# 3D 地质示意图分割模型实验 PRD

> 文档类型：实验设计 / 产品需求文档
> 版本：v1.0
> 日期：2026-06-09
> 负责人：Agent
> 目标：以 text_removal 最优输出为输入，系统性实验并确定 3D 地质示意图的最优分割策略

---

## 1. 背景与动机

### 1.1 上游交付

`text_removal.py` 已达到当前最优配置：

- **MSER + Laplacian(unfiltered)** 检测文字区域（亮度 filter 仅作用于 MSER，Laplacian 边缘绕过 brightness filter 捕获 anti-aliased 文字边缘）
- **Gaussian mask expansion** (sigma=7, threshold=0.3) 平滑扩展掩码
- **Telea inpaint** (r=7) 回填文字区域
- **Stage 2 residual cleanup** 清除残余

mask 覆盖率：~10-14%（Panel 1: 10.9%, Panel 2: 10.1%, Panel 3: 13.8%）。文字去除干净，地质结构保持完好。

### 1.2 当前分割问题

现有 `process_final_v3.py` 使用 **felzenszwalb + post-merge** 流程：

1. V-channel CLAHE enhancement
2. felzenszwalb (scale=300, sigma=0.5, min_size=30)
3. two-phase post-merge（小碎片合并 + 梯度感知保守合并）
4. fragment detection（LoG + HSV 蓝色过滤）

**失效模式**：

| 问题 | Panel 1 | Panel 2 | Panel 3 |
|------|---------|---------|---------|
| 文字修复痕迹被误分为独立区域 | 轻微 | 中等 | 严重（"weak zone"→蓝色区） |
| 过度分割（碎片过多） | 中 | 中 | 严重（绿色碎屑纹理） |
| 欠分割（相邻色层合并） | 轻微 | 轻微 | 中（黄色/橙色过渡层） |
| 边界锯齿/不连续 | 轻微 | 中（结构线干扰） | 轻微 |
| 地幔柱/断层结构断裂 | 无 | 轻微 | 无 |

### 1.3 为什么需要系统性实验

单一 felzenszwalb 方案无法同时解决：
- **文字残留污染**：即使 text_removal 已极限优化，inpaint 痕迹仍可能被敏感的分割算法捕获
- **Panel 间差异**：三个面板的地质表达风格差异大，单一参数无法适配
- **传统 CV vs ML 分割**：geoseg 引擎族已积累多种算法，但未在 3D schematic 上系统验证

---

## 2. 核心假设

### 假设 1：叠层隔离优于修复后分割

`design_diff_overlay.py` 提出的差分叠层方案——将高频细节（文字、标注、碎片）隔离为独立叠层标签，不参与地质分区竞争——比分割前修复文字更可靠。

### 假设 2：Panel 间需要差异化策略

- Panel 1（平坦渐变）：颜色聚类（k-means）或 felzenszwalb 即可
- Panel 2（结构线多）：edge-guided 或水平约束分割更优
- Panel 3（复杂纹理）：叠层隔离 + 大尺度 felzenszwalb，或 SLIC 超像素预聚合

### 假设 3：geoseg 引擎族可直接复用

geoseg v2 的 `segment_engines`（v4_kmeans, slic_kmeans, edge_guided, ensemble 等）经过多图验证，具备在 schematic 上运行的基础能力，只需调整参数。

---

## 3. 实验设计

### 3.1 数据集

输入统一为 text_removal 最优输出：

```python
from text_removal import remove_text_two_pass
final_img = remove_text_two_pass(image)[0]  # (H, W, 3) RGB
```

| Panel | 特征 | 预期难点 |
|-------|------|----------|
| Panel 1 | 灰→蓝→橙→黄 平坦渐变 | 色层过渡带模糊 |
| Panel 2 | 渐变 + 黑色结构线 + 箭头 | 结构线被误分为边界 |
| Panel 3 | 复杂纹理（绿色碎屑）+ 渐变 | 纹理导致过度分割 |

### 3.2 实验分组

#### 组 A：geoseg 引擎族 baseline

在 text-removed 图像上直接运行 geoseg segment_engines：

| 引擎 | 来源 | 核心机制 | 预期适配性 |
|------|------|----------|-----------|
| v4_kmeans | geoseg | LAB 空间 K-Means + colorbar seeds + shape filter | 中（依赖 colorbar） |
| slic_kmeans | geoseg | SLIC 超像素 + 超像素级 K-Means | 高（文字鲁棒） |
| edge_guided | geoseg | Canny 边缘 + 区域生长 | 中（Panel 2 结构线干扰） |
| grayscale | geoseg | 灰度 agglomerative | 低（丢失颜色信息） |
| ensemble | geoseg | 多引擎融合 + 投票 | 高（互补优势） |

**参数**：每个引擎使用默认参数 + 3 组 n_layers（4, 6, 8）
**产出数**：5 引擎 × 3 n_layers × 3 panels = 45 个任务

#### 组 B：diff-overlay 叠层隔离 pipeline

基于 `design_diff_overlay.py` 的完整实验：

**B1：差分提取参数网格**

| 参数 | 取值 |
|------|------|
| `blur_ksize` | 11, 15, 21, 31 |
| `blur_sigma` | 2.0, 3.0, 5.0 |
| `diff_thresh` | 15, 20, 30, 50 |
| `expand_radius` | 10, 15, 20 |

产出数：4 × 3 × 4 × 3 × 3 panels = 432（过多，采用拉丁超立方采样降至 36 个配置）

**B2：叠层 + 分割引擎组合**

对 B1 的最优配置，叠层隔离后分别用以下引擎分割非叠层区域：
- felzenszwalb (scale=300/500)
- v4_kmeans (n_layers=6)
- slic_kmeans (n_layers=6)

产出数：3 引擎 × 3 panels = 9 个任务

#### 组 C：3D-schematic 原生 pipeline 调优

优化现有 `process_final_v3.py` 的参数：

| 参数 | 当前值 | 扫描范围 |
|------|--------|----------|
| `felz_scale` | 300 | 100, 200, 300, 500, 800 |
| `felz_sigma` | 0.5 | 0.3, 0.5, 0.8, 1.0, 1.5 |
| `felz_min_size` | 30 | 10, 30, 50, 100 |
| `small_ratio` (post-merge) | 0.015 | 0.005, 0.01, 0.015, 0.02, 0.03 |
| `max_score` (post-merge) | 0.8 | 0.5, 0.6, 0.7, 0.8, 1.0 |
| `max_color` (post-merge) | 45 | 30, 45, 60, 80 |

产出数：5 × 5 × 4 × 5 × 5 × 4 = 10,000（过多，采用网格采样降至 60 个配置）

#### 组 D：跨策略对比

在统一评估标准下对比各组最优结果：

| 策略 | 来源 |
|------|------|
| REF_v3 | `process_final_v3.py` 当前参数 |
| REF_v3_best | 组 C 调优后最优 |
| GEO_v4_kmeans | 组 A 最优 v4_kmeans |
| GEO_slic | 组 A 最优 slic_kmeans |
| GEO_ensemble | 组 A 最优 ensemble |
| OVERLAY_felz | 组 B 最优 diff-overlay + felzenszwalb |
| OVERLAY_kmeans | 组 B 最优 diff-overlay + v4_kmeans |
| HYBRID | diff-overlay + ensemble 融合 |

产出数：8 策略 × 3 panels = 24 个任务

---

## 4. 评估指标

### 4.1 客观指标

| 指标 | 计算方法 | 说明 |
|------|----------|------|
| **n_labels** | 最终标签数量 | 地质层数 + 碎片数。Panel 1/2/3 地质层约 4-6 层，n_labels > 15 提示过度分割 |
| **fragment_ratio** | 面积 < 1% 的标签数 / 总标签数 | 碎片比例，越低越好 |
| **boundary_alignment** | 检测边界与颜色梯度的 IoU | 边界与真实颜色过渡的一致性 |
| **color_purity** | 每个标签内部颜色标准差均值 | 越低表示标签内部颜色越一致 |
| **overlay_coverage** | 叠层 mask 面积 / 总面级 | diff-overlay 专用，应覆盖所有文字/标注区域 |

### 4.2 主观视觉审计

沿用 `acceptance.md` 的 5 维度评分，权重针对分割场景调整：

| 维度 | 权重 | 说明 | 评分标准 |
|------|------|------|----------|
| 层位准确 | 1.5x | 地质层是否被正确分离，无合并/错分 | 5=完美分离；1=严重合并或错分 |
| 碎片控制 | 1.3x | 无面积 < 1% 的孤立碎片 | 5=无碎片；1=碎片遍布 |
| 边界质量 | 1.3x | 层位交界线是否平滑、连续、与地质结构一致 | 5=平滑连续；1=锯齿/断裂 |
| 文字免疫 | 1.5x | 文字/标注区域不产生虚假独立分区 | 5=完全免疫；1=文字变成独立区域 |
| 结构完整 | 1.0x | 地幔柱、断层、箭头等地质结构是否保持 | 5=完整；1=结构断裂/消失 |

**通过标准**：
- 层位准确 ≥ 4
- 碎片控制 ≥ 3
- 边界质量 ≥ 3
- 文字免疫 ≥ 4
- 加权总分 ≥ 16/20

### 4.3 下游任务指标（可选）

若时间允许，评估分割结果对后续 polygon extraction + SPECFEM export 的影响：
- 提取的多边形数量与人工标注的匹配度
- 属性赋值正确率（基于颜色→地质单元映射）

---

## 5. 资源需求

### 5.1 计算资源

| 资源 | 需求 | 说明 |
|------|------|------|
| CPU | Mac mini M4 | 纯 CPU 算法，无需 GPU |
| 内存 | < 2GB 峰值 | 图像尺寸 1740×3480，多引擎并行 |
| 磁盘 | ~200MB | 实验产出图 + 评估数据 |
| 时间 | 组 A 30min + 组 B 1h + 组 C 2h + 组 D 30min + 审计 2h = ~6h | 不含运行时间 |

### 5.2 软件依赖

```python
# 必须（已有）
numpy, opencv-python, scikit-image, scipy, Pillow

# geoseg 模块（已有）
geoseg.modules.segment_engines.{v4_kmeans, slic_kmeans, edge_guided, ensemble, grayscale}

# 评估（已有）
geoseg.modules.segment_engines.metrics
```

### 5.3 人力投入

| 阶段 | 时间 | 产出 |
|------|------|------|
| 环境准备 + 组 A 实现 | 2h | geoseg 引擎族 wrapper |
| 组 B diff-overlay 实验 | 2h | 参数扫描 + 叠层可视化 |
| 组 C pipeline 调优 | 2h | 参数网格运行 |
| 组 D 跨策略对比 | 1h | 统一对比图生成 |
| 视觉审计 | 2h | 5 维度评分表 |
| 报告撰写 | 1h | PRD 更新、集成建议 |
| **总计** | **10h** | |

---

## 6. 实现方案

### 6.1 代码模块结构

```
experiments/segmentation_experiment/
├── core/
│   ├── __init__.py
│   ├── geoseg_bridge.py         # 封装 geoseg segment_engines 调用
│   ├── diff_overlay.py          # diff-overlay pipeline（基于 design_diff_overlay.py）
│   ├── v3_pipeline.py           # process_final_v3.py 参数化版本
│   └── evaluator.py             # 客观指标计算
├── runners/
│   ├── group_a_geoseg.py        # geoseg 引擎族实验
│   ├── group_b_overlay.py       # diff-overlay 参数扫描
│   ├── group_c_v3_tune.py       # v3 pipeline 调优
│   ├── group_d_comparison.py    # 跨策略对比
│   └── evaluate.py              # 批量评估脚本
├── visualization/
│   ├── compare_grid.py          # 多策略对比网格图
│   ├── overlay_vis.py           # 叠层掩码可视化
│   └── score_table.py           # 评分表生成
└── README.md
```

### 6.2 核心接口设计

```python
from pathlib import Path
import numpy as np
from typing import Protocol

class SegmentationStrategy(Protocol):
    """统一分割策略接口。"""

    def segment(self, image: np.ndarray, **kwargs) -> dict:
        """
        Returns: dict with keys:
            - labels: (H, W) int array
            - overlay: (H, W, 3) RGB visualization
            - meta: dict with n_labels, fragment_ratio, etc.
        """
        ...


class DiffOverlayStrategy:
    """差分叠层隔离策略。"""

    def __init__(self, base_strategy: SegmentationStrategy,
                 blur_ksize: int = 15, blur_sigma: float = 3.0,
                 diff_thresh: float = 20.0, expand_radius: int = 15):
        self.base = base_strategy
        self.blur_ksize = blur_ksize
        self.blur_sigma = blur_sigma
        self.diff_thresh = diff_thresh
        self.expand_radius = expand_radius

    def segment(self, image: np.ndarray, **kwargs) -> dict:
        from diff_overlay import diff_overlay_pipeline
        result = diff_overlay_pipeline(
            image,
            blur_ksize=self.blur_ksize,
            blur_sigma=self.blur_sigma,
            diff_thresh=self.diff_thresh,
            expand_radius=self.expand_radius,
            **kwargs
        )
        return {
            "labels": result["final_labels"],
            "overlay": render_overlay(image, result["final_labels"]),
            "meta": {
                "overlay_coverage": result["overlay_mask"].mean(),
                "n_geo_labels": len(np.unique(result["geo_labels"])),
            }
        }
```

### 6.3 评估脚本

```python
def evaluate_segmentation(image: np.ndarray, labels: np.ndarray) -> dict:
    """计算客观指标。"""
    unique, counts = np.unique(labels, return_counts=True)
    total = image.shape[0] * image.shape[1]

    # n_labels
    n_labels = len(unique)

    # fragment_ratio
    small = counts < total * 0.01
    fragment_ratio = small.sum() / len(unique)

    # color_purity
    purities = []
    for lbl in unique:
        if lbl < 0:  # skip overlay label
            continue
        mask = labels == lbl
        colors = image[mask]
        purity = np.std(colors, axis=0).mean()
        purities.append(purity)
    color_purity = np.mean(purities)

    return {
        "n_labels": n_labels,
        "fragment_ratio": fragment_ratio,
        "color_purity": color_purity,
    }
```

---

## 7. 风险与缓解

### 7.1 技术风险

| 风险 | 影响 | 可能性 | 缓解措施 |
|------|------|--------|----------|
| geoseg 引擎依赖 colorbar 输入 | 高 | 高 | 3D schematic 无 colorbar，v4_kmeans 可能降级到 pastel_faded fallback；记录降级情况 |
| diff-overlay 过度覆盖地质结构 | 中 | 中 | 视觉审计重点检查叠层 mask 是否覆盖真实地质边界；调整 diff_thresh |
| ensemble 运行时间过长 | 中 | 低 | ensemble 内部已优化，单图 < 5s；若超时则跳过 |
| 视觉审计主观偏差 | 中 | 中 | 使用 5 维度结构化评分表 + 多 crop 对比，减少主观性 |
| 参数空间过大导致实验爆炸 | 高 | 高 | 组 B/C 使用采样而非全网格；优先测试关键配置 |

### 7.2 资源风险

| 风险 | 影响 | 可能性 | 缓解措施 |
|------|------|--------|----------|
| 实验产出图片过多 | 中 | 高 | 自动过滤 fragment_ratio > 0.5 的明显失败结果，减少审计负担 |
| Mac mini M4 内存不足 | 低 | 低 | 单图处理，峰值 < 2GB |

---

## 8. 验收标准

### 8.1 最低通过标准

- [ ] 完成组 A 至少 3 个引擎的测试（v4_kmeans, slic_kmeans, ensemble）
- [ ] 完成组 B diff-overlay 的可行性验证（至少 6 个配置）
- [ ] 完成组 C 至少 20 个参数的测试
- [ ] 完成组 D 跨策略对比
- [ ] 至少 1 个策略在所有 panel 上达到：
  - 层位准确 ≥ 4
  - 文字免疫 ≥ 4
  - 碎片控制 ≥ 3
- [ ] 输出集成代码和参数推荐

### 8.2 优秀标准

- [ ] diff-overlay 方案在 Panel 3 上显著优于 v3 baseline
- [ ] ensemble 或混合策略在所有 panel 上达到加权总分 ≥ 18/20
- [ ] 输出 per-panel 最优策略映射表
- [ ] 分割结果可直接用于下游 polygon extraction

---

## 9. 时间计划

```
Day 1 (3h)
  ├── 环境准备：创建 experiments/segmentation_experiment/ 目录结构
  ├── 实现 core/ 模块（geoseg_bridge, diff_overlay, evaluator）
  └── 运行组 A 实验（geoseg 引擎族）

Day 2 (3h)
  ├── 运行组 B 实验（diff-overlay 参数扫描）
  ├── 运行组 C 实验（v3 pipeline 调优）
  └── 初步筛选最优配置

Day 3 (3h)
  ├── 运行组 D 跨策略对比
  ├── 生成对比可视化（grid + crops）
  └── 5 维度视觉审计

Day 4 (1h)
  ├── 汇总评分表
  ├── 撰写实验报告
  └── 输出集成建议和 PRD 更新
```

---

## 10. 预期产出

1. **实验代码**：`experiments/segmentation_experiment/` 完整模块
2. **结果目录**：`results/segmentation_experiment/` 含所有输出图和评分
3. **审计报告**：`docs/experiment_plan_repair/audit_segmentation.md`
4. **集成代码**：可直接使用的 `SegmentationStrategy` 接口 + 最优策略实现
5. **参数推荐表**：per-panel 最优参数映射
6. **策略决策树**：输入图像特征 → 推荐策略的决策逻辑

---

## 11. 关键设计决策

### 决策 1：是否使用 diff-overlay 作为必选项？

**推荐：作为核心候选方案，非必须。**

- diff-overlay 解决文字污染问题的思路独特，但需验证是否过度覆盖地质结构
- 若 diff-overlay 在 Panel 1/2 上表现不佳，可仅用于 Panel 3
- 最终策略可能是"v3 pipeline + diff-overlay fallback"

### 决策 2：geoseg 引擎 vs 3D-schematic 原生 pipeline？

**推荐：两者并行实验，不预设胜者。**

- geoseg 引擎族经过更多图像验证，通用性更强
- 3D-schematic 原生 pipeline（felzenszwalb + post-merge）有领域特定优化（fragment detection）
- 若 geoseg 引擎显著更优，则迁移；否则保留原生 pipeline 并优化

### 决策 3：评估以视觉审计为主还是客观指标为主？

**推荐：视觉审计为主，客观指标为辅。**

- 地质示意图分割没有 ground truth，视觉判断是最终标准
- 客观指标（n_labels, fragment_ratio）用于快速筛选明显失败结果，减少审计负担
- 关键决策仍由人工视觉审计做出

### 决策 4：是否实验深度学习分割（如 SAM）？

**推荐：本次不实验。**

- 3D schematic 是概念模型图，非自然图像，SAM 等预训练模型可能不适用
- 当前资源和时间约束下，传统 CV + geoseg 引擎族已足够探索
- 若传统 CV 全部失败，再考虑引入 SAM

---

## 12. 附录：与现有实验的衔接

本次实验是 `experiment_plan_repair` 的延伸：

- **上游输入**：沿用 text_removal 最优输出（`text_removal.py` v3 Laplacian-unfiltered）
- **评估标准**：沿用 5 维度视觉审计，权重针对分割场景调整
- **代码复用**：复用 `design_diff_overlay.py` 的 diff-overlay 实现
- **引擎复用**：复用 `geoseg/modules/segment_engines/` 的引擎族
- **产出衔接**：分割最优结果将用于下游 `post_process`（polygon extraction + SPECFEM export）

### 与 text_removal 实验的关系

```
text_removal 最优输出
    ├──→ 组 A: geoseg 引擎族直接分割
    ├──→ 组 B: diff-overlay 隔离后分割
    ├──→ 组 C: v3 pipeline 调优分割
    └──→ 组 D: 跨策略对比
                └──→ 最优分割策略
                        └──→ polygon extraction
                                └──→ SPECFEM export
```

最终目标：确定 3D schematic 的最优分割策略，形成与 text_removal 同等质量的极限输出，作为 geoseg v2 管线的标准组件。
