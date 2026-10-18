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
| e028 | FH + SLIC segmentation | **放弃** | 真实图像上 FH 严重过度分割（72 segment vs 目标 5），SLIC+K-Means 碎片为零但欠分割严重 |
| e029 | Spatial regularized clustering | **部分** | x-y-LAB 五维 K-Means **完全无效**；SLIC 超像素级聚类 **强力替代**（fragments→0）；median 后处理立即可集成 |
| e030 | Anisotropic preprocessing | **保留** | row_median(size=5) 替换 adaptive_blur 为最佳预处理；水平开运算有方向性局限；直方图过滤效果有限 |

---

## 2. 候选算法引擎（按适用场景分类）

### A. 颜色聚类引擎（无 VLM seeds）

| 引擎 | 核心 | 适用 | 速度 | 质量 |
|------|------|------|------|------|
| **e026** | KMeans RGB + NN | 简单 jet-colormap，已知 n_layers | ~0.1s | 中 |
| **e027** | SLIC + Graph Cut (ICM) | Conceptual model panels，需平滑边界 | ~0.3s | 高 |
| **v4_kmeans** | KMeans LAB + shape filter + row_median + median_post | Vivid jet/rainbow，有/无 VLM seeds，**文字鲁棒** | ~0.2s | 高 |
| **slic_kmeans** | SLIC superpixel + K-Means | 文字覆盖严重的 panel，碎片零容忍 | ~0.3s | 中（欠分割） |
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
  ├─ 单 panel + vivid + 文字覆盖严重 + 碎片零容忍 → slic_kmeans (e029)
  ├─ 单 panel + vivid + 平衡速度/质量 → v4_kmeans (row_median + median_post)
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
| P0 | row_median + median_post (e030) | **已验证**：6 张真实图像 + 视觉确认，碎片 -50%~-70%，立即集成 |
| P0 | SLIC superpixel + K-Means (e029) | **已验证**：文字鲁棒性极佳，但欠分割严重，仅作辅助引擎 |
| P3 | GMM (e011) | 虽然之前结论负面，但 multimodal 提供更好种子后可能改善 |
| P3 | Mean Shift (e008) | 自动 k 发现，新种子质量下可能改善 |

---

## 7. 2026-07 Fig.6 迭代精修战役（e031–e038）

这段工作围绕 Fig.6 的五条剖面，为目标"分区几何 → 速度模型"链路写了约 50 个脚本。代码统一归档在
`experiments/archive/2026-07-fig6-colorbar/`、`experiments/archive/2026-07-pm-repair/` 和
`scripts/archive/2026-07-fig6-palette/`；下面只保留结论，避免后来者重走同一条路。

| 编号 | 尝试 | 结论 | 处置 |
|------|------|------|------|
| e031 | 16-zone 颜色条种子匹配：从色条等距取 16 个种子色，LAB 最近邻分配，再按相邻关系合并成 5 层 | **有效**。有颜色条时，以色条为锚比直接全图 K-means 更稳定，且给出绝对色阶顺序。16 区可视为多尺度量化的中间层 | 代码留档；作为下一步多尺度分区（8/16/24/32/60 + 合并树）的算法起点 |
| e032 | 文本伪影去除：11 个变体（小连通域、暗像素、边缘、inpaint、双边、行中值） | **行中值滤波（size=7）+ 暗像素中值替换（阈值 40）是最优保守组合**。小连通域法无效（文字嵌在大 label 内）；全图 inpaint 会把区域洗白；掩膜过激（约 20% 像素）会误伤真实地质。行中值能成立是因为文字笔画与地层同为水平方向，用方向先验区分不了，只能靠"短程脉冲 vs 长程界面"的尺度差异 | `row_median` 已进入 v4 主路径；其余变体留档，最终策略见 `remove_text_artifacts_final_delivery.py` |
| e033 | 逐标签调色迭代：`verify_iter1`–`verify_iter16` 共 16 轮，每轮手工改若干 label 的调色板并回滚变差的条目 | **不收敛**。每次修好一个 label 会让另一个变差，属于没有全局约束的局部搜索。正确做法是以颜色条为锚做全局调色板拟合，再用残差审计挑异常标签 | 反模式，代码留档；不再新增逐标签补丁 |
| e034 | label 12 残差纠正：先按残差把文字从地质里拆开，再做颜色拟合 | **拆分会劣化结果**。文字与地质混在同一 label 时，保留 label 统一、只用非文字像素拟合颜色，效果更好；文字像素允许保持高残差，不参与分区 | 代码留档，结论写入本表 |
| e035 | 欠分割修复：profile 07 浅色区只分出 1–2 层 | **LAB L 通道引导的过分割 + 自适应合并（Variant D）有效**，比调 K 或换引擎更直接。可作为"层数不足自动重试"的实现参考 | 代码留档；产品化仍是 P0 待办 |
| e036 | PM/OCR 定点修复：ROI inpaint + 最近色局部重分割（round1–3） | **有效但不可推广**：ROI 内修复能保住 ROI 外边界；最近色比在 ROI 上跑 K-means 稳。PaddleOCR 可自动定位文字框，但需要独立 `.venv_ocr`，且每张图要单独调 ROI | 整体留档；`.venv_ocr` 已移到 `~/geoseg-artifacts/2026-10/venvs/` 并软链回来 |
| e037 | 候选图筛选与随机性利用：37 个 `test_*` 脚本，含"多次重跑 kmeans"、"同一 overlay 反复问 VLM 直到 ACCEPT" | **不是有效评估**。重试直到随机性给出好结果会虚高通过率，也无法复现。正确做法是固定种子、报告方差、用 `synthetic_benchmark` 做对照 | 反模式，代码整体归档到 `scripts/archive/2026-07-candidate-screening/` |
| e038 | 3D 示意图羽流/管状结构：暖色合并、L 通道分割、edge_guided、regional fusion | **2D 分层引擎在羽流上系统性失败**（18/18 配置被拒）。去文字对 Panel 1/2 有帮助，Panel 3 仍不达标；需要形状先验或专门的 3D 预处理 | 留在 `experiments/3d_schematic/` 子项目，不并入主管线 |

### 7.1 可复用遗产

- **颜色条种子采样**：`scripts/archive/2026-07-fig6-palette/` 里的 16-zone 脚本包含 `_sample_colorbar_seeds` 的完整用法，是多尺度量化的现成起点。
- **行中值预处理**：`row_median_filter` 已在 `segment_engines/_shared.py`，参数 size=6–8 是实测甜区。
- **残差审计**：逐标签 RGB 残差是发现"文字混入"和"调色板错配"的有效诊断信号，但不适合直接当作修复目标。
- **定点修复的边界处理**：ROI 内重分割、ROI 外冻结，是 regional fusion 的正确用法。

### 7.2 反模式清单（新代码禁区）

1. **不要用重试随机性换通过率**（重跑 kmeans、反复询问 VLM 直到接受）。固定种子，报告均值与方差。
2. **不要逐标签手工调色**。任何调色板调整必须是带全局约束的拟合，可回滚、可复现。
3. **不要为了去文字而拆碎 label**。先预处理压文字，再分割；修复时保拓扑。
4. **不要用过激的文字掩膜**。超过 ~10% 像素的去字掩膜基本会伤到地质。
5. **不要在错误的表示层修复**。overlay 上的视觉问题不等于 labels 的问题；inpaint 原图会污染后续颜色统计。
6. **欠分割不要只调 K**。优先用亮度通道过分割 + 合并，或换需要不同信息的引擎。
7. **实验产物写入 `runs/`，不要写在代码目录**。本轮约 50 个脚本把中间结果写进 `experiments/`，是这次目录臃肿的直接原因。
