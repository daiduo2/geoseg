# 文字残留修复算法对比实验计划

> 基于 `research_repair_algorithms.md` 调研结论，针对 3D 地质示意图三个 panel 的文字残留问题设计系统化实验。
> 实验对象：原 panel（未处理）+ text_removal_v2 输出（单阶段 / 双阶段）

---

## 1. 实验目标

1. 找到比 text_removal_v2 当前 71x71 median blur 更优的残留修复方案
2. 验证不同背景类型（平坦渐变 vs 复杂纹理）对算法选择的敏感性
3. 确定最小可行算法集（拒绝过度工程化的深度学习方案，除非传统 CV 明显落后）
4. 为 geoseg v2 管线提供可集成的修复模块参数

---

## 2. 实验对象

### 2.1 输入图像（6 张）

| 编号 | 路径 | 说明 |
|------|------|------|
| P1-orig | `figures/panels/panel_1.png` | 原图，含白色文字标注，背景为灰→蓝→橙→黄渐变 |
| P1-single | `experiments/text_removal_v2/final_pipeline/panel_1_single.png` | text_removal_v2 单阶段输出（inpaint only） |
| P1-final | `experiments/text_removal_v2/final_pipeline/panel_1_final.png` | text_removal_v2 双阶段输出（inpaint + median blur） |
| P2-orig / P2-single / P2-final | 同上，panel_2 | 文字更多，背景渐变类似 |
| P3-orig / P3-single / P3-final | 同上，panel_3 | 含绿色碎屑纹理，背景更复杂 |

### 2.2 掩码输入

- text_removal_v2 一阶段 mask：`final_pipeline/panel_*_mask.png`
- text_removal_v2 二阶段 residual mask：`final_pipeline/panel_*_residual.png`
- 实验需要时自行生成检测 mask

---

## 3. 实验组设计

### 3.1 实验组 A：修复算法单变量替换（核心对照）

**目标**：固定检测 mask 为 text_removal_v2 的一阶段 mask，仅替换修复算法，直接对比修复质量。

**输入**：P1/P2/P3 原图 + 对应 mask
**输出目录**：`results/experiment_plan_repair/group_a_replacement/`

| 子组 | 算法 | 参数网格 | 预估产出数 |
|------|------|----------|-----------|
| A1 | OpenCV Telea | `inpaintRadius ∈ {3, 5, 7, 9}` | 3 panels × 4 = 12 |
| A2 | OpenCV Navier-Stokes | `inpaintRadius ∈ {3, 5, 7, 9}` | 12 |
| A3 | Biharmonic Inpainting (skimage) | 固定参数，逐通道 | 3 |
| A4 | Median blur 替换 | `ksize ∈ {21, 41, 51, 71, 91}` | 15 |
| A5 | **LaMa** (simple-lama) | 固定参数，MPS 加速 | 3 |
| A6 | **当前基线** (inpaint r=3 + median 71) | 固定 | 3 |

**注**：A5 LaMa 为探索性子组，若首次运行内存超过 8GB 或单图耗时 >30s，则标记为"资源不可行"并停止该子组。

### 3.2 实验组 B：检测+修复组合（精准修复）

**目标**：在 text_removal_v2 已修复输出上，自动检测残留并局部修复。

**输入**：P1/P2/P3-final（text_removal_v2 双阶段输出）
**输出目录**：`results/experiment_plan_repair/group_b_detect_repair/`

| 子组 | 检测策略 | 修复算法 | 参数 | 产出数 |
|------|----------|----------|------|--------|
| B1 | 局部亮度异常（medianBlur 15 背景估计，diff > threshold） | Telea | `threshold ∈ {10, 15, 20}`, `radius ∈ {3, 5}` | 3×3×2 = 18 |
| B2 | 局部对比度异常（DoG 差异） | Telea | `sigma_ratio ∈ {(1,2), (1.5,3)}`, `threshold ∈ {15, 25}` | 3×2×2 = 12 |
| B3 | 在 final 上重跑 MSER+Laplacian → 与一阶段 mask 取交集得种子 → 区域生长 | median blur 替换 | `grow_threshold ∈ {15, 20, 25}`, `ksize ∈ {51, 71}` | 3×3×2 = 18 |
| B4 | 二值化残留检测（final 与原图 diff > threshold） | Telea | `threshold ∈ {10, 20, 30}` | 9 |

### 3.3 实验组 C：边缘保持后处理（全局平滑去残留）

**目标**：不依赖 mask，对 text_removal_v2 最终输出做全局平滑，观察微弱残留是否可被消除而不伤边界。

**输入**：P1/P2/P3-final
**输出目录**：`results/experiment_plan_repair/group_c_post_smooth/`

| 子组 | 算法 | 参数网格 | 产出数 |
|------|------|----------|--------|
| C1 | Guided Filter | `radius ∈ {2, 4, 8}`, `eps ∈ {0.001, 0.01, 0.1}` | 3×3×3 = 27 |
| C2 | Domain Transform Filter | `sigmaSpatial ∈ {10, 20, 40}`, `sigmaColor ∈ {0.05, 0.1, 0.2}` | 3×3×3 = 27 |
| C3 | Bilateral Filter | `d ∈ {5, 9, 15}`, `sigmaColor ∈ {30, 75}`, `sigmaSpace ∈ {30, 75}` | 3×3×2×2 = 36 |
| C4 | **对照：高斯模糊** | `ksize ∈ {5, 11, 21}`, `sigma ∈ {1, 2}` | 3×3×2 = 18 |

**注意**：C 组重点观察"边界保持度"——地质层位之间的分界线（如灰/蓝、蓝/橙交界处）不能模糊。

### 3.4 实验组 D：全流程替代方案（端到端对比）

**目标**：用调研文档推荐方案替代 text_removal_v2 的完整检测+修复流程。

**输入**：P1/P2/P3 原图
**输出目录**：`results/experiment_plan_repair/group_d_full_pipeline/`

| 子组 | 方案 | 实现要点 |
|------|------|----------|
| D1 | OpenCV Telea 全流程 | 原图 → MSER+Laplacian 检测（v2 参数）→ Telea r=3 → 输出 |
| D2 | Biharmonic 全流程 | 原图 → MSER+Laplacian 检测 → Biharmonic 修复 → 输出 |
| D3 | LaMa 全流程 | 原图 → text_removal_v2 mask → LaMa 修复 → 输出 |
| D4 | Telea + 二阶段 Biharmonic | 一阶段 Telea → 残留检测 → Biharmonic 局部修复 |

---

## 4. 参数选择逻辑

### 4.1 为什么这些参数

- **Telea radius {3,5,7,9}**：3 是小瑕疵最优，5-7 是 text_removal_v2 当前值，9 测试大区域容忍度
- **Median ksize {21,41,51,71,91}**：当前用 71，测试更小（更锐）和更大（更平滑）的权衡
- **Guided Filter eps {0.001, 0.01, 0.1}**：0.001 保守（几乎不平滑），0.1 激进，0.01 居中
- **DTF sigmaColor {0.05, 0.1, 0.2}**：float 图像归一化后值；0.05 仅平滑极相似颜色，0.2 混合更大差异

### 4.2 背景类型假设

| Panel | 主导背景类型 | 预期最优算法 |
|-------|-------------|-------------|
| Panel 1 | 平坦渐变（灰→蓝→橙→黄） | Telea r=3 / Biharmonic |
| Panel 2 | 平坦渐变 + 更多结构线 | Telea r=3 / 区域生长+median |
| Panel 3 | 复杂纹理（绿色碎屑）+ 渐变 | LaMa / DTF / 大 radius Telea |

---

## 5. 实验执行步骤

### Step 0：环境准备（预计 30min）

```bash
# 1. 确认依赖
python -c "import cv2, numpy, skimage, scipy, PIL; print('OK')"
# 2. 尝试安装 LaMa（可选，若失败则跳过 A5/D3）
pip install simple-lama-inpainting 2>/dev/null || echo "LaMa unavailable"
# 3. 确认 opencv-contrib（Guided Filter / DTF 需要）
python -c "import cv2.ximgproc; print('ximgproc OK')" 2>/dev/null || echo "ximgproc missing"
# 4. 创建输出目录
mkdir -p results/experiment_plan_repair/{group_a,group_b,group_c,group_d}
```

### Step 1：运行实验组 A（预计 1-2h，不含 LaMa）

- 使用 `run_all_text_removal.py` 的 `run_baseline` 变体，将修复步骤替换为各算法
- 每张图记录运行时间
- LaMa 单独运行，内存/时间超标则记录并放弃

### Step 2：运行实验组 B（预计 1-2h）

- 基于 `second_pass_final.py` 框架扩展
- 重点调参 `grow_threshold` 和 `threshold`，避免过度生长波及地质结构

### Step 3：运行实验组 C（预计 1-2h）

- 对 P1/P2/P3-final 逐一张贴参数网格
- Guided Filter / DTF 若 `cv2.ximgproc` 不可用，用 scikit-image bilateral 替代并记录

### Step 4：运行实验组 D（预计 30min，不含 LaMa）

- 复用已有代码框架快速跑通

### Step 5：视觉审计（预计 2-3h）

- 按第 6 节审计协议执行
- 产出评分表

### Step 6：结论汇总（预计 1h）

- 汇总最优参数组合
- 输出推荐集成方案

---

## 6. 视觉审计协议

见同目录 `acceptance.md`。

---

## 7. 风险控制

| 风险 | 缓解措施 |
|------|----------|
| LaMa 内存爆炸 | 首次运行即监控 `psutil.virtual_memory().percent`，>85% 立即终止 |
| ximgproc 未安装 | 提前检测，缺失时 C1/C2 用 scikit-image bilateral 替代 |
| 产出图片过多导致审计疲劳 | A/C 组参数网格已压缩到最少必要组合；优先审计 A 组和 B 组 |
| 残留检测误伤地质结构 | B 组所有结果必须人工确认结构线完整性 |

---

## 8. 附录：快速参考代码片段

见 `research_repair_algorithms.md` 附录 A-E。本实验额外需要：

### 8.1 区域生长残留检测（B3）

```python
def detect_residual_region_growing(repaired, first_mask, grow_threshold=20):
    """在已修复图像上检测残留，基于区域生长。"""
    gray = cv2.cvtColor(repaired, cv2.COLOR_RGB2GRAY)
    # 重跑 MSER+Laplacian 得种子
    seeds = detect_text_post_repair(gray) & first_mask.astype(bool)
    # 区域生长...
    return residual_mask
```

### 8.2 亮度异常检测（B1）

```python
def detect_brightness_anomaly(image, blur_ksize=15, threshold=15):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bg = cv2.medianBlur(gray, blur_ksize)
    diff = cv2.subtract(gray, bg)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask
```
