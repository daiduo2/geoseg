# PyPatchMatch 修复算法实验 PRD

> 文档类型：实验设计 / 产品需求文档
> 版本：v1.0
> 日期：2026-06-08
> 负责人：Agent
> 目标：评估 PyPatchMatch 对地质示意图文字残留修复的适用性，并与现有基线对比

---

## 1. 背景与动机

### 1.1 当前问题

在 geoseg v2 的 3D 地质示意图处理管线中，`text_removal_v2` 基于 **MSER + Laplacian 检测 + OpenCV Telea 修复 + median blur** 的流程存在以下问题：
- **文字残留未完全清除**（Panel 1/2/3 均有微弱痕迹）
- **median blur 伤边界**（将层位交界线模糊化）
- 修复结果影响下游分割引擎（k-means / SLIC / 区域融合）的精度

### 1.2 为什么选择 PyPatchMatch

通过对 LaMa、ViTEraser、PyPatchMatch 三个开源项目的代码级分析，发现 PyPatchMatch 具备以下独特优势：

| 特性 | PyPatchMatch | Telea | LaMa | ViTEraser |
|------|-------------|-------|------|-----------|
| 是否需要训练数据/预训练模型 | 否 | 否 | 是（180MB） | 是（28-88MB） |
| 是否能限制 patch 源区域 | **是（global_mask）** | 否 | 否 | 否 |
| 对层位边界的保护能力 | **强** | 中 | 弱 | 弱 |
| M4 16GB 推理资源占用 | 极低（<100MB） | 极低 | 中（~500MB） | 中（~1-2GB） |
| 对纯色/渐变背景的适应性 | **极佳** | 好 | 好 | 需微调 |
| 对纹理背景的适应性 | 好 | 中 | **极佳** | **极佳** |

**核心差异点**：PyPatchMatch 的 `global_mask` 功能可以禁止从层位边界另一侧采样 patch，这是其他算法不具备的能力。

### 1.3 实验目标

1. 验证 PyPatchMatch 在地质示意图文字残留修复上的效果
2. 探索 `global_mask` 对层位边界保护的实际收益
3. 找出最优参数组合（patch_size、迭代次数、搜索策略）
4. 与当前基线（Telea r=3）做定量对比
5. 输出可集成到 geoseg 管线的修复模块设计

---

## 2. 核心假设

### 假设 1：层位边界可保护
对于水平分层的地质示意图，通过 `global_mask` 限制 patch 只能从文字所在层的同侧采样，可以避免跨边界颜色污染。

### 假设 2：PatchMatch 优于 Telea 的场景
- 大面积文字块（>20px）：PatchMatch 可以从远处找到结构匹配的 patch，避免 Telea 的"糊状"平滑
- 纹理背景（如 Panel 3 碎屑区域）：PatchMatch 的块匹配能更好地保持纹理一致性
- 层内纯色/渐变：PatchMatch 与 Telea 等效或略优

### 假设 3：存在最优参数
- `patch_size`：小值（7-11）适合细小残留，大值（15-21）适合大文字块
- 金字塔层级：更多层级提升大区域质量，但增加时间
- `global_mask`：在文字靠近边界时收益最大

---

## 3. 实验设计

### 3.1 数据集

使用已有的 3D Schematic 三个 panel：
- **Panel 1**：平坦纯色/微渐变（灰→蓝→橙→黄）
- **Panel 2**：平坦渐变 + 结构线
- **Panel 3**：复杂纹理（绿色碎屑）

输入数据：
- 原图：`figures/panels/panel_{1,2,3}.png`
- 掩码：`experiments/text_removal_v2/final_pipeline/panel_{1,2,3}_mask.png`
- 基线结果：`experiments/text_removal_v2/final_pipeline/panel_{1,2,3}_final.png`

### 3.2 实验分组

#### 组 A：PatchMatch 参数网格

固定问题，探索 PatchMatch 参数空间：

| 参数 | 取值 |
|------|------|
| `patch_size` | 7, 11, 15, 21 |
| `pyramid_levels` | 1（单尺度）, 3（粗到细） |
| `iterations_per_level` | 3, 5, 7 |

任务数：4 × 2 × 3 = 24 个配置 × 3 panels = 72 个任务

#### 组 B：global_mask 边界保护实验

针对层位边界附近的文字残留，测试 global_mask 的效果：

| 配置 | global_mask 策略 |
|------|-----------------|
| B1 | 无 global_mask（基线 PatchMatch） |
| B2 | 水平半平面掩码（文字所在层为源区域） |
| B3 | 层位感知掩码（基于层边界 y 坐标精确限制） |
| B4 | 膨胀后的文字掩码作为源区域限制 |

任务数：4 × 3 panels = 12 个任务（每个 panel 选 2-3 个边界文字区域重点测试）

#### 组 C：跨算法对比

在统一掩码上对比多种算法：

| 算法 | 配置 |
|------|------|
| REF_v2_final | text_removal_v2 最终输出 |
| REF_v2_single | text_removal_v2 仅 inpaint（无 median blur） |
| Telea_r3 | cv2.inpaint(r=3, TELEA) — 当前最优 |
| NS_r3 | cv2.inpaint(r=3, NS) |
| PM_best | 组 A 最优 PatchMatch 配置 |
| PM_best + GM | 组 A 最优 + global_mask |
| LaMa | big-lama 预训练模型 |

任务数：7 × 3 panels = 21 个任务

#### 组 D：失败案例分析

收集 PatchMatch 的失效模式：
- 文字跨层边界
- 大面积文字块（>50px）
- 复杂纹理背景
- 重复模式背景（条纹、网格）

### 3.3 全局掩码生成策略

**B2 水平半平面**（快速近似）：
```python
# 检测文字区域的质心 y 坐标，选择包含文字的那一半作为源区域
hy, hx = np.where(mask > 0)
centroid_y = int(np.mean(hy))
global_mask = np.zeros_like(mask)
global_mask[centroid_y:, :] = 255  # 或上半部分，根据文字位置动态选择
```

**B3 层位感知掩码**（精确版，需预处理）：
```python
# 使用颜色聚类或边缘检测提取层位边界
layer_boundaries = detect_layer_boundaries(image)  # 返回 y 坐标列表
text_y = np.mean(np.where(mask > 0)[0])
# 找到文字所在层段
layer_idx = find_containing_layer(layer_boundaries, text_y)
y1, y2 = layer_boundaries[layer_idx], layer_boundaries[layer_idx + 1]
global_mask = np.zeros_like(mask)
global_mask[y1:y2, :] = 255
```

**B4 膨胀文字掩码**（保守版）：
```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
global_mask = cv2.dilate(mask, kernel, iterations=3)
# invert: 只允许从膨胀区域外的同层区域采样
```

---

## 4. 评估指标

### 4.1 客观指标

| 指标 | 计算方法 | 通过阈值 |
|------|----------|----------|
| **PSNR** | 与 Telea 修复结果对比（无 ground truth） | > 35dB（无显著退化） |
| **SSIM** | 与 Telea 修复结果对比 | > 0.95（结构保持） |
| **LPIPS**（可选） | 感知距离，使用 AlexNet 特征 | < 0.05（感知相似） |
| **Mask Coverage 差异** | 检测修复后是否仍有残留 | 残留掩码面积 < 原始 10% |
| **运行时间** | 单图推理耗时 | < 5s（M4 CPU） |
| **内存占用** | 峰值内存 | < 500MB |

### 4.2 主观视觉审计（沿用验收标准）

沿用 `docs/experiment_plan_repair/acceptance.md` 的 5 维度评分：

| 维度 | 权重 | 说明 |
|------|------|------|
| 残留清除 | 1.2x | 文字痕迹是否完全不可见 |
| 边界保持 | 1.5x | 地质层位交界线是否锐利 |
| 颜色一致 | 1.0x | 修复区域是否与背景融为一体 |
| 伪影引入 | 1.3x | 是否有重复纹理、色带、模糊 |
| 结构完整 | 1.5x | 地质结构（褶皱、断层）是否保持 |

**通过标准**：
- 残留清除 ≥ 3
- 边界保持 ≥ 4
- 结构完整 ≥ 4
- 加权总分 ≥ 16/20

### 4.3 下游任务指标

对修复后的图像运行 geoseg 的 k-means / SLIC 分割引擎：
- 分割边界 IoU（与 manual ground truth 或 v2_single 结果对比）
- 聚类中心偏移量

---

## 5. 资源需求

### 5.1 计算资源

| 资源 | 需求 | 说明 |
|------|------|------|
| CPU | Mac mini M4 | PyPatchMatch 为 CPU 算法 |
| 内存 | < 500MB 峰值 | 单图推理 |
| 磁盘 | ~100MB | 代码 + 依赖 |
| 时间 | 每组 10-30 分钟 | 取决于参数和并发 |

### 5.2 软件依赖

```
# 必须
Python 3.9+
numpy
opencv-python (Python 绑定)
Pillow

# PyPatchMatch C++ 编译需要（仅限使用原始 C++ 实现时）
g++ with C++14
OpenCV C++ development headers
make

# 可选评估
scikit-image
lpips
```

### 5.3 人力投入

| 阶段 | 时间 | 产出 |
|------|------|------|
| 环境搭建 | 2h | 编译 libpatchmatch.so 或实现纯 Python fallback |
| 实验实现 | 4h | 分组实验代码、评估脚本 |
| 运行实验 | 4h | 105 个任务分批运行 |
| 视觉审计 | 4h | 5 维度评分 |
| 报告撰写 | 2h | PRD 更新、集成建议 |
| **总计** | **16h** | |

---

## 6. 实现方案

### 6.1 代码模块结构

```
experiments/patchmatch_experiment/
├── core/
│   ├── __init__.py
│   ├── patchmatch_bridge.py       # 封装 PyPatchMatch ctypes 调用
│   ├── patchmatch_pure.py         # 纯 Python fallback（无 C++ 依赖）
│   └── global_mask_generator.py   # 生成层位感知 global_mask
├── runners/
│   ├── group_a_param_grid.py      # 参数网格实验
│   ├── group_b_global_mask.py     # global_mask 实验
│   ├── group_c_cross_algorithm.py # 跨算法对比
│   └── evaluate.py                # 评估脚本
├── visualization/
│   ├── crop_compare.py            # 裁剪对比图生成
│   ├── edge_compare.py            # 边界保持对比
│   └── score_table.py             # 评分表
└── README.md
```

### 6.2 核心接口设计

```python
from pathlib import Path
import numpy as np

class PatchMatchRepairer:
    """PyPatchMatch 修复器封装。"""

    def __init__(self, patch_size: int = 11, pyramid_levels: int = 3,
                 iterations: int = 5, use_cpp: bool = True):
        self.patch_size = patch_size
        self.pyramid_levels = pyramid_levels
        self.iterations = iterations
        self.use_cpp = use_cpp

        if use_cpp:
            import patch_match  # from PyPatchMatch
            self._pm = patch_match
        else:
            from .patchmatch_pure import patchmatch_crop
            self._pm = patchmatch_crop

    def repair(self, image: np.ndarray, mask: np.ndarray,
               global_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if self.use_cpp:
            return self._pm.inpaint(image, mask=mask, patch_size=self.patch_size)
        else:
            return self._pm(image, mask, patch_size=self.patch_size)

    def repair_with_layer_mask(self, image: np.ndarray, mask: np.ndarray,
                               layer_boundaries: list[int]) -> np.ndarray:
        """使用层位边界生成 global_mask 后修复。"""
        global_mask = generate_layer_aware_mask(
            image.shape[:2], mask, layer_boundaries
        )
        return self.repair(image, mask, global_mask=global_mask)
```

### 6.3 层位边界检测（预处理）

```python
def detect_layer_boundaries(image: np.ndarray, n_layers: int = 4) -> list[int]:
    """
    检测水平地质层位边界。
    方法：对灰度图做水平投影 + 颜色聚类，找到层间过渡带。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 水平方向均值（对 x 方向做平均，突出 y 方向的颜色变化）
    profile = np.mean(gray, axis=1)
    # 计算梯度
    grad = np.abs(np.diff(profile))
    # 找局部极大值作为边界
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(grad, height=np.percentile(grad, 80), distance=100)
    return sorted(peaks.tolist())
```

---

## 7. 风险与缓解

### 7.1 技术风险

| 风险 | 影响 | 可能性 | 缓解措施 |
|------|------|--------|----------|
| PyPatchMatch C++ 编译失败 | 高 | 中 | 准备纯 Python fallback；预编译 .so |
| 纯 Python 版本太慢 | 高 | 高 | 限制测试区域大小；使用 C++ 版本 |
| global_mask 生成错误 | 中 | 中 | 人工校验边界；提供可视化调试 |
| 纹理区域 PatchMatch 失效 | 中 | 中 | 组 C 引入 LaMa 作为 fallback |
| 大文字块修复伪影 | 中 | 中 | 增大 patch_size；使用金字塔 |

### 7.2 资源风险

| 风险 | 影响 | 可能性 | 缓解措施 |
|------|------|--------|----------|
| 实验任务过多导致时间过长 | 中 | 中 | 每组减少参数组合；优先测关键配置 |
| M4 内存不足 | 低 | 低 | PatchMatch 内存占用 <100MB |

---

## 8. 验收标准

### 8.1 最低通过标准

- [ ] 完成组 A 至少 12 个关键配置的测试
- [ ] 完成组 B 的 global_mask 效果验证
- [ ] 完成组 C 的跨算法对比
- [ ] 至少 1 个 PatchMatch 配置在所有 panel 上达到：
  - 残留清除 ≥ 3
  - 边界保持 ≥ 4
  - 结构完整 ≥ 4
- [ ] 输出集成代码和参数推荐

### 8.2 优秀标准

- [ ] PatchMatch 在 Panel 3 纹理区域显著优于 Telea
- [ ] global_mask 在层位边界处明显减少颜色污染
- [ ] 修复速度 < 2s/图
- [ ] 输出可直接集成的 `PatchMatchRepairer` 类

---

## 9. 时间计划

```
Day 1 (4h)
  ├── 环境搭建：编译 PyPatchMatch 或 fallback
  ├── 实现 core/ 模块
  └── 实现 group_a 参数网格

Day 2 (4h)
  ├── 运行组 A 实验
  ├── 运行组 B global_mask 实验
  └── 初步视觉审计

Day 3 (4h)
  ├── 运行组 C 跨算法对比
  ├── 生成对比可视化
  └── 撰写评分表

Day 4 (4h)
  ├── 失败案例分析
  ├── 撰写实验报告
  └── 输出集成建议和 PRD 更新
```

---

## 10. 预期产出

1. **实验代码**：`experiments/patchmatch_experiment/` 完整模块
2. **结果目录**：`results/patchmatch_experiment/` 含所有输出图和评分
3. **审计报告**：`docs/experiment_plan_repair/audit_pypatchmatch.md`
4. **集成代码**：可直接使用的 `PatchMatchRepairer` 类
5. **参数推荐表**：per-panel 最优参数
6. **global_mask 策略**：边界保护的最佳实践

---

## 11. 关键设计决策

### 决策 1：使用 PyPatchMatch C++ 还是纯 Python fallback？

**推荐：优先 C++，fallback 为纯 Python。**

- C++ 版本性能是 Python 的 10-50 倍
- 但编译需要 OpenCV C++ headers
- 准备 `patchmatch_bridge.py` 自动检测并 fallback

### 决策 2：global_mask 自动生成还是手动标注？

**推荐：自动检测 + 可视化校验。**

- 水平分层地质图可用颜色投影自动检测
- 复杂情况允许手动修正
- 实验阶段先用自动方法，验证可行性

### 决策 3：评估时是否使用 LaMa 作为上界？

**推荐：是。**

- LaMa 是当前开源修复的强基线
- 用于验证 PatchMatch 是否达到"足够好"水平
- 若 PatchMatch 显著弱于 LaMa，则推荐混合策略

### 决策 4：是否引入下游分割指标？

**推荐：作为可选指标，非必须。**

- 主指标仍是视觉审计
- 下游分割指标用于验证"修复是否真正帮助了分割"
- 如果时间允许则测量

---

## 12. 附录：与现有实验的衔接

本次 PatchMatch 实验是在 `experiment_plan_repair` 基础上的延伸：
- 沿用相同的 3 个 panel 和 mask
- 沿用 5 维度视觉审计标准
- 沿用 `audit_grids`、`audit_crops`、`audit_edges` 可视化格式
- 结果将与现有 `audit_report/scores.md` 合并更新

最终目标：将 PatchMatch 作为 geoseg v2 修复模块的候选算法之一，与 Telea 形成互补：
- **Telea r=3**：小残留、纯色/渐变背景、速度优先
- **PatchMatch + global_mask**：边界敏感、大文字块、纹理背景
