# 文字残留修复算法对比实验

## 目录结构

```
experiment_plan_repair/
├── README.md           # 本文件：实验上下文与总览
├── plan.md             # 详细实验计划（4 组实验 + 参数网格 + 执行步骤）
└── acceptance.md       # 视觉审计协议与验收标准
```

## 实验上下文

### 当前基线

text_removal_v2（`../experiments/text_removal_v2/final_pipeline/`）已实现两阶段文字移除：

1. **一阶段**：MSER + Laplacian 检测 → inpaint (Telea, r=3) → 文字主体移除
2. **二阶段**：修复后重跑检测 → 区域生长得 residual mask → 71x71 median blur 替换

**问题**：视觉审计显示 Panel 1/2/3 仍有不同程度残留：
- Panel 1：顶部 "Continental crust" 区域、底部有微弱痕迹
- Panel 2："removed mantle lithosphere" 残留较清晰
- Panel 3：左下角文字区域残留明显

### 调研结论

`../research_repair_algorithms.md` 调研了 20+ 种修复/平滑算法，推荐 6 个方案：

| 优先级 | 方案 | 特点 |
|--------|------|------|
| 1 | OpenCV Telea | 零依赖、极速、适合小区域 |
| 2 | 检测+修复 | 最精准，只修问题像素 |
| 3 | LaMa | 深度学习，质量最高 |
| 4 | Guided/DTF Filter | O(1) 边缘保持平滑 |
| 5 | Biharmonic | 最平滑填充 |
| 6 | ViTEraser | 专用文字擦除，需微调 |

本实验验证方案 1-5 在 3D 地质示意图 3 个 panel 上的实际效果。

### 实验对象

| 对象 | 路径 | 说明 |
|------|------|------|
| 原图 | `figures/panels/panel_{1,2,3}.png` | 含文字标注的原始 panel |
| text_removal_v2 单阶段 | `experiments/text_removal_v2/final_pipeline/panel_{1,2,3}_single.png` | 仅 inpaint |
| text_removal_v2 双阶段 | `experiments/text_removal_v2/final_pipeline/panel_{1,2,3}_final.png` | inpaint + median blur |
| text_removal_v2 mask | `experiments/text_removal_v2/final_pipeline/panel_{1,2,3}_mask.png` | 一阶段检测掩码 |
| text_removal_v2 residual | `experiments/text_removal_v2/final_pipeline/panel_{1,2,3}_residual.png` | 二阶段残留掩码 |

## 快速开始

```bash
# 1. 阅读实验计划
cat plan.md

# 2. 阅读验收标准
cat acceptance.md

# 3. 检查环境
python -c "import cv2, numpy, skimage; print('OK')"

# 4. 执行实验（按 plan.md Step 0-6）
# 实验脚本应放在 experiments/ 下，产出到 results/experiment_plan_repair/
```

## 关键决策点

1. **是否运行 LaMa（A5/D3）**：取决于 `pip install simple-lama-inpainting` 是否成功 + Mac mini M4 内存是否足够
2. **是否运行 Guided/DTF Filter（C1/C2）**：取决于 `cv2.ximgproc` 是否可用
3. **Panel 3 是否单独处理**：Panel 3 背景含复杂纹理（绿色碎屑），可能与 Panel 1/2 的最优方案不同

## 预期产出

- 最优修复算法及参数（per-panel 或全局）
- 可集成到 geoseg v2 管线的 Python 模块接口
- 视觉审计评分表（含失败案例分析）
