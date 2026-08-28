# Kimi PPT 素材包说明

本目录用于上传给 Kimi 生成一版新的学术技术展示 PPT。

## 目录结构

```text
ppt_kimi_package/
├── README.md
├── kimi_ppt_prompt.md
├── images/
│   ├── case1_inversion_velocity_segmentation.jpg
│   ├── case2_fig5a_concept_model_segmentation.png
│   └── case3_fig6_geometry_boundary_segmentation.png
└── flowcharts/
    ├── agent_geoseg_pipeline.png
    ├── agent_geoseg_pipeline.svg
    ├── agent_geoseg_pipeline.mmd
    └── render_pipeline_png.py
```

## 图片用途

### `flowcharts/agent_geoseg_pipeline.png`

总体流程图，用于第 3 页“Agent + 图像识别的自动化建模流程”。

### `images/case1_inversion_velocity_segmentation.jpg`

案例 1：反演速度模型/速度结构图的自动分区识别。

建议页面标题：

“案例 1：从反演速度模型图中提取速度分区”

### `images/case2_fig5a_concept_model_segmentation.png`

案例 2：paper22 图 5a 概念模型图分区识别。

建议页面标题：

“案例 2：从概念模型图中自动识别分区”

### `images/case3_fig6_geometry_boundary_segmentation.png`

案例 3：paper22 图 6 复杂几何/边界识别结果。

建议页面标题：

“案例 3：复杂几何图中的边界与区域识别”

## 给 Kimi 的使用方式

1. 上传整个 `ppt_kimi_package` 目录中的图片。
2. 将 `kimi_ppt_prompt.md` 的内容完整粘贴给 Kimi。
3. 要求 Kimi 严格使用这些素材，并按 9 页结构生成 PPT。
4. 如果 Kimi 对长横图排版不理想，要求它使用“整页横向大图 + 少量标题/标注”的方式重排案例页。

## 设计原则

- 新 PPT 不沿用旧项目状态汇报结构。
- 核心主线是“Agent 自动化 + 图像识别 + SEM 正演建模”。
- 案例页以图片为主，文字为辅。
- 风格建议为深色学术科技风。

