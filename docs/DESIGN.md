# geoseg v2 设计文档（一页）

> v0.1 写于 2026-05-18(初稿)
> v0.2 修订于 2026-05-18(模块独立验证 + e026 复制进项目)
> v0.3 修订于 2026-05-18(PDF 提取嵌入图 + e025 schema 扩展 + VLM 受控智能参数)
> v0.4 修订于 2026-05-19(MinerU 主提取 + PyMuPDF rasterize fallback)
> v1（`photo/geo-segment-gui/`）失败终止后的第二次尝试。
> **未签字前不写任何代码。** 签字后再开实现工单。

---

## 1. 目标 & 验收准则

**导师导入文献 PDF → Agent 自动分析 → e026 分割 → 编辑分区 → 一键导出 SPECFEM。**

验收（缺一不可）：
1. 在固定 fixture `tests/fixtures/ph01/` 上端到端跑通：PDF → 提取嵌入图 → VLM 分析 → 分割 → overlay 显示 → SPECFEM 文件落盘。
2. 组装阶段集成测试覆盖 `controller → pipeline` 整条路径（mock VLM,无真实 API）,CI 可跑。
3. 任意 JSON schema 改动都同步改了**所有 consumer** 并通过集成测试,才允许提交。
4. **每个模块在合入主线前,必须有 standalone `demo.py` 跑通并产出可视化/JSON 验证产物。**

---

## 2. 设计原则（吸取 v1 教训,对应 `MISTAKE_LOG.md`）

| # | 原则 | v1 反面案例 |
|---|------|-------------|
| 1 | **CV 先 detect 候选 panel,VLM 仅返回语义描述,CV crop 精确 bbox** | Mistake 1/2:让 VLM 给 bbox 坐标,落白底 |
| 2 | **Schema 改动 = grep 全仓库 consumer + 原子提交** | Mistake 3:删 `colorbar.bbox` 没改 pipeline,运行时 KeyError |
| 3 | **模块独立可丢弃**:每个 module 先 standalone demo 跑通,再进入组装阶段。组装阶段才写 e2e 集成测试。 | Mistake 4:249 单测全绿但 e2e 一点就崩;Mistake 5:多 agent 各看一文件,缝合处全是 bug |
| 4 | **单一所有人开发 Phase 0,不分发给子 agent** | Mistake 5 |
| 5 | **QThread 每次 `Analyze` 重建 worker 实例,不复用** | Mistake 7:QThread 不能 restart,二次点击崩 |
| 6 | **初版 VLM 走 Claude Code 内置多模态能力,不管理外部 API key**。代码只封装 Prompt 组装 + JSON 解析 + 重试/降级逻辑,调用层利用当前 Claude Code session。 | (v0.2 未覆盖:初版不搞重 API key 管理) |
| 7 | **MinerU 主提取 + PyMuPDF rasterize fallback**:MinerU API 提取 PDF 中的 figure 图片 + 自动生成 caption markdown,作为 VLM 和 CV 的输入。MinerU 对嵌入图和位图的提取能力优于 PyMuPDF XObject。多 panel figure 被拆分时,用 PyMuPDF rasterize 该 figure 区域补全。 | (v0.3 卡点:PyMuPDF XObject 仅提取 32 张图,漏掉 velocity model;full-page rasterize 噪声太多 VLM confidence 仅 0.15-0.45) |
| 8 | **VLM 受控智能参数化**:每个 review 点有明确的重试上限(2 次)、confidence 阈值(0.7)、单页调用预算(5 次)。超限或低 confidence → 降级到人工 review。 | (v0.1/0.2 未覆盖:无限重试会烧预算,无 confidence 阈值会漏判) |
| 9 | **前端技术栈决策记录**:当前 PySide6 GUI 判定为「废品」（功能完整但不可用），废弃而非迁移。重写选型 Tauri + TS + Svelte + Fabric.js。决策依据：① PySide6 桌面外观差、打包体积大；② Tauri 前端可用现代 Web 生态（Canvas 库、PDF 渲染）；③ Python 后端保留，通过 FastAPI 暴露 HTTP API，前后端解耦。 | v1 教训:GUI 写完后发现交互体验差，但已沉没成本导致不愿重写 |

---

## 3. 流水线（无用户介入的自动段）

```
PDF
 ↓ M0.5: mineru_client.upload_and_extract(PDF) → 等待提取完成
   → images/ (figure 图片) + full.md (caption) + content_list.json (page/bbox/caption 元数据)
   → 用 content_list.json 构建 figure 索引: [{page_idx, img_path, caption, bbox, width, height}]
   → 若 figure 被拆分或尺寸 < 800px: PyMuPDF rasterize 该 figure 区域补全
 ↓ [FastAPI] POST /api/pdf/import → job_id
   → 前端轮询 GET /api/pdf/status/{job_id}
   → 返回 figure 列表 [{page_idx, img_path, caption, thumbnail_base64}]
 ↓ 用户选择 figure（前端 FigureSelector）

【Pipeline B — 纯Agent流】
 ↓ [FastAPI] POST /api/agent/process-figure
   → M1: vlm_client.classify_figure(figure 图片)
   → M1a: vlm_client.review_page_overview(figure 图片 + caption + text_blocks)
   → M1b: cv_detect.detect_panels(figure 图片)
   → M3: segment_engines.route_and_segment(panel 图片, reps, colorbar)
   → 返回 {classification, panels[], summary, review_warnings}
 ↓ 前端 QualityReview 弹窗 → [接受] / [人工修正] / [跳过]
 ↓ [接受] → PanelSelector → SegmentationCanvas（只读/微调）
 ↓ [人工修正] → 切换到 Pipeline A，预加载 Agent 结果

【Pipeline A — 纯人工流】
 ↓ 用户手动选择工具（多边形 / 矩形+智能收缩 / 画笔+区域生长）
 ↓ [FastAPI] POST /api/manual/segment-from-polygon / segment-from-rect / segment-from-stroke
   → 后端分割 → 返回 SegmentationResult（labels + contours JSON）
 ↓ 前端 Canvas 显示轮廓，用户可拖拽顶点修正
 ↓ [FastAPI] POST /api/export/specfem → tomography_file.xyz + Par_file_snippet.txt

tomography_file.xyz + Par_file_snippet.txt
```

**关键变化(v0.7 vs v0.6)**:
- 前端从 PySide6 改为 **Tauri + TypeScript + Svelte**，通过 **FastAPI HTTP API** 与 Python 后端通信
- 后端模块职责不变（`cv_detect` / `vlm_client` / `segment_engines` / `exporter`），仅新增 `geoseg/server.py` FastAPI 包装层
- 前端不直接处理原始 label ndarray — 后端预计算轮廓为 JSON 多边形列表
- Pipeline A / B 共享同一套 FastAPI endpoint，仅调用路径不同
- VLM review 输入新增 `text_blocks`（MinerU content_list 中的文本+bbox 空间信息）

---

## 3.5 双管线架构（v0.7）

为解决「人机混合操作边界不清」的问题，将 pipeline 拆为两套独立管线，通过统一接口 `pipeline_interfaces.py` 衔接。v0.7 在此基础上增加 **FastAPI HTTP 层** 和 **Tauri 前端**，前后端通过 REST API 通信。

### 前端架构（Tauri + TypeScript + Svelte）

| 层 | 技术 | 理由 |
|---|---|---|
| UI Framework | **Svelte** | 轻量、编译时优化、单文件组件，比 React 更适合桌面工具 |
| Canvas 库 | **Fabric.js** | 成熟的多边形/路径/选择框支持，内置拖拽、事件系统，TypeScript 友好 |
| 状态管理 | Svelte Stores | 足够轻量，无需 Redux/Zustand |
| HTTP Client | 原生 `fetch` | 直接调用 FastAPI |
| PDF 预览 | `pdfjs-dist` | 浏览器内渲染 PDF 页面 |
| 样式 | TailwindCSS + Lucide | 快速构建工具型 UI |

**核心画布：SegmentationCanvas**（Fabric.js 三层叠加）
1. **底图层**：原始图像 `fabric.Image`
2. **标签叠加层**：半透明彩色区域（来自后端预计算的轮廓 JSON）
3. **交互层**：当前正在绘制的工具图形（多边形顶点、矩形框、画笔路径）

### FastAPI 后端 API 设计

```python
# geoseg/server.py

class PanelInput(BaseModel):
    id: int
    bbox: tuple[int, int, int, int]
    source: str
    confidence: float | None

class SegmentationResult(BaseModel):
    labels: list[list[int]]   # 仅用于调试；前端不直接渲染
    contours: list[list[dict]]  # 预计算轮廓 JSON，前端主渲染源
    overlay_base64: str | None  # base64 PNG
    meta: dict

# Pipeline B (Agent)
POST /api/agent/process-figure
  → {image: multipart, caption: str, text_blocks: str(JSON), n_layers: int}
  ← {classification, panels[], summary, review_warnings}

POST /api/agent/detect-panels
  → {image: multipart}
  ← list[PanelInput]

POST /api/agent/segment
  → {image: multipart, n_layers: int, reps: str|None}
  ← SegmentationResult

# Pipeline A (Manual)
POST /api/manual/segment-from-polygon
  → {image: multipart, polygon: str(JSON [{x,y}...]), n_layers: int}
  ← SegmentationResult

POST /api/manual/segment-from-rect
  → {image: multipart, bbox: tuple[int,int,int,int], n_layers: int}
  ← SegmentationResult
  # 后端：在 bbox 内用 grab-cut / graph-cut / 颜色聚类自动收缩到精确边界

POST /api/manual/segment-from-stroke
  → {image: multipart, strokes: str(JSON [{x,y,label}...]), n_layers: int}
  ← SegmentationResult
  # 后端：以笔触像素为种子，用区域生长 / watershed 得到完整区域

# Shared
POST /api/export/specfem
  → {labels: multipart(NPZ), color_names: str(JSON)}
  ← {tomo_xyz: str, parfile_snippet: str}

# PDF
POST /api/pdf/import
  → {pdf: multipart}
  ← {job_id: str}

GET /api/pdf/status/{job_id}
  ← {status: "pending"|"done"|"error", figures: list[dict], message: str}
```

**NumPy 序列化约定**：label map 用压缩 NPZ → multipart；图像统一 PNG → multipart。交互时后端预计算轮廓为 JSON 多边形列表，前端不处理原始 label 数组。

### Pipeline A：纯人工流

```
PDF → FastAPI /api/pdf/import → 前端 FigureSelector
  ↓ 用户选择 figure
  ↓ 前端加载 figure 底图到 FigureReview 面板
  ↓ 【环节 1：Figure Review】用户观察 caption、colorbar、panel 数量
  ↓ 【环节 2：Panel 获取】
     ├─ 模式 A1: 用户手动框选 panel（矩形框工具）
     ├─ 模式 A2: 调用 CV detect_panels → 用户从候选框选择/修正/删除
     └─ 模式 A3: 整图作为一个 panel（fallback）
  ↓ 【环节 3：Panel Review】用户确认 panel 列表
  ↓ 对每个 panel：
     ↓ 前端加载 panel 底图到 SegmentationCanvas
     ↓ 【环节 4：分割】
        ├─ 模式 A1: 多边形绘制 → POST /api/manual/segment-from-polygon
        ├─ 模式 A2: 矩形框选+智能收缩 → POST /api/manual/segment-from-rect
        └─ 模式 A3: 画笔涂抹+区域生长 → POST /api/manual/segment-from-stroke
     ↓ 后端分割 → 返回 contours JSON
     ↓ 前端渲染彩色区域 + 可编辑顶点
  ↓ 用户修正 → 重新提交（可选）
  ↓ FastAPI /api/export/specfem → SPECFEM 文件
```

#### 环节 1：Figure Review（人工）
用户在前端 FigureReview 面板查看：
- figure 原图 + caption 文本
- 可手动标记：`has_colorbar`、`n_panels_estimate`、`figure_type`
- 可调用 `POST /api/agent/detect-panels` 获取 CV 候选框作为参考（但不强制使用）
- 决定：跳过 / 进入 Panel 获取

#### 环节 2：Panel 获取（人工 + 可选 CV 辅助）
在 figure 底图上，用户有三种方式获取 panel：

**模式 A1：手动框选**
- 矩形框工具拖拽画框
- 每个框生成一个 `PanelInput`，`source="manual"`

**模式 A2：CV 辅助 + 人工修正**
- 前端调用 `POST /api/agent/detect-panels`
- 返回候选框叠加在底图上
- 用户可：接受 / 删除某个框 / 新增框 / 调整框大小
- 最终确认后生成 `list[PanelInput]`

**模式 A3：整图 fallback**
- 不框选任何 panel，整张 figure 作为一个 panel
- `source="fallback_whole"`

#### 环节 3：Panel Review（人工）
- 显示 panel 缩略图列表
- 用户可选择跳过某个 panel
- 对每个 panel 决定进入分割环节

#### 环节 4：分割（人工交互工具）

**工具 1：多边形绘制**
- 单击添加顶点，显示实线连接
- 双击 / 回车闭合 → 提交 `POST /api/manual/segment-from-polygon`
- ESC 取消当前绘制
- 后端：根据多边形内像素做区域分割（类似 mask-prompt）

**工具 2：矩形框选 + 智能边缘收缩**
- 拖拽画矩形 → 显示虚线框
- 松开 → 提交 `POST /api/manual/segment-from-rect`
- 后端：在 bbox 内用 grab-cut / graph-cut / 颜色聚类自动收缩到精确边界
- 结果显示为精确的多边形轮廓

**工具 3：画笔涂抹 + 区域生长**
- 按住鼠标涂抹 → 显示彩色笔触
- 松开 → 提交 `POST /api/manual/segment-from-stroke`
- 可切换「前景画笔」和「背景画笔」（类似 Photoshop 快速选择工具）
- 后端：以笔触像素为种子，用区域生长 / watershed 得到完整区域

#### 结果修正
所有工具结果都显示为可编辑的多边形：
- **拖拽顶点**：调整边界
- **右键菜单**：删除区域、重新分配层 ID
- **阈值滑块**：过滤小区域（面积 < threshold%）
- **撤销/重做**：CommandStack 记录每次操作

### Pipeline B：纯Agent流

```
PDF → FastAPI /api/pdf/import → 前端 FigureSelector（缩略图网格 + CV 分类标记）
  ↓ 用户选择 figure
  ↓ FastAPI POST /api/agent/process-figure
    → 【环节 1】M1 classify_figure
    → 【环节 2】M1a review_page_overview（含 panel 检测）
    → 【环节 3】M1b detect_panels（CV 精确 bbox）
    → 【环节 4】M3 route_and_segment
  ← 返回结果 + QualityReview
  ↓ 前端 QualityReview 弹窗
```

#### QualityReview 弹窗
```
┌─────────────────────────────────────┐
│  Agent 结果 Review                   │
│  • Figure 类型: velocity_model       │
│  • 检测到 3 个 panels                │
│  • VLM 置信度: 0.85                  │
│  • 警告: panel_mismatch              │
│                                      │
│  [接受]  [人工修正]  [跳过]           │
└─────────────────────────────────────┘
```
- **[接受]** → 进入 PanelSelector → SegmentationCanvas（只读/微调模式）
- **[人工修正]** → 无缝切换到 Pipeline A 画布，预加载 Agent 结果（panel bbox + 分割轮廓）
- **[跳过]** → 返回 FigureSelector，可选其他 figure

### Agent → Manual 无缝衔接

Agent 流的输出（`list[PanelInput]` + `SegmentationResult`）可直接作为 Pipeline A 画布的初始状态加载。用户不需要重新画边界，只需要微调。

```typescript
function loadAgentResult(result: AgentResult) {
  result.panels.forEach(p => canvas.addPanelBBox(p.bbox));
  result.segmentation.contours.forEach((contour, i) => {
    canvas.addRegion({
      label: i + 1,
      color: getColor(result.segmentation.meta.color_names[i]),
      polygon: contour,
    });
  });
}
```

### 接口衔接点（v0.7 更新）

| 环节 | 数据契约 | 人工流实现 | Agent流实现 | 切换场景 |
|------|----------|-----------|------------|---------|
| PDF 提取 | figure 图片列表 | `mineru_client`（共享） | 同左 | 无 |
| Figure 选择 | figure 索引 | `FigureSelector`（共享） | 同左 | 无 |
| **Figure Review** | `{figure_type, has_colorbar, n_panels}` | 人工观察 + 手动标记面板 | `vlm_client.classify_figure` + `review_page_overview` | VLM confidence < 0.7 → 弹窗让用户确认 |
| **Panel 获取** | `list[PanelInput]` | ① 手动框选 ② CV 辅助+人工修正 ③ 整图 fallback | `cv_detect.detect_panels` | Agent 检测后用户可修正；或用户直接放弃检测自己框选 |
| **Panel Review** | `list[PanelInput]`（过滤后） | 人工确认 panel 列表 | `QualityReview` 自动判断 | Agent 检测到 panel_mismatch 时人工介入 |
| **分割** | `SegmentationResult` | `POST /api/manual/segment-from-*`（多边形/矩形/画笔） | `segment_engines.router.route_and_segment` | Agent 分割不满意 → 切到交互工具重画 |
| **后处理/导出** | SPECFEM 文件 | `POST /api/export/specfem`（共享） | 同左 | 无 |

关键原则：**每个环节两种实现，同一数据契约，任意切换**。Pipeline A 可以在 Panel 获取环节调用 Agent 的 `detect_panels`；Pipeline B 可以在 Review 环节降级到人工确认。模块只认 `PanelInput` / `SegmentationResult` / `QualityReview`，不认上游来源。

## 4. Schema 契约

> **改动需更新本文档 + 全 consumer + 跑 `tests/test_integration_ph01.py`**

### 4.1 HTTP API 契约（TS 前端 ↔ FastAPI 后端）

#### `POST /api/pdf/import` Request / Response
```json
// Request: multipart/form-data, pdf=file
// Response:
{
  "job_id": "uuid-string",
  "status": "accepted"
}
```

#### `GET /api/pdf/status/{job_id}` Response
```json
{
  "status": "done",
  "figures": [
    {
      "page_idx": 7,
      "img_path": "/tmp/.../figure_01.png",
      "caption": "Figure 2. Vs model...",
      "thumbnail_base64": "iVBORw0KGgo...",
      "width": 1343,
      "height": 691,
      "cv_class": "conceptual_model"
    }
  ],
  "message": "MinerU extracted 42 figures"
}
```

#### `POST /api/agent/process-figure` Request / Response
```json
// Request: multipart/form-data
//   image=file, caption=str, text_blocks=str(JSON), n_layers=int

// Response:
{
  "classification": {
    "figure_type": "conceptual_model",
    "vlm_classification": "velocity_model",
    "vlm_confidence": 0.92,
    "vlm_reason": "Colored regions..."
  },
  "panels": [
    {
      "panel_id": 0,
      "bbox": [0, 0, 1343, 691],
      "classification": {"figure_type": "conceptual_model"},
      "segmentation": {
        "labels": "<base64 NPZ>",
        "overlay": "<base64 PNG>",
        "meta": {
          "engine": "vivid_nn",
          "color_names": ["slow_red", "fast_blue"],
          "n_layers": 2,
          "quality_score": 0.85
        }
      },
      "review": {
        "n_layers_found": 2,
        "is_target_panel": true
      }
    }
  ],
  "summary": {
    "status": "ok",
    "n_panels": 1,
    "total_layers": 2,
    "engines_used": ["vivid_nn"],
    "saturation_ratio": 0.72,
    "review_warnings": [],
    "vlm_has_colorbar": true,
    "vlm_target_panel_id": 0
  }
}
```

#### `POST /api/manual/segment-from-polygon` Request / Response
```json
// Request: multipart/form-data
//   image=file, polygon=str(JSON [{"x":int,"y":int},...]), n_layers=int

// Response: SegmentationResult
{
  "labels": "<base64 NPZ>",
  "contours": [
    [{"x": 100, "y": 200}, {"x": 150, "y": 250}, ...],
    [{"x": 300, "y": 400}, ...]
  ],
  "overlay_base64": "iVBORw0KGgo...",
  "meta": {
    "engine": "manual_polygon",
    "color_names": ["layer_1", "layer_2"],
    "n_layers": 2,
    "quality_score": 1.0
  }
}
```

#### `POST /api/export/specfem` Request / Response
```json
// Request: multipart/form-data
//   labels=file(.npz), color_names=str(JSON ["layer_1", ...])

// Response:
{
  "tomo_xyz": "<file content as string or download url>",
  "parfile_snippet": "<file content as string>"
}
```

### 4.2 后端内部 JSON Schema 契约（Python 模块间）

### `figure_classification`（VLM-C 输出,语义分类）
```json
{
  "figure_type": "velocity_model",
  "confidence": 0.92,
  "reason": "Colored regions representing seismic velocities with colorbar"
}
```
- `figure_type` 枚举: `velocity_model` / `geological_cross_section` / `shot_gather` / `waveform_plot` / `equation` / `statistical_plot` / `data_table` / `tomography_map` / `flowchart` / `other`。
- `confidence` 必填,低于 0.7 触发降级。
- 只有 `velocity_model` 和 `geological_cross_section` 是有效分割目标。

### `page_overview`（VLM-0 输出,看嵌入图 + caption）
```json
{
  "page_idx": 7,
  "image_size": {"width": 1343, "height": 691},
  "figure_type": "velocity_model_cross_section",
  "panels": [
    {"id": 0, "description": "Absolute isotropic Vs at 5 km depth (panel a)"},
    {"id": 1, "description": "Absolute isotropic Vs at 15 km depth (panel b)"}
  ],
  "target_panel_id": 0,
  "has_colorbar": true,
  "noise_elements": [
    {"kind": "colorbar", "description": "Vs colorbar 1000-5000 m/s"},
    {"kind": "axis_label", "description": "40°N"},
    {"kind": "annotation", "description": "KMP"}
  ],
  "color_zones": [
    {"color_name": "slow_red", "colorbar_value": 0},
    {"color_name": "slow_orange", "colorbar_value": 25},
    {"color_name": "medium_yellow", "colorbar_value": 50},
    {"color_name": "fast_cyan", "colorbar_value": 75},
    {"color_name": "fast_blue", "colorbar_value": 100}
  ],
  "confidence": 0.92
}
```
- **VLM 不给坐标**!只有语义描述 (`description`)。
- `figure_type` 枚举: `velocity_model_cross_section` / `velocity_model_map` / `3d_isosurface` / `reflection_amplitude` / `uncertain`。
- `confidence` 必填,低于 0.7 触发降级。
- `color_zones` 的 `colorbar_value` 是归一化 0-100(百分比),后续结合 colorbar 物理范围换算为真实速度。

### `segmentation_result`（pipeline 返回,不变）
```json
{
  "components": [{"id":int,"layer_id":int,"bbox":[x,y,w,h],"area":int,"centroid":[cx,cy]}],
  "layers": [{"id":int,"vp":float,"vs":float,"rho":float,"color":[r,g,b]}],
  "overlay_path": "/abs/path/to/overlay.png"
}
```

### `PanelInput`（panel 检测/选择的最小契约）
```json
{
  "id": 0,
  "bbox": [0, 0, 1343, 691],
  "source": "cv_detect",
  "confidence": 0.95
}
```
- `source` 枚举: `cv_detect` / `manual` / `vlm_hint` / `fallback_whole`
- 空列表 → 调用 `make_whole_image_panel()` 生成全图 fallback

### `SegmentationResult`（分割引擎统一输出）
```json
{
  "labels": "<ndarray HxW int32>",
  "overlay": "<ndarray HxWx3 uint8, optional>",
  "meta": {
    "engine": "vivid_nn",
    "color_names": ["slow_red", "medium_yellow", "fast_blue"],
    "n_layers": 3,
    "quality_score": 0.85
  }
}
```
- `labels` 是唯一的 truth source，`overlay` 仅用于显示
- `meta.engine` 记录实际使用的引擎，用于 audit 追溯

### `QualityReview`（review 统一输出）
```json
{
  "warnings": ["panel_mismatch: vlm_sees_3_panels cv_detects_2_panels"],
  "score": 0.6,
  "can_auto_fix": true,
  "suggested_action": "retry"
}
```
- `suggested_action` 枚举: `continue` / `retry` / `manual_intervention` / `skip`
- `can_auto_fix=true` 时 controller 可用 repair_hints 重试对应模块

---

## 5. 模块责任表（模块化目录 + 行预算,超预算拆分）

### 5.1 Python 后端模块（独立 demo 跑通,顺序见 §8）

| 路径 | 职责 | 行预算 |
|------|------|--------|
| `tests/fixtures/ph01/` | `gxae11701.pdf` + `ph01_vlm.json`(e025 格式参考) | — |
| `geoseg/modules/mineru_client/client.py` | **M0.5-MinerU**:MinerU v4 API 客户端。`upload_and_extract(pdf_path)` → 上传 PDF → 轮询结果 → 下载 zip → 解压为 `images/` + `full.md` + `content_list.json` | <150 |
| `geoseg/modules/mineru_client/demo.py` | M0.5-MinerU standalone:在 ph01 PDF 上验证提取出 figure 图片 + caption + 元数据 | <60 |
| `geoseg/modules/pdf_extractor/extract.py` | **M0.5-Fallback**:PyMuPDF 提取 `{XObject(Image) + 页面文字块}`;`rasterize_page(pdf_path, page_idx, dpi=300)` 整页/区域 rasterize。当 MinerU 拆分 figure 或提取尺寸过小时 fallback | <200 |
| `geoseg/modules/pdf_extractor/demo.py` | M0.5-Fallback standalone:验证 rasterize 能产出 2481×3260 page PNG;验证 XObject 提取 | <60 |
| `geoseg/modules/cv_detect/panel_detector.py` | M1b:在**提取后的 figure 图**上 detect panel 候选 bbox(连通域 + 矩形面积过滤)。实现 `PanelDetector` Protocol | <150 |
| `geoseg/modules/cv_detect/figure_classifier.py` | CV heuristic figure 分类（饱和度/颜色分布） | <100 |
| `geoseg/modules/cv_detect/colorbar_extractor.py` | 从 figure 中提取 colorbar 区域 | <100 |
| `geoseg/modules/cv_detect/demo.py` | M1b standalone:在 ph01 嵌入图上画框,人眼比对 e026 已知的 4 个 pattern panel | <60 |
| `geoseg/modules/vlm_client/client.py` | M2: **唯一 LLM 出口**。2 个调用:`classify_figure` / `review_page_overview`。实现 `QualityReviewer` Protocol。每个调用含 Prompt 组装 + pydantic schema 校验 + confidence 校验 + 重试/降级逻辑。初版走 Claude Code CLI 多模态能力(`claude -p` + `--json-schema`),不管理外部 API key | <300 |
| `geoseg/modules/vlm_client/prompts.py` | M2: Prompt 模板 + JSON schema 定义(pydantic model)。版本号写进每条 audit JSON | <150 |
| `geoseg/modules/vlm_client/demo.py` | M2 standalone:在当前 Claude Code session 中验证 2 个调用点的 Prompt 组装 + schema 校验 + confidence 校验 + audit 落盘 | <80 |
| `geoseg/modules/segment_engines/router.py` | M3: 根据图像特征（饱和度、颜色 vividness）自动路由到合适的分割引擎。实现 `Segmenter` Protocol | <200 |
| `geoseg/modules/segment_engines/vivid_nn.py` | vivid color 神经网络分割引擎 | <250 |
| `geoseg/modules/segment_engines/v4_kmeans.py` | k-means 颜色聚类分割引擎 | <200 |
| `geoseg/modules/segment_engines/edge_guided.py` | 边缘引导分割引擎 | <200 |
| `geoseg/modules/segment_engines/grayscale_agglomerative.py` | 灰度图层次聚类分割引擎 | <150 |
| `geoseg/modules/segment_engines/full_pipeline.py` | M3 组装：`classify → detect panels → VLM review → segment`。编排各模块 | <300 |
| `geoseg/modules/segment_engines/demo.py` | M3 standalone:读 ph01 嵌入图 → 复现分割 overlay | <80 |
| `geoseg/modules/post_process/polygon.py` | M3.5: 从 label map 提取多边形轮廓（contours）+ GeoJSON 导出 | <200 |
| `geoseg/modules/post_process/properties.py` | M3.5: 为每层分配 Vp/Vs/ρ 属性值 | <150 |
| `geoseg/modules/post_process/demo.py` | M3.5 standalone:验证 polygon 提取 + 属性分配 | <60 |
| `geoseg/modules/exporter/specfem.py` | M4: SPECFEM tomography_file + Par_file snippet | <200 |
| `geoseg/modules/exporter/demo.py` | M4 standalone:mock layers + components 输入,验证文件格式 | <60 |
| `geoseg/pipeline_interfaces.py` | **接口契约**:双管线共享数据结构(`PanelInput`/`SegmentationResult`/`QualityReview`) + Protocol 定义。改契约 = 更新所有 consumer | <150 |
| `geoseg/controller.py` | 后端流水线编排(M0.5→M1→M1a→M1b→M3→M4)。被 `server.py` 调用，不感知 HTTP/GUI | <400 |
| `geoseg/server.py` | **M7-FastAPI**: HTTP API 包装层。暴露 `/api/agent/*` / `/api/manual/*` / `/api/pdf/*` / `/api/export/*`。启动 Python 为 sidecar | <300 |
| `tests/test_integration_ph01.py` | day-final e2e(mock VLM,2 个调用点分别 mock)| <200 |

### 5.2 Tauri 前端模块（`geoseg-gui/`，与 Python 同级目录）

```
geoseg-gui/
├── src/
│   ├── lib/
│   │   ├── api.ts          # FastAPI 客户端封装（fetch wrapper + 类型）
│   │   ├── types.ts        # PanelInput, SegmentationResult, QualityReview 等 TS 类型
│   │   └── stores.ts       # 全局 Svelte Stores（currentImage, labels, tools, regions）
│   ├── components/
│   │   ├── AppShell.svelte         # 主布局：菜单栏 + 工作区 + 状态栏
│   │   ├── Toolbar.svelte          # 工具栏：多边形 / 矩形 / 画笔 / 撤销 redo
│   │   ├── StatusBar.svelte        # 底部状态栏
│   │   ├── PipelineSelector.svelte # A/B pipeline 切换
│   │   ├── FigureSelector.svelte   # 缩略图网格 + CV 分类标记
│   │   ├── PanelSelector.svelte    # panel 选择弹窗
│   │   ├── QualityReviewDialog.svelte  # Agent 结果 review 弹窗
│   │   ├── ExportPanel.svelte      # 导出配置面板
│   │   └── ThresholdSlider.svelte  # 面积阈值滑块
│   ├── canvas/
│   │   ├── SegmentationCanvas.svelte   # Fabric.js 画布主组件（三层叠加）
│   │   ├── tools/
│   │   │   ├── PolygonTool.ts          # 多边形顶点绘制
│   │   │   ├── RectSmartTool.ts        # 矩形框 + 后端智能收缩
│   │   │   ├── BrushTool.ts            # 画笔涂抹 + 区域生长
│   │   │   └── ToolManager.ts          # 工具切换 + 状态管理
│   │   └── renderers/
│   │       ├── LabelOverlay.ts         # 标签颜色叠加层
│   │       └── RegionRenderer.ts       # 多边形区域渲染（可拖拽顶点）
│   ├── routes/
│   │   ├── AgentPipeline.svelte        # Pipeline B 流程页
│   │   └── ManualPipeline.svelte       # Pipeline A 画布页
│   └── App.svelte
├── src-tauri/
│   └── src/main.rs         # Tauri 入口，spawn Python sidecar (`python -m geoseg.server`)
└── package.json
```

| 路径 | 职责 | 行预算 |
|------|------|--------|
| `src/lib/api.ts` | FastAPI HTTP 客户端。所有 API 调用集中在此，含错误处理、重试、base64/JSON 序列化 | <200 |
| `src/lib/types.ts` | TypeScript 类型定义，与 Python `pipeline_interfaces.py` 保持同步 | <100 |
| `src/lib/stores.ts` | Svelte writable/derived stores：当前图像、labels、regions、工具状态、undo stack | <100 |
| `src/canvas/SegmentationCanvas.svelte` | Fabric.js 画布主组件。三层叠加管理 + 缩放/平移 + 事件路由到当前工具 | <300 |
| `src/canvas/tools/*.ts` | 三种交互工具 + ToolManager。每种工具独立文件 <150 行 | <150×4 |
| `src/canvas/renderers/*.ts` | LabelOverlay（半透明颜色层）+ RegionRenderer（多边形+顶点拖拽） | <150×2 |
| `src/components/*.svelte` | UI 组件。每个组件单一职责，<200 行 | <200×8 |
| `src/routes/*.svelte` | Pipeline A/B 页面级组件 | <200×2 |
| `src-tauri/src/main.rs` | Tauri 入口。启动 Python FastAPI sidecar，管理进程生命周期 | <100 |

### 5.3 已废弃模块（v0.7 起不再维护）

以下 PySide6 GUI 模块将在 Tauri 前端完成后删除，v0.7 期间保留但不更新：
- `geoseg/gui/main_window.py`
- `geoseg/gui/segmentation_view.py`
- `geoseg/gui/figure_selector.py`
- `geoseg/gui/panel_selector.py`
- `geoseg/gui/pdf_import_worker.py`
- `geoseg/gui/pdf_page_review_worker.py`
- `geoseg/gui/demo.py`

### 5.4 产物目录（gitignore）

```
runs/M0/       # fixture 移植 + e026 复刻 baseline
runs/M0.5/     # PDF 提取出的 figure 图片 + caption markdown + content_list.json
runs/M0.5-fb/  # PyMuPDF fallback: rasterize 补全的 figure 区域
runs/M1/       # VLM 的 figure_classification + page_overview JSON + audit
runs/M1b/      # CV panel 候选标注图
runs/M3/       # pipeline overlay + segmentation_result.json
runs/M4/       # 导出的 SPECFEM 文件
runs/M5/       # Tauri 前端截图 / 录屏
runs/audit/    # 每次 VLM 调用的 {prompt, model_version, input_images, output_json, confidence, timestamp}
geoseg-gui/dist/   # Tauri 构建产物
```

---

## 6. 集成测试（组装阶段才写,先于 controller 实现）

```python
# tests/test_integration_ph01.py
def test_analyze_segment_export_with_ph01(tmp_path, monkeypatch):
    fixture = Path("tests/fixtures/ph01")
    pdf_path = fixture / "gxae11701.pdf"
    vlm_cls = json.loads((fixture / "ph01_vlm_figure_class.json").read_text())
    vlm_overview = json.loads((fixture / "ph01_vlm_overview.json").read_text())

    # Mock the only LLM exit at 2 call points
    monkeypatch.setattr("geoseg.modules.vlm_client.client.classify_figure",
                        lambda img, **kw: vlm_cls)
    monkeypatch.setattr("geoseg.modules.vlm_client.client.review_page_overview",
                        lambda img, text, **kw: vlm_overview)

    ctrl = MainController(window=DummyWindow())
    ctrl.load_pdf(pdf_path)          # M0.5 提取嵌入图
    ctrl.analyze_page()              # M1 + M1a + M1b
    ctrl.run_segmentation()           # M3
    ctrl.export_specfem(tmp_path)     # M4

    assert len(ctrl.state.components) >= 5
    assert (tmp_path / "tomography_file.xyz").exists()
    assert (tmp_path / "Par_file_snippet.txt").exists()
```
本测试是**主路径回归**,组装阶段每次改动必跑通才合并。

---

## 7. 显式不做（v1 教训 + DECISIONS.md）

- ❌ VLM 提供 bbox 坐标（任何用途）
- ❌ Agent 聊天面板（Phase 2）
- ❌ Swarm 多 agent 并行写代码（Phase 0 单 owner）
- ❌ 任何 `geo_segment_skill/lib` 之外的"重写算法"冲动
- ❌ 依赖 `~/.claude/skills/geo-segment/lib/`(不存在)
- ❌ **VLM 黑盒决策**:每个 review 点必须有 JSON schema + Prompt 版本号 + audit 落盘
- ❌ **VLM 无限重试**:每个 review 点最多 5 次重试,超限强制人工 review
- ❌ **跨模块共享状态**:模块阶段每个 module 只通过函数参数和返回值通信
- ❌ **维护 PySide6 GUI**:v0.7 起全面废弃 `geoseg/gui/`，不迁移、不修复、不增加功能
- ❌ **前端直接调 Python 函数**:前后端必须通过 HTTP API 通信，禁止共享内存或直接 import
- ❌ **前端处理原始 label ndarray**:后端预计算轮廓 JSON，前端只渲染矢量图形

---

## 8. 开工顺序（签字后，按周推进，每周独立可验证）

> 原则：后端模块（M0-M4）已在 v0.6 前基本完成。v0.7 重点是 **FastAPI 包装层 + Tauri 前端重写**。按模块独立验证 → 前后端联调 → 完整 workflow 三阶段推进。

### Week 1: FastAPI 后端骨架（M7）

1. **`geoseg/server.py`**：包装现有模块为 FastAPI HTTP 服务
   - `POST /api/pdf/import` + `GET /api/pdf/status/{job_id}`
   - `POST /api/agent/process-figure` / `detect-panels` / `segment`
   - `POST /api/manual/segment-from-polygon` / `segment-from-rect` / `segment-from-stroke`
   - `POST /api/export/specfem`
   - **跑通判据**: `curl -X POST http://localhost:8000/api/agent/detect-panels -F image=@test.png` 返回 `list[PanelInput]` JSON

2. **验证后端模块仍为绿色**：跑 `pytest tests/test_integration_ph01.py`

### Week 2: Tauri + TS 骨架 + FigureSelector + FigureReview

3. **初始化 `geoseg-gui/`**：`npm create tauri-app` → 选 Svelte + TypeScript
4. **`src-tauri/src/main.rs`**：spawn Python sidecar（`python -m geoseg.server`）
5. **`src/lib/api.ts`** + **`src/lib/types.ts`**：HTTP 客户端 + 类型定义
6. **`src/components/FigureSelector.svelte`**：缩略图网格 + CV 分类标记
7. **`src/components/FigureReview.svelte`**：figure 原图 + caption 显示 + 人工标记面板（has_colorbar / n_panels / figure_type）
   - **跑通判据**: Tauri 窗口可加载测试图像，FigureSelector 显示缩略图网格，点击后进入 FigureReview 面板

### Week 3: Pipeline A — Panel 获取 + 多边形分割

8. **Panel 获取（手动框选）**：`src/canvas/tools/RectBBoxTool.ts`
   - 在 figure 底图上拖拽画矩形框生成 panel
   - 显示已框选的 panel 列表，可删除/调整
9. **Panel 获取（CV 辅助）**：前端调用 `POST /api/agent/detect-panels` → 候选框叠加 → 用户确认/修正
10. **PanelSelector**：显示 panel 缩略图列表，用户选择进入分割
11. **`src/canvas/tools/PolygonTool.ts`**：单击加点、双击闭合
12. **`POST /api/manual/segment-from-polygon`**：后端根据多边形 mask 分割
13. **`src/canvas/renderers/RegionRenderer.ts`**：显示后端返回的轮廓多边形
    - **跑通判据**: ① 手动框选 2 个 panel → PanelSelector 显示列表；② 选择 panel → 画多边形 → 闭合 → 后端返回 contours → 前端显示彩色区域

### Week 4: Pipeline A — 矩形+智能收缩 + 画笔+区域生长

14. **`src/canvas/tools/RectSmartTool.ts`**：在 panel 底图上拖拽画矩形 → 提交 `POST /api/manual/segment-from-rect`
15. **后端 grab-cut / graph-cut**：在 bbox 内自动收缩到精确边界
16. **`src/canvas/tools/BrushTool.ts`**：画笔涂抹 → 提交 `POST /api/manual/segment-from-stroke`
17. **后端区域生长 / watershed**：以笔触为种子分割
18. **`src/components/Toolbar.svelte`**：三种分割工具切换
    - **跑通判据**: 三种分割工具分别能成功分割并显示可编辑轮廓

### Week 5: Pipeline B — Agent 流 + PDF 导入

19. **`src/components/QualityReviewDialog.svelte`**：Agent 结果 review 弹窗
20. **`src/routes/AgentPipeline.svelte`**：Pipeline B 流程页
21. **PDF 导入前端链路**：上传 PDF → 轮询 status → 显示 FigureSelector
    - **跑通判据**: PDF → Figure 选择 → Agent 自动运行（classify → review → detect → segment）→ QualityReviewDialog → [接受/人工修正/跳过]

### Week 6: 衔接 + 导出 + 打磨

22. **Agent → Manual 无缝切换**：`loadAgentResult()` 预加载 Agent 的 panel bbox + 分割轮廓到画布
23. **`src/components/ExportPanel.svelte`** + `POST /api/export/specfem`
24. **撤销/重做**：前端 CommandStack（每次 API 调用 + 用户编辑）
25. **阈值滑块**：过滤小区域
26. **UI 打磨**：状态栏、错误提示、加载动画
    - **跑通判据**: 完整 workflow：PDF → Figure 选择 → Agent 分割 → 人工修正 → SPECFEM 导出

**总计：~6 周**

| 模块 | 行数估算 | 时间 |
|---|---|---|
| FastAPI 后端（server.py） | ~300 行 Python | 1 周 |
| Tauri + TS 骨架 + FigureSelector/FigureReview | ~400 行 TS/Svelte | 1 周 |
| Panel 获取 + 多边形分割 | ~300 行 TS + Python | 1 周 |
| 矩形+智能收缩 / 画笔+区域生长 | ~300 行 TS + Python | 1 周 |
| Pipeline B Agent 流 | ~400 行 TS/Svelte | 1 周 |
| 衔接、导出、打磨 | ~300 行 TS | 1 周 |
| **总计** | **~2,000 行** | **~6 周** |

---

## 9. 模块跑通验收清单（硬门禁）

### 后端模块（v0.6 已完成，v0.7 保持回归）

| 模块 | 跑通判据(必须全部满足) | 产物路径 |
|------|------|----------|
| **M0** | ① 两个 e026 脚本核心函数已搬入 `e026_algo/{core,components}.py` 且去硬编码; ② demo.py 跑出的 `pattern1_overlay.png` 和 e026 原产物像素级一致(MSE ≤ 1%) | `runs/M0/pattern1_overlay.png` + diff 报告 |
| **M0.5** | ① `extract.py` 能提取 ph01 PDF 全部 17 页的嵌入图; ② Page 7 提取出 2 个嵌入图(1343×874 + 1343×802); ③ caption 文字块与对应图正确配对 | `runs/M0.5/page_7_images/` + `page_7_text.json` |
| **M1** | ① `classify_figure` 返回 JSON 通过 schema 校验; ② `figure_type` 为 `velocity_model`; ③ `confidence ≥ 0.7` | `runs/M1/figure_classification.json` + `audit/cls.json` |
| **M1a** | ① `review_page_overview` 返回 JSON 通过 schema 校验; ② `color_zones` 非空; ③ `confidence ≥ 0.7`; ④ `panels` 数 ≥ 2 | `runs/M1a/page_overview.json` + `audit/001.json` |
| **M1b** | demo 在提取后的 figure 图上画出 ≥2 个候选框,人眼比对 panel 中心点偏差 ≤ 5px | `runs/M1b/ph01_candidates.png` |
| **M3** | demo 输入 ph01 pattern1 → 输出 `segmentation_result` 通过 schema 校验,`components` 数 ≥ 5,`overlay_path` 文件存在 | `runs/M3/segmentation_result.json` + `overlay.png` |
| **M4** | demo 输入 7 层 mock layers → 输出文件 `tomography_file.xyz` 行数 = 网格点数,`Par_file_snippet.txt` 包含 `nbmodels = 7` | `runs/M4/tomography_file.xyz` + `Par_file_snippet.txt` |

### v0.7 新增模块

| 模块 | 跑通判据(必须全部满足) | 产物路径 |
|------|------|----------|
| **M7-FastAPI** | ① `python -m geoseg.server` 启动成功，监听 8000 端口; ② `curl -F image=@test.png http://localhost:8000/api/agent/detect-panels` 返回 JSON 数组且每个元素含 `id`/`bbox`/`source`; ③ `curl -F image=@test.png -F polygon='[{"x":0,"y":0},...]' http://localhost:8000/api/manual/segment-from-polygon` 返回 JSON 含 `contours` + `overlay_base64` | `runs/M7/curl_test.log` |
| **M8-Tauri骨架** | ① `npm run tauri dev` 启动窗口无报错; ② 窗口可加载测试图像并显示; ③ Python sidecar 自动启动且 API 可访问 | `runs/M8/screenshot.png` |
| **M9-FigureReview** | ① FigureSelector 显示缩略图网格，点击后进入 FigureReview; ② FigureReview 面板显示原图 + caption + 人工标记选项（has_colorbar / n_panels / figure_type） | `runs/M9/figure_review.png` |
| **M10-Panel获取** | ① 手动框选：在 figure 底图上拖拽生成 panel 框，PanelSelector 显示列表; ② CV 辅助：调用 `POST /api/agent/detect-panels` 获取候选框，叠加显示后用户确认/修正/删除; ③ 整图 fallback：不框选时整张图作为一个 panel | `runs/M10/panel_selection.png` |
| **M11-多边形分割** | ① 选择 panel 进入 SegmentationCanvas; ② 单击添加顶点、实线连接; ③ 双击闭合后调用 `POST /api/manual/segment-from-polygon`; ④ 后端返回 contours 后前端正确渲染彩色区域; ⑤ 区域可拖拽顶点修正 | `runs/M11/polygon_demo.png` |
| **M12-矩形/画笔分割** | ① 矩形拖拽画框 → `POST /api/manual/segment-from-rect` → 返回收缩后轮廓; ② 画笔涂抹 → `POST /api/manual/segment-from-stroke` → 返回区域生长结果; ③ 两种工具结果均可编辑 | `runs/M12/rect_brush_demo.png` |
| **M13-Agent流** | ① 上传 PDF → 显示 FigureSelector; ② 选择 figure → Agent 自动运行（classify → review → detect → segment）→ 显示 QualityReviewDialog; ③ [接受] → PanelSelector → Canvas（只读/微调）; ④ [人工修正] → 切换到 Pipeline A 画布并预加载 Agent 结果 | `runs/M13/agent_flow.png` |
| **M14-导出** | ① Canvas 中修正后的区域可导出 SPECFEM; ② 导出文件格式正确 (`tomo.xyz` + `parfile_snippet.txt`) | `runs/M14/export_files/` |

任一模块未通过验收,**不**进入下一周。

---

## 10. VLM 受控智能参数表

| 参数 | 值 | 说明 |
|---|---|---|
| **单 review 点重试上限** | 5 次 | `missing_panels` 非空或 `quality == degraded` 时,用 `repair_hints` 重跑对应模块 |
| **Confidence 阈值** | 0.7 | 低于此值触发降级,不进入重试 |
| **单页 VLM 调用预算上限** | 10 次 | 2 个正常 review 点 + 最多 8 次重试(全局预算约束)。超限强制人工 review |
| **Prompt 版本控制** | `prompts.py` 中每个模板有 `VERSION = "x.y"`,写进每条 audit JSON | 可重放,可对比不同版本效果 |
| **Audit 保留策略** | 每次调用落 `{timestamp}_{step}_{retry}.json`,保留最近 100 次 | 人工 review 时可追溯 |
| **降级路径** | confidence < 0.7 或 retry ≥ 2 → UI 弹窗展示 VLM 判断 + audit 链接,等人工确认 | 不静默失败 |

---

## 11. v0.4 vs v0.3 改动摘要

| 节 | 改动 |
|---|------|
| §2 | 原则 7 更新:MinerU API 主提取 + PyMuPDF rasterize fallback(替代纯 PyMuPDF XObject) |
| §3 | 流水线起点改为 `mineru_client.upload_and_extract` → figure 图片 + caption markdown + content_list.json;增加 MinerU 拆分 figure 时的 PyMuPDF fallback 路径 |
| §5 | 新增 `mineru_client` 模块(M0.5-MinerU);`pdf_extractor` 改为 fallback 角色(M0.5-Fallback);产物目录新增 `runs/M0.5-fb/` |
| §8 | M0.5 拆分为 mineru_client 主提取 + pdf_extractor fallback 补全 |
| §9 | M0.5 验收更新:MinerU 提取 ≥30 figure + caption;Fig 14 完整提取;拆分 figure fallback 补全。M1a 面板数 ≥2(原 ≥4,因实际 figure 为 2-panel)。M1b 输入改为"提取后的 figure 图" |
| §11 | **新增**:v0.4 改动摘要(本节) |

## 12. v0.3 vs v0.2 改动摘要

| 节 | 改动 |
|---|------|
| §2 | 新增原则 7(PDF 提取嵌入图而非 rasterize) + 原则 8(VLM 受控智能参数化) |
| §3 | 流水线起点从 `pdf2image(300dpi)` 改为 `pdf_extractor` 提取嵌入图 + 页面文字;CV detect 输入改为**嵌入图**;增加 VLM 重试/降级路径 |
| §4 | Schema 从 2 个扩展为 4 个:`page_overview` / `panel_review` / `segmentation_review` / `segmentation_result`(不变) |
| §5 | 新增 M0.5 `pdf_extractor` 模块;`vlm_client` 拆为 3 个调用点 + prompts.py;行预算调整;产物目录新增 `runs/M0.5/` 和 `runs/audit/` |
| §6 | 集成测试更新为 mock 3 个 VLM 调用点;fixture 准备 3 个 mock JSON |
| §7 | 显式不做新增 3 条:不黑盒决策、不无限重试、不跨模块共享状态 |
| §8 | 阶段 A 新增 M0.5,顺序重排;阶段 B 不变 |
| §9 | 验收清单新增 M0.5/M1a/M1c/M3.5,更新 M1b 判据(嵌入图输入) |
| §10 | **新增**:VLM 受控智能参数表(重试/阈值/预算/版本控制/降级路径) |

---

## 13. v0.5 改动摘要

| 节 | 改动 |
|---|------|
| §3 | 移除 M1c（review_panel_detection）和 M3.5（review_segmentation）;新增 M1（classify_figure）作为首道语义过滤;流水线简化为 M0.5 → M1 → M1a → M1b → M3 → M4 |
| §4 | 新增 `figure_classification` schema;移除 `panel_review` 和 `segmentation_review` schema |
| §5 | vlm_client 调用点从 3 个减为 2 个;产物目录移除 `runs/M1c/` 和 `runs/M3.5/`;集成测试 mock 从 3 个点减为 2 个 |
| §8 | 阶段 A 移除 M1c 和 M3.5 步骤,新增 M1 步骤 |
| §9 | 验收清单新增 M1（classify_figure）,移除 M1c 和 M3.5 |
| §10 | 单页预算说明从"3 个正常 review 点"改为"2 个正常 review 点" |

**v0.5 已签字（2026-05-22）。**

---

## 14. v0.6 改动摘要

| 节 | 改动 |
|---|------|
| §3.5 | **新增**:双管线架构（Pipeline A 纯人工 + Pipeline B 纯Agent），统一接口 `pipeline_interfaces.py` |
| §4 | 新增 `PanelInput` / `SegmentationResult` / `QualityReview` schema 契约 |
| §5.1 | 新增 `pipeline_interfaces.py` 模块;标注各模块实现的 Protocol(`PanelDetector`/`QualityReviewer`/`Segmenter`) |
| §5.2 | `controller.py` 职责更新:根据 `QualityReview.suggested_action` 决定后续动作 |

**v0.6 已签字（2026-05-22）。**

---

## 15. v0.7 改动摘要

| 节 | 改动 |
|---|------|
| §2 | 新增原则 9：前端技术栈选型决策记录（Tauri + TS + Svelte vs PySide6 废弃理由） |
| §3 | 流水线图更新：前端 → FastAPI → Python 后端；移除 PySide6 直接调用 |
| §3.5 | **大幅扩展**：Pipeline A 补充 Figure Review + Panel 获取环节（手动框选 / CV 辅助 / 整图 fallback），与 Agent 流环节一一对应（7 个环节各有两种实现，可任意切换）；三种分割交互工具；Pipeline B Review 弹窗；Agent → Manual 无缝衔接 |
| §4 | 新增 HTTP API schema 契约（TS 前端 ↔ FastAPI 后端）；保留原有 JSON schema 作为后端内部契约 |
| §5 | 废弃 PySide6 GUI 模块（`geoseg/gui/` 全部）；新增 `geoseg/server.py`（FastAPI）和 `geoseg-gui/`（Tauri 前端）模块表 |
| §7 | 显式不做新增：不再维护 PySide6 GUI、不做前端直接调 Python 函数 |
| §8 | 实现顺序重排：Week 1 FastAPI 骨架 → Week 2 Tauri 骨架 → Week 3 多边形工具 → Week 4 矩形/画笔 → Week 5 Agent 流 → Week 6 衔接+导出 |
| §9 | 验收清单更新：M5 改为 Tauri 画布加载测试；新增 M7 FastAPI endpoint 测试 |

**v0.7 已签字（2026-05-23）。**

---

**v0.4 已签字（2026-05-19）。**
