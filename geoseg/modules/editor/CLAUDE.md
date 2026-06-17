# editor — napari-based segmentation editor (Shapes-primary)

> **核心原则**：Shapes 层是唯一交互层。Label 判定由拓扑自动完成——单连通区域即为一个 label。
> **不做**：不暴露 label ID 给用户；不手动分配/选择 label；不直接操作像素。

## 架构：Shapes → Topology → Labels

```
用户操作 Shapes Layer（画线、拖拽顶点、删除线）
    ↓
后端: 将所有 Shapes rasterize 为边界 mask
    ↓
对非边界区域做连通域分析 (ndimage.label)
    ↓
每个连通域 = 一个 label → 更新 Labels Layer（只读展示）
    ↓
用户看到彩色填充区域
```

| Layer | 作用 | 用户操作 |
|-------|------|----------|
| **Shapes** | **唯一交互层** — 边界线集合 | 画线、画多边形、拖拽顶点、删除线 |
| **Labels** | **只读展示层** — 拓扑计算出的填充区域 | 不可操作，由 Shapes 变更驱动刷新 |
| **Image** | 原始图参考 | 可开关、调透明度 |

**关键设计**：
- 整个画布被边界线分割为若干单连通区域，没有"空区域"
- Label 0 = 边界线（separator），不是背景
- 用户只感知"区域"和"边界线"，不感知 label ID
- Label ID 是临时分配的，仅用于可视化配色，关闭后无需保留

## 边界线模型

所有 Shapes 都是**边界线**，分为两类：

| Shape 类型 | 语义 | 效果 |
|---|---|---|
| **闭合多边形** | 外边界 | 圈出一块区域，与外部隔离 |
| **开放线** | 内部边界/分割线 | 穿过一个区域，将其一分为二 |
| **开放线（两端抵达其他边界）** | 内部分割线 | 连接两条现有边界，分割中间区域 |

**操作即拓扑变更**：
- **添加边界线** → 穿过区域的线将该区域一分为二；闭合多边形在内部创建独立区域
- **删除边界线** → 被该线隔开的两侧区域合并为一个
- **移动顶点** → 边界形状变化 → 相邻区域面积/形状变化
- **画线不闭合** → 线所在区域被分割（线两端自动延伸到最近边界）

## 核心算法：shapes_to_labels

```python
def shapes_to_labels(shapes_layer, image_shape) -> np.ndarray:
    """将 Shapes layer 中的所有线/多边形 rasterize 为边界 mask，
    对非边界区域做连通域分析，返回 label 数组。

    Args:
        shapes_layer: napari Shapes layer
        image_shape: (H, W)

    Returns:
        labels: (H, W) int32，label 0 = 边界线，1..N = 各区域
    """
    # 1. 初始化边界 mask
    boundary = np.zeros(image_shape, dtype=bool)

    # 2. Rasterize 所有 shapes
    for shape_data, shape_type in zip(shapes_layer.data, shapes_layer.shape_type):
        if shape_type == 'polygon':
            # 闭合多边形：画边界线（不填充内部）
            rr, cc = polygon_perimeter(shape_data[:, 0], shape_data[:, 1])
        else:  # line / path
            # 开放线：直接画线
            rr, cc = line(shape_data[:, 0], shape_data[:, 1])
        valid = (rr >= 0) & (rr < image_shape[0]) & (cc >= 0) & (cc < image_shape[1])
        boundary[rr[valid], cc[valid]] = True

    # 3. 对非边界区域做连通域分析
    # 背景 = 边界(boundary=True) + 图像外框
    background = boundary.copy()
    background[0, :] = background[-1, :] = True  # 上下边框
    background[:, 0] = background[:, -1] = True  # 左右边框

    # 反转：非背景区域 = 可填充区域
    fillable = ~background
    labels, n = ndimage.label(fillable)

    # 4. 边界像素设为 0
    result = labels.astype(np.int32)
    result[boundary] = 0

    return result
```

**关键性质**：
- 边界线上的像素值为 0（黑色/透明）
- 每个单连通区域获得唯一的 label ID（临时）
- Label ID 每次重新计算，不持久化——关闭编辑器后只保留 Shapes 数据
- 重新打开时：Shapes → shapes_to_labels → 重新分配 label ID（颜色可能变化，但拓扑一致）

## 交互设计

### Layer 配置

```python
# 1. Image Layer — 原始图（参考层）
viewer.add_image(image, name='reference', opacity=0.3, visible=True)

# 2. Labels Layer — 填充区域（只读展示）
# 由 shapes_to_labels() 计算，用户不可直接操作
viewer.add_labels(np.zeros_like(image, dtype=np.int32), name='regions')

# 3. Shapes Layer — 边界线（唯一交互层）
viewer.add_shapes([], name='boundaries', shape_type='line')
```

### 原生工具映射

| napari 原生工具 | 用户动作 | 拓扑效果 |
|---|---|---|
| **Add Line** (快捷键 `L`) | 画 2-vertex 开放线穿过区域 | 线穿过区域 → 该区域一分为二；两端自动吸附到边界 |
| **Add Path** (工具栏折线图标) | 画多顶点开放折线穿过区域 | 折线作为整体边界分割区域；**仅两端点自动吸附**，中间顶点保留 |
| **Add Polygon** (快捷键 `P`) | 画闭合多边形 | 多边形内部成为独立区域（与外部隔离）；不吸附 |
| **Select + Delete** (快捷键 `Delete`) | 选中一条边界线 → 删除 | 被该线分隔的区域合并 |
| **Direct** (快捷键 `D`) | 拖拽顶点 | 边界变形 → 相邻区域 reshape |
| **Pan/Zoom** | 平移/缩放 | — |
| **Ctrl+Z** | Undo | Shapes 的 undo 自动触发 labels 重算 |

**不需要自定义快捷键**。所有操作通过 napari Shapes layer 的原生工具完成。

### 画开放线（分割区域）

#### Add Line — 2-vertex 线段

```
用户按 'L' 或点击工具栏 Add Line
  ↓
点击 p1（在区域内或边界附近，无需精确）
  ↓
拖动到 p2（在区域内或边界附近，无需精确）
  ↓
释放鼠标 → 线被添加到 Shapes layer
  ↓
后端自动执行边界吸附：
  1. 取线中点 → 确定目标区域（该点当前所在 label）
  2. 调用 snap_line_endpoints(mask=目标区域, p1, p2)
     → 两端点分别吸附到该区域边界
  3. 用吸附后的线替换 Shapes layer 中的原始线
  ↓
触发 shapes_to_labels()：
  - 精确分割线将区域一分为二
  - Labels layer 自动刷新（显示两个新区域）
  ↓
线保留在 Shapes layer（可后续拖拽顶点微调）
```

#### Add Path — 多顶点折线

```
用户点击工具栏 Add Path（折线图标）
  ↓
逐点击添加顶点（可添加任意数量）
  ↓
双击闭合 或 按 Enter 结束
  ↓
折线被添加到 Shapes layer
  ↓
后端自动执行端点吸附：
  1. 取首段中点 → 确定起点目标区域
  2. 取末段中点 → 确定终点目标区域（可与起点不同）
  3. 调用 snap_path_endpoints → 仅两端点吸附到各自区域边界
     → 中间顶点完全保留
  4. 用吸附后的折线替换原始折线
  ↓
触发 shapes_to_labels()：
  - 折线整体作为边界分割区域
  - Labels layer 自动刷新
```

**边界吸附算法**（`snap_line_endpoints` / `snap_path_endpoints`）：
1. **阈值圆判定**：以端点为圆心、`threshold=25px` 为半径，沿切方向双向采样
2. 若采样路径上出现 `inside→outside` 或 `outside→inside` 状态转换 → 边界在附近
3. 二分法精化：在转换点附近亚像素插值，找到精确边界穿越点
4. 若最近边界距离 ≤ threshold → 端点吸附到边界；否则保持原端点

**统一处理两种场景**：
- **伸长**：端点在区域内，边界在"向外"方向 → 线延长至边界
- **裁剪**：端点在区域外，边界在"向内"方向 → 线裁剪至边界

**失败处理**：
- 端点附近无边界（距离 > threshold）→ 保持原端点
- 吸附后线段太短（< 5px）→ 拒绝吸附 → 保留原始线
- 吸附后的线与现有边界重合 → 不重复添加（去重）
- 首段/末段中点在 label 0（边界）上 → 跳过该端点吸附

### 画闭合多边形（新建独立区域）

```
用户按 'P' 或点击工具栏 Add Polygon
  ↓
逐点击顶点
  ↓
双击闭合 或 按 Enter
  ↓
多边形被添加到 Shapes layer
  ↓
自动触发 shapes_to_labels()：
  - 多边形边界圈出的区域成为独立 label
  - 多边形内部与外部被隔离
  - Labels layer 自动刷新
```

### 删除边界线（合并区域）

```
用户按 'S' 或点击工具栏 Select
  ↓
点击一条边界线 → 线高亮选中
  ↓
按 Delete 键
  ↓
线从 Shapes layer 移除
  ↓
自动触发 shapes_to_labels()：
  - 被该线隔开的区域连通 → 合并为一个区域
  - Labels layer 自动刷新
```

### 拖拽顶点（微调边界）

```
用户按 'D' 或点击工具栏 Direct
  ↓
点击边界线上的一个顶点 → 顶点高亮
  ↓
拖拽到新位置 → 边界形状实时变化
  ↓
释放鼠标 → 触发 shapes_to_labels() → Labels 刷新
```

## 属性管理

每个区域（连通域）可以关联物理属性。由于 label ID 是临时的，属性绑定需要稳定的标识。

**方案**：属性绑定到区域的"质心坐标 + 面积"指纹，或让用户手动标记区域名称。

```python
class RegionProperties:
    """区域属性，独立于临时 label ID。"""

    def __init__(self):
        # {region_fingerprint: {"name": str, "Vp": float, "Vs": float, "rho": float}}
        self._props: dict[str, dict] = {}

    def _fingerprint(self, labels: np.ndarray, label_id: int) -> str:
        """基于区域几何特征生成稳定指纹。"""
        mask = labels == label_id
        cy, cx = ndimage.center_of_mass(mask)
        area = int(mask.sum())
        return f"{cy:.1f},{cx:.1f},{area}"

    def get(self, labels: np.ndarray, label_id: int) -> dict | None:
        fp = self._fingerprint(labels, label_id)
        return self._props.get(fp)

    def set(self, labels: np.ndarray, label_id: int, props: dict) -> None:
        fp = self._fingerprint(labels, label_id)
        self._props[fp] = props
```

**交互**：
- 鼠标悬停/点击 Labels 上的区域 → 右侧面板显示属性
- 属性可编辑 → 保存到 region_fingerprint
- 拓扑变更（split/merge）后，属性通过 fingerprint 自动迁移

## 数据流

```
geoseg pipeline 自动分割
    ↓
输出: labels.npz + overlay.png + 原始图.png
    ↓
CLI: python -m geoseg.modules.editor.napari_app --labels path.npz --image path.png
    ↓
初始化:
  - Image layer: 原始图（参考层，30% 透明度）
  - Labels layer: labels.npz（初始显示）
  - Shapes layer: 从 labels 提取边界线（通过 find_contours / skeletonize）
    ↓
用户操作 Shapes layer（原生工具：画线、多边形、删线、拖拽顶点）
    ↓
每次 shapes 变更 → shapes_to_labels() → 更新 Labels layer
    ↓
保存: Ctrl+S
  - 导出 Shapes 数据（JSON/GeoJSON，边界线集合）
  - 导出 Labels（可选，通过 shapes_to_labels 重新计算）
  - 导出 Properties（region_fingerprint → 属性映射）
    ↓
geoseg exporter: labels_to_grids() → SPECFEM
```

**持久化格式**：
```json
{
  "image_shape": [400, 600],
  "shapes": [
    {"type": "polygon", "vertices": [[y1,x1], [y2,x2], ...]},
    {"type": "line", "vertices": [[y1,x1], [y2,x2], ...]}
  ],
  "properties": {
    "fingerprint1": {"name": "sedimentary_1", "Vp": 3000, "Vs": 1732, "rho": 2200},
    "fingerprint2": {"name": "basement", "Vp": 6000, "Vs": 3464, "rho": 2700}
  }
}
```

## 从 labels 初始化 Shapes

首次加载 geoseg 自动分割结果时，需要从 labels 数组反向生成初始 Shapes：

```python
def labels_to_shapes(labels: np.ndarray) -> list[dict]:
    """从 labels 数组提取边界线，生成 Shapes layer 数据。"""
    # 1. 找到所有 label 之间的边界
    # 方法 A: 对每个 label，find_contours 提取外轮廓
    # 方法 B: 对 labels 做梯度检测，边界像素构成骨架，再提取线段

    # 推荐方法 A（简单可靠）:
    shapes = []
    for label_id in range(1, labels.max() + 1):
        mask = labels == label_id
        if not mask.any():
            continue
        # 提取轮廓（napari Shapes 期望 [y, x]）
        from skimage.measure import find_contours
        contours = find_contours(mask, level=0.5)
        for contour in contours:
            # contour 是 (N, 2) array of [row, col] = [y, x]
            shapes.append({"type": "polygon", "vertices": contour.tolist()})
    return shapes
```

**注意**：
- `find_contours` 返回的是 `[y, x]`（即 `[row, col]`），与 napari Shapes 的坐标系一致
- 每个区域可能有多条轮廓（有孔洞时），都作为独立 polygon 添加
- 相邻区域共享的边界会出现两次（各区域一条），这是正常的——删除时需要同时删除两条才会真正合并。可在初始化时做去重优化。

## Edge Cases

| 场景 | 检测 | 处理 |
|---|---|---|
| **线完全在区域外** | shapes_to_labels 后该区域未分割 | 线成为孤立边界，不影响拓扑（视觉上可接受） |
| **线太短未穿过区域** | 连通域数量未增加 | 线仍被保留，但无分割效果——用户可拖拽延长 |
| **多边形自相交** | skimage polygon 处理 | 自相交多边形的内部定义模糊，暂不处理，依赖用户正确绘制 |
| **删除线后区域不合并** | 仍有其他边界阻隔 | 正确行为——需删除所有共享边界才会合并 |
| **区域面积过小** | 连通域面积 < min_area | 过滤：小区域合并到最大相邻区域（可选后处理） |
| **Undo/Redo** | napari Shapes 原生支持 | shapes_to_labels 在 events.data 回调中触发，undo 自动重算 |

## 测试策略

### 单元测试（test_editor_core.py）

| 测试 | 覆盖 |
|---|---|
| `test_shapes_to_labels_empty` | 无 shapes → 全图一个区域 |
| `test_shapes_to_labels_single_line` | 单条线分割 → 两个区域 |
| `test_shapes_to_labels_cross_lines` | 十字交叉线 → 四个区域 |
| `test_shapes_to_labels_polygon` | 闭合多边形 → 内外分离 |
| `test_shapes_to_labels_nested_polygons` | 嵌套多边形 → 正确层级 |
| `test_labels_to_shapes_roundtrip` | labels → shapes → labels 拓扑一致 |
| `test_region_properties_fingerprint` | 属性绑定稳定 |

### 集成测试（test_napari_app.py）

- 加载真实 labels → 初始化 Shapes → 验证拓扑一致
- 在 Shapes 画线 → 验证 Labels 正确分割
- 删除 Shapes 线 → 验证 Labels 正确合并
- 拖拽顶点 → 验证 Labels 正确更新
- save/load Shapes JSON → 验证持久化

## 依赖

```
napari[all]>=0.5
scikit-image>=0.22
scipy>=1.11
numpy>=1.26
```

## 启动方式

```bash
# 从文件直接启动
python -m geoseg.modules.editor.napari_app --labels path.npz --image path.png

# 从 session_state 启动（加载 labels + properties）
python -m geoseg.modules.editor.napari_app --session runs/session_001.json --figure fig1
```

## 文件组织

```
geoseg/modules/editor/
├── CLAUDE.md              # 本文件
├── __init__.py
├── editor_core.py         # shapes_to_labels, labels_to_shapes, RegionProperties,
│                          #   snap_line_endpoints, snap_path_endpoints
├── napari_app.py          # napari 三层架构、events 绑定
├── test_editor_core.py    # 单元测试
└── test_napari_app.py     # 集成测试
```

## 工作量估算

| 模块 | 代码量 | 工作量 |
|---|---|---|
| shapes_to_labels | ~50 行 | 1h |
| labels_to_shapes | ~30 行 | 1h |
| RegionProperties | ~50 行 | 1h |
| napari_app.py（Shapes-primary） | ~150 行 | 3h |
| 单元测试 | ~150 行 | 2h |
| 集成测试 + 调试 | — | 2h |
| **总计** | **~430 行** | **~10h（1.5 天）** |
