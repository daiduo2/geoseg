# 控制点界面编辑器（curve_editor）设计

> 2026-10-19。替代 `modules/editor` 的 Shapes-primary 编辑器，主链路仍保留旧编辑器作为兼容入口。
> 目标：把**既有速度分区/labels**的边界变成带稳定控制点的参数曲线，支持鼠标微调，并为后续边界合并、人工矫正和消除提供统一模型。
>
> 方向修正（2026-10-21）：边界必须是既有标签/速度映射中同值区域的闭包，不再从图像颜色重新构造分区；T1/T2/T3 的多尺度颜色分区方向已废弃。

## 1. 为什么重做

旧编辑器以 napari Shapes 为唯一状态：

- `0` 同时表示边界和背景，导致保存/重开时 label 数变化（paper22 8→9，newimage_p2 3→52，ph01 6→29）。
- 只有多边形顶点，没有“界面 / 控制点 / 闭合曲线”语义；曲线平滑、曲率、弧长都无从表达。
- 属性依赖 centroid/area 指纹，顶点移动后无法稳定对应。
- 保存只写 labels，不保存可编辑几何。

新模型把“界面曲线 + 控制点”作为事实来源，labels 是派生产物：

```
InterfaceModel
  ├── background_mask     显式背景，永远 label 0
  ├── base_labels         未编辑时逐像素返回，保证幂等
  ├── interfaces[]        开放/闭合参数曲线
  │     └── ControlPoint[] 稳定 ID、locked、kind
  └── regions{id: seed}   区域身份，编辑后不重编号
        ↓ splinebox
      dense curve
        ↓ rasterize
      boundary mask + connected components
        ↓
      labels (0 = background/boundary)
```

## 2. 关键语义

| 问题 | 旧实现 | 新实现 |
|------|--------|--------|
| 0 的含义 | 边界或背景 | 背景；边界也是 0，但背景 mask 独立保存 |
| 未编辑保存/重开 | 重新拓扑，可能变 label 数 | 直接返回 `base_labels`，逐像素一致 |
| 控制点 | 无 | `p_0001` 稳定 ID，插入/删除不重排 |
| 区域 ID | 每次重新分配 | seeds 锚定；编辑后保持 |
| 闭合区域 | polygon | `Interface.closed=True`，曲线闭合 |
| 平滑 | 折线 | splinebox B1/B3/CatmullRom/Exponential |
| 自由手绘 | 顶点即折线 | 弧长采样为 knots，拟合平滑曲线 |

## 3. 从已有 labels 提取曲线

`extract.py` 不提取“区域轮廓”，而是提取**label 之间的边界**：

1. `boundary_mask =` 4 邻域 label 不同的像素。
2. `skeletonize` 成 1 px 图。
3. 按节点（度 != 2）拆成开放路径；无节点的纯环拆成闭合路径。
4. 路径坐标转 `(x, y)`，RDP 简化（tolerance 1.5 px，最多 28 点）。
5. 以 `B1` knots 保存，使曲线贴近原始边界；用户后续可切换 B3 平滑。

提取结果用于建立可编辑模型；未编辑时仍返回原始 labels，因此提取误差不会污染“打开即保存”的场景。

### 3.1 两个严格分离的层

保留判定不能经过曲线简化或平滑，否则“边界是否该存在”会被几何拟合悄悄改变。当前分为：

**Layer 1 — exact closure（`exact_boundaries.py`）**

- 对每一对相邻 label，直接取分区结果中两者相接的原始像素；
- 按 8 连通拆成 boundary object；`length` 即像素数；
- 不做 skeleton、RDP、spline、曲率或任何平滑；
- `decide_exact_boundaries` 只依据 label pair、背景角色（exterior / internal hole）和原始长度做保守的 keep/review；当前不自动 remove；
- 视觉审计的 baseline 是 labels 重新上色的 mask；闭边界以粗线 + 深色 halo 绘制，不使用照片做 baseline。

**Layer 2 — geometry（`refine.py`）**

- 只接收 Layer 1 已判定 keep 的边界；
- 做平滑样条、端点固定、曲率限制；
- Layer 2 无权改变保留决定；失败只能回到 review。

`extract.py` 的 skeleton + RDP 仅用于交互编辑器构造可编辑曲线，不属于 Layer 1，也不能作为审计依据。

## 4. 编辑后的重栅格化

首次真实编辑会丢弃 `base_labels`，之后：

1. 每条 curve 采样成 polyline，画到 boundary mask（默认加粗 1 px，闭合小缺口）。
2. 非边界、非背景像素做 4 连通分析（边界 8 连通，填充 4 连通，防止对角漏填）。
3. 每个连通域用 region seeds 认领：
   - 一个 seed → 直接使用该区域 ID。
   - 多个 seed → 在连通域内按**测地线距离**分配给最近 seed（避免边界缺口让一个小区域吞掉整个连通域）。
   - 无 seed → 导入模型合并到最近 seed；手绘模型则创建新区域。
4. 边界与背景重置为 0。

`merge_unseeded=True` 是导入模型的默认值，防止骨架分叉/缝隙产生几十个新 label。

## 5. curated 语料验证

`validation/curated/manifest.jsonl` 的 11 个 panel，全部：

- `model_from_labels(labels).to_labels()` 与输入逐像素一致；
- 首次重栅格化不产生 base 之外的新区域 ID；
- 重栅格化像素一致率（当前实现，边界加粗 1 px + 测地线分配）：

| panel | accuracy | 备注 |
|-------|---------:|------|
| newimage_p0 | 0.915 | 分层 |
| newimage_p1 | 0.949 | 分层 |
| newimage_p2 | 0.919 | 分层 |
| newimage_p3 | 0.960 | 分层 |
| newimage_p4 | 0.910 | 分层 |
| paper22_fig5a | 0.818 | 无背景，断层/尖灭 |
| paper22_fig6 | 0.953 | 分层 |
| ph01_page8_panel3 | 0.697 | 文字遮挡多、细界面 |
| paper23_fig7a | 0.943 | 复杂分层 |
| schematic_panel2 | 0.845 | 3480×1740，3D 示意 |
| paper23_fault_guided | 0.968 | 断层 |
| **mean** | **~0.90** | |

低分案例的误差主要来自：边界在文字/遮挡处断开的缺口、极细区域、曲线简化后的端点偏移。后续方向：

- 曲线端点吸附到相邻边界（junction snapping）；
- 在缺口处用切向延长/桥接（已在 `horizon/occlusion.py` 有可用算法）；
- 从 labels 提取时保留每个 label 的多个连通域 seed；
- 用户编辑后只重栅格化局部窗口。

## 6. napari 界面

只用公开 API：

- `Image`：原图。
- `Labels`：只读，展示当前 labels。
- `Shapes(shape_type="path")`：只读，展示采样后的曲线（napari 0.7 已移除 Paths 层）。
- `Points`：黄色控制点，`select` 模式拖动；数据变化同步回模型。
- `Points`（草稿）：`add` 模式点选新界面顶点，Enter 提交。

手势：

- 拖动点：局部移动。
- `n` → 点击 → `Enter`：新建开放曲线；`Shift-Enter` 闭合。
- `r` → 按住拖一个闭合圈 → 点击内部：自由手绘区域模式（见下）。
- `i`：在选中点后插入点（鼠标位置）。
- `Delete`/`Backspace`：删点。
- `Ctrl-S`：按 CLI 参数保存 labels / curves / session。

## 6.1 区域模式：画圈 → 贴边 → 点内部

导师给的参考流程，作为第二种交互加入，不替代控制点画线：

1. 鼠标按住拖出粗略闭合笔迹（`viewer.mouse_drag_callbacks` 捕获），允许非凸/不规则形状。
2. `edge_strength()` 在 Lab 或灰度图上算平滑梯度；`snap_closed_curve()` 对等弧长采样的每个点，沿法向在 ±`search_radius` 内搜索“梯度 - 距离惩罚”最大的位置，再做环向平滑。笔迹只是初始化，最终边界来自图像。
3. 对 snapped 边界按**弧长**重采样；`select_control_points()` 在曲率峰和梯度峰合并的候选点中按最小间距选控制点（平滑边界则均匀补点），因此控制点优先落在转折/强边界处。
4. 用户点击内部；`add_region_interface()` 用点击点作为区域 seed（点在多边形外时自动取最近内部点），栅格化时该 seed 保证新区域 ID 不被 `merge_unseeded` 合并掉。

边界情况：

- 笔迹超出 panel：先 `clip_polygon_to_rect()` 裁到图像矩形，缺失部分沿图像边缘闭合，不用一条任意弦切掉区域。
- 自交笔迹：`self_intersects()` 检测后拒绝，提示重画一个简单闭合圈（当前不做拓扑拆分）。
- 开放界面端点：`extend_open_curve()` 以笔迹覆盖最多的区域为主区域；端点在区域内则沿切向延伸到边界，越过边界则回缩到最近的本区域边界，无区域时延伸到 panel 边缘。添加开放界面时 `add_split_interface()` 在旧 seed 另一侧补新 seed，保证切开的两半得到两个稳定区域 ID。
- 采样点过于靠近边界：`interior_point()` 会把 seed 移到最近的、离边界至少约 2 px 的内部点，避免 spline 把 seed 挤出区域。

合成验证（`tests/modules/curve_editor/test_curve_region.py` 和 `test_boundary_ops.py`）：矩形目标上粗略笔迹 IoU 0.81 → 贴边后 0.96；圆形 0.98；矩形四角都能选到控制点；跨出 panel 的笔迹裁剪后所有点都在图像内；band 内的开放线会延伸到上下边界并切成两个区域。

## 7. 文件格式

- `session.npz`：`model_json`（shape/interfaces/points/regions/merge flag）+ `background_mask` + 未编辑时的 `base_labels`。
- `curves.json`：便于 diff 和 agent 读取的曲线几何；不含 raster 数据。
- `labels.npz`：当前 `to_labels()` 结果。

## 8. 当前边界与下一步

已完成（headless + GUI）：

- 稳定 ID、背景语义、未编辑精确回放、局部编辑、闭合曲线；
- labels → skeleton → spline 导入；
- 自由手绘区域贴边（仅作手工辅助，不作为自动分区）；
- 11 panel 语料验收与 napari 冒烟测试。

尚未做：

- 自交笔迹的拓扑拆分/修复（目前检测后拒绝）；
- 跨多个区域的开放笔迹自动拆成多段（目前按主区域延伸/回缩，内部交界处可能仍有交叉）；
- 基于既有 labels 的连通域合并/拆分、边界消除和属性保持；
- 端点吸附到既有 label boundary/junction 与缺口桥接；
- 语言命令（“point 3 up 1 km”）解析到 `EditorCommands`；
- 曲线置信度/多段 fit 状态的人工确认；
- 局部增量栅格化（大图拖动性能）。

多尺度颜色分区与 merge tree（T1/T2/T3）已从主线移除；边界来源回到既有 labels/速度映射的等值闭包。
