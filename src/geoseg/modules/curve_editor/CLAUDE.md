# curve_editor — control-point interface editor

> **核心原则**：控制点模型是唯一事实来源；napari 只负责显示和手势。
> 编辑后由曲线栅格化为标签，0 永远表示背景/边界，不会把背景变成区域。
>
> **方向修正（2026-10-21）**：主路径是既有 labels/速度映射 → 边界曲线 → 合并/矫正/消除。
> `region.py` 的自由手绘贴边仅作手工辅助，不作为自动分区来源；T1/T2/T3 多尺度颜色分区已移除。

## 模块结构

| 文件 | 职责 |
|------|------|
| `model.py` | `InterfaceModel`、`Interface`、`ControlPoint`；稳定 ID、背景 mask、区域种子、`base_labels` 精确回放 |
| `spline.py` | splinebox 开放/闭合曲线、knots/control-points 两种模式、弧长重采样、曲率、freehand 拟合 |
| `rasterize.py` | 曲线 → 边界 mask → 4 连通填充；多 seed 组件用测地线分配；未播种组件合并到最近区域 |
| `extract.py` | 从已有 labels 提取边界骨架 → 曲线路径；支持开放层位与闭合透镜 |
| `region.py` | 自由手绘区域：梯度贴边、弧长重采样、曲率/梯度控制点、内部点 |
| `boundary_ops.py` | 超出图像边界的多边形裁剪、开放界面端点延伸/回缩、自交检测 |
| `commands.py` | 与 UI/语言无关的命令层：move/insert/delete/set_locked/add/remove |
| `io.py` | labels → model；session（npz）与 curves（json）读写 |
| `view.py` | napari 层：Image + 只读 Labels + Shapes(path) + Points；草稿层画新界面 |

## 不能破坏的约束

1. **未编辑时逐像素一致**：`model_from_labels(labels).to_labels()` 必须等于输入 labels；`base_labels` 只有第一次真实编辑才丢弃。
2. **控制点 ID 稳定**：插入/删除点不重排其他点；区域 ID 通过 seeds 保持稳定。
3. **背景显式**：`background_mask` 中的像素永远是 0；没有“0 兼作边界和背景”的歧义。
4. **只用 napari 公开 API**：不碰 `_qt`、`_data_view`；曲线显示用 Shapes `path`，不用已移除的 Paths 层。
5. **headless 优先**：所有编辑逻辑必须能在没有 napari 的测试中运行；`view.py` 才导入 napari。

## 从已有 labels 导入

`model_from_labels(labels)`：

1. 保存 `base_labels` 并建立每区域一个 seed（距离变换最深处）。
2. 对 label 差异 mask 做 skeleton，按度 != 2 的节点拆成开放路径/闭合环。
3. 用 RDP 简化到控制点（默认 ≤28 点/曲线），以 `B1` 基函数保存，尽量贴合原始边界。
4. 首次编辑后重新栅格化：边界加厚 1 px 闭合小缺口；一个连通域含多个 seed 时按**测地线距离**分配，避免边界缺口导致整个区域被一个 ID 吞掉。

## 手势

| 操作 | 效果 |
|------|------|
| 拖动黄色控制点 | 移动点，曲线局部变形 |
| `n` → 点击 → `Enter` | 新建开放界面；`Shift-Enter` 闭合 |
| `r` → 按住拖一个闭合圈 → 点击内部 | 自由手绘区域：贴到图像边界、按曲率/梯度取控制点、点击处作为区域 seed 填充 |
| `i` | 在选中点后插入控制点（当前鼠标位置） |
| `Delete` / `Backspace` | 删除选中控制点（保留最少 2/3 点） |
| `Esc` | 取消草稿/区域模式 |
| `Ctrl-S` | 保存到 `--output-labels/-shapes/-session` |

### 区域模式（advisor workflow）

`region.py` 实现：

1. 鼠标按住拖出粗略闭合笔迹（允许非凸、任意形状；自交会被拒绝并提示重画）。
2. 超出图像/panel 的部分先用 `clip_polygon_to_rect()` 裁到矩形，缺口沿图像边缘闭合，避免用任意弦切掉区域。
3. `edge_strength()` 计算 Lab/灰度梯度，`snap_closed_curve()` 让每个采样点沿法向在 `search_radius` 内找最强边界，并带距离惩罚，避免跳到远处的强边缘。
4. 边界按弧长重采样；`select_control_points()` 在曲率峰 + 图像梯度峰上取控制点（最小间距、最多 24 点；过于光滑时均匀补点）。
5. 用户点击内部，`InterfaceModel.add_region_interface()` 把点击点（或最近内部点）登记为新区域 seed，再交给栅格化填满内部。

注意：区域必须是简单闭合曲线；自交笔迹会拒绝，不做拓扑修复。

### 开放界面的边界处理

`boundary_ops.extend_open_curve()`：

- 取笔迹覆盖最多的区域作为主区域；
- 端点在同一区域内部 → 沿切向延伸到该区域边界；
- 端点越过边界进入邻区/背景 → 沿切向回缩到最近的本区域边界；
- 没有区域边界（或全背景）→ 延伸到 panel 边缘；
- 添加开放界面时 `add_split_interface()` 会在旧区域 seed 的另一侧补一个新区域 seed，保证切开后的两半不会因为 `merge_unseeded` 得到同一个 ID。

限制：跨多个区域的笔迹目前按主区域处理，不自动拆成多段；内部交界处可能仍有一条线穿过。

## 入口

```bash
make curve-edit LABELS=validation/curated/labels/newimage_p0.npz \
  IMAGE=validation/curated/images/newimage_p0.jpg EDIT_OUT=runs/curve_edit

# 或直接
uv run python -m geoseg.modules.curve_editor.view \
  --labels ... --image ... --output-labels ... --output-shapes ...
```

## 测试

```bash
uv run pytest tests/modules/curve_editor -q
QT_QPA_PLATFORM=offscreen uv run pytest tests/modules/curve_editor/test_curve_view.py -q -m gui
```

验收：合成三层/闭合透镜可重建；11 个 curated panel 无编辑逐像素回放；curated 重栅格化平均像素一致率约 0.90（复杂 ph01 / paper22 较低，见 `docs/design/curve_editor.md`）。
