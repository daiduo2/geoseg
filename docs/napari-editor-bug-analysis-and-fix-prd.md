# Napari 编辑器 Bug 分析与修复 PRD

> 范围：`geoseg/modules/editor/napari_app.py`、`geoseg/modules/editor/editor_core.py`
> 状态：仅分析，未改动代码
> 目标：修复当前 Shapes-primary napari 编辑器中的显式 bug，使其满足 CLI HITL 交互模型要求

---

## 1. 执行摘要

当前 napari 编辑器已搭建起三层架构（Image / Labels / Shapes），并实现了 `shapes_to_labels`、`labels_to_shapes`、`snap_line_endpoints` 等核心算法。单元测试（`test_editor_core.py`）全部通过，说明算法层基本正确。

**但应用层（`napari_app.py`）存在多处明显 bug，主要集中在事件绑定与边界吸附逻辑：**

1. `events.data` 回调在 shape 绘制过程中就触发，导致 incomplete shape 被重算 labels。
2. 每次数据变化都对**所有** line/path 重新吸附，会覆盖用户手动微调。
3. 使用 napari 私有 API `_data_view.edit` 修改内部状态，存在数据同步与版本兼容风险。
4. Labels 层未正确锁定，用户可能直接涂改像素。
5. 空 shapes 时 labels 被错误置为全 0。
6. `save_labels` 的 `fill_boundary_gaps` 会误填真实背景区域。
7. 相邻区域共享边界被拆成两条重叠 polygon，删除一条不会合并。
8. 通过 `--labels` 重新编辑时丢失 `RegionProperties`。

本 PRD 按 **Critical / High / Medium** 分级给出根因、修复方案、测试计划与验收标准。

---

## 2. Bug 分级清单

### 2.1 Critical

#### Bug-1：绘制过程中就重算 labels

- **位置**：`napari_app.py:75-90`
- **现象**：
  - 用户画 path/polygon 时，每点击一个顶点，labels 就刷新一次。
  - 多边形未闭合时会被当成开放线处理，区域被错误分割。
  - 拖拽顶点时 labels 持续闪烁/重算。
- **根因**：`_on_shapes_changed` 直接连接 `self.shapes_layer.events.data`，而该事件在顶点添加、移动、删除时都会同步触发，不区分"绘制中"与"已完成"。
- **修复方案**：
  - 监听 shape 完成事件：优先使用 napari Shapes layer 的 `mode` 变化或鼠标释放事件来判断绘制完成。
  - 最小改动：在 `_on_shapes_changed` 内检查 `self.shapes_layer.mode` 是否为 `add_*` 模式；若处于 add 模式，跳过 recompute。
  - 更稳健：改为在 `mouse_release` 或 shape 的 `finished` 回调中触发一次性的 recompute。
  - 推荐：引入 `_pending_recompute` 标志 + `QTimer`（若可用）做 50-100ms debounce，仅在用户停止操作后重算。
- **验收标准**：
  - 绘制多边形过程中 labels 不更新；双击/回车闭合后才更新。
  - 拖拽顶点时 labels 在释放鼠标后才更新，不闪烁。

#### Bug-2：对所有 line/path 反复重新吸附

- **位置**：`napari_app.py:92-205`
- **现象**：
  - 用户用 Direct 工具把已吸附的端点拉开，下一次事件又把它吸回边界。
  - 删除一条边界后，剩余边界被重新评估并可能发生偏移。
- **根因**：`_snap_new_lines` 没有识别"本次新增/修改的 shape"，而是遍历全部 shape 做全量吸附。
- **修复方案**：
  - 新增状态标记：记录每个 shape 的"已吸附"状态（或基于 shape 内容哈希判断是否需要再吸附）。
  - 只在 shape 从"未完成"变为"完成"时执行一次吸附；之后 Direct 模式下的修改不再自动吸附。
  - 提供显式"吸附"入口：例如用户按 `A` 键时对选中 shape 执行吸附，而不是自动吸附。
- **验收标准**：
  - 用户手动拖拽端点后，端点位置保持不动，不会被自动拉回。
  - 新画的 line/path 在完成后自动吸附一次；已吸附的 shape 不再被重复处理。

#### Bug-3：使用 napari 私有 API `_data_view.edit`

- **位置**：`napari_app.py:191-202`
- **现象**：
  - 代码里已经写了 `try / except AttributeError` fallback，说明接口不稳定。
  - 更新私有状态后，公共 `data` 与内部 `_data_view` 的同步依赖 napari 内部实现。
- **根因**：绕过 napari 公共 API 直接修改 `_data_view`。
- **修复方案**：
  - **首选**：使用 `self.shapes_layer.data[i] = new_vertices` 或 `self.shapes_layer.data = new_data` 公共 setter。虽然 setter 会重建 `_data_view`，但可以在 shape 完成后再做，避免绘制中崩溃。
  - **次选**：若公共 setter 确实导致 mid-creation 崩溃，则把吸附逻辑从 `events.data` 回调中移出，改在 shape 完成后执行，此时重建 `_data_view` 是安全的。
  - 删除 `_data_view.edit` 调用及相关 fallback。
- **验收标准**：
  - 不再引用任何 napari 私有 API（`_data_view`、`_moving_value` 等）。
  - 吸附功能在 napari 0.7+ 公共 API 下稳定工作。

---

### 2.2 High

#### Bug-4：Labels 层未正确锁定

- **位置**：`napari_app.py:49-52`
- **现象**：用户可能切换到 Labels layer 的画笔/填充工具，直接涂改像素，破坏 Shapes-primary 设计。
- **根因**：仅设置 `self.labels_layer.editable = False`，未设置 `mode = "pan_zoom"`。
- **修复方案**：
  - 初始化时设置 `self.labels_layer.mode = "pan_zoom"`。
  - 同时保留 `editable = False` 作为双重保护。
  - 在 `_on_shapes_changed` 中检测到 labels 被直接修改时，可重置为 computed labels 或记录 warning。
- **验收标准**：
  - Labels layer 无法进入 paint/fill/erase 模式。
  - 用户只能操作 Shapes layer。

#### Bug-5：空 shapes 时 labels 被错误置为全 0

- **位置**：`napari_app.py:206-210`
- **现象**：用户删除所有边界后，labels layer 显示全黑（全 0），而不是一个完整区域。
- **根因**：`_recompute_labels` 对空 shapes 返回 `np.zeros(...)`，与 `shapes_to_labels([], [], shape)` 的语义（全图为一个区域）不一致。
- **修复方案**：
  - 空 shapes 时调用 `shapes_to_labels([], [], self._image_shape)`，返回单区域 labels。
- **验收标准**：
  - 删除所有 shapes 后，labels layer 显示一个彩色区域（label 1）。
  - `test_empty_shapes` 的语义在应用层也得到保持。

#### Bug-6：`fill_boundary_gaps` 误填真实背景

- **位置**：`napari_app.py:243-255`、`editor_core.py:663-696`
- **现象**：
  - 用户仅打开 napari 查看、未编辑，保存后的 `labels_edited.npz` 把原图中的背景区域（label 0）合并到相邻区域。
  - 导出 SPECFEM 时背景被错误赋予物理属性。
- **根因**：`fill_boundary_gaps` 对除最外圈 1px 外的所有 label 0 做最近邻填充，无法区分"薄边界"和"真实背景"。
- **修复方案**：
  - **方案 A（推荐）**：只对 editor 自己产生的 label 0 做填充。Editor 产生的 label 0 是 shapes rasterize 出的 1px 宽边界，可通过形态学操作（如先膨胀再腐蚀）识别出连通的小面积 0 区域。
  - **方案 B**：保留 label 0 作为 background，导出时由 `controller.run_post_process_and_export` 自行处理背景。即 `save_labels` 不再调用 `fill_boundary_gaps`，保持 label 0 语义。
  - 建议采用 **方案 B**，因为 pipeline 其他部分（controller、post_process）已经把 0 当作 background/boundary 处理，editor 不应擅自改变语义。
- **验收标准**：
  - 保存后的 labels 中，原 segmentation 的 label 0 背景区域仍然为 0。
  - 薄边界（由 shapes 产生）可视情况保留为 0 或填充；不得把大面积背景误填。

---

### 2.3 Medium

#### Bug-7：相邻区域共享边界被拆成两条重叠 polygon

- **位置**：`editor_core.py:106-139`
- **现象**：删除两个区域之间的共享边界时，必须同时删除两条几乎重合的 polygon 才能真正合并。
- **根因**：`labels_to_shapes` 对每个 label 独立 `find_contours`，共享边界自然出现两次。
- **修复方案**：
  - 初始化时提取所有 label 间边界，并做去重/合并，生成唯一的共享边界线集合。
  - 简单实现：对 `labels != 0` 做边缘检测 + 细化（skeletonize），再提取线段/多边形。
  - 注意：该方法可能把闭合外边界也变成线，需要区分 interior 边界与 image frame 边界。
- **验收标准**：
  - 相邻区域之间只显示一条可删除的边界。
  - 删除该边界后，两个区域立即合并为一个。

#### Bug-8：通过 `--labels` 重新编辑时丢失 `RegionProperties`

- **位置**：`napari_app.py:330-344`
- **现象**：按 skill 文档重新编辑时只带 `--labels labels_edited.npz --image ...`，properties 不会被恢复。
- **根因**：CLI 的 `--labels` 分支没有尝试加载同目录下的 `shapes.json` 或 `properties.json`。
- **修复方案**：
  - 在 `--labels` 分支中，若 `--properties` 未提供，自动查找与 labels 文件同目录的 `shapes.json`，读取其中的 `properties` 字段。
  - 或者新增 `--shapes` 参数，明确传入 shapes.json。
- **验收标准**：
  - 通过 `--labels` 重开编辑器时，若能找到对应 shapes.json，则恢复 `RegionProperties`。

---

## 3. 修复实施计划

### Phase 1：事件层重构（最高优先级）

1. 重构 `_on_shapes_changed`：
   - 移除直接绑定到 `events.data` 的全量处理。
   - 新增 `_schedule_recompute()`，使用 debounce（50-100ms）或 shape 完成事件触发。
   - 在 add 模式下跳过 recompute。
2. 重构 `_snap_new_lines`：
   - 只处理新增/未吸附的 shape。
   - 用公共 `data` setter 替换 `_data_view.edit`。
3. 修复 Labels layer 锁定：
   - 设置 `mode = "pan_zoom"`。

### Phase 2：语义修正

1. 修复空 shapes 处理。
2. 移除/修正 `save_labels` 中的 `fill_boundary_gaps`，或改为可选参数。
3. 优化 `labels_to_shapes` 的共享边界去重（可选，视人力）。

### Phase 3：CLI 与持久化

1. `--labels` 分支自动加载同目录 `shapes.json` 的 properties。
2. 增加 `--shapes` 参数支持。

### Phase 4：测试

1. 新增 `test_napari_app.py` 集成测试：
   - 启动 editor、模拟添加 line、删除 shape、拖拽顶点后 labels 正确。
   - 空 shapes → 单区域。
   - 保存后 label 0 背景保留。
2. 更新 `test_editor_core.py`：
   - 补充共享边界去重测试（若实现）。
   - 补充 `fill_boundary_gaps` 对大面积背景的保留测试。

---

## 4. 测试策略

| 测试类型 | 覆盖点 | 目标 |
|---------|--------|------|
| 单元测试 | `shapes_to_labels`、`labels_to_shapes`、`snap_*`、`fill_boundary_gaps` | 保持现有 38 项通过，新增 4-6 项 |
| 集成测试 | `GeoSegEditor` 的事件绑定、吸附、保存 | 新增 `test_napari_app.py`，覆盖关键交互 |
| 手工测试 | 启动 napari、画线、画多边形、删线、拖拽、保存、重开 | 验证真实交互 |
| 回归测试 | `tests/test_integration_ph01.py` | schema/接口不变时必须通过 |

---

## 5. 验收标准

- [ ] 绘制 path/polygon 过程中 labels 不闪烁，完成后才更新。
- [ ] 拖拽顶点释放后 labels 更新一次，过程中不更新。
- [ ] 新 line/path 完成后自动吸附一次；已吸附 shape 手动微调后不再被自动拉回。
- [ ] 不再使用 napari 私有 API `_data_view.edit`。
- [ ] Labels layer 无法被用户直接涂改。
- [ ] 删除所有 shapes 后，labels 显示一个完整区域（非全 0）。
- [ ] 保存 `labels_edited.npz` 时，真实背景区域（label 0）不被误填。
- [ ] 相邻区域共享边界只显示一条可删除线（可选，建议实现）。
- [ ] 通过 `--labels` 重开编辑器时自动恢复 properties（若 shapes.json 存在）。
- [ ] `test_editor_core.py` 与新增集成测试全部通过。

---

## 6. 风险与回退

| 风险 | 影响 | 回退方案 |
|------|------|---------|
| 公共 API `shapes_layer.data` setter 在事件回调中仍不稳定 | 吸附失败或崩溃 | 把吸附逻辑移到 shape 完成后的独立函数，避免回调中重建 data |
| 移除 `fill_boundary_gaps` 导致 exporter 需要处理 label 0 | post_process 出错 | 在 exporter 增加显式 background 处理，保持 0 = background 语义 |
| 共享边界去重算法改变初始 shapes 数量 | 现有测试或用户习惯受影响 | 作为可选功能，默认保持原 behavior，通过参数开启 |
| napari 版本差异 | 事件名称或 API 不同 | CI 固定 napari 版本，并在 PRD 中记录 target version |

---

## 7. 附录：关键代码引用

- `napari_app.py:49-52`：Labels layer 初始化与 `editable` 设置
- `napari_app.py:75-90`：`events.data` 绑定与 `_on_shapes_changed`
- `napari_app.py:92-205`：`_snap_new_lines` 全量吸附逻辑
- `napari_app.py:191-202`：`_data_view.edit` 私有 API 调用
- `napari_app.py:206-210`：空 shapes 错误处理
- `napari_app.py:243-255`：`save_labels` + `fill_boundary_gaps`
- `napari_app.py:330-344`：`--labels` 分支 properties 加载
- `editor_core.py:106-139`：`labels_to_shapes` 共享边界重复提取
- `editor_core.py:663-696`：`fill_boundary_gaps` 实现
