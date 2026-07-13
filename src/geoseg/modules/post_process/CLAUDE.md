# post_process — 模块契约

> **M3.5→M4 桥梁**：从分割 labels 提取几何与物理属性，为 SPECFEM 导出做准备。

## 职责

- `polygon.py`：从 label map 提取连通域（`extract_components`）+ 轮廓多边形（`labels_to_polygons`）+ GeoJSON 保存
- `properties.py`：颜色名称 → 弹性物理属性映射（`assign_properties`）。默认 crustal-scale 模板（Vp/Vs/rho），支持自定义 JSON 覆盖

## 与上下游的边界

| 上游 | 下游 | 本模块 |
|------|------|--------|
| `segment_engines/` 输出 label array | `exporter/` 需要 grids + properties | 几何提取 + 属性分配 |

## 不做

- 不修改分割结果（不 smoothing、不手动修正）—— 修正交给 GUI 或重新 segmentation
- 不在本模块调外部 LLM（属性表由用户提供 JSON 或默认模板，不走 VLM）

## 测试

```bash
python -m geoseg.modules.post_process.demo
```
