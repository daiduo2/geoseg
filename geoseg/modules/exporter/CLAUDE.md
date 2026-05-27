# exporter — 模块契约

> **M4**：SPECFEM2D/3D 模型导出。

## 职责

- `labels_to_grids(labels, properties) → (vp_grid, vs_grid, rho_grid)`：将分割 label 映射 + 物理属性转换为 SPECFEM 速度/密度网格
- `write_tomography_file(path, vp, vs, rho)`：导出 `.xyz` 格式的层析成像文件
- `write_parfile_snippet(path, n_layers, ...)`：生成 `Par_file` 配置片段

## 输入

来自 `post_process/` 的：
- `labels`：(H, W) int array，背景为 0
- `properties`：`{color_name: {"Vp": float, "Vs": float, "rho": float}}`

## 不做

- 不解析 SPECFEM 源码或运行正演模拟（纯文件格式输出）
- 不在本模块做物理单位换算（输入即输出单位，m/s 或 km/s 由调用方统一）

## 测试

```bash
python -m geoseg.modules.exporter.demo
```
