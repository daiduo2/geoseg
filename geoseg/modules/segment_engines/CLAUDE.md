# segment_engines — 模块契约

> **Segmenter Protocol 实现者**。路由 + 执行分割算法，输入 PanelInput，输出 SegmentationResult。

## 职责

- 提供多算法分割引擎族，供 agent / controller 按需调用
- 支持 `n_layers`、`reps`（种子点）、`colorbar_rgb` 等参数
- 返回标准 `SegmentationResult`（`labels` + `overlay` + `meta`）
- `metrics.py`：多引擎评估与对比
- `strategy_memory.py`：agent 策略学习（引擎选择历史与效果追踪）
- `_shared.py`：引擎间共享工具函数
- `batch_test.py`：批量测试 runner
- `compare_results.py`：多结果对比可视化

## 实现 Protocol

| Protocol | 函数 | 说明 |
|----------|------|------|
| `Segmenter` | `route_and_segment(img_rgb, **kwargs)` | 路由到可用引擎（agent 自主决策优先） |

## 引擎族

| 文件 | 引擎 |
|------|------|
| `v4_kmeans.py` | v4 K-Means |
| `edge_guided.py` | Edge-guided 分割 |
| `edge_grow.py` | Edge-grow 区域生长 |
| `e027_slic_graphcut.py` | SLIC + GraphCut（e027） |
| `kmeans_full.py` | K-Means 全图版 |
| `grayscale.py` | 灰度 agglomerative |
| `ensemble.py` | 多引擎融合 |
| `full_pipeline.py` | 完整流水线组合 |
| `vlm_reps.py` | VLM 种子点辅助 |

## 与 cv_detect 的边界

- `cv_detect` 输出 `PanelInput` 列表（bbox）
- `segment_engines` 接收 `PanelInput`，对 crop 后的图像执行分割
- 分割结果通过 `SegmentationResult` 统一接口返回

## 不做

- 不做 panel 检测（交给 `cv_detect`）
- 不做 VLM 语义分析（交给 `vlm_client`）
- 不直接操作 GUI（纯后端模块）

## 测试

```bash
python -m geoseg.modules.segment_engines.demo
```
