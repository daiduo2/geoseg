# cv_detect — 模块契约

> **PanelDetector Protocol 实现者**。CV 检测 panel 边框，输出标准 `PanelInput` 列表。

## 实现 Protocol

| Protocol | 函数 | 说明 |
|----------|------|------|
| `PanelDetector` | `detect_panels(img_rgb)` | 连通域 + 布局聚类 → `list[PanelInput]` |

## 子模块

| 文件 | 职责 |
|------|------|
| `detect.py` | 主入口，调度各子模块 |
| `figure_classifier.py` | figure 分类器 |
| `panel_detector.py` | panel 检测器（默认版） |
| `panel_detector_e026.py` | panel 检测器（e026 实验版） |
| `colorbar_extractor.py` | colorbar 提取器 |
| `quality_filter.py` | 质量过滤器 |

## 与 vlm_client 的边界

| 职责 | 谁负责 |
|------|--------|
| 语义描述（`description` / `repair_hints`） | `vlm_client`（VLM） |
| 精确 bbox / 像素级 mask | **`cv_detect`** |
| 像素阈值 / 形态学参数 | `cv_detect`（不要让 VLM 给数字） |

VLM 给「左上角那块灰色的疑似断层」→ CV 给出 `[(x1,y1),(x2,y2)]`。

## 输出格式

```python
[
    {"id": 0, "bbox": (x, y, w, h), "source": "cv_detect", "confidence": 0.95},
    ...
]
```

空列表 → 上游 `make_whole_image_panel()` fallback。

## 不做

- 不依赖 VLM 给的任何坐标（VLM 协议禁止给坐标）
- 不在本模块调外部 LLM（唯一出口在 `vlm_client`）
- 不做分割（交给 `segment_engines`）

## 测试

```bash
python -m geoseg.modules.cv_detect.demo
```
