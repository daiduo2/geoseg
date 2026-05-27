# PDF 矢量图形提取 — 技术规格（M0.5v）

> 状态：**待开发**（独立 session，不阻塞主流水线）
> 依赖：PyMuPDF ≥ 1.24
> 与主线关系：产出替换 `ph01.jpg` 作为 M1b/M3 的高分辨率输入，pipeline 内部接口不变

---

## 1. 背景与问题

当前 `M0.5 pdf_extractor` 只提取 PDF 中的 **XObject(Image)** 嵌入位图。在 `gxae11701.pdf` Page 7 上：

| 内容 | XObject? | 现状 |
|------|----------|------|
| 两个位图嵌入图（地质剖面） | ✅ 是 | 已提取，但分辨率低、无 panel |
| 概念模型图（4-panel velocity cross-section） | ❌ 否 | **矢量图形**，未出现在 XObject 列表 |

概念模型图是 geoseg 的核心处理目标，当前靠外部位图 `ph01.jpg`（1039×691）作为 fallback。

**目标**：直接从 PDF 提取该矢量图形，获得 ≥ 当前 fallback 的分辨率，使主流水线不再依赖外部文件。

---

## 2. 技术方案对比

| 方案 | 原理 | 优点 | 缺点 | 推荐度 |
|------|------|------|------|--------|
| **A. 高分辨率页面 rasterize** | `page.get_pixmap(dpi=600)`，裁剪 figure 区域 | 简单；捕获所有内容（矢量+位图+文字）；输出直接兼容现有 pipeline | 丢失矢量信息；文件大 | ⭐ 首选 |
| **B. SVG 矢量化** | `page.get_svg_image()` 提取 SVG XML | 保留矢量精度；理论上无限缩放 | 复杂；颜色/样式转换可能不一致；需额外转位图才能进 CV | 备选 |
| **C. 矢量路径解析** | 逐路径提取颜色填充区域 | 精确；可获得原始色值 | 极复杂；PyMuPDF 不直接支持；需写解析器 | 不推荐 |

**决策**：先实现方案 A（rasterize），满足 pipeline 输入需求；方案 B 作为后续优化项。

---

## 3. 接口契约

### 3.1 函数签名

```python
from pathlib import Path
from typing import Literal
import numpy as np


def extract_page_figure_as_bitmap(
    pdf_path: Path,
    page_idx: int,
    dpi: int = 600,
    crop_bbox: tuple[float, float, float, float] | None = None,
) -> dict:
    """将 PDF 指定页面 rasterize 为高分辨率位图。

    用于提取 PDF 中的矢量图形内容（如图件），补充 XObject 提取未覆盖的场景。

    Args:
        pdf_path: PDF 文件路径。
        page_idx: 页码（0-based）。
        dpi: 渲染分辨率。概念模型图建议 600（打印级）。
        crop_bbox: 可选，页面坐标系上的裁剪区域 (x0, y0, x1, y1)。
                   若页面有多个图件，通过 bbox 隔离目标。

    Returns:
        {
            "page_idx": int,
            "dpi": int,
            "width": int,      # 像素
            "height": int,     # 像素
            "image": np.ndarray,  # RGB, shape (H, W, 3)
            "source": str,     # "pdf_rasterize"
        }

    Raises:
        FileNotFoundError: pdf_path 不存在。
        ValueError: page_idx 超出范围。

    Test scenario:
        >>> result = extract_page_figure_as_bitmap(
        ...     Path("tests/fixtures/ph01/gxae11701.pdf"), page_idx=6, dpi=600
        ... )
        >>> assert result["width"] >= 1343
        >>> assert result["height"] >= 691
        >>> assert result["image"].ndim == 3
    """
    ...
```

### 3.2 与现有模块的关系

```
pdf_extractor/
  ├── extract.py         # M0.5: XObject + text_blocks（已有）
  ├── vector_extract.py  # M0.5v: 本规格（新增）
  ├── demo.py            # M0.5: XObject demo（已有）
  └── demo_vector.py     # M0.5v: rasterize demo（新增）
```

- `vector_extract.py` **不改动** `extract.py` 的任何代码。
- `vector_extract.py` **不依赖** `extract.py`，可独立 import。
- 两者在 controller 层由调用方按需选择：先尝试 XObject，若无结果再用 rasterize。

---

## 4. 实现要点

### 4.1 PyMuPDF rasterize

```python
import fitz

page = doc[page_idx]
mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72dpi 是 PDF 默认
pix = page.get_pixmap(matrix=mat)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
if pix.n == 4:
    img = img[:, :, :3]  # RGBA → RGB
```

### 4.2 裁剪（可选）

若已知 figure 在页面上的大致位置（可通过 `extract.py` 的 text_blocks 定位 caption 辅助推断），`crop_bbox` 按页面坐标（pt）裁剪：

```python
if crop_bbox:
    x0, y0, x1, y1 = crop_bbox
    rect = fitz.Rect(x0, y0, x1, y1)
    pix = page.get_pixmap(matrix=mat, clip=rect)
```

**初始版本可跳过 crop_bbox，整页 rasterize**，由 controller 或 CV detect 后续处理。

### 4.3 颜色空间

PyMuPDF `get_pixmap` 默认输出 RGB（若页面含 CMYK 会自动转换）。无需像 XObject 提取那样手动处理 colorspace。

---

## 5. 开发要求（TDD）

按 geoseg v2 模块阶段规范：

1. **接口定义**：先写 `vector_extract.py` 的函数签名 + docstring + type hints（即 §3.1）。
2. **写 `demo_vector.py`**：
   - 输入：`tests/fixtures/ph01/gxae11701.pdf`，page 7（index 6）
   - 输出：`runs/M0.5v/page_7_rasterize_600dpi.png`
   - **跑，应 FAIL**（红）
3. **最小实现**：写 `extract_page_figure_as_bitmap` 函数体。
4. **跑 `demo_vector.py`，应 PASS**（绿）：
   - 输出图片存在
   - 宽度 ≥ 1343，高度 ≥ 691（参考 ph01.jpg 尺寸，600dpi 下应远大于此）
   - 人眼可辨识 4 个 panel 和 colorbar
5. **行预算**：`vector_extract.py` < 100 行，`demo_vector.py` < 60 行。

---

## 6. 验收判据

| 检查项 | 通过标准 | 验证方式 |
|--------|----------|----------|
| 文件存在 | `runs/M0.5v/page_7_rasterize_600dpi.png` 存在 | `assert path.exists()` |
| 分辨率 | width × height ≥ 2000×1000（600dpi 下应远超） | 打印 shape |
| 内容完整性 | 4 个 panel + colorbar + 坐标轴 人眼可见 | 人眼比对 ph01.jpg |
| 白底验证 | 背景为白色（灰度均值 > 240） | `np.mean(gray) > 240` |
| 无文字模糊 | 文字（如 "Vs"、"5 km"）清晰可读 | 人眼检查 |

---

## 7. 与主线 pipeline 的衔接（未来）

```python
# controller.py 中（组装阶段）的伪代码
def _resolve_source_image(self, page_data):
    """决定使用 XObject 还是 rasterize 作为输入。"""
    if page_data["images"]:
        # 优先使用 XObject（原生分辨率更高）
        return page_data["images"][0]["data"]
    # fallback: rasterize 矢量图形
    result = extract_page_figure_as_bitmap(
        self.pdf_path, page_idx=page_data["page_idx"], dpi=600
    )
    return result["image"]
```

**注意**：此衔接逻辑在组装阶段才写，模块阶段 `vector_extract.py` 保持独立。

---

## 8. 已知风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| 600dpi 下文件过大（>10MB） | 内存/IO 压力 | 提供 `dpi` 参数，默认 300，按需调 600 |
| 矢量图形包含半透明/渐变 | rasterize 后颜色与 ph01.jpg 不一致 | 与 ph01.jpg 做颜色直方图比对，偏差大时告警 |
| 页面包含多个图件 | 整页 rasterize 会混入无关内容 | 后续用 text_blocks 定位 + crop_bbox 裁剪 |
| 文字渲染为位图后抗锯齿 | CV detect 的 white_threshold 需调整 | 在 M1b demo 中验证阈值兼容性 |

---

## 9. 产物路径

```
runs/
  M0.5/               # XObject 提取产物（已有）
  M0.5v/              # 矢量图提取产物（本规格）
    page_7_rasterize_600dpi.png
    page_7_metrics.json   # {width, height, dpi, mean_brightness, file_size_bytes}
```

---

**编写日期**: 2026-05-18
**版本**: 1.0
**签名状态**: 等待签字
