"""
# 差分叠层分割方案设计文档 (Diff-Overlay Segmentation Design)

> 针对问题：地质示意图中文字/标注被误分为独立地质分区
> 提出者：导师 (高频差分 + 叠层隔离思路)
> 实验验证：多 agent 并行参数扫描

---

## 1. 问题诊断 (Problem Statement)

### 1.1 当前 pipeline (v3) 的失效模式

从 `result_final_v3.png` 观察：

- Panel 1/2/3 的白色文字标注（"Mantle", "Lithosphere", "weak zone" 等）
  在 `remove_text()` 阶段被 cv2.inpaint 涂白/模糊
- 但文字原位置在 Label Fill 中变成了**独立的彩色小区域**
- 例如 Panel 3: "weak zone" → 蓝色小区域；"refractory..." → 绿色区域
- **文字区域被当作地质分区**，严重污染分割结果

### 1.2 根本原因

`remove_text()` 的假设：**文字可以被完美修复回底层地质纹理**。

现实：
- inpaint 是插值算法，无法"猜"出被文字遮盖的真实地质颜色
- 总会留下微弱的颜色痕迹/模糊过渡带
- felzenszwalb (scale=300, sigma=0.5) 对这些细微差异敏感
- → 文字区域变成虚假独立分区

---

## 2. 核心原理 (Core Principle)

### 2.1 高频差分提取 (High-Pass Detail Extraction)

数学表达：

    D(x,y) = |I(x,y) - G_σ * I(x,y)|

其中：
- I(x,y): 原始图像
- G_σ * I: 高斯模糊（低通滤波，保留大尺度结构）
- D(x,y): 差分图（高频细节）

### 2.2 为什么差分能分离文字与地质？

| 特征维度 | 文字/标注 | 地质分层 |
|---------|----------|----------|
| 空间尺度 | 小（5~50px） | 大（100~500px） |
| 频率成分 | 高频（锐利边缘） | 低频（渐变过渡） |
| 模糊前后变化 | **大**（文字被抹平） | **小**（渐变几乎不变） |
| 差分响应 | **强** | **弱** |

### 2.3 导师三步法

```
Step 1: 差分提取细节层
    detail = |original - GaussianBlur(original, ksize, sigma)|

Step 2: 细节区域单独分区（叠层）
    overlay_mask = threshold(detail, thresh)  # 文字/标注/碎片区域
    overlay_mask = label_as_overlay(overlay_mask)  # 不参与地质分区竞争

Step 3: 平滑扩展范围
    overlay_mask = dilate/gaussian_blur(overlay_mask, radius)
    # 确保文字边缘过渡带也被纳入叠层
```

---

## 3. 方案设计 (Solution Design)

### 3.1 架构：叠层作为独立标签

```
输入图像 I
    │
    ├──→ 高斯模糊 B = G_σ * I ──→ 差分 D = |I - B| ──→ 阈值化 ──→ overlay_mask
    │                                                              │
    │    ┌─────────────────────────────────────────────────────────┘
    │    │
    │    └──→ 平滑扩展 overlay_mask (膨胀 + 高斯)
    │            │
    ├──→ 在 I 上叠层区域做 inpaint（可选，或直接用周围颜色填充）
    │            │
    ├──→ 对非叠层区域运行 felzenszwalb ──→ geo_labels（地质分区）
    │            │
    └────┴──────→ 合并：final_labels
                        - overlay_mask 区域 → label = OVERLAY_LABEL
                        - 其余区域 → geo_labels
```

### 3.2 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 叠层处理方式 | 独立 label，不合并到地质分区 | 叠层是标注信息，不是地质结构 |
| 叠层区域分割前处理 | inpaint 后送入 felzenszwalb | 避免叠层边缘影响邻接区域的分割 |
| 扩展策略 | 高斯模糊 + 阈值（优于形态学膨胀） | 更平滑的过渡，减少硬边 |
| 多通道差分 | 在 RGB 各通道分别差分后取 max | 文字颜色可能与某通道对比度更高 |

### 3.3 参数设计空间

| 参数 | 含义 | 扫描范围 | 推荐初值 |
|------|------|----------|----------|
| `blur_ksize` | 高斯模糊核大小 | [7, 11, 15, 21, 31] | 15 |
| `blur_sigma` | 高斯模糊 sigma | [1.0, 2.0, 3.0, 5.0] | 3.0 |
| `diff_thresh` | 差分阈值（0-255） | [10, 15, 20, 30, 50] | 20 |
| `expand_radius` | 叠层扩展半径 | [5, 10, 15, 20, 30] | 15 |
| `felz_scale` | felzenszwalb scale | [100, 200, 300, 500] | 300 |
| `felz_sigma` | felzenszwalb sigma | [0.3, 0.5, 0.8, 1.0] | 0.5 |

---

## 4. 实验计划 (Experiment Plan)

### Phase A: 单参数敏感性分析（独立 agent 并行）

**Agent-1: 差分提取参数扫描**
- 固定 blur_sigma=3.0，扫描 blur_ksize = [7, 11, 15, 21, 31]
- 固定 blur_ksize=15，扫描 blur_sigma = [1.0, 2.0, 3.0, 5.0]
- 输出：差分图可视化矩阵

**Agent-2: 阈值化 + 扩展实验**
- 固定 blur_ksize=15, sigma=3.0
- 扫描 diff_thresh = [10, 15, 20, 30, 50]
- 对每个 thresh，扫描 expand_radius = [5, 10, 15, 20, 30]
- 输出：叠层掩码可视化

### Phase B: 完整 pipeline 对比（独立 agent 并行）

**Agent-3: 叠层分割 pipeline**
- 选取 Phase A 最优参数组合
- 跑完整 pipeline：差分 → 叠层 → 非叠层 felzenszwalb → 合并
- 对比当前 v3 pipeline

**Agent-4: 综合评估与可视化**
- 汇总所有 agent 结果
- 生成对比图：v3 vs diff-overlay（各参数组合）
- 视觉评估：文字区域是否不再产生虚假分区

### 验收标准 (Acceptance Criteria)

1. **文字区域不再产生独立分区**：Label Fill 中文字原位置不出现面积 < 5% 的孤立小区域
2. **地质分层完整性**：非文字区域的分区结果与 v3 一致或更优
3. **叠层覆盖完整**：文字边缘过渡带被完整纳入叠层，无残留虚假分区
4. **参数鲁棒性**：在参数变化 ±30% 范围内，结果定性一致

---

## 5. 参考实现框架

见下方 `run_experiment()` 和 `diff_overlay_pipeline()` 函数。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import felzenszwalb


def extract_detail_layer(
    image: np.ndarray,
    blur_ksize: int = 15,
    blur_sigma: float = 3.0,
) -> np.ndarray:
    """Step 1: 差分提取细节层.

    Returns:
        detail: float32 array, shape (H, W), 值域 [0, 255]
                亮区 = 高频细节（文字、边缘、碎片）
                暗区 = 低频区域（地质分层主体）
    """
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), sigmaX=blur_sigma)
    # RGB 各通道分别差分后取 max，确保任意颜色文字都被捕捉
    diff = np.abs(image.astype(np.float32) - blurred.astype(np.float32))
    detail = diff.max(axis=2)  # (H, W)
    return detail


def create_overlay_mask(
    detail: np.ndarray,
    diff_thresh: float = 20.0,
    expand_radius: int = 15,
) -> np.ndarray:
    """Step 2+3: 阈值化 + 平滑扩展叠层掩码.

    Returns:
        overlay_mask: bool array, shape (H, W), True = 叠层区域
    """
    # 阈值化
    binary = (detail > diff_thresh).astype(np.uint8) * 255

    # 平滑扩展：高斯模糊 + 重新阈值化
    # 比形态学膨胀更平滑，减少硬边
    if expand_radius > 0:
        ksize = expand_radius * 2 + 1
        blurred = cv2.GaussianBlur(binary, (ksize, ksize), sigmaX=expand_radius)
        # 重新阈值化：扩展后的区域
        overlay_mask = blurred > 64  # 经验阈值，可微调
    else:
        overlay_mask = binary > 0

    return overlay_mask


def diff_overlay_pipeline(
    image: np.ndarray,
    blur_ksize: int = 15,
    blur_sigma: float = 3.0,
    diff_thresh: float = 20.0,
    expand_radius: int = 15,
    felz_scale: float = 300.0,
    felz_sigma: float = 0.5,
    overlay_label: int = -1,
) -> dict:
    """完整差分叠层分割 pipeline.

    Returns:
        dict with keys:
            - detail: 差分图 (H, W) float32
            - overlay_mask: 叠层掩码 (H, W) bool
            - geo_labels: 非叠层区域地质分区 (H, W) int
            - final_labels: 合并后标签 (H, W) int，叠层区域 = overlay_label
            - overlay_only: 仅叠层区域可视化
    """
    h, w = image.shape[:2]

    # Step 1: 差分提取
    detail = extract_detail_layer(image, blur_ksize, blur_sigma)

    # Step 2+3: 叠层掩码
    overlay_mask = create_overlay_mask(detail, diff_thresh, expand_radius)

    # Step 4: 叠层区域 inpaint（用周围颜色填充，避免影响 felzenszwalb）
    inpaint_mask = overlay_mask.astype(np.uint8) * 255
    inpainted = cv2.inpaint(image, inpaint_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # Step 5: 非叠层区域 felzenszwalb
    geo_labels = felzenszwalb(inpainted, scale=felz_scale, sigma=felz_sigma, min_size=30)

    # Step 6: 合并标签
    final_labels = geo_labels.copy()
    final_labels[overlay_mask] = overlay_label

    # 可视化：叠层区域用特殊颜色标记
    overlay_vis = image.copy()
    overlay_vis[overlay_mask] = [255, 0, 255]  # 洋红色 = 叠层

    return {
        "detail": detail,
        "overlay_mask": overlay_mask,
        "geo_labels": geo_labels,
        "final_labels": final_labels,
        "overlay_only": overlay_vis,
        "inpainted": inpainted,
    }


def render_label_fill(labels: np.ndarray, overlay_label: int = -1) -> np.ndarray:
    """渲染标签填充图，叠层区域用灰色标记."""
    import colorsys

    unique = sorted(np.unique(labels))
    h, w = labels.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    colors = []
    for i, lbl in enumerate(unique):
        if lbl == overlay_label:
            colors.append([128, 128, 128])  # 灰色 = 叠层
        else:
            hue = (i * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.92)
            colors.append([int(c * 255) for c in rgb])

    for i, lbl in enumerate(unique):
        mask = labels == lbl
        result[mask] = colors[i]
    return result


def run_experiment(
    image_path: Path,
    out_dir: Path,
    params: dict,
) -> dict:
    """运行单次实验并保存结果."""
    out_dir.mkdir(parents=True, exist_ok=True)

    img = np.array(Image.open(image_path).convert("RGB"))
    result = diff_overlay_pipeline(img, **params)

    # 保存可视化
    suffix = f"_k{params['blur_ksize']}_s{params['blur_sigma']}_t{params['diff_thresh']}_e{params['expand_radius']}"

    # 差分图
    detail_norm = (result["detail"] / result["detail"].max() * 255).astype(np.uint8)
    Image.fromarray(detail_norm).save(out_dir / f"detail{suffix}.png")

    # 叠层掩码
    overlay_uint8 = result["overlay_mask"].astype(np.uint8) * 255
    Image.fromarray(overlay_uint8).save(out_dir / f"overlay{suffix}.png")

    # 叠层可视化
    Image.fromarray(result["overlay_only"]).save(out_dir / f"overlay_vis{suffix}.png")

    # 最终标签填充
    fill = render_label_fill(result["final_labels"], params.get("overlay_label", -1))
    Image.fromarray(fill).save(out_dir / f"fill{suffix}.png")

    # 参数记录
    with open(out_dir / f"params{suffix}.txt", "w") as f:
        for k, v in params.items():
            f.write(f"{k}={v}\n")

    return result


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent
    out_dir = base / "diff_overlay_experiments"

    # 基准参数
    base_params = {
        "blur_ksize": 15,
        "blur_sigma": 3.0,
        "diff_thresh": 20.0,
        "expand_radius": 15,
        "felz_scale": 300.0,
        "felz_sigma": 0.5,
        "overlay_label": -1,
    }

    # 跑 Panel 3（文字最多的）
    img_path = base / "panel_3_front.png"
    print(f"Running baseline experiment on {img_path.name}...")
    run_experiment(img_path, out_dir, base_params)
    print(f"Results saved to {out_dir}")
