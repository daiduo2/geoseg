# 开源修复项目分析报告

> 分析日期: 2026-06-08
> 分析项目: LaMa, ViTEraser, PyPatchMatch
> 分析目的: 评估其对地质示意图文字残留修复问题的适用性

---

## 项目速览

| 项目 | 类型 | 语言 | 大小 | 核心算法 | 预训练模型 |
|------|------|------|------|----------|-----------|
| **LaMa** | 深度学习修复 | Python/PyTorch | 14M | FFC (Fast Fourier Convolution) | big-lama (~180MB) |
| **ViTEraser** | 深度学习文字擦除 | Python/PyTorch | 1.6M | SwinV2 + SegMIM | Tiny/Small/Base (~28-88MB) |
| **PyPatchMatch** | 传统算法修复 | Python/C++ | 2.1M | PatchMatch (随机最近邻) | 无 |

---

## 1. LaMa (Large Mask Inpainting)

### 1.1 核心架构

**关键创新：Fast Fourier Convolution (FFC)**
- 每层分裂为两条路径：
  - **Local pathway**: 标准空间卷积
  - **Global pathway**: 频域变换（`torch.fft.rfftn` / `irfftn`）
- 有效感受野无限大，无需膨胀卷积或注意力机制
- 训练于 256x256，但可泛化到 ~2k 分辨率

**Generator: `FFCResNetGenerator`**
```
Input (RGB + Mask, 4ch)
  → 3x Downsample (Encoder)
  → 18x FFCResNetBlock (Bottleneck, ratio_g=0.75)
  → 3x Upsample (Decoder)
  → Output (RGB, 3ch)
```

**Training Loss**: L1 + Perceptual (VGG) + Adversarial (PatchGAN) + Feature Matching

### 1.2 推理流程

```python
# predict.py 核心逻辑
inpainted = mask * predicted + (1 - mask) * original
```

1. 加载 `config.yaml` + `best.ckpt`
2. 图像自动 pad 到 8 的倍数
3. 拼接 `masked_image + mask` 输入生成器
4. 用 mask 混合预测结果和原始图像
5. 去除 pad，保存 PNG

**Refinement Mode**: 多尺度金字塔优化（latent feature + Adam），质量更高但慢得多，且需要多 GPU。

### 1.3 预训练权重

| 模型 | 训练数据 | 大小 | 获取 |
|------|----------|------|------|
| big-lama | Places2 Challenge | ~180MB | HuggingFace/Google Drive |
| lama-fourier | Places2 Standard | ~50MB | 同上 |
| lama-regular | Places2 Standard | ~50MB | 同上 |

### 1.4 依赖

- PyTorch 1.8.0 + torchvision 0.9.0
- pytorch-lightning 1.2.9
- hydra-core, omegaconf
- opencv-python, scikit-image, albumentations, kornia
- **问题**: 依赖版本非常老旧（2021年），升级需大量迁移工作

### 1.5 适用性评估

**Pros:**
- 大感受野适合大面积修复
- 分辨率鲁棒性强
- 推理简单（单前向传播）
- big-lama 是通用最强模型之一

**Cons:**
- **领域不匹配**: 训练于自然图像（Places2），地质示意图是结构化非真实感图形
- 可能在层位边界处产生自然纹理幻觉
- 需要显式二值掩码
- 无 MPS 原生支持（需 patch）

**M4 16GB 可行性**: CPU 推理可行（~1-3s/512x512），内存占用小。

---

## 2. ViTEraser

### 2.1 核心架构

**关键创新：SwinV2 + SegMIM 预训练**
- **Encoder/Decoder**: 分层 Swin Transformer V2
- **SegMIM Pretraining**: 联合学习 (a) 掩码图像重建 (MIM) + (b) 文字分割预测
- **GAN Training**: PatchGAN 判别器（全局 + 局部双分支）

```
ViTEraser(nn.Module)
├── encoder: SwinV2Encoder          # 4-stage 分层特征提取
├── decoder: SwinTransformerV2Decoder  # 5-stage + PatchSplit 上采样 + skip connections
├── pixel_embed: nn.Linear          # 维度桥接
└── vgg16: VGG16 (frozen)           # Perceptual loss 特征提取
```

**Forward Flow:**
1. `images` → `encoder` → 多尺度特征
2. 最后一层特征 → `pixel_embed` → `decoder`
3. `decoder` → 中间输出列表 + 预测掩码
4. 训练时: 组合输出 + VGG16 perceptual/style losses

### 2.2 训练/推理流程

**Training Loss:**
- `MSR_loss`: 多尺度重建 L1（mask 加权）
- `prc_loss`: VGG perceptual loss
- `style_loss`: Gram matrix style loss
- `D_fake`: 生成器对抗 loss
- `mask_loss`: 文字掩码 Dice loss

**Inference:**
```bash
python -m torch.distributed.launch --nproc_per_node 1 main.py \
  --eval --resume path/to/weights --data_root data/TextErase/
```

### 2.3 预训练权重

| 模型 | 参数量 | 来源 |
|------|--------|------|
| ViTEraser-Tiny | ~28M | BaiduNetDisk / Google Drive |
| ViTEraser-Small | ~50M | 同上 |
| ViTEraser-Base | ~88M | 同上 |

**注意**: 场景文字擦除的预训练权重可用，但 SegMIM 预训练权重需另下载。

### 2.4 依赖

- PyTorch 1.8.2 (CUDA 11.1)
- timm 0.6.11 (Swin Transformer 工具)
- opencv_python, scikit_image, pytorch-fid
- **问题**: PyTorch 版本锁定在 1.8.2，现代 PyTorch 2.x 可能有兼容性问题

### 2.5 适用性评估

**Pros:**
- 专为文字擦除设计，任务匹配度高
- SwinV2 的层次化结构适合多尺度文字
- SegMIM 预训练提供强文字感知表征

**Cons:**
- **领域差距大**: 训练于自然场景（街景、招牌），地质示意图是合成图形
- 对**微弱残留**可能不敏感（训练数据是高对比度粗体文字）
- 需要**微调**: 建议准备 50-100 对地质图样本 fine-tune
- GAN 训练复杂，集成难度高

**M4 16GB 可行性**: Tiny 模型推理可行（~1-2GB VRAM），fine-tune 需 batch_size=1。

---

## 3. PyPatchMatch

### 3.1 核心算法

**PatchMatch (Barnes et al., SIGGRAPH 2009)**

随机近似最近邻算法，核心三步：

1. **Random Initialization**: 每个像素随机分配一个目标坐标
2. **Propagation**: 好匹配向空间邻居传播（扫描线交替方向）
3. **Random Search**: 指数递减半径的局部细化搜索

**复杂度**: O(N log N) vs. O(N²) 暴力搜索

### 3.2 实现细节

| 方面 | 详情 |
|------|------|
| 语言 | C++14 核心 + Python ctypes 绑定 |
| 构建 | `make` 编译 `libpatchmatch.so` |
| 依赖 | OpenCV (C++), numpy, PIL |
| 距离度量 | Patch SSD + 图像梯度 (5通道: RGB + grad_x + grad_y) |
| 扩展 | 正则引导变体 (`ijmap`) 用于结构化图案 |

**粗到细金字塔** (`inpaint.cpp`):
- 重复下采样构建图像金字塔
- 从粗到细逐层运行 PatchMatch
- 每层 EM 迭代：E-step（NNF 投票）→ M-step（平均重建）

### 3.3 使用方式

```python
import patch_match

# 基础用法（默认纯白色像素为掩码）
result = patch_match.inpaint(image, patch_size=15)

# 显式掩码
result = patch_match.inpaint(image, mask=mask, patch_size=15)

# 全局掩码（限制源区域）
result = patch_match.inpaint(image, mask=hole_mask, global_mask=restricted_mask)

# 正则引导（结构化图案）
result = patch_match.inpaint_regularity(image, mask, ijmap, patch_size=15)
```

### 3.4 适用性评估

**Pros:**
- 轻量级，无 ML 依赖
- 纯色/均匀背景上是 PatchMatch 的强项
- 可保护层位边界（`global_mask` 限制源区域）
- 小孔洞非常快（sub-second）
- M4 上运行零压力

**Cons:**
- 大区域可能模糊或重复伪影
- 层位边界处可能从错误侧拉取 patch
- 需要 OpenCV C++ 开发头文件编译
- 无 pip 包，需手动 make

**M4 16GB 可行性**: 极高。CPU 单线程，内存 <100MB（1Kx1K 图像）。

---

## 4. 三项目横向对比

| 维度 | LaMa | ViTEraser | PyPatchMatch |
|------|------|-----------|--------------|
| **算法类型** | 深度学习 (频域卷积) | 深度学习 (ViT+GAN) | 传统算法 (块匹配) |
| **任务匹配度** | 中（通用修复） | 高（文字擦除） | 中（通用修复） |
| **领域适配** | 需验证（自然→示意图） | 需微调（街景→地质图） | 即开即用 |
| **预训练模型** | big-lama (180MB) | Tiny (28MB) / Small (50MB) | 无 |
| **推理速度** | ~1-3s (CPU, 512²) | ~1-2s (CPU, Tiny) | ~0.1-1s (CPU) |
| **内存占用** | ~500MB | ~1-2GB | ~100MB |
| **代码质量** | 良（模块化但依赖老旧） | 良（研究代码需清理） | 中（C++ 核心简洁） |
| **集成难度** | 中（需封装为库） | 中高（需解耦分布式训练） | 低（但需编译 C++） |
| **M4 可行性** | ✅ CPU 可行 | ✅ Tiny 可行 | ✅ 极佳 |

---

## 5. 针对地质示意图的推荐策略

### 5.1 短期验证（1-2 天）

**优先级 1: PyPatchMatch**
- 原因：零模型依赖，编译后即可测试
- 测试方案：
  - 用 text_removal_v2 的 mask 作为输入
  - 对比 PatchMatch vs. Telea r=3 的修复质量
  - 测试 `global_mask` 保护层位边界的效果
- 预期：纯色层内效果好，边界处需小心

**优先级 2: LaMa (big-lama)**
- 原因：通用最强模型，分辨率鲁棒
- 测试方案：
  - 下载 big-lama 权重
  - 用 v2 mask 跑推理
  - 重点关注层位边界是否模糊
- 预期：简单背景效果好，边界处可能过度平滑

### 5.2 中期验证（3-5 天）

**ViTEraser 微调**
- 若 LaMa/PatchMatch 在复杂纹理（Panel 3）上效果不佳
- 准备 50-100 对样本：
  - `image/`: text_removal_v2 后的残留图
  - `label/`: Telea r=3 修复后的"干净"图（作为 pseudo-ground-truth）
  - `mask/`: v2 检测到的文字掩码
- 用 ViTEraser-Tiny 做轻量 fine-tune（batch_size=1, 简化 loss: L1 + perceptual only）

### 5.3 推荐集成方案

```python
# 推荐的分级修复策略
def smart_repair(image: np.ndarray, mask: np.ndarray, 
                 background_type: str = "auto") -> np.ndarray:
    """
    根据背景类型自动选择修复算法。
    """
    if background_type == "auto":
        # 简单启发式：局部方差低 = 纯色/渐变，方差高 = 纹理
        local_var = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        background_type = "uniform" if local_var < 100 else "texture"
    
    if background_type == "uniform":
        # 纯色/渐变背景：Telea r=3 已足够（实验已验证）
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    
    elif background_type == "texture":
        # 纹理背景（如 Panel 3 碎屑）：尝试 PatchMatch 或 LaMa
        # PyPatchMatch 对纹理保持更好
        try:
            import patch_match
            return patch_match.inpaint(image, mask=mask, patch_size=15)
        except ImportError:
            # fallback to LaMa
            return lama_inpaint(image, mask)
    
    else:
        raise ValueError(f"Unknown background_type: {background_type}")
```

---

## 6. 代码质量与集成风险评估

| 项目 | 风险 | 缓解措施 |
|------|------|----------|
| **LaMa** | 依赖老旧（PyTorch 1.8） | 使用 `lama-cleaner` 封装包替代原始仓库 |
| **ViTEraser** | 分布式训练耦合 | 提取模型类，写独立推理 wrapper |
| **PyPatchMatch** | 需编译 C++ 共享库 | 预编译 `.so`/`.dylib`，添加 Python 输入校验 |

---

## 7. 结论

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| **快速验证 / 纯色背景** | OpenCV Telea r=3 | 已实验验证满分，零额外依赖 |
| **纹理背景 / 边界敏感** | PyPatchMatch | 轻量，可限制源区域保护边界 |
| **大面积文字块 / 复杂背景** | LaMa big-lama | 大感受野，分辨率鲁棒 |
| **追求 SOTA 文字擦除质量** | ViTEraser + Fine-tune | 专为文字设计，需地质图微调数据 |

**对于当前 3D Schematic 问题**：
- Panel 1/2（纯色/渐变背景）: **Telea r=3 已是最优**，无需引入复杂模型
- Panel 3（复杂纹理）: 可尝试 **PyPatchMatch** 或 **LaMa** 作为升级
- 长期来看，若需处理更多样化的地质图，**ViTEraser 微调** 是最佳投资方向
