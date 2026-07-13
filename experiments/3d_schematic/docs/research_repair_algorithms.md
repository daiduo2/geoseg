# 小瑕疵修复算法调研文档

> 调研范围：图像修复（Inpainting）、文字擦除（Scene Text Removal）、小区域去噪/平滑算法
> 调研日期：2026-06-07
> 适用场景：地质示意图文字移除后的微弱残留修复

---

## 1. 调研背景

### 1.1 当前问题

在 geoseg v2 的 3D 地质示意图处理管线中，文字移除后仍可能存在微弱残留（faint text residue）。这些残留表现为：
- 浅色文字轮廓在彩色填充区域上的微弱痕迹
- 抗锯齿边缘留下的灰度/半透明像素
- 与背景颜色接近但可辨识的结构性瑕疵

这些残留对后续 CV 分割引擎（如 k-means、SLIC、区域融合等）产生干扰：
- 分割边界沿残留文字轮廓断裂
- 颜色统计被污染，导致聚类中心偏移
- 层位边界检测出现虚假边缘

### 1.2 目标

找到能有效清除小瑕疵（small blemish / spot / scratch）的算法，要求：
1. **针对性**：对"小区域、平滑背景上的微弱残留"有效
2. **边缘保持**：不模糊地质层位之间的真实边界
3. **可集成**：能在 Python/OpenCV 生态中快速落地
4. **可控性**：参数可调，避免过度修复引入伪影
5. **效率**：在 Mac mini M4 (16GB) 上可接受

---

## 2. 算法分类总览

| 算法名 | 类型 | 适用场景 | 计算成本 | 可获取性 | Python 支持 |
|--------|------|----------|----------|----------|-------------|
| OpenCV Telea / NS | 传统 CV 修复 | 小区域缺陷、平滑背景 | 极低 | 内置 | 是（cv2.inpaint） |
| Criminisi  exemplar | 传统 CV 修复 | 中等区域、纹理背景 | 低 | 开源实现 | 是（纯 Python） |
| PatchMatch | 传统 CV 修复 | 纹理合成、大区域 | 中等 | 开源实现 | 是（C++ + Python 绑定） |
| LaMa (big-lama) | 深度学习修复 | 大掩码、复杂纹理 | 中等（GPU）/ 高（CPU） | 开源 | 是（simple-lama） |
| MAT | 深度学习修复 | 高分辨率、人脸/场景 | 高 | 开源 | 是（需 CUDA） |
| FLUX.1 Fill | 扩散模型修复 | 高质量、语义一致性 | 很高 | 开源权重 | 是（diffusers） |
| BrushNet | 扩散适配器 | 精确掩码控制 | 很高 | 开源 | 是 |
| PowerPaint | 扩散修复 | 多任务（移除/插入/外扩） | 很高 | 开源 | 是（IOPaint） |
| ViTEraser | Transformer 文字擦除 | 场景文字移除 | 中等 | 研究代码 | 是 |
| Uformer-B + TMIM | Transformer 文字擦除 | 场景文字移除 SOTA | 中等 | 研究代码 | 是 |
| Bilateral Filter | 边缘保持平滑 | 全局去噪、斑点去除 | 低 | 内置 | 是（cv2.bilateralFilter） |
| Guided Filter | 边缘保持平滑 | 快速平滑、细节保持 | 极低 | contrib | 是（cv2.ximgproc） |
| Domain Transform Filter | 边缘保持平滑 | 实时平滑、纹理去除 | 极低 | contrib | 是（cv2.ximgproc） |
| Non-Local Means | 非局部去噪 | 随机噪声、纹理保持 | 高 | 内置 | 是（cv2.fastNlMeans） |
| Biharmonic Inpainting | PDE 修复 | 小划痕、平滑填充 | 低 | skimage | 是（skimage.restoration） |
| Photoshop CAF | 商业工具 | 专业修图 | N/A | 商业软件 | 否 |
| Snapseed Healing | 移动应用 | 快速小区域修复 | N/A | 移动应用 | 否 |
| TouchRetouch | 商业应用 | 线条/网格/对象移除 | N/A | 商业应用 | 否 |

---

## 3. 各类算法详细分析

### 3.1 深度学习修复算法

#### 3.1.1 Stable Diffusion Inpainting (v1.5 / v2 / SDXL / SD 3.5)

- **核心思想**：在潜在空间中，以二进制掩码为条件，通过迭代去噪生成掩码区域内的合理内容
- **优势**：语义一致性强；生态系统庞大（ControlNet、LoRA 微调）；对平坦色块区域效果好
- **劣势**：需要 GPU（4-8GB VRAM）；速度慢（512x512 约 5-20s）；提示词模糊时可能 hallucinate 虚假地质特征；掩码边缘可能有伪影
- **适用性**：高。若文字覆盖在平坦层状背景上，配合提示词 "clean geological cross-section, continuous rock layers, no text" 效果良好。建议搭配 ControlNet（depth/edge）保留结构线
- **获取**：PyTorch（diffusers、AUTOMATIC1111、ComfyUI），预训练权重在 HuggingFace

#### 3.1.2 FLUX.1 Fill (Black Forest Labs, 2024)

- **核心思想**：基于 DiT（Diffusion Transformer）的专用修复/外扩模型，原生 1024x1024 分辨率，提示词遵循性强
- **优势**：文本到图像对齐度极高；原生分辨率质量高；对象移除的掩码控制精确
- **劣势**：需要较大 VRAM（12GB+ 推荐 dev 模型）；比 SD 慢；pro 模型仅 API 可用
- **适用性**：高。适合出版质量的地质图修复。"Fill" 变体专门针对掩码区域补全优化
- **获取**：PyTorch via diffusers、ComfyUI 节点。权重在 HuggingFace（black-forest-labs/FLUX.1-Fill-dev）

#### 3.1.3 BrushNet (ECCV 2024)

- **核心思想**：即插即用的双分支扩散适配器，将掩码图像特征和噪声潜在变量分解到独立分支，提供密集的逐像素控制
- **优势**：比原生 SD 修复更好地保留未掩码区域；即插即用于 SD 1.5 / SDXL；掩码遵循性强
- **劣势**：增加复杂度；仍是扩散模型速度；需要基础扩散模型
- **适用性**：好。若已使用 SD，希望在移除文字的同时更好地保留地质层位边界，BrushNet 是优选
- **获取**：PyTorch。GitHub: TencentARC/BrushNet

#### 3.1.4 LaMa (WACV 2022)

- **核心思想**：基于 Fast Fourier Convolution（FFC）的大掩码修复网络，具有全图像感受野，对分辨率鲁棒
- **优势**：分辨率鲁棒性强；大掩码修复效果极佳；可在 CPU 运行（慢）或 GPU 运行（快）；预训练于 Places2；对周期性结构处理良好
- **劣势**：对极小瑕疵可能过度修复；CPU 上速度较慢
- **适用性**：高。对于地质图中的较大文字块或背景有微妙渐变/纹理的情况，LaMa 是最佳速度/质量权衡
- **获取**：`pip install simple-lama-inpainting`（轻量库）或 `pip install lama-cleaner`（完整 GUI）
- **Mac mini M4**：支持 `--device=mps`（Metal Performance Shaders），推荐

#### 3.1.5 MAT (Mask-Aware Transformer, CVPR 2022)

- **核心思想**：基于 StyleGAN2-ADA 的 Transformer 修复网络，在 512x512 上达到 SOTA
- **优势**：人脸/场景修复质量极高
- **劣势**：需要 CUDA；无官方 CPU 支持；图像尺寸需为 512 倍数
- **适用性**：中。质量高但集成难度大，不推荐作为首选
- **获取**：GitHub: fenglinglwb/MAT

#### 3.1.6 ViTEraser (ICCV 2023)

- **核心思想**：基于 Vision Transformer 的场景文字擦除，使用 SegMIM（Segmentation-aware Masked Image Modeling）预训练，利用可见 patch 的全局上下文预测掩码文字区域
- **优势**：首个基于 ViT 的 STR，在 SCUT-EnsText 上检测评估误差 <1%；背景恢复强；处理多样文字风格
- **劣势**：模型大（191.9M 参数）；需要预训练数据；推理比 CNN 慢
- **适用性**：高。专为文字移除设计，若对地质标注图进行微调，效果极佳
- **获取**：PyTorch。需联系作者获取代码和权重

#### 3.1.7 Uformer-B + TMIM (2024) — 当前 STR SOTA

- **核心思想**：Uformer 架构 + Text-aware Masked Image Modeling 预训练，利用大规模场景文字检测数据进行弱监督学习
- **优势**：SCUT-EnsText 上最佳 PSNR/SSIM；参数量仅为 ViTEraser 的 1/4（50.88M）；预训练更轻量
- **劣势**：仍是 Transformer 规模；检测评估略低于 ViTEraser
- **适用性**：高。文字移除的最佳准确率/效率权衡
- **获取**：PyTorch。需检查论文作者仓库

#### 3.1.8 扩散模型文字擦除（DiffSTR / PSSTRNet）

- **DiffSTR (2024)**：结合 TR-MAE 编码器与 Stable Diffusion 的高分辨率（512x512）文字擦除。质量高但极慢（1000 步），实用性低
- **PSSTRNet (2023)**：渐进式分割引导场景文字移除，极轻量（4.88M 参数），速度快但质量低于 Uformer/ViTEraser
- **适用性**：中。DiffSTR 质量高但不适合批处理；PSSTRNet 适合速度受限场景

### 3.2 传统 CV 修复算法

#### 3.2.1 OpenCV Inpainting（Telea / Navier-Stokes）

- **Telea (Fast Marching Method, 2004)**：将修复问题视为边值问题，使用快速行进方法从边界向内传播边界像素，基于已知邻居的加权平均
  - 优势：极快；比 Navier-Stokes 更适合文字移除；合理保留边缘；零依赖
  - 劣势：模糊精细细节；无纹理合成；大区域可见平滑
  - 参数：`inpaintRadius=3` 通常足够小文字修复
  - Python：`cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)`

- **Navier-Stokes (Bertalmio et al., CVPR 2001)**：将图像修复建模为流体动力学，将等照度线（level lines）传播到孔洞同时保持连续性
  - 优势：对平滑/渐变背景效果好；保留平滑颜色过渡
  - 劣势：比 Telea 更模糊；更慢；纹理区域表现差
  - 适用：仅当背景有平滑渐变（如阴影地质层）时使用
  - Python：`cv2.inpaint(image, mask, 5, cv2.INPAINT_NS)`

- **适用性（地质图文字残留）**：**极高**。推荐默认方案。若背景真正均匀，扩散模型是过度设计且可能引入伪影

#### 3.2.2 Criminisi et al. (2004) — 基于优先级的块填充

- **核心思想**：通过从图像边界复制最佳匹配块来填充孔洞，优先处理结构（边缘）而非纹理
- **优势**：无需训练；确定性；小区域极快；同质/平坦背景上的完美匹配
- **劣势**：大孔洞或复杂结构表现差；可能产生"垃圾收集"伪影；无语义理解
- **适用性**：极高。若文字在均匀色块上，采样边界颜色并填充本质上是最佳方案
- **获取**：OpenCV `cv2.INPAINT_NS`（Navier-Stokes 变体）；纯 Python 实现可用

#### 3.2.3 PatchMatch (Barnes et al., SIGGRAPH 2009)

- **核心思想**：随机化最近邻算法，快速找到图像中近似最佳匹配块用于孔洞填充
- **优势**：比暴力块搜索快得多；纹理合成好；处理重复模式
- **劣势**：无语义一致性；可能传播错误；大孔洞质量下降
- **适用性**：中等。适合纹理地质背景（如地震波形图），但平坦色块上过度设计
- **获取**：GIMP/Photoshop 中的 C++ 实现；Python 绑定：PyPatchMatch；OpenCV xphoto 模块有 Shift-Map

#### 3.2.4 Biharmonic Inpainting

- **核心思想**：在掩码区域求解双调和方程（4 阶 PDE），边界条件来自已知像素。产生非常平滑、无缝的填充
- **优势**：平滑填充效果极佳；无缝融合
- **劣势**：需要掩码；对纹理区域效果差
- **适用性**：极高。地质色块填充是平滑渐变，双调和修复将产生无缝结果
- **获取**：`skimage.restoration.inpaint_biharmonic(image, mask, channel_axis=-1)`

### 3.3 修图工具技术解析

#### 3.3.1 Adobe Photoshop

- **核心算法**：PatchMatch（Barnes et al., 2009, Princeton + Adobe）
  - 随机初始化 → 传播（好匹配扩散到邻居）→ 随机搜索（指数递减半径局部细化）
  - 时间复杂度：近线性 O(N log N) vs. O(N^2) 暴力搜索
- **现代架构（2024-2025）**：混合栈
  - PatchMatch 用于纹理合成和小区域修复
  - Adobe Sensei / Firefly 生成式扩散模型用于大孔洞和语义填充
  - 多尺度金字塔（粗到细）+ NNF 上采样
- **小瑕疵处理**：Spot Healing Brush（"Content-Aware" 模式）执行实时局部 PatchMatch
  - 自动采样周围像素；无需手动源
  - 匹配纹理、光照、透明度和阴影
  - 高斯加权混合 + 增益/偏置颜色适应
- **Python 复现**：PatchMatch 部分可复现（gaurav-behera/PatchMatch、CaptainHarryChen/PatchMatchInpainting、vacancy/PyPatchMatch）；Firefly 生成式 AI 不可复现

#### 3.3.2 Google Snapseed

- **核心算法（推断）**：快速、移动优化的基于 exemplar 的修复（Criminisi 风格或简化 PatchMatch 变体）
  - 自动从周围区域采样块（无手动源控制）
  - 局部搜索窗口（速度受限）
  - 固定块大小，GPU 加速（OpenGL/Metal）
  - 羽化 alpha 混合（非泊松编辑）
- **行为证据**：
  - >100px^2 区域困难 → 有限搜索窗口
  - 可见"块效应"/涂抹 → 简单混合无高级纹理合成
  - 均匀背景效果好 → 基于 exemplar 或扩散修复
  - 产生模糊；需要后锐化 → 混合中的平滑
- **Python 复现**：容易。简化的基于 exemplar 的修复器 + 局部搜索 + 羽化混合可在纯 Python/NumPy 中实现

#### 3.3.3 TouchRetouch (ADVA Soft)

- **核心算法**：基于 exemplar 的修复 + AI 边缘感知增强
- **架构（推断）**：用户选择 → 边缘感知分割 → 背景分析 → 源块搜索 → Exemplar 填充 → 无缝混合
- **特色功能**：
  - 专用线条移除工具（文字常有线性结构）
  - 网格检测处理重复模式
  - AI 边缘感知保留背景结构
- **Python 复现**：核心 exemplar 修复可复现；专用线条/网格检测可用 OpenCV Hough 变换、形态学操作和轮廓分析近似；"Erase AI" 自动检测需要训练好的分割模型（如 YOLO、SAM）

#### 3.3.4 iPhone (iOS 18) Clean Up / Samsung Galaxy AI

- **iPhone Clean Up (iOS 18.1+)**：仅设备端，多 ML 模型（对象检测 + 分割 + 智能修复）
  - 使用 2-bit 每权重量化、4-bit 嵌入表、8-bit KV cache
  - 具体架构未公开
- **Samsung Galaxy AI**：混合架构（设备端 Exynos 2400 NPU + 加密云）
  - Exynos 2400 NPU AI 性能比 Exynos 2200 提升 14.7 倍
  - 可能使用量化扩散模型或 GAN 变体，INT8/INT4 推理
- **Python 复现**：设备端神经模型不可直接复现，但预训练开源模型可达到类似效果（LaMa、Stable Diffusion Inpainting、MAT）

#### 3.3.5 GIMP Resynthesizer / Heal Selection

- **核心算法**：基于 Paul Harrison 博士论文 "Image Texture Tools" 的最近邻非参数纹理合成
  - 逐像素（或逐块）合成
  - 为每个待合成像素搜索源语料库中的最佳匹配邻域
  - "Heal Selection" 使用边界像素作为纹理源和匹配约束
- **架构**：
  - 插件 UI：Python (Python-Fu) — 掩码/语料库设置、用户界面
  - 核心引擎：C — 邻域匹配、合成
- **关键参数**：
  - `0.117` — 邻居概率（~1/8.5，8 连通邻域）
  - `16` — 截断（搜索加速，仅检查 N 个最佳候选）
  - `500` — 随机种子 / 迭代限制
- **Python 复现**：算法概念可复现，但 C 引擎性能在纯 Python 中难以复制。OpenCV `cv2.inpaint()` 提供类似功能

### 3.4 小区域去噪/平滑算法

#### 3.4.1 Bilateral Filter（双边滤波）

- **工作原理**：非线性边缘保持滤波。每个输出像素是邻居的加权平均，权重同时取决于空间距离（高斯）和光度（颜色/强度）相似性。跨强边缘的像素颜色差异大，权重降低，从而在平滑均匀区域的同时保留边缘
- **侵略性参数**：
  - `d`：邻域直径（px）。小值（3-5）针对微小斑点；大值（9-15）平滑更广区域
  - `sigmaColor`：颜色空间容差。越大 = 混合越多不相似颜色 = 更激进的斑点移除。典型值：10-75（8-bit）或 0.02-0.1（float）
  - `sigmaSpace`：空间范围。典型值：与 `d` 同阶
- **计算成本**：O(N * d^2) 每像素。明显慢于高斯模糊
- **Python**：`cv2.bilateralFilter()`（OpenCV 核心）；`skimage.restoration.denoise_bilateral()`（scikit-image）
- **适用性（地质图文字残留）**：好。平滑色块填充正是双边滤波的优势 — 不同强度的斑点被平滑而层位边界被保留。风险：若文字残留强度接近填充颜色，`sigmaColor` 需仔细调参

#### 3.4.2 Guided Filter（引导滤波）

- **工作原理**：假设输出是引导图像的局部线性变换。在窗口中计算局部线性系数 (a, b) 并重建。保留引导图像中的边缘，同时平滑其他区域。比双边滤波快，避免梯度反转伪影
- **侵略性参数**：
  - `radius`：局部窗口半径（px）。越小 = 更精细控制
  - `eps`：正则化参数。越大 = 更多平滑（更激进）。典型值：0.01-1.0
- **计算成本**：O(N) 每次迭代，与窗口大小无关。远快于双边滤波
- **Python**：`cv2.ximgproc.createGuidedFilter()`（OpenCV contrib）
- **适用性**：**极佳**。O(1) 复杂度，无梯度反转。使用原始图像作为引导来平滑文字残留，同时保持层位边缘清晰

#### 3.4.3 Domain Transform Filter（域变换滤波）

- **工作原理**：将图像变换到同时考虑空间和颜色差异的域，然后应用递归 1D 滤波（每维度 O(1)）。实时边缘感知平滑
- **侵略性参数**：
  - `sigmaSpatial`：空间标准差（px）。控制空间范围
  - `sigmaColor`：颜色/范围标准差。控制容许多大颜色差异
  - `numIters`：迭代次数。越多 = 越平滑
  - `mode`：`DTF_NC`、`DTF_IC`、`DTF_RF`（不同递归公式）
- **计算成本**：O(N) 每次迭代，极快。为实时应用设计
- **Python**：`cv2.ximgproc.createDTFilter()` + `filter()`（OpenCV contrib）；`cv2.ximgproc.amFilter()` 为 Adaptive Manifold 变体（更简单的一行调用）
- **适用性**：**极佳**。实时或批处理管线适用。`cv2.ximgproc` 中的 `bilateralTextureFilter` 专为结构保持纹理去除设计 — 直接相关于去除微弱文字残留

#### 3.4.4 Adaptive Manifold Filter（自适应流形滤波）

- **工作原理**：使用自适应流形近似高维双边滤波。比精确双边滤波快，质量相似
- **侵略性参数**：
  - `sigma_s`：空间 sigma
  - `sigma_r`：范围/颜色 sigma
  - `adjust_outliers`：是否调整异常值
- **计算成本**：O(N)，比精确双边滤波快
- **Python**：`cv2.ximgproc.createAMFilter()` 或 `cv2.ximgproc.amFilter()`（OpenCV contrib）
- **适用性**：好。当速度重要时的双边滤波替代方案

#### 3.4.5 Non-Local Means（非局部均值去噪）

- **工作原理**：对每个像素，在大窗口中搜索与局部块相似的块，然后基于块相似性加权平均。与局部滤波不同，它可以在远处找到匹配纹理/结构。在保留纹理的同时去除噪声方面表现极佳
- **侵略性参数**：
  - `h` / filter strength：控制平滑强度。越高 = 越激进。典型值：10-30（8-bit）或 0.6-1.15 * sigma_est（float）
  - `templateWindowSize` / patch_size：比较块大小（5x5 到 11x11）。更大的块 = 更稳健但更难找到精确匹配
  - `searchWindowSize` / patch_distance：搜索区域大小（13x13 到 31x31）。越大 = 质量越好但慢得多
  - `fast_mode`（scikit-image）：均匀空间权重 vs 高斯。快速模式更快但质量略低
- **计算成本**：O(N * searchWindowSize^2 * patchSize^2)。非常慢。`searchWindowSize` 主导运行时间
- **Python**：
  - OpenCV：`cv2.fastNlMeansDenoising()`（灰度）、`cv2.fastNlMeansDenoisingColored()`（彩色，在 CIELAB 中分别处理 L 和 AB 通道）
  - scikit-image：`skimage.restoration.denoise_nl_means()`（`fast_mode=True/False`）、`skimage.restoration.estimate_sigma()` 自动 `h` 调参
- **适用性（地质图文字残留）**：**中等**。NLM 为随机噪声设计，非结构化缺陷如文字。可能模糊文字残留而非干净移除。最好用作预处理步骤或结合掩码 + 修复。`h` 参数可激进调参，但过度平滑层位边界的风险存在

#### 3.4.6 形态学操作

- **开运算（Opening）**：腐蚀后膨胀。通过腐蚀消除小亮对象（"盐噪声"），然后膨胀回剩余结构。保留较大对象的形状/大小
  - 侵略性参数：`kernel` / footprint（结构元素形状和大小）、`iterations`
  - 适用性：**有限**。开运算在二进制或近二进制图像上工作。对于彩色填充上的微弱文字，需要先阈值化，可能破坏颜色信息。最好用作检测斑点后的掩码清理步骤

- **形态学重建（Morphological Reconstruction）**：掩码图像约束下的标记图像迭代测地膨胀。可去除小峰、填充孔洞或执行高级噪声去除同时保留结构
  - 适用性：中等。适合去除小亮峰（文字残留），如果能构建合适的标记。比滤波器更复杂

- **Remove Small Objects**：标记连通组件并移除低于大小阈值的组件
  - 适用性：需要二进制掩码。在缺陷掩码上的后处理步骤有用，不直接用于彩色图像

#### 3.4.7 频域方法

- **傅里叶低通/陷波滤波**：斑点（小、局部化缺陷）表现为高频成分。低通掩码衰减高频。陷波滤波针对特定周期性斑点模式
  - 适用性：**差**。傅里叶方法是全局的 — 模糊所有高频内容包括地质层之间的合法锐利边缘。除非文字有非常特定的周期性模式（不太可能），否则不适合局部文字残留移除

- **小波去噪**：将图像分解为多尺度的近似 + 细节系数。小斑点出现在高频细节系数（尤其对角线）中。阈值化这些系数去除斑点同时比傅里叶更好地保留边缘
  - 侵略性参数：`wavelet`（小波族）、`wavelet_levels`（分解层数）、`mode`（阈值模式：'soft'/'hard'）、`sigma` / threshold、`method`（'BayesShrink'/'VisuShrink'）
  - 适用性：中等。比傅里叶更适合局部缺陷。可针对文字残留出现的特定尺度。然而，层位边界也有高频内容可能受影响。最好作为预处理步骤或结合空间掩码

- **块 DCT（离散余弦变换）**：类似 JPEG 压缩 — 将重叠块变换到频域，阈值化高频系数，然后重建。局部化频率分析
  - 适用性：中等。比全局傅里叶更好的局部化。可按块调参。比小波去噪实现更复杂

---

## 4. 推荐方案（排序）

### 方案 1：OpenCV Telea 修复（首选 — 零依赖、极速）

- **推荐理由**：
  - 对"小区域、平滑背景上的微弱残留"是最佳匹配
  - 零 ML 依赖，<10ms 每区域
  - 若背景真正均匀，扩散模型是过度设计且可能引入伪影
  - 已在 OpenCV 内置，一行代码调用
- **预期效果**：
  - 平坦色块上的文字残留完全消除
  - 层位边界保持清晰
  - 对极小残留（1-3px）效果最佳
- **实现难度**：极低
  ```python
  import cv2
  # mask: 文字残留区域为 255 的二值掩码
  result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
  ```
- **风险**：
  - 大残留区域（>20px）可能产生可见平滑
  - 需要准确的掩码；若掩码遗漏残留像素，修复不完整
  - 对复杂纹理背景（如地震波形）效果差

### 方案 2：检测 + 修复（Detect + Inpaint — 最精准）

- **推荐理由**：
  - 针对性最强：只修复检测到的问题像素，不触碰正常区域
  - 避免全局平滑带来的边界模糊
  - 可组合多种检测策略（颜色阈值、亮度阈值、局部对比度）
- **预期效果**：
  - 残留文字像素被精确替换为周围填充色
  - 无过度修复伪影
  - 可处理多种残留类型（白色文字、灰色抗锯齿、半透明覆盖）
- **实现难度**：低
  ```python
  import cv2
  import numpy as np
  from skimage.morphology import disk, opening

  # 1. 检测：局部亮度异常（文字残留通常比周围亮）
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  # 使用局部均值作为背景估计
  bg = cv2.medianBlur(gray, 15)
  diff = cv2.subtract(gray, bg)
  _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

  # 2. 清理掩码
  mask = opening(mask, disk(1))

  # 3. 修复
  result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
  ```
- **风险**：
  - 检测阈值需要针对不同图像调参
  - 误检可能将合法细节（如薄层位线）标记为残留
  - 需要为每类地质图建立检测规则

### 方案 3：LaMa（深度学习 — 质量最高，需模型）

- **推荐理由**：
  - 大掩码修复的最佳速度/质量权衡
  - 对微妙渐变和纹理背景效果好
  - Apple Silicon 支持 MPS 加速
  - 3 行 Python API，集成简单
- **预期效果**：
  - 较大文字块（>20px）的完美移除
  - 背景纹理自然延续
  - 对复杂地质特征（裂缝、褶皱）上的文字效果好
- **实现难度**：低
  ```python
  from simple_lama_inpainting import SimpleLama
  from PIL import Image

  lama = SimpleLama()
  image = Image.open("panel.png")
  mask = Image.open("mask.png").convert('L')
  result = lama(image, mask)
  ```
- **风险**：
  - 模型文件 ~180MB，增加部署体积
  - MPS 上速度不如 CUDA，但比 CPU 快很多
  - 对小瑕疵可能"过度修复"，改变背景细节
  - 16GB RAM 在批处理大图像时可能紧张

### 方案 4：Guided Filter / Domain Transform Filter（预处理平滑）

- **推荐理由**：
  - 当掩码难以获取时的最佳全局平滑方案
  - O(1) 复杂度，实时速度
  - 结构保持，不模糊层位边界
  - 可作为修复前的预处理步骤
- **预期效果**：
  - 微弱文字残留被平滑融入背景
  - 层位边缘保持锐利
  - 适合"几乎看不见但分割引擎敏感"的残留
- **实现难度**：极低
  ```python
  import cv2

  # Guided Filter
  gf = cv2.ximgproc.createGuidedFilter(guide=image, radius=4, eps=0.01)
  result = gf.filter(image)

  # 或 Domain Transform Filter
  result = cv2.ximgproc.dtFilter(image, image, sigmaSpatial=20, sigmaColor=0.1, mode=cv2.ximgproc.DTF_NC)
  ```
- **风险**：
  - 全局操作，可能平滑合法细节
  - 参数（radius, eps, sigmaSpatial, sigmaColor）需针对具体图像调参
  - 对高对比度残留效果有限

### 方案 5：Biharmonic Inpainting（skimage — 最平滑填充）

- **推荐理由**：
  - 对平滑填充产生最无缝的结果
  - 地质色块填充是平滑渐变，双调和修复天然适合
  - 纯算法，零模型依赖
- **预期效果**：
  - 修复区域与周围背景完全无缝
  - 无可见边界或纹理不匹配
  - 适合小划痕和点状残留
- **实现难度**：极低
  ```python
  from skimage.restoration import inpaint_biharmonic

  result = inpaint_biharmonic(image, mask, channel_axis=-1)
  ```
- **风险**：
  - 比 OpenCV Telea 慢（求解稀疏线性系统）
  - 大区域修复计算成本高
  - 需要准确掩码

### 方案 6：ViTEraser / Uformer-B + TMIM（专用文字擦除 — 需微调）

- **推荐理由**：
  - 专为场景文字移除设计，检测评估指标最佳
  - 若收集地质图文字标注数据集进行微调，效果将极佳
  - 无残留文字可检测（detection-eval 最优）
- **预期效果**：
  - 文字残留完全不可检测
  - 背景恢复自然
  - 对多种文字风格（字体、大小、颜色）鲁棒
- **实现难度**：高
  - 需要获取研究代码和预训练权重
  - 需要地质图文字数据集进行微调
  - 模型较大（ViTEraser 191.9M，Uformer-B 50.9M）
- **风险**：
  - 研究代码可用性不确定
  - 微调需要标注数据
  - 推理速度比传统方法慢

---

## 5. 下一步实验建议

### 5.1 短期实验（1-2 天）

1. **建立评估基准**
   - 收集 10-20 张有文字残留的地质示意图（不同背景类型：平坦色块、渐变、纹理）
   - 标注残留区域掩码（Ground Truth）
   - 定义评估指标：PSNR、SSIM、分割边界 F1-score（修复前后对比）

2. **快速验证方案 1 和 2**
   - 实现 OpenCV Telea + 自动掩码检测（局部亮度异常）
   - 在基准集上测试，记录：
     - 修复质量（视觉检查 + PSNR/SSIM）
     - 对后续分割的影响（修复前后跑同一分割引擎，比较 IoU）
     - 运行时间
   - 调参：`inpaintRadius`（2, 3, 5）、检测阈值（10, 15, 20）

3. **快速验证方案 4**
   - 测试 Guided Filter 和 Domain Transform Filter 作为预处理
   - 对比参数组合：
     - Guided Filter: radius=[2, 4, 8], eps=[0.001, 0.01, 0.1]
     - DTF: sigmaSpatial=[10, 20, 40], sigmaColor=[0.05, 0.1, 0.2]
   - 评估对分割引擎的影响

### 5.2 中期实验（3-5 天）

4. **验证方案 3（LaMa）**
   - 安装 `simple-lama-inpainting`
   - 在 MPS 和 CPU 上分别测试速度
   - 对比 LaMa 与 OpenCV Telea 的质量差异
   - 评估内存占用（16GB 限制）

5. **验证方案 5（Biharmonic）**
   - 使用 `skimage.restoration.inpaint_biharmonic`
   - 重点测试小划痕和点状残留
   - 对比 OpenCV Telea 的平滑度

6. **掩码生成策略研究**
   - 颜色阈值：基于 HSV/LAB 空间的残留检测
   - 亮度阈值：局部对比度异常检测
   - 简单分类器：用少量标注数据训练像素级分类器（如 Random Forest）
   - 对比不同掩码生成策略对修复质量的影响

### 5.3 长期方向（如有需求）

7. **专用模型微调（方案 6）**
   - 若前序方案在特定场景（如复杂纹理背景）效果不佳，考虑：
     - 收集 100+ 张地质图文字标注数据
     - 微调 ViTEraser 或 Uformer-B + TMIM
     - 或基于 LaMa 在地质图数据上继续训练

8. **混合策略**
   - 根据背景类型自动选择算法：
     - 平坦色块 → OpenCV Telea
     - 渐变背景 → Biharmonic Inpainting
     - 纹理背景 → LaMa
     - 复杂结构 → FLUX.1 Fill + ControlNet
   - 实现简单的背景类型分类器（基于局部方差、边缘密度）

9. **集成到 geoseg 管线**
   - 在 `post_process` 或 `segment_engines` 模块中增加修复步骤
   - 作为可选预处理：在分割前自动检测并修复文字残留
   - 参数暴露给用户配置（侵略性级别：保守/标准/激进）

---

## 附录：快速参考代码

### A. OpenCV Telea 修复（推荐首选）

```python
import cv2
import numpy as np

def repair_with_telea(image: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """使用 OpenCV Telea 算法修复图像缺陷。

    Args:
        image: 输入图像 (H, W, 3), uint8
        mask:  二值掩码 (H, W), 255=缺陷区域
        radius: 修复半径，小瑕疵用 2-3，较大区域用 5-7

    Returns:
        修复后的图像
    """
    return cv2.inpaint(image, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
```

### B. 自动检测 + 修复

```python
import cv2
import numpy as np
from skimage.morphology import disk, opening

def detect_and_repair(image: np.ndarray,
                      blur_kernel: int = 15,
                      threshold: int = 15,
                      repair_radius: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """自动检测亮度异常区域并修复。

    Returns:
        (repaired_image, mask)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bg = cv2.medianBlur(gray, blur_kernel)
    diff = cv2.subtract(gray, bg)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    mask = opening(mask, disk(1)).astype(np.uint8) * 255
    repaired = cv2.inpaint(image, mask, repair_radius, cv2.INPAINT_TELEA)
    return repaired, mask
```

### C. LaMa 修复

```python
from simple_lama_inpainting import SimpleLama
from PIL import Image

def repair_with_lama(image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
    """使用 LaMa 模型修复图像缺陷。"""
    lama = SimpleLama()
    return lama(image_pil, mask_pil.convert('L'))
```

### D. Guided Filter 平滑

```python
import cv2

def smooth_with_guided_filter(image: np.ndarray, radius: int = 4, eps: float = 0.01) -> np.ndarray:
    """使用引导滤波进行边缘保持平滑。"""
    gf = cv2.ximgproc.createGuidedFilter(guide=image, radius=radius, eps=eps)
    return gf.filter(image)
```

### E. Biharmonic 修复

```python
from skimage.restoration import inpaint_biharmonic

def repair_with_biharmonic(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """使用双调和方程修复图像缺陷。"""
    return inpaint_biharmonic(image, mask, channel_axis=-1)
```

---

## 参考来源

- LaMa GitHub: https://github.com/advimman/lama
- LaMa Paper: https://arxiv.org/abs/2109.07161
- BrushNet GitHub: https://github.com/TencentARC/BrushNet
- BrushNet Paper: https://arxiv.org/abs/2403.06976
- PowerPaint GitHub: https://github.com/open-mmlab/PowerPaint
- ACE++ GitHub: https://github.com/ali-vilab/ACE_plus
- ViTEraser Paper: https://arxiv.org/abs/2306.12106
- Uformer+TMIM Paper: https://arxiv.org/abs/2409.13431
- DiffSTR Paper: https://arxiv.org/abs/2410.21721
- MAT Paper: https://arxiv.org/abs/2203.15270
- MAT GitHub: https://github.com/fenglinglwb/MAT
- FLUX.1 Tools: https://bfl.ai/flux-1-tools/
- OpenCV Inpainting: https://opencv.org/text-detection-and-removal-using-opencv/
- Navier-Stokes Inpainting Paper: https://www.math.ucla.edu/~bertozzi/papers/cvpr01.pdf
- PatchMatch Paper: https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf
- PyPatchMatch: https://github.com/vacancy/PyPatchMatch
- Scene Text Removal Survey: https://arxiv.org/abs/2409.13431
- TextDestroyer Paper: https://arxiv.org/abs/2411.00355
- simple-lama-inpainting: https://github.com/enesmsahin/simple-lama-inpainting
- lama-cleaner: https://github.com/Sanster/IOPaint
- IOPaint Models: https://www.iopaint.com/models
- OpenCV ximgproc: https://docs.opencv.org/4.x/df/d6c/group__ximgproc.html
- scikit-image Restoration: https://scikit-image.org/docs/stable/api/skimage.restoration.html
- bootchk/resynthesizer: https://github.com/bootchk/resynthesizer
- Apple On-Device Foundation Models 2025: https://machinelearning.apple.com/research/apple-foundation-models-2025-updates
- Adobe Inpainting: https://www.adobe.com/products/photoshop/inpainting.html
- Barnes et al. 2009 PatchMatch: http://www.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/index.php
- IPOL Non-Local Patch-Based Inpainting: https://www.ipol.im/pub/art/2017/189/article.pdf
- Image Quilting: https://www.ipol.im/pub/art/2017/171/article_lr.pdf
- GraphCut Textures: http://gamma.cs.unc.edu/kwatra/publications/gc-final-lowres.pdf
