# Layer 2 平滑修正：地球物理先验与相关工作调研

> 2026-10-21。范围：Layer 1 已经确定保留的**分区闭边界**，如何做几何平滑修正。
> 本文只做调研与设计，不实现平滑算法。
>
> 前置约束（来自当前项目共识）：
> 1. Layer 1 的边界是 labels 的直接闭边界，不做 skeleton、RDP、spline；
> 2. Layer 2 只改几何，不改变 keep/review/remove，不重新分割；
> 3. 视觉审计 baseline 是 labels 重新上色的 mask，不是照片；
> 4. 不允许用平滑去弥补 labels 缺层或错分。

---

## 0. 结论先行

在本次检索范围内，没有发现能直接套用到“离散 labels 闭边界网络”的现成方法。原因是我们处理的对象不是地震振幅、不是层位拾取点，而是**离散 labels 的边界网络**；文献中的平滑方法大多属于以下四类之一：

1. 对地震图像做结构导向/各向异性扩散，平滑同相轴、保护断层；
2. 对层位拾取点做最小曲率、样条、动态规划或 RGT 重建；
3. 在速度模型反演中加平滑/TV/交叉梯度/界面约束；
4. 在图像分割中加拓扑/连通性约束。

这些工作提供了四类可直接转化为 Layer 2 的约束：

- **方向先验**：沿边界切向平滑，法向保持；断层/不整合是终止面，不是噪声。
- **几何先验**：最小曲率、张力、曲率变化惩罚、端点固定；避免过冲和虚假拐点。
- **不连续先验**：断层、不整合、尖灭点把边界网络切成独立平滑域，不能跨域拟合。
- **拓扑先验**：节点固定、禁止自交/互交、禁止新增或丢失区域/孔洞；离散分割中加入全局拓扑约束通常是 NP-hard，工程上采用“局部约束 + 全局校验 + 回退”。

推荐路线不是全局扩散、水平集或图割重分割，而是：

```
exact closure
  → 边界网络（junction/endpoint/fault 切成 segment）
  → 分段约束样条（最小曲率 + 张力 + 端点/切向约束 + 位移走廊）
  → 全局拓扑/几何校验
  → 违规则局部降低平滑或回退 exact，绝不删除
  → 在 recolor mask 上做粗线视觉审计
```

最重要的否定结论：

- 不做跨 junction / 跨 fault 的全局平滑；
- 不用一条全局样条拟合多个 segment；
- 不用 curvature 超限作为删除理由；
- 不用 per-panel 参数调优；
- 不把平滑后的 labels 当作新的分区事实来源。

---

## 1. 任务定义与约束

### 1.1 输入/输出

输入：

- `labels`：Layer 1 的事实来源；
- `keep`：Layer 1 已确定保留的 exact boundary objects；
- 每个 boundary 的 `labels_pair`、原始像素、长度、背景角色；
- 可选的断层/不整合 mask、panel 边界、labels 置信度。

输出：

- 每条 keep boundary 的平滑几何（曲线或折线）；
- 每个 segment 的端点、切向、类别、平滑参数、残差、违规记录；
- 不输出新的 keep/remove 决定，不覆盖原始 labels。

### 1.2 硬约束（不可违反）

| 约束 | 含义 | 违反后果 |
|---|---|---|
| retention invariant | Layer 1 的 keep 集合不变 | 等价于重新做保留判定 |
| endpoint/junction pinning | 边界在 panel 边缘、三叉点、断层交点处固定 | 拓扑漂移、区域粘连/开裂 |
| no self-intersection | 单条曲线不自交 | 非法几何 |
| no cross-intersection | 非相邻边界不互交 | 区域被切碎或吞并 |
| adjacency preservation | label 邻接关系不变 | 区域拓扑改变 |
| component/hole preservation | 区域连通分量和孔洞数不变 | 等价于删除/新建区域 |
| fault offset preservation | 断层两侧位移不被抹平 | 地质错误 |
| bounded deviation | 平滑曲线不超出 exact closure 的允许走廊 | 细层被吃掉 |
| thin-layer preservation | 不因平滑使薄层厚度趋零 | 区域消失 |

### 1.3 软约束（可权衡）

- 沿边界切向的低曲率；
- 曲率变化平缓（避免折线拐点）；
- 端点切向与相邻 segment 连续；
- 层厚变化平缓；
- 与图像局部结构方向一致（只作弱证据）；
- 闭合体（透镜、盐体、河道）的面积/宽度尽量保持。

### 1.4 与“图像去噪”的区别

exact closure 的“噪声”主要不是随机像素噪声，而是：

- label 栅格化的 staircase；
- 文字/遮挡造成的缺口或绕行；
- 分割器在弱边界处的锯齿；
- 标签本身缺层/合并层造成的错误拓扑。

Layer 2 可以处理 staircase、遮挡缺口和弱边界锯齿；缺层/合并层造成的错误拓扑必须回到 Layer 1 / labels 修正。用平滑掩盖缺层，会把错误固化。

### 1.5 地质/地球物理先验清单

| 先验 | 地质含义 | 对 Layer 2 的直接含义 |
|---|---|---|
| 层序/沉积 | 层位近似等时面，层序有上下关系，层厚通常为正 | 不跨层拉直；保持层序和薄层厚度；junction 固定 |
| 断层 | 位移不连续面，断层两侧错距、断层带/损伤带 | 断层处切开平滑域；保持错距；不把断层当 outlier |
| 不整合 | 侵蚀/沉积间断，可有角度不整合和 truncation | 允许角度和 C0/C1 不连续；降低平滑强度 |
| 褶皱 | 曲率有限但可较大，有 hinge 和陡翼 | 不能把 hinge 拉平；用 knot/anchor 或局部降平滑 |
| 盐体/侵入体/河道 | 高 contrast、形状不规则、窄体/闭合体 | 面积/宽度保持；避免曲率流收缩；小体低平滑 |
| 尖灭/超覆 | 层厚趋零，边界终止/合并 | 保留 termination；禁止把尖灭点抹成圆角 |
| 各向异性/倾角 | 层理方向控制连续性 | 沿边界切向平滑，法向位移受限 |
| 井/剖面控制点 | 解释约束，不应被平滑违背 | 控制点硬约束或高权重 |
| 解释不确定性 | 遮挡、弱反射、标签错误导致置信度不同 | 高不确定区域保留 exact，进入 review |

这些先验不是“图像滤波参数”，而是决定**哪些地方可以平滑、哪些地方必须保持不连续**的结构规则。Layer 2 的核心不是选一个更强的平滑器，而是先识别这些结构域。

---

## 2. 仓库现有实现审计

### 2.1 `curve_editor/refine.py`

当前实现：

- 对每条 `ExtractedCurve` 做 `splprep` 平滑；
- 开放曲线端点给高权重，拟合后强制回到原端点；
- 曲率超过 `max_curvature` 就增大 smoothing，最多 8 次；仍超限则返回 `None`；
- `ignore_background=True` 时直接丢弃含 0 的边界；
- `refine_decisions` 只对 keep 做平滑，review 保留原始曲线。

问题：

1. **会丢弃边界**：`refine_curve` 返回 `None`，`refine_curves` 静默不输出；这违反 Layer 2 不得删除。
2. **没有 junction/fault 概念**：整条曲线全局拟合，三叉点、断层交点会漂移。
3. **没有拓扑校验**：自交、互交、薄层消失都未检查。
4. **闭合曲线会收缩/圆化**：`splprep(per=True)` 的平滑会改变小闭合体的面积。
5. **曲率阈值是像素尺度**：`0.35` 对 191 px 高和 1740 px 高的 panel 不等价。
6. **只有几何，没有类别**：horizon、fault、unconformity、closed body 用同一套参数。
7. **背景边界处理仍是旧语义**：`ignore_background` 与 Layer 1 的 exact 决策不一致。

结论：`refine.py` 只能作为“单段平滑”的算法原型，不能直接扩展成 Layer 2 主流程。

### 2.2 `segment_engines/horizon/fitting.py`

已有方法：

- Hampel outlier 去除；
- Savitzky–Golay；
- multiscale Savgol（细/粗尺度融合）；
- knot-constrained spline（保留显著极值）；
- quintic spline（惩罚三阶导/曲率变化）；
- B-spline；
- LOESS。

这些是 Layer 2 可复用的**拟合原语**，尤其：

- `knot-constrained` 对应“保留褶皱转折”；
- `quintic` 对应“曲率变化惩罚”；
- Hampel 对应“去除局部尖刺”。

限制：

- 以 `y(x)` 单值 horizon 为前提，不能处理任意参数曲线、闭合体、断层错断；
- 没有拓扑/端点/薄层约束；
- 某些方法会改变 labels 或依赖 column repartition。

### 2.3 `horizon/occlusion.py`

`bridge_boundary_with_endpoint_tangents` 用两端切向做三次 Hermite 桥接，并对过冲和曲率变化设阈值。这个思路对 Layer 2 的“缺口/遮挡段”有价值，但它解决的是**桥接**，不是整条边界平滑；且阈值仍是像素尺度。

### 2.4 `horizon/refine.py`

该管线会对 labels 做重新分配（column repartition、label-space Gaussian blur），并可能合并/删除小碎片。它属于“分割后处理”，不是 Layer 2 几何平滑。其 quality gate（层数保持、碎片率、最小层厚）可以作为 Layer 2 的**校验指标**参考，但不能直接调用。

---

## 3. 文献地图

| 家族 | 代表工作 | 核心先验 | 对 Layer 2 的用途 | 主要风险 |
|---|---|---|---|---|
| 结构导向滤波 | Fehmers & Höcker 2003；Lavialle et al. 2007；Weickert 1998 | 沿同相轴平滑，跨断层不扩散 | 切向/法向各向异性；断层终止 | 直接作用于图像，可能移动边界 |
| 局部方向估计 | Fomel 2002；Wu & Janson 2017；Wang et al. 2008 | 平面波/结构张量/方向导数给出局部倾角 | 作为弱方向先验或置信度 | 陡倾角、无主导方向时不稳定 |
| 层位追踪 | Hale 2013；Yan & Wu 2021；Wu & Hale 2015；Wu & Zhong 2012；Zhou et al. 2023；Bugge et al. 2019 | 斜率约束、控制点、RGT、断层/不整合作为约束 | 端点和斜率约束、断层两侧独立 | 重新追踪会改变几何事实 |
| 最小曲率/样条 | Briggs 1974；Smith & Wessel 1990；Wahba 1990；Eilers & Marx 1996；Bookstein 1989；Mallet 1992 | 最小弯曲能、张力、惩罚样条、DSI | 分段样条、曲率惩罚、张力防过冲 | 全局拟合会跨 junction；最小曲率会产生虚假拐点 |
| 变分/曲率正则 | Mumford & Shah 1989；Rudin et al. 1992；Chan et al. 2003；Burger et al. 2007 | 分段光滑、TV、Euler elastica、曲率流 | 曲率变化惩罚、缺口桥接 | 全局重分割改变拓扑 |
| 主动轮廓/拓扑 | Caselles et al. 1997；Chan & Vese 2001；Han et al. 2003；Le Guyader & Vese 2008；Zeng et al. 2008；Vicente et al. 2008 | 拓扑保持、连通性、简单点 | 拓扑约束概念、junction pinning | 重新分割/改变 labels |
| 拓扑简化/制图 | Saalfeld 1999 | 简化时保持拓扑一致 | 边界网络简化、互交检查 | 只解决 2D 拓扑，不含地质先验 |
| 反演/速度建模 | Haber & Oldenburg 1997；Gallardo & Meju 2004；Lin & Huang 2017；Zhou 2006；Birdus 2007 | 结构耦合、界面引导、层状参数化、断层约束 | 界面作为硬/软约束；层状模型 | 反演机器过重，不是几何后处理 |
| 隐式地质建模 | Wellmann & Caumon 2018；Calcagno et al. 2008；Grose et al. 2021 | 断层切开插值域、step function、fault frame | 断层两侧独立平滑、拓扑切开 | 3D 建模，超出当前 2D 需求 |

---

## 4. 关键发现

### 4.1 结构导向平滑：沿结构平滑，不跨终止面

Fehmers & Höcker (2003) 的核心不是“高斯平滑”，而是三件事：

1. orientation analysis；
2. edge detection；
3. edge-preserving oriented smoothing。

其摘要明确说明：平滑平行于地震反射，**不跨越反射终止（断层）**。这直接支持 Layer 2 的第一个原则：

> 平滑方向应沿边界切向；边界的法向位移应受约束；断层/不整合/尖灭是平滑域的边界，不是 outlier。

Lavialle et al. (2007) 的 SFPD 用结构张量给出局部层方向，并用置信度控制扩散张量特征值，目标是在去噪的同时保护断层。Weickert (1998) 的 anisotropic diffusion 提供了 edge-enhancing / coherence-enhancing 的一般框架。

对我们的启发：

- 不需要对 labels 做二维扩散；只需把边界参数化为弧长曲线，在**一维切向**上平滑。
- 局部方向可以从 exact boundary 自身估计；图像结构张量只能作为弱证据，不能覆盖 labels。
- 在断层交点附近降低平滑强度，而不是把断层当噪声抹平。

### 4.2 最小曲率与张力：不要只优化曲率

Briggs (1974) 的 minimum curvature 用最小总曲率（biharmonic）插值，是地球物理网格化的经典方法。Smith & Wessel (1990) 指出：最小曲率面可能产生**大幅振荡和虚假拐点**；加入 tension 参数可以在同一线性系统内抑制过冲。

这对应 Layer 2 的一个关键设计：

- 只用 `∫||f''||²` 会把真实褶皱的 hinge 拉平，也可能在稀疏控制点之间产生过冲；
- 应同时使用 tension / 曲率变化惩罚 `∫||f'''||²` 或 P-spline 差分惩罚；
- 端点必须固定；junction 不能自由移动；
- 平滑强度应由数据拟合残差和几何约束共同决定，而不是固定 `s`。

Wahba (1990) 的 smoothing spline、Eilers & Marx (1996) 的 P-spline、Bookstein (1989) 的 thin-plate spline 都提供了“数据保真 + 弯曲能”的统一形式。Mallet (1992) 的 DSI 更进一步，把地质约束写成线性约束/正则项，适合“某些点必须过、某些方向必须满足”的场景。

### 4.3 层位追踪：斜率约束、控制点和断层/不整合约束

Hale (2013) 的动态图像 warping 用非线性累积对齐地震图像；Yan & Wu (2021) 用动态规划提取层位，控制点作为约束，可在断层/噪声处保持相位一致；Wu & Hale (2015) 用地震法向量构造层位体，并把解释控制点作为预条件约束；Wu & Zhong (2012) 在 3D graph-cut phase unwrapping 中显式加入 horizon 和 unconformity 约束；Zhou et al. (2023) 在追踪中加曲率阈值；Bugge et al. (2019) 用非局部道匹配处理断层错断层位。

这些工作说明：

- 层位几何的**斜率/曲率约束**是成熟做法；
- 断层和不整合必须作为**约束或屏障**，而不是普通平滑项；
- 少量控制点可以显著稳定全局结果。

但 Layer 2 不应重新追踪层位。我们已经有 labels 给出的 exact closure；追踪算法会引入新的解释，可能改变边界位置和拓扑。可借用的是：

- 端点切向/斜率约束；
- 断层两侧独立；
- 控制点（junction/端点）硬约束；
- 曲率阈值作为诊断，不作为删除理由。

### 4.4 断层与不整合：硬边界，不是噪声

多篇工作把断层作为插值域的边界：

- Fehmers & Höcker：平滑不跨越反射终止；
- Lavialle et al.：扩散张量保护断层；
- Wu & Zhong：不整合约束恢复相位不连续；
- Wu & Hale：控制点可在断层/不整合附近提高层位提取精度；
- Lin & Huang (2017)：interface-guided modified TV，把偏移成像得到的界面信息加入 FWI 正则，改善 sharp interfaces 和陡倾断层；
- Birdus (2007)：fault-constrained tomography，用断层约束速度模型；
- 隐式建模：Calcagno et al. (2008) 用 step function 表示断层两侧的标量场跳变；Grose et al. (2021) 用 fault frame 和 time-aware 顺序建模；Wellmann & Caumon (2018) 系统讨论了断层/不整合作为结构约束。

对 Layer 2 的直接结论：

1. 断层/不整合处的边界必须**切开**，两侧 segment 独立平滑；
2. 断层错距（offset）是硬约束，不能被平滑抹平；
3. 不整合可以有角度不整合，不能强制 C1 连续；
4. 若当前 labels 没有显式断层 label，只能用 junction/几何/图像证据标记“疑似断层”，并进入 review，不能自动假定。

### 4.5 拓扑保持：硬约束通常 NP-hard，工程上要“约束 + 校验”

Han et al. (2003) 用 digital topology 的 simple point 保持 level set 拓扑；Le Guyader & Vese (2008) 用 self-repelling snakes 防止轮廓合并/分裂；Zeng et al. (2008) 的 Topology Cuts 把拓扑作为全局约束嵌入 min-cut/max-flow，并证明该优化是 NP-hard，只能给近似解；Vicente et al. (2008) 的 connectivity prior 同样是 NP-hard，并用启发式求解。

Saalfeld (1999) 在 Douglas–Peucker 简化中加一个拓扑一致性测试，保证简化后的线不与自身或邻线冲突。这是非常实用的思路：

> 不需要全局拓扑优化；在每次几何修改后做局部拓扑校验，违规就拒绝/回退。

制图/地质图综合还提供了另一条重要规则：**共享边界只能平滑一次**，不能把相邻两个区域各自当成独立多边形去平滑，否则会产生缝隙/重叠，或者让薄层被两侧同时吃掉。因此 Layer 2 应以“边界对象/segment”为单位，而不是以“每个 label 的 polygon”为单位。

对 Layer 2：

- junction、panel 边缘端点、断层交点固定；
- 平滑后检查自交、互交、最小间距、区域/孔洞数、薄层厚度；
- 违规时局部降低平滑强度或回退 exact；
- 绝不通过删除边界来“修复”拓扑。

### 4.6 反演与速度建模：提供正则化思想，不提供直接算法

Haber & Oldenburg (1997) 的 joint inversion structural approach、Gallardo & Meju (2004) 的 cross-gradient、Lin & Huang (2017) 的 interface-guided TV、Zhou (2006) 的 deformable-layer tomography、Koren & Ravve (2005) 的 constrained velocity inversion，都属于“用界面/结构约束反演物性”的范畴。

可借鉴的只有三点：

- 界面位置可以作为正则化的引导，而不是被平滑掉；
- 层状参数化比网格参数化更符合地质；
- 平滑/TV/结构耦合的权重需要由数据与先验共同决定。

不应把 FWI/层析反演引入 Layer 2：输入是 labels 边界，不是地震波形；引入反演会重新解释数据，违背“几何后处理”的定位。

### 4.7 变分/曲率正则：可用于局部能量，不宜全局重分割

Mumford–Shah 提供 piecewise smooth 区域模型；TV 提供边缘保持；Euler elastica 和 anisotropic curvature flow 提供曲率/长度联合正则。这些思想可以转化为曲线能量：

- 长度项防止锯齿；
- 曲率项防止折线；
- 曲率变化项防止过冲；
- 数据项保持贴近 exact closure。

但全局 level set / TV 分割会改变区域拓扑和 labels，不适合 Layer 2。应只在**单段曲线**上最小化这些能量。

### 4.8 学习类方法：不作为主路径

DeepRGT、神经网络层位追踪等方法在断层/不整合场景有进展，但需要训练数据、存在泛化与幻觉风险。当前任务已经有 labels 和 exact closure，不需要学习“边界在哪里”；学习模型最多用于提供方向/置信度先验，不能替代几何约束。

---

## 5. 适用性判断

| 方法 | 是否直接用 | 原因 |
|---|---|---|
| 最小曲率 / 张力样条 | **是** | 单段曲线天然适用；可加端点/曲率约束 |
| smoothing spline / P-spline / DSI | **是** | 数据保真 + 弯曲能 + 线性约束 |
| knot-constrained / quintic | **是** | 保留褶皱 hinge、惩罚曲率变化 |
| Hampel / robust weights | **是** | 去除局部尖刺，但需避免把真实尖灭当 outlier |
| 结构张量 / plane-wave dip | **条件使用** | 只作为弱方向/置信度先验；不能覆盖 labels |
| 各向异性扩散（图像域） | **否** | 会移动边界、改变 labels；不是几何后处理 |
| DP / RGT / 层位追踪 | **否（可借约束）** | 会重新解释层位；我们的几何已由 labels 给出 |
| 水平集 / 主动轮廓 / graph cuts | **否** | 重新分割，改变 keep 集合和拓扑 |
| Topology Cuts / connectivity priors | **借概念** | 硬拓扑约束 NP-hard；采用局部校验 + 回退 |
| 拓扑保持简化（Saalfeld） | **是** | 简化/平滑时保持拓扑一致 |
| 断层约束层析 / interface-guided TV | **借概念** | 断层/界面作为硬边界；不引入反演 |
| 隐式地质建模 / fault frame | **借概念** | 断层切开插值域；当前 2D 不需要完整 3D 建模 |
| 学习类层位追踪 | **否（主路径）** | 需要训练、可能幻觉；不适合几何约束 |

---

## 6. 推荐的 Layer 2 设计

### 6.1 第一步：构建边界网络，而不是一堆独立曲线

从 exact boundary pixels 构建：

1. 对每个 label pair 的 boundary object 排序成 pixel chain；
2. 标记节点：
   - **junction**：3 个及以上 label 相接；
   - **endpoint**：panel 边缘；
   - **fault intersection**：与断层/不整合 mask 相交；
   - **pinch-out**：两条边界合并/消失处；
3. 在节点处切开，得到 **segment**；
4. 每个 segment 记录：
   - `labels_pair`；
   - 类别：`horizon` / `fault` / `unconformity` / `closed_body` / `panel_edge` / `review`；
   - 端点、端点切向；
   - 长度、曲率、相邻 segment；
   - 最小层厚/走廊宽度；
   - 置信度（来自 labels 审计、遮挡、碎片）。

没有节点切分的独立曲线平滑，是当前 `refine.py` 最大的结构性问题。

### 6.2 第二步：分段约束样条

对每个 segment 的参数点 `p_i`，拟合 `f(s)`（B-spline / P-spline）：

```
E(f) = Σ_i w_i ||f(s_i) - p_i||²
     + λ ∫ ||f''(s)||² ds
     + μ ∫ ||f'''(s)||² ds
     + ν · (端点/切向约束)
```

约束：

- `f(0) = p_start`，`f(1) = p_end`（硬约束）；
- 端点切向与相邻 segment 的夹角在允许范围内（软约束）；
- `||f(s) - p(s)|| <= d_max`（位移走廊）；
- 曲率 `κ(s) <= κ_max`（按 panel 尺度和 segment 类别归一化）；
- 闭合体用周期样条，并加面积/宽度保持项；
- robust 权重 `w_i` 降低局部尖刺影响，但不能把真实尖灭/断层错断当 outlier。

数据项的实现要注意：exact boundary 的像素点是有顺序的，但平滑允许沿切向轻微滑动。因此数据项更合理的做法是惩罚**法向位移**（到 exact boundary 的距离/距离场），而不是惩罚每个采样点到固定像素点的欧氏距离；后者会把切向滑动误判为误差，导致曲线被“钉”在 staircase 上。可以保留弧长对应作为正则，但数据项用 normal-band 距离。

建议：

- 用 **tension** 或三阶惩罚抑制最小曲率的过冲/虚假拐点；
- 对褶皱 hinge 使用 knot anchor 或降低局部平滑；
- 对薄层，`d_max` 取小于最小层厚的一半；
- 对 closed body，避免纯曲率流（会收缩），加面积保持。

### 6.3 第三步：全局几何/拓扑校验

对全部 segment 平滑结果做：

1. 自交检查；
2. 非相邻 segment 的最小距离/互交检查；
3. junction 端点一致性；
4. label adjacency graph 是否改变；
5. 每个 label 的连通分量数、孔洞数、面积/最小宽度；
6. 断层错距是否保持；
7. 位移残差分布（mean/p95/max）；
8. 曲率/曲率变化是否下降且未抹平 hinge。

为了检查区域拓扑，可以在内存中把平滑曲线栅格化回 labels 做 validation；该 raster 只用于校验，不回写、不作为新的事实来源。

处理策略：

- 违规 segment：局部降低平滑强度，重新拟合；
- 仍违规：回退到 exact geometry；
- 任何情况下都不删除 boundary，不改变 Layer 1 决定。

### 6.4 第四步：参数选择

不用 per-panel 手调。建议：

- 平滑尺度用 panel 尺寸归一化（例如相对短边的比例），而不是固定像素；
- 由 GCV / L-curve 在**有界范围**内选择 `λ`，并记录选择依据；
- 对类别使用全局参数（horizon/fault/closed body 各一套），不做 panel 特调；
- 平滑上限由几何约束（走廊、薄层、曲率）截断，而不是由“看起来更顺”决定；
- 所有参数和残差写入输出，便于跨 panel 审计。

### 6.5 类别策略

| 类别 | 平滑策略 |
|---|---|
| horizon/contact | 沿切向平滑；最小曲率 + 张力；junction 固定；保留褶皱 hinge |
| fault | 只沿断层迹线平滑；不跨断层；保持错距和端点 |
| unconformity | 低平滑强度；允许角度不整合；不强制 C1 |
| closed body（透镜/盐体/河道） | 周期样条；面积/宽度保持；小闭合体低平滑 |
| panel edge | 端点固定；只做单侧平滑；不把边界拉出 panel |
| review/unknown | 不自动平滑，保留 exact；等待视觉审计 |

### 6.6 与视觉审计的接口

审计图必须是：

- baseline：labels 重着色 mask；
- exact closure：细线/低透明度；
- smoothed：粗线、深色 halo；
- 颜色：keep=green，review=yellow，remove=red；
- 额外显示：位移残差、拓扑违规、junction 固定点、fault 段。

不要用降噪照片作为 baseline；照片只能辅助确认断层/不整合，不能替代 mask。

---

## 7. 验证与 benchmark

### 7.1 合成用例

构造带 ground truth 的 label masks：

- 水平/倾斜层；
- 褶皱（hinge、陡翼）；
- 正断层/逆断层（错距、断层带）；
- 角度不整合（truncation）；
- 尖灭/薄层；
- 透镜/河道（闭合、窄体）；
- 文字孔洞/遮挡；
- 像素 staircase 和 label 抖动。

每个用例有 exact 曲线和期望平滑曲线，可计算真实误差。

### 7.2 指标

| 指标 | 目的 |
|---|---|
| mean / p95 / max displacement | 平滑是否偏离 exact |
| symmetric Hausdorff | 最坏偏离 |
| curvature / curvature variation | 是否真的更顺 |
| length / area / width change | 是否吃薄层/收缩闭合体 |
| topology violation count | 自交、互交、区域/孔洞变化 |
| adjacency graph edit distance | 区域邻接是否改变 |
| fault offset error | 断层错距是否保持 |
| endpoint/junction error | 节点是否漂移 |
| visual audit preference | 地质合理性 |

### 7.3 评测协议

- 固定一套全局参数，在 11 个 curated panel 上做 leave-one-panel-out；禁止 per-panel 调参；
- 保护集：断层边界、junction、薄层、review 边界，单独报告违规率；
- 合成用例用来校准参数，真实 panel 只做留出验证；
- 视觉审计用双盲/强制选择：exact vs smoothed，哪条更像地质边界；
- 任何“平滑后更好看但拓扑变了”的结果判为失败。

### 7.4 初始验收假设

以下是待验证的初始假设，不是定论：

- 拓扑违规 = 0；
- 断层错距误差 = 0；
- 最大位移 < 最小层厚的一半；
- p95 位移在 1–2 px 量级（按 panel 尺度归一化）；
- 曲率/曲率变化显著下降，但褶皱 hinge 和尖灭点保留；
- 视觉审计中，不得出现“跨 junction 拉直”“断层被抹平”“薄层被吃掉”。

---

## 8. 研究空白与本项目的定位

1. **在本次检索范围内，没有发现现成的“labels 闭边界网络平滑”方法**。文献主要处理地震图像、层位点、速度模型或 3D 地质面。
2. **拓扑保持与地质先验很少结合**：Topology Cuts / topology-preserving level set 面向医学分割；GIS 拓扑简化不含断层/层序先验。
3. **缺少边界级 benchmark**：现有工作多用图像相似度或层位拾取误差，很少评估“平滑是否改变分区拓扑”。
4. **2D 论文图件的断层/不整合标注不完整**：很多 panel 没有显式 fault label，只能靠 junction/几何/图像证据，必须允许 `review`。
5. **3D 扩展未解决**：当前 curated 是 2D panel；3D 需要面片网络、fault frame、体积拓扑，不能直接套 2D 曲线方法。

因此本项目可以形成一个新的、范围明确的贡献：

> 对既有地质分区 labels 的 keep 边界，构建带断层/不整合切分的边界网络，做拓扑约束的分段最小曲率/张力样条平滑，并以 recolor mask 为 baseline 做边界级审计。

---

## 9. 建议的实施顺序

### Phase A：无断层特化的最小闭环

- exact boundaries → 排序/节点切分；
- 单段 smoothing spline / P-spline，端点固定，位移走廊；
- 自交/互交/区域拓扑校验；
- 违规回退 exact；
- recolor mask 粗线审计。

目的：先验证“分段 + 约束 + 校验”是否比当前全局 `refine.py` 好，不引入断层分类。

### Phase B：断层/不整合/闭合体类别

- 用 labels、junction、图像证据标记 fault/unconformity/closed body；
- 断层两侧独立平滑，保持错距；
- 闭合体周期样条 + 面积保持；
- 类别级全局参数，不做 panel 特调。

### Phase C：弱图像先验

- 结构张量/局部 dip 作为置信度和方向软约束；
- 只在 labels 几何不确定时使用；
- 不允许图像证据覆盖 labels 边界。

### Phase D：benchmark 与泛化

- 合成 ground truth + curated 留出；
- 保护集与违规率；
- 视觉审计；
- 固定参数跨 panel 验证。

---

## 10. 参考文献

1. Fehmers, G. C., & Höcker, C. F. W. (2003). Fast structural interpretation with structure-oriented filtering. *Geophysics*, 68(4), 1286–1293. https://doi.org/10.1190/1.1598121
2. Lavialle, O., Pop, S., Germain, C., Donias, M., Guillon, S., Keskes, N., & Berthoumieu, Y. (2007). Seismic fault preserving diffusion. *Journal of Applied Geophysics*, 61(2), 132–141. https://doi.org/10.1016/j.jappgeo.2006.06.002
3. Weickert, J. (1998). *Anisotropic Diffusion in Image Processing*. Teubner.
4. Fomel, S. (2002). Applications of plane-wave destruction filters. *Geophysics*, 67(6), 1946–1960. https://doi.org/10.1190/1.1527095
5. Wu, X., & Janson, X. (2017). Directional structure tensors in estimating seismic structural and stratigraphic orientations. *Geophysical Journal International*, 210(1), 534–548. https://doi.org/10.1093/gji/ggx194
6. Wang, Y. E., Luo, Y., & Alfaraj, M. (2008). Inverse-vector approach for smoothing dips and azimuths. *Geophysics*, 73(6), P9–P14. https://doi.org/10.1190/1.2976157
7. Hale, D. (2013). Dynamic warping of seismic images. *Geophysics*, 78(2), S105–S115. https://doi.org/10.1190/geo2012-0327.1
8. Yan, S., & Wu, X. (2021). Seismic horizon extraction with dynamic programming. *Geophysics*, 86(2), IM51–IM62. https://doi.org/10.1190/geo2020-0039.1
9. Wu, X., & Hale, D. (2015). Horizon volumes with interpreted constraints. *Geophysics*, 80(2), IM21–IM33. https://doi.org/10.1190/geo2014-0212.1
10. Wu, X., & Zhong, G. (2012). Generating a relative geologic time volume by 3D graph-cut phase unwrapping method with horizon and unconformity constraints. *Geophysics*, 77(4), O21–O34. https://doi.org/10.1190/geo2011-0351.1
11. Zhou, J., Xu, H., Yang, P., & Yang, M. (2023). Automatic horizon tracking algorithm with curvature constraint. *Annals of Geophysics*, 66(1), PE117. https://doi.org/10.4401/ag-8899
12. Bugge, A. J., Lie, J. E., Evensen, A. K., Faleide, J. I., & Clark, S. (2019). Automatic extraction of dislocated horizons from 3D seismic data using nonlocal trace matching. *Geophysics*, 84(6), IM77–IM86. https://doi.org/10.1190/geo2019-0029.1
13. Briggs, I. C. (1974). Machine contouring using minimum curvature. *Geophysics*, 39(1), 39–48. https://doi.org/10.1190/1.1440410
14. Smith, W. H. F., & Wessel, P. (1990). Gridding with continuous curvature splines in tension. *Geophysics*, 55(3), 293–305. https://doi.org/10.1190/1.1442837
15. Wahba, G. (1990). *Spline Models for Observational Data*. SIAM. https://doi.org/10.1137/1.9781611970128
16. Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89–121. https://doi.org/10.1214/ss/1038425655
17. Bookstein, F. L. (1989). Principal warps: thin-plate splines and the decomposition of deformations. *IEEE TPAMI*, 11(6), 567–585. https://doi.org/10.1109/34.24792
18. Mallet, J.-L. (1992). Discrete smooth interpolation in geometric modelling. *Computer-Aided Design*, 24(4), 178–191. https://doi.org/10.1016/0010-4485(92)90054-E
19. Mumford, D., & Shah, J. (1989). Optimal approximations by piecewise smooth functions and associated variational problems. *Communications on Pure and Applied Mathematics*, 42(5), 577–685. https://doi.org/10.1002/cpa.3160420503
20. Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. *Physica D*, 60(1–4), 259–268. https://doi.org/10.1016/0167-2789(92)90242-F
21. Chan, T. F., Kang, S. H., & Shen, J. (2003). Euler's elastica and curvature-based inpainting. *SIAM Journal on Applied Mathematics*, 63(2), 564–592. https://doi.org/10.1137/S0036139901390088
22. Burger, M., Haußer, F., Stöcker, C., & Voigt, A. (2007). A level set approach to anisotropic flows with curvature regularization. *Journal of Computational Physics*, 225(1), 183–205. https://doi.org/10.1016/j.jcp.2006.11.026
23. Caselles, V., Kimmel, R., & Sapiro, G. (1997). Geodesic active contours. *International Journal of Computer Vision*, 22(1), 61–79. https://doi.org/10.1023/A:1007979827043
24. Chan, T. F., & Vese, L. A. (2001). Active contours without edges. *IEEE Transactions on Image Processing*, 10(2), 266–277. https://doi.org/10.1109/83.902291
25. Han, X., Xu, C., & Prince, J. L. (2003). A topology preserving level set method for geometric deformable models. *IEEE TPAMI*, 25(6), 755–768. https://doi.org/10.1109/TPAMI.2003.1201824
26. Le Guyader, C., & Vese, L. A. (2008). Self-repelling snakes for topology-preserving segmentation models. *IEEE Transactions on Image Processing*, 17(5), 767–779. https://doi.org/10.1109/TIP.2008.919951
27. Zeng, Y., Samaras, D., Chen, W., & Peng, Q. (2008). Topology cuts: A novel min-cut/max-flow algorithm for topology preserving segmentation in N-D images. *Computer Vision and Image Understanding*, 112(1), 81–90. https://doi.org/10.1016/j.cviu.2008.07.008
28. Vicente, S., Kolmogorov, V., & Rother, C. (2008). Graph cut based image segmentation with connectivity priors. *CVPR*. https://doi.org/10.1109/CVPR.2008.4587440
29. Saalfeld, A. (1999). Topologically consistent line simplification with the Douglas-Peucker algorithm. *Cartography and Geographic Information Science*, 26(1), 7–18. https://doi.org/10.1559/152304099782424901
30. Haber, E., & Oldenburg, D. (1997). Joint inversion: a structural approach. *Inverse Problems*, 13(1), 63–77. https://doi.org/10.1088/0266-5611/13/1/006
31. Gallardo, L. A., & Meju, M. A. (2004). Joint two-dimensional DC resistivity and seismic travel time inversion with cross-gradients constraints. *Journal of Geophysical Research: Solid Earth*, 109, B03311. https://doi.org/10.1029/2003JB002716
32. Lin, Y., & Huang, L. (2017). Building subsurface velocity models with sharp interfaces using interface-guided seismic full-waveform inversion. *Pure and Applied Geophysics*, 174(11), 4035–4055. https://doi.org/10.1007/s00024-017-1628-5
33. Zhou, H.-W. (2006). Multiscale deformable-layer tomography. *Geophysics*, 71(3), R11–R19. https://doi.org/10.1190/1.2194519
34. Birdus, S. (2007). Removing fault shadow distortions by fault-constrained tomography. *SEG Technical Program Expanded Abstracts*. https://doi.org/10.1190/1.2793102
35. Koren, Z., & Ravve, I. (2005). Constrained velocity inversion. *SEG Technical Program Expanded Abstracts*. https://doi.org/10.1190/1.2148174
36. Wellmann, F., & Caumon, G. (2018). 3-D structural geological models: Concepts, methods, and uncertainties. *Advances in Geophysics*, 59, 1–121. https://doi.org/10.1016/bs.agph.2018.09.001
37. Calcagno, P., Chilès, J.-P., Courrioux, G., & Guillen, A. (2008). Geological modelling from field data and geological knowledge: Part I. *Physics of the Earth and Planetary Interiors*, 171(1–4), 147–157. https://doi.org/10.1016/j.pepi.2008.06.013
38. Grose, L., Ailleres, L., Laurent, G., & Jessell, M. (2021). LoopStructural 1.0: time-aware geological modelling. *Geoscientific Model Development*, 14, 3915–3937. https://doi.org/10.5194/gmd-14-3915-2021

---

## 附录：检索说明

检索时间：2026-10-21。

数据库/来源：Crossref、OpenAlex、SEG/Geophysics、JGR、GJI、IEEE TPAMI/TIP、CVPR、CVIU、SIAM、GMD、arXiv、LoopStructural/GIS 文档。

主要检索词：

- seismic horizon smoothing / tracking / curvature constraint / dynamic programming
- structure-oriented filtering / anisotropic diffusion / structure tensor / dip estimation
- fault-preserving smoothing / fault-constrained tomography / unconformity constraints
- minimum curvature / splines in tension / thin-plate spline / P-spline / discrete smooth interpolation
- topology preserving segmentation / topology cuts / connectivity prior
- topology preserving line simplification / geological map generalization
- interface-guided full-waveform inversion / cross-gradient / layer tomography
- implicit geological modeling / fault frame / structural geological models

纳入标准：与边界/层位/界面的平滑、几何约束、断层/不整合处理、拓扑保持或速度模型界面约束直接相关；优先同行评议论文和经典专著。
排除标准：纯地震采集/成像算法、与边界几何无关的机器学习性能研究、缺少方法细节的专利/商业宣传。
