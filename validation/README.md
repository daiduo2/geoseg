# validation — 分割实例语料

从 `runs/` 中积累的历史分割结果整理出的验证集，用于评估多尺度分区、界面参数化、控制点编辑这条主线。**它是对照基线，不是像素级 ground truth。**

## 目录

```
validation/
├── manifest.jsonl        123 个可用 panel（图像 + 标签 + 元数据）
├── labels_only.jsonl     74 个有标签但找不到对齐原图的 panel（仅做标签级分析）
├── images/<id>.jpg       干净 panel 图，和标签逐像素对齐
├── labels/<id>.npz       int32 标签，key = "labels"
├── overlays/<id>.jpg     历史 overlay（若有）
├── splits/{dev,holdout,rare,negative}.txt
└── report.json           统计、缺口、图像来源、尺寸校验结果
```

## 可信度分层（`tier`）

| tier | 含义 | 数量 |
|---|---|---|
| `human` | 人工编辑/接受（目前 0 个） | 0 |
| `audited` | agent/VLM 审阅或人工挑选过的结果（fig6 精选、j_tecto 清理、agent_review_vlm、readme 示例） | 46 |
| `auto` | 流水线自动输出，未经审阅 | 77 |

`audited` 不等于人工确认，只表示经过了一次 agent 视觉审阅或人工挑选。

## 划分（`split`）

| split | 用途 | 数量 |
|---|---|---|
| `dev` | 调参、开发 | 50 |
| `holdout` | 只用于最终对比，不参与调参 | 20 |
| `negative` | 只分出 1 个区域的失败案例，用于检验欠分割检测/修复 | 51 |
| `rare` | 启发式判定含薄层的样本，用于检查稀有地层是否被先验抹掉 | 2 |

`rare` 目前只有 2 个，**不足以验证稀有地层**。薄层和透镜体需要用合成数据补，或在人工审计时手工标注更多（见下文）。

## 标签约定（`labels.convention`）

| 约定 | 含义 | 数量 |
|---|---|---|
| `zero_is_background` | 0 = 背景，1..N = 区域（主流水线输出） | 116 |
| `zero_is_region` | 0 也是区域（fig6 精选分区） | 5 |
| `negative_is_background` | -1 = 面板外，0..N = 区域（j_tecto） | 2 |

使用前必须读 `manifest.jsonl` 里的 `labels.convention`，不要假设 0 一定是背景。

## 字段

每条 manifest 记录关键字段：

- `panel_id`：稳定 ID，推荐作为缓存/结果文件名
- `tier` / `split`
- `source`：原始 label 路径、图像路径、paper、figure hash、面板序号、`image_source`
- `labels`：`n_labels`、`background_fraction`、`tiny_fragments`、`thin_layers`、`min_area_fraction`、`shape`、`convention`、`sha256`
- `flags`：`single_region`、`thin_layer_candidate`、`many_tiny_fragments`、`large_background`、`image_label_shape_mismatch`（应为 0）
- `variants`：同一 panel 的其他引擎/运行结果（label 路径 + n_labels + 碎片数）
- `files`：语料内相对路径 + sha256

`source.image_path` 对 `redetected_crop` 记录可能指向临时裁剪文件，仅作溯源；实际使用 `files.image`。

## best 子集（按时间/迭代次数筛出）

`validation/best/` 不是人工挑“好看”的图，而是按历史迭代密度筛出的成熟样本：

- 5 个 `newimage_p0..p4`：2026-06-06 → 2026-07-06，11–12 个 campaign、15–37 次运行，最终稳定在 5 区域结果；
- 每个 panel 选的是**稳定的 merge/visual-audit 版本**，不是最后一次过度分割的 `labels_best_split_v3`（后者 8–12 区域）；
- 查看证据：每个 panel 有 11 个查看目录、10 个 overlay-with-legend、5 个 comparison render；
- 导出证据：本次补做了 SPECFEM 导出，`tomo.xyz` 在 `~/geoseg-artifacts/2026-10/best_exports/`，`Par_file` snippet / properties / polygons 在 `validation/best/exports/`；
- 注意：历史数据里“导出”和“查看”是两条分离的流水线（432 个 tomo 集中在 agent_review/literature_test，查看集中在 fig6/newimage），所以 best 是“被频繁查看的集合 + 本次补导出”；
- 物性目前是 `synthetic_gradient_placeholder`，不是从色条反演的真实速度；
- 未纳入：`feng_fig6_p03..p07`（迭代 37–43 次，但最新结果碎片化）、`jtecto`（13–16 区域，过复杂）、silixa 系列（2 层，参考价值低）、3D schematic。

```bash
make best          # 重建 validation/best
make history       # panel -> campaign/run 时间线
make review        # 重新生成带编号的 review sheets
```

`best/` 内的 `manifest.jsonl` 记录每个 panel 的 campaign 数、run 数、首次/最后日期和所用版本。复杂案例可加 `--include-optional`（c11b8db、jtecto fig7/fig6b）。

## curated 子集（来自 agent session 的人工干预记录）

`validation/curated/` 是从 Codex session（2026-07-16/17）里找到的、你实际多轮纠正并固化的结果：

| panel | 结构 | 区域数 | session 证据 |
|---|---|---:|---|
| `paper22_fig5a` | layered | 8 | OCR ROI + 残差分类 + `label_blur sigma=2.5`，并定向追踪 2.91/3.3 界面；带真实密度 2.65–3.5 g/cm³ |
| `paper22_fig6` | boundary_driven | 6 | 你指出 fig6 的边界由短黑/红线段决定而非颜色；边界驱动 watershed 为最终结果，boundary alignment 0.9306、碎片 0 |
| `ph01_page8_panel3` | layered | 7 | 2026-06-20 你要求把 ph01 page8 放进 PPT；干净图从 fixture Page 8 第二行结构模型重建，重建 overlay 与历史 overlay 一致；沿用历史 tomo/Par_file |
| `paper23_fig7a` | faulted | 15 | 11 步干预历史：图例吸收、F3 右延伸、谷底 vs 内部交点修正，最终 `reviewed/` |
| `paper23_fault_guided` | faulted | 14 | 你指出图例误分割和 3 处漏边界；最终 color labels + red barriers + tangent completion |

另外从 session 里补进了 3 个你明确接受/称赞过的 3D 示意图结果：

| panel | 结构 | 区域数 | session 证据 |
|---|---|---:|---|
| `schematic_panel1` | schematic_3d | 5 | 2026-06-19 接受 panel2 后要求用同一 workflow 重跑 panel1；这是重跑结果 |
| `schematic_panel2` | schematic_3d | 4 | “我接受，你先把当前的 panel2 归档到此前的最佳 overlay 目录下” |
| `schematic_panel3_front` | schematic_3d_front | 6 | “当前的效果远远优于此前任何一次”；front-face fused 结果 |

说明：`schematic_*` 是 3D 羽流/示意图压力样本，不参与 2D 层状界面测试；`paper22_fig5a`/`fig6` 仍是最干净的主验证集。`ph01_page8` 只找到 labels + overlay + tomo，干净原图缺失，暂不纳入。fig6 的后续迭代你已明确说“问题越改越多”，所以不纳入 best。

你在做 PPT 时选的三个例子正是 **newimage + paper22 + paper23**；加上 3D 示意图后，showcase 覆盖 2D 层状、边界驱动、断层、3D 羽流四类。

```bash
make curated     # 重建 validation/curated 并补导出
```

## 使用

```python
import json
from pathlib import Path
import numpy as np
from PIL import Image

root = Path("validation")
rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
dev = [r for r in rows if r["split"] == "dev"]

for r in dev:
    image = np.array(Image.open(root / r["files"]["image"]).convert("RGB"))
    labels = np.load(root / r["files"]["labels"])["labels"]
    assert image.shape[:2] == labels.shape
```

或用现成加载器：

```bash
uv run python scripts/dev/corpus.py stats
uv run python scripts/dev/corpus.py list --split dev
uv run python scripts/dev/corpus.py check      # 文件存在、尺寸、hash 校验
```

## 重建

```bash
make corpus          # 重新扫描 runs/ 并生成 validation/
make corpus-check    # 校验语料完整性
```

扫描逻辑在 `scripts/dev/build_corpus.py`。它只读 `runs/`（`~/geoseg-artifacts/2026-10/runs`），不会修改历史产物；同一图像通过 sha256 二次去重，同一 panel 的不同引擎结果收进 `variants`。

## 已知缺口

1. **没有像素级人工 GT**：`audited` 只是 agent/人工挑选，不是逐像素标注。
2. **51 个负例**：这些 panel 的参考结果只分出 1 个区域，属于当前流水线的欠分割失败；它们适合检验“能否自动发现并修复”，不适合当正样本。
3. **rare 只有 2 个**：薄层/透镜体覆盖不足，必须在合成基准里补齐。
4. **74 个 labels-only**：原文图像已找不到，只能用标签做界面级分析，不能做重新分割。
5. **图像来源混合**：`clean_crop` 38、`redetected_crop` 85；另有 74 个 panel 找不到对齐原图，只能进 `labels_only.jsonl`。`redetected_crop` 是用当前 panel detector 从原始图重新裁的；若与当年 detector 版本有 1px 级差异，已在 `image_label_shape_mismatch` 中检查（当前为 0）。

## 建议的验收协议

1. 在 `dev` 上开发，在 `holdout` 上只做最终对比；
2. 核心指标用 `core`（n_labels ≥ 2）：边界 F1、控制点数、无编辑 roundtrip 幂等；
3. 用 `negative` 检验欠分割是否被识别为失败，而不是被“修复”成一个假的多层结果；
4. 用 `rare` + 合成薄层/透镜体检验几何先验是否抹掉稀有结构；
5. 任何物理先验实验都应跑 `prior_mode=off` 的对照，确认稀有地层指标不变。

## 下一步建议

- 从 `dev` 里挑 20 个 panel 做人工界面标注（10 个 core、5 个 negative、5 个含薄层），把 `tier` 升级为 `human`；
- 把 3D 示意图作为独立的 stress 集，不与 2D 分层语料混在一起；
- 每次新方法实验都把结果写回 `runs/`，并在 manifest 中追加新候选，而不是覆盖本语料。
