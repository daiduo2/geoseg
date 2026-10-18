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
