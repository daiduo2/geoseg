# Agent-driven segmentation audit sandbox

> 2026-10-21。把此前临时进行的“跑一次 → 看图 → 改参数 → 再跑”固化为可回放的基础设施。
> **固定的是流程，不是动作。** 不做自动 repair 策略树。

## 1. 为什么不是固定动作

固定动作策略会把 sandbox 变成窄管道。ph01 的迭代结果补出了新区域，却把左侧坐标轴也分进去了；这不是通过“再加一层/换一个 engine”的规则能修的，而是 **panel 进入引擎前没有裁到数据区**。agent 看图后应直接要求裁剪，而不是让审计循环猜。

因此当前实现只固定：

- 裁剪前置；
- 每次引擎尝试立即审计；
- 每次尝试/修复落盘；
- 修复后必须复审计；
- accepted attempt 可追溯。

动作选择完全交给 agent。

## 2. Panel 预裁剪

`modules/iteration/preprocess.py`

- `detect_panel_bbox(image)`：饱和度/非白内容的 bbox；
- `crop_panel(image, bbox=None)`：返回裁剪图、bbox、offset、method；
- 强制所有尝试使用裁剪图；
- 自动 bbox 不对时，agent 直接给 manual bbox；
- ph01 page8 panel3 已知正确 bbox：`(28, 0, 258, 335)`，去掉左侧深度轴。

## 3. 审计证据

`modules/iteration/audit.py`

每个 label 输出：

- 是否包含两个显著主色（缺层/合并层）；
- 白/黑文字、背景占比；
- 同标签碎片数；
- 与 expected min/max 层数的关系；
- `AuditReport` + `suggested_action`。

`suggested_action` 只是建议。agent 可以：

- 如果缺层：提高 n_layers、换 engine、或直接按颜色拆 label；
- 如果文字/轴：重裁 panel、删 label、清 pixel；
- 如果碎片：合并/过滤；
- 如果已经正确：accept。

## 4. Session 记录

`modules/iteration/session.py`

```python
from geoseg import api

session = api.IterationSession(
    image,
    output_dir="runs/sandbox/ph01",
    crop_bbox=(28, 0, 258, 335),
    audit_config=api.AuditConfig(expected_min_layers=8),
)
a1 = session.run_attempt("v4_kmeans", 8, note="initial cropped pass")
a2 = session.run_attempt("v4_kmeans", 10, note="split missing layers")
session.accept(a2.attempt)
session.save_best()
```

每个 attempt 保存 labels/overlay/audit/params/note/hash；`session.json` 记录 crop、attempt、accepted attempt。

## 5. ph01 现状

- full panel（258×335）审计：text pixel fraction **8.3%**，v4 n=10 把坐标轴当成区域；
- 裁剪 `(28,0,258,335)` 后：text pixel fraction **0.17%**，坐标轴消失；
- 裁剪后 v4 n=8/10/12 仍存在缺层/内部文字线；agent 需要继续选择更高层数、换 engine 或清理内部文字，而不是固定规则自动接受。
- curated 6 层标签本身也缺层：label 2 合并浅蓝+灰紫，label 4 合并棕色+灰褐。

## 6. 测试

```bash
uv run pytest tests/modules/iteration -q
```

覆盖自动/手动裁剪、坐标轴排除、缺层审计、文字 label 审计、session 尝试记录、accept、compact labels。

## 7. 下一步

- 把 panel crop 的自动 bbox 接入 agent skill 的强制 preflight；
- 让 agent 的每次 repair 通过 `record_attempt` 落盘；
- 对裁剪后的 ph01 继续做 agent 选择的尝试（增加层数/换 engine/去内部文字）；
- 最后才进入“从正确 labels 派生边界并合并/矫正/消除”。
