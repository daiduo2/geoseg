# iteration — agent-driven audit sandbox

> **核心原则**：固定的是流程基础设施，不是动作。
> agent 自己决定换引擎、加层数、裁图、合并、删文字层或 refine；系统负责裁剪、审计、记录和复审计。

## 为什么不再做固定动作策略

固定策略会把 sandbox 变成一条窄管道。ph01 的失败就是例子：iterative 结果虽然补出了区域，但把 Y 轴一起分进去了；这是 **panel 预裁剪** 问题，不是“再加一层/再换一个 engine”的规则能解决的。agent 看图后应直接要求裁剪，而不是让审计循环去猜。

## 固定基础设施

### 1. Panel 预裁剪（强制）

`preprocess.py`

- `detect_panel_bbox(image)`：用饱和度/非白内容估计数据区 bbox，去掉白边、刻度、坐标轴文字。
- `crop_panel(image, bbox=None)`：返回 `CropResult`，包含裁剪图、`bbox (x0,y0,x1,y1)`、offset 和 method。
- 所有 engine 尝试必须使用裁剪后的 panel；记录 bbox，后续坐标可加回 offset。
- 自动 bbox 不对时，agent 直接给 manual bbox；ph01 的正确 bbox 是 `(28, 0, 258, 335)`。

### 2. 审计（证据，不是判决）

`audit.py`

- 每个 label 检查：
  - 内部是否存在两个显著主色（缺层/合并层）；
  - 白色/黑色文字、背景污染；
  - 同标签的碎片数；
  - 标签数相对 expected min/max。
- 输出 `AuditReport`：under-segmented labels、text labels、fragmented labels、text pixel fraction、score、suggested_action。
- `suggested_action` 只是建议，不是自动执行策略。

### 3. Session（记录所有尝试）

`session.py`

```python
from geoseg import api

session = api.IterationSession(
    image,
    output_dir="runs/sandbox/ph01",
    crop_bbox=(28, 0, 258, 335),
    audit_config=api.AuditConfig(expected_min_layers=8),
)
a1 = session.run_attempt("v4_kmeans", 8, note="first pass")
# agent inspects overlay + audit, then chooses any next attempt:
a2 = session.run_attempt("v4_kmeans", 10, note="more layers after missing-layer audit")
# record manual cleanup/merge/refine as its own attempt:
a3 = session.record_attempt(cleaned_labels, engine="manual", note="remove text label")
session.accept(a2.attempt)
session.save_best()
```

每个 attempt 自动保存：

- `labels.npz`
- `overlay.jpg`
- `audit.json`
- `session.json`（crop、attempt 参数、审计、接受项、artifact 路径）

没有自动 `run_iterative_segmentation` 动作树；agent 用 audit/overlay 自由试错，但每次都必须落盘和复审计。

## 推荐 agent 流程

1. 先裁剪 panel，保存 `crop.json`，确认没有轴/刻度/边距。
2. 运行初始 engine，读 overlay 和 audit。
3. 若 audit 报缺层，agent 可：增加 n_layers、换 engine、或对特定 label 做颜色拆分。
4. 若报文字/背景，agent 可删 label、清 pixel 或重新裁剪；不要误删真实白色地层。
5. 任何修复后用 `record_attempt` 记录并重新 audit。
6. 视觉判断通过后 `accept`，保存 best。

## 测试

```bash
uv run pytest tests/modules/iteration -q
```

`test_sandbox_session.py` 覆盖裁剪、audit、session 记录、accept、compact labels。
