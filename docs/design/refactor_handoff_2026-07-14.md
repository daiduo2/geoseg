# geoseg 重构交接单

> 作用：给下一位 session 接手用的状态快照。这里只记录当前这条重构主线，不试图解释全部历史。

## 当前判断

项目已经从“实验驱动堆叠”进入“必须收束边界”的阶段。当前工作树里同时存在较大范围的结构整理与兼容层迁移，说明重构已经不是局部修补，而是要先把分层和依赖方向定住，再继续扩张功能。

## 当前进展

### 已落地

1. `segment_engines` 的内部工具开始拆分到更细的 private 模块。
2. `regional_fusion` 不再直接暴露/依赖某些底层实现细节，而是改走公共 facade。
3. `visual_audit/rendering.py` 已改为只依赖公开入口，不再碰 engine 内部实现。
4. 已新增针对区域标签重排的公共入口 `segment_engines/regions.py`。
5. 已补了 import boundary 测试，开始用测试约束“公共入口 vs 内部实现”的边界。

### 进行中

1. `full_pipeline` 仍然是兼容 facade，但它现在只是第一步，后面要继续拆成真正的分层入口。
2. `pipeline.segment` / `pipeline.stages` / `controller.py` 之间的职责还需要继续收口。
3. `segment_engines` 内仍有不少历史兼容导入，需要逐个迁移到稳定 facade。
4. `docs/CODEBASE.md` 已在往新分层方向收敛，但还没完全和代码状态同步。

## 这次主线的目标

### Goal 1: 继续拆 `full_pipeline`

把旧式“一个入口串完 classify -> detect -> review -> segment -> export”的模式，拆成更清晰的 stage 编排。

目标状态：

- `full_pipeline.py` 只保留兼容层语义
- 真正的流程编排放到 `pipeline/` 下
- 上游调用只依赖稳定 stage/API，不直接黏住历史入口

### Goal 2: 继续做模块解耦

优先把这些边界固定住：

- `segment_engines` 对外只暴露稳定 facade
- `internal/*` 只给模块内部使用
- `visual_audit` 只读公开渲染接口
- `scripts/`、`examples/` 不再把实验 helper 当成产品 API 用

### Goal 3: 让测试替代口头约束

继续补 import boundary / compat / facade 测试，避免以后再把内部实现直接暴露出去。

## 当前值得改的架构问题

1. 旧入口太多，语义重复。
2. 兼容层和产品层边界还不够硬。
3. `segment_engines` 里既有稳定 API，又有实验 helper，容易误导调用方。
4. `pipeline` 和 `controller` 的职责边界还可以更清晰。
5. `scripts/` 里大量历史试验脚本仍在直接依赖具体引擎实现。
6. 文档已经开始向新架构描述靠拢，但还没完全和代码同步。

## 建议下一步怎么做

1. 先把 `full_pipeline` 的调用链梳理完整，列出当前谁在调用它。
2. 把调用方逐个迁移到 `pipeline.segment` 或新的 stage facade。
3. 把 `full_pipeline` 最终压成纯 compatibility shim。
4. 再清理 `segment_engines` 内部的旧导入和重复 helper。
5. 最后补一轮针对边界的测试和文档同步。

## 交接备注

- 当前工作树里还有大量未提交改动，且不全是本轮新增；这是一个持续中的大重构现场。
- 新 session 不要先追求一次性收完，先沿着 `full_pipeline -> pipeline -> stable facade` 这条线继续收口。
- 如果要继续改代码，优先保持向后兼容，先迁移调用，再删除旧路径。

