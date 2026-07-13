# e026_algo — 已弃用（Deprecated）

> **v2 不再维护**。`segment_engines/` 已完全替代本模块功能。保留本目录仅作 v1 算法历史参考。
>
> 日期: 2026-05-20

## 来源

**已从 `~/.claude/skills/geo-segment/lib/` 复制进项目**：
- `core.py` ← skill 的 e026 算法主体
- `components.py` ← 子组件

**不再依赖外部 skill 路径**。修改时只改本目录文件。

## 升级 skill 时

如果 `~/.claude/skills/geo-segment/` 有新版本需要同步：
1. **手动复制**（不要软链 / 动态加载 / sys.path hack）
2. 同步后跑 `demo.py` + `demo_m3.py` 确认无回归
3. 在本文件下方记录同步日期 + skill 版本

## 同步记录

| 日期 | skill 版本 / commit | 同步范围 | 验收 |
|------|---------------------|----------|------|
| _(待首次同步时填) _ | — | — | — |

## 测试

```bash
python -m geoseg.modules.e026_algo.demo
python -m geoseg.modules.e026_algo.demo_m3
```

## 不做

- 不依赖 `~/.claude/skills/geo-segment/lib/` 的任何路径（已 deny 在 `.claude/settings.json`）
- 不直接修改 `components.py` 中从 skill 复制来的公共算法逻辑而不留同步记录
