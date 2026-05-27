# geoseg 项目 Ruflo CLI 使用指南

> 配置时间：2026-05-18
> 用途：开发稳定与质量保障（Phase 0 单人开发）
> 方式：**CLI 命令**（MCP 已移除，避免上下文挤占）

---

## 已启用的 CLI 功能

| 类别 | CLI 命令 | 用途 |
|------|----------|------|
| **Memory** | `claude-flow memory store/search` | 存储设计决策、Schema 变更、bug 修复记录 |
| **AIDefence** | `claude-flow security scan` | 扫描 API key 泄露、prompt 注入风险 |
| **Task** | `claude-flow task create/list` | 跟踪开发进度 |
| **Performance** | `claude-flow performance bottleneck` | 识别性能瓶颈 |
| **Hooks** | `claude-flow hooks post-task` | 记录任务完成质量 |

## 明确不用的功能

- 不开 swarm / agent 团队（Phase 0 单人）
- 不启用 hive-mind  collective intelligence
- 不启用 coordination consensus

---

## 预置 Memory（已存储）

```bash
# 查看项目核心记忆
claude-flow memory search --namespace geoseg ""
```

| Key | 内容 |
|-----|------|
| `geoseg-design-principles` | v1 教训 + 6 条核心设计原则 |
| `geoseg-json-schema-contracts` | page_analysis + segmentation_result Schema 契约 |
| `geoseg-dev-sequence` | 8 步开工顺序 + 模块行预算 |
| `geoseg-explicitly-not-doing` | Phase 0 不做清单 |

---

## 使用场景

### 1. Schema 变更时 — 存储决策

```bash
claude-flow memory store \
  --namespace geoseg \
  --key "schema-change-YYYY-MM-DD" \
  --value "改了 colorbar.bbox 为可选字段。原因：xxx。影响的 consumer：agent/prompts.py, core/pipeline.py"
```

### 2. 写代码前 — 检索设计约束

```bash
claude-flow memory search \
  --namespace geoseg \
  --query "QThread worker 重建"
```

### 3. 提交前 — 安全检查

```bash
claude-flow security scan --input "$(git diff)"
```

### 4. 跟踪开发进度

```bash
claude-flow task create \
  --type feature \
  --description "实现 core/cv_detect.py 连通域检测" \
  --priority high
```

### 5. 性能回归

```bash
claude-flow performance bottleneck --component "e026 segmentation"
```

---

## 配置位置

| 文件 | 作用 |
|------|------|
| `~/.claude.json` → `/Users/daiduo2/geoseg` | 项目级 Claude Code 配置（MCP 已移除） |
| `geoseg/.claude/settings.json` | 项目级权限配置 |
| `geoseg/.claude/RUFLO.md` | 本使用指南 |
