# geoseg Skills

Project-level skills for geophysics figure segmentation.

## Skill Index

| Skill | Purpose | Entry Point |
|-------|---------|-------------|
| `geo-segment` | End-to-end: figure → SPECFEM model | Single figure processing |
| `figure-classify` | Classify if a figure is a valid velocity model | Standalone classification |
| `sandbox-segment` | Autonomous panel segmentation (agent selects engines) | Single panel, agent-driven |
| `batch-segment` | Batch process a directory (≤5 parallel agents) | Multiple figures |
| `module-demo` | Run a module's demo.py to verify it works | Module testing |
| `schema-bump` | Schema change protocol with consumer sync | Schema updates |

## Usage

Skills are activated by Claude Code context matching. You can also reference
them explicitly in prompts:

- "Classify this figure" → `figure-classify`
- "Segment this panel" → `sandbox-segment`
- "Process all figures in this directory" → `batch-segment`
- "Convert this figure to SPECFEM" → `geo-segment`

## Architecture

```
geo-segment (end-to-end orchestrator)
    ├── figure-classify (agent Read 看图 → 分类 JSON)
    ├── cv_detect (panel detection — Bash 运行 Python 工具)
    ├── sandbox-segment (agent 自主分割：选引擎 → 评估 → 融合)
    │   ├── strategy_memory (历史策略查询)
    │   └── metrics (客观指标辅助)
    └── post_process (SPECFEM export — Bash 运行 Python 工具)

batch-segment (coordinator)
    ├── spawn multiple sandbox-segment agents (≤5 concurrent)
    ├── collect results
    └── meta-learning: analyze batch → update strategy_templates
```

## Rules

- **All VLM/semantic reasoning happens inside Claude Code agent sessions.** Agent directly reads images with the Read tool; no Python subprocess calls to `claude -p`.
- **`vlm_client/` is schema-only.** `client.py` functions (`_call_claude_cli`, `classify_figure`, etc.) are DEPRECATED. New code uses skill + agent-native reasoning.
- Agent communicates with Python tools via inline Bash scripts (`python -c "..."`) and file system (约定路径 in `runs/sandbox/`).
