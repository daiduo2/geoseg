# geoseg Skills

Project-level skills for geophysics figure segmentation.

## Skill Index

| Skill | Purpose | Entry Point |
|-------|---------|-------------|
| `geo-segment` | End-to-end: figure → SPECFEM model | Single figure processing |
| `figure-classify` | Classify if a figure is a valid velocity model | Standalone classification |
| `sandbox-segment` | Autonomous panel segmentation (agent selects engines) | Single panel, agent-driven |
| `batch-segment` | Batch process a directory (≤5 parallel agents) | Multiple figures |
| `visual-audit` | Agent-driven visual critic: reads overlay-with-legend, outputs structured RegionalAudit | After segmentation, before export |
| `segment-export` | Export accepted segmentation to txt labels/palette + reconstructed + original-vs-reconstructed comparison | After visual-audit, before archiving |
| `preprocess-artifact` | Absorb red fault lines / black crosses before segmentation | Single figure, parameter-driven |
| `module-demo` | Run an example under `examples/geoseg/` to verify a module workflow | Example testing |
| `schema-bump` | Schema change protocol with consumer sync | Schema updates |

## Usage

Skills are activated by Claude Code context matching. You can also reference
them explicitly in prompts:

- "Classify this figure" → `figure-classify`
- "Segment this panel" → `sandbox-segment`
- "Process all figures in this directory" → `batch-segment`
- "Convert this figure to SPECFEM" → `geo-segment`
- "Export this segmentation" → `segment-export`

## Architecture

```
geo-segment (end-to-end orchestrator)
    ├── figure-classify (agent Read 看图 → 分类 JSON)
    ├── cv_detect (panel detection — Bash 运行 Python 工具)
    ├── sandbox-segment (agent 自主分割：选引擎 → 评估 → 融合)
    │   ├── strategy_memory (历史策略查询)
    │   ├── metrics (客观指标辅助)
    │   └── visual-audit (agent 视觉批评 → RegionalAudit)
    ├── post_process (SPECFEM export — Bash 运行 Python 工具)
    └── segment-export (txt labels/palette + reconstructed + comparison)

batch-segment (coordinator)
    ├── spawn multiple sandbox-segment agents (≤5 concurrent)
    ├── collect results
    └── summarize batch behavior; strategy template changes require explicit review
```

## Rules

- **All VLM/semantic reasoning happens inside Claude Code agent sessions.** Agent directly reads images with the Read tool; no Python subprocess calls to `claude -p`.
- **`vlm_client/` is schema-only.** `client.py` functions (`_call_claude_cli`, `classify_figure`, etc.) are DEPRECATED. New code uses skill + agent-native reasoning.
- Agent communicates with Python tools via packaged CLI entry points, `uv run python -m ...`, or short inline Bash scripts, plus file-system artifacts under agreed `runs/` paths.
- Product code must not depend on private engine helpers such as `_run_engine` or `_shared`; experiments may do so temporarily but must graduate through the project promotion rules in `CLAUDE.md`.
