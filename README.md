# geoseg v2

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Agent-native velocity-zone extraction from geophysics interpretation figures.

**geoseg** turns published geophysics figures — colored cross-sections, tomography maps, MATLAB-rendered seismic profiles — into [SPECFEM2D/3D](https://github.com/SPECFEM/specfem2d)-ready velocity-zone models. The pipeline is driven by Claude Code agents, not by a GUI: agents look at the image, decide what to do, and ask for help only at the overlay-review stage.

## Scope

Concept-model extraction only. No reflection-amplitude forward modeling, no waveform inversion, no full-waveform parameter tuning. The `figure_classifier` is **conservative** — false negatives are acceptable, false positives are not. A 1-D well-log plot should be rejected outright, not forced into the layered pipeline.

## Why agent-native

Traditional segmentation tools require manual tracing, parameter tuning, and domain expertise. v2 replaces clicks with conversation — natural language is the interface, and Claude Code skills orchestrate the whole pipeline. The agent **sees** the figure via the `Read` tool and **decides** which engine, mask, or refinement to apply.

## Key features

- **Agent-native orchestration** — figure-classify → cv_detect → sandbox-segment → visual-audit → export, driven by Claude Code skills.
- **CLI human-in-the-loop** — agent auto-runs the pipeline, presents the overlay, then waits for natural-language feedback. *"Remove the colorbar"* or *"split the bottom layer"* triggers an immediate re-segment.
- **Multi-engine sandbox** — `sandbox-segment` tries engines (`colorbar_guided`, `regional_fusion`, `edge_guided`, `ensemble`, `kmeans_full`, `grayscale`, …), evaluates by VLM visual judgment + objective metrics, and picks or fuses the best.
- **Artifact-aware preprocessing** — red fault lines, black crosses, white gaps, label merges, and small-component cleanup happen before zone extraction.
- **Visual audit** — `visual-audit` reads overlay-with-legend and emits a structured `RegionalAudit`. No PASS/FAIL scoring — the agent decides.
- **Strategy memory** — past segmentations inform engine selection on similar figures.
- **Session state with backtracking** — full lifecycle `pending → classified → segmented → reviewed → exported`, with backtrack to any upstream stage.
- **Batch processing** — directory mode with ≤5 concurrent segmenter agents (M-series Mac ~1.5 GB per agent).
- **Napari Shapes editor** — blocking GUI for fine label editing when natural-language feedback isn't enough.

## Architecture

```
PDF / Image
    ↓
[Agent: figure-classify] → velocity_model / skip
    ↓
[cv_detect] → panels + colorbar
    ↓
[Agent: sandbox-segment] → best labels (engine pick + fuse)
    ↓
[Agent: visual-audit] → RegionalAudit (semantic summary)
    ↓
[napari editor]  ←── optional, only when natural language isn't enough
    ↓
[post_process + exporter] → polygons + properties + SPECFEM2D/3D
```

All VLM reasoning runs inside Claude Code agent sessions via the `Read` tool. No Python subprocess to `claude -p`.

## Skills

| Skill | What it does |
|-------|--------------|
| `geo-segment` | End-to-end: figure → SPECFEM, dialog HITL. |
| `figure-classify` | Look at the image, decide velocity model vs. skip. |
| `sandbox-segment` | Try multiple engines, evaluate, fuse. |
| `visual-audit` | Read overlay-with-legend, emit structured audit. |
| `batch-segment` | Directory mode, ≤5 concurrent agents. |
| `segment-export` | Export accepted segmentation to SPECFEM. |
| `preprocess-artifact` | Red fault lines, black crosses, white-gap repair. |
| `module-demo` | Run `examples/geoseg/` flows to verify modules. |
| `schema-bump` | Schema change protocol. |

Skill workflow graphs and failure handling live in `.claude/skills/<skill>/SKILL.md`.

## Quick start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — Python package manager
- [Claude Code](https://claude.ai/code) CLI
- Optional: MinerU API key (PDF ingestion)
- Optional: [rmux](https://github.com/joshmedeski/rmux) (HTML report ↔ CLI feedback bridge)

### Package manager rules

| Ecosystem | Tool | Notes |
|-----------|------|-------|
| Python | **uv** | `uv sync`, `uv run python -m geoseg…` |
| TS/JS | **pnpm** | `pnpm install`, `pnpm exec …` |

Do **not** mix `pip` / `python3 -m venv`, `npm`, or `yarn` inside this project. The virtualenv is managed exclusively by `uv`.

### Install

```bash
git clone https://github.com/daiduo2/geoseg.git
cd geoseg
uv sync
uv run scripts/env_check.py
```

### Single figure

```
User: /geo-segment runs/M0.5/fig1.png --n-layers=5

Agent: [auto-runs classify → detect → segment → audit]
       fig1.png  分割完成
         类型: velocity_model (0.92)
         引擎: colorbar_guided → 5 层
         audit: ok
       [shows overlay]

       Accept / Modify / Skip / Backtrack ?

User: 修改。去掉右上角颜色条，底层分两层。

Agent: [re-segments with mask + n_layers+1]
       [shows new overlay]
       Accept / Modify / Skip / Backtrack ?

User: 接受

Agent: [exports SPECFEM]
       ✅ tomo.xyz + Par_file_snippet.txt
```

### Batch

```
User: /batch-segment runs/M0.5/ --n-layers=5

Agent: [Stage 1-3: scans → classifies all → segments all]
       📦 5 张目标图已处理完毕，请 review。

       [1] fig1.png  ✅  0.85  5层
       [2] fig3.png  ✅  0.91  4层
       [3] fig4.png  ⚠️   0.62  3层  ← 建议修改
       [4] fig7.png  ✅  0.78  6层
       [5] fig9.png  ⚠️   0.58  2层  ← 建议修改

User: 1,2,4 接受；3 修改：底层应分两层；5 跳过

Agent: [exports 1,2,4; re-segments 3; skips 5]
```

## PDF ingestion (optional)

Published models are usually embedded in papers, not released as raw data tables. geoseg provides two ingestion layers:

- **MinerU** (`modules/mineru_client/`) — structured extraction (figures + caption markdown + content_list.json). Requires `MINERU_API_KEY`.
- **PyMuPDF fallback** (`modules/pdf_extractor/`) — `{XObject + text block}` extraction and `rasterize_page()`. Used when MinerU splits a figure or sizes are too small.

```bash
export MINERU_API_KEY="your-api-key"
```

## Real-time feedback (optional)

The HTML report chatbox can drive the CLI session in real time via `rmux`:

```bash
# Terminal 1 — Claude Code inside a named rmux session
rmux new-session -s geoseg
# (inside rmux) cd /path/to/geoseg && cc

# Terminal 2 — feedback bridge
uv run python -m geoseg.feedback_bridge --rmux-session=geoseg

# Generate and open the dashboard
uv run python -m geoseg.generate_report runs/sessions/batch_xxx.json
open runs/reports/batch_xxx.html
```

## Module structure

```
src/geoseg/
├── core/                  # Stable data contracts + cross-module facade
│   ├── models.py
│   └── image_ops.py
├── pipeline/              # Stage orchestration
│   ├── segment.py
│   ├── export.py
│   └── stages/
├── modules/
│   ├── cv_detect/         # Panel detection + colorbar extraction
│   ├── segment_engines/   # Engine family (registry / runner / policy / retry)
│   │   ├── v4/            # colorbar_guided, palette
│   │   ├── regional/      # regional_fusion
│   │   ├── edge/          # edge_guided helpers
│   │   ├── horizon/       # horizon refinement internals
│   │   ├── strategy/      # strategy memory
│   │   ├── diagnostics/   # metrics, batch_test, compare_results
│   │   └── internal/      # shared helpers
│   ├── post_process/      # Polygons + physical properties
│   ├── exporter/          # SPECFEM2D/3D output
│   ├── editor/            # Napari Shapes-primary editor
│   ├── visual_audit/      # overlay-with-legend views
│   ├── mineru_client/     # MinerU PDF ingestion
│   ├── pdf_extractor/     # PyMuPDF fallback
│   └── vlm_client/        # Schema + prompt templates (no LLM call)
├── cli/                   # Packaged CLI entrypoints
├── batch/                 # Batch directory processing
├── api/                   # FastAPI schema/routes (legacy compat)
├── experiments.py         # Script-side facade for CV / VLM / engine helpers
├── session_state.py       # Persistent session state with backtracking
├── controller.py          # End-to-end compat facade
├── pipeline_interfaces.py # Old import compat layer
└── server.py              # FastAPI compat entrypoint
```

For the full code map and module contracts see [`docs/CODEBASE.md`](docs/CODEBASE.md).

## Design philosophy

1. **Agent-native over GUI** — conversation is the interface.
2. **HITL only at review** — auto-run everything; stop only for overlay confirmation.
3. **Upstream backtracking** — user can backtrack to `classify` / `panel` / `segment`.
4. **Conservative classification** — prefer false negatives over false positives.
5. **VLM judgment primary** — visual evaluation trumps quantitative metrics.
6. **Immutable state** — session-state updates return new objects; every step persists.
7. **No new Tauri/FastAPI product frontend** — `api/` and `server.py` are historical compatibility only.

## Testing

```bash
uv run pytest                              # full suite
uv run pytest tests/test_integration_ph01.py  # smoke after schema changes
```

GUI tests are excluded from CI on macOS runners. See [`tests/`](tests/).

## License

[MIT](LICENSE).
