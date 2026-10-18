# experiments/archive — 历史实验索引

这里的脚本只用于追溯"当时怎么试的"，不再维护。结论已提炼到
[`docs/ALGORITHM_FAMILY.md`](../../docs/ALGORITHM_FAMILY.md) 第 7 节，先读结论再决定要不要翻代码。

| 目录 | 内容 | 结论 |
|------|------|------|
| `2026-07-fig6-colorbar/` | Fig.6 的 16-zone 颜色条匹配、调色板残差纠正、去文字、欠分割修复，共 47 个脚本 | e031–e035；16-zone 与行中值可复用，逐标签调色是反模式 |
| `2026-07-pm-repair/` | PM/OCR 文字定点修复三轮（round1–3） | e036；有效但不可推广，OCR 依赖独立 `.venv_ocr` |
| `2026-06-engine-experiments/` | e028 FH+SLIC、e029 空间正则化聚类、e030 各向异性预处理 | 已在 ALGORITHM_FAMILY 第 1 节；e030 已并入主路径 |

配套的 `scripts/archive/` 下另有 `2026-07-fig6-palette/`（22 个调色迭代脚本）和
`2026-07-candidate-screening/`（37 个候选筛选脚本）。

## 注意

- 归档脚本保留原始相对路径假设（如 `Path(__file__).parents[1] / "src"`），移动后**不保证能原地运行**。
- 部分脚本硬编码了 `/Users/daiduo2/geoseg/runs/...`；`runs/` 现在是指向
  `~/geoseg-artifacts/2026-10/runs` 的软链，路径仍可解析。
- 需要重跑某个实验时，用 `git log --follow <file>` 找到原始位置，再 `git mv` 回去。
