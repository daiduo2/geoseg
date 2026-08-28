---
name: segment-export
description: >
  Export an accepted segmentation run to txt label + palette files and generate
  reconstructed/comparison images. Runs after visual-audit/review, using the
  best available labels (best_v3 > best_v2 > best > raw) and the original unsmoothed image
  as the reconstruction origin.
  Triggers: "export this segmentation", "导出分割结果", "txt export",
  "original vs reconstructed", "export labels and palette"
argument-hint: <run_dir> [--output-dir=path] [--profiles=name1 name2 ...] [--label-version=auto|best_v3|best_v2|best|raw] [--zip]
allowed-tools: Bash, Read, Write, Edit
---

# segment-export

Export a finalized segmentation run to the same txt format used by
`scripts/reconstruct_from_txt.py`, plus per-panel **original vs reconstructed**
comparison images.

## Quick Start

```
User: /segment-export runs/preprocess_newimage_merged --zip
Agent: [runs scripts/export_segmentation_txt.py --zip]
       "Exported 5 profiles to runs/preprocess_newimage_merged/txt_export/"
       "Created zip archive: runs/preprocess_newimage_merged/txt_export.zip"
```

## When to Use

- After segmentation has been audited and accepted (post `visual-audit`).
- When the user asks for txt labels, palette, or reconstructed images.
- When the user wants an original vs reconstructed side-by-side comparison.
- Before archiving or sharing results outside the pipeline.

## When NOT to Use

- Do NOT use before `visual-audit` has run and issues are resolved.
- Do NOT use for partial/throwaway segmentations.
- Do NOT invent profile names; use the figure's panel/profile names or panel_N.

## Export Workflow

### Step 0: Locate Best Labels

Pick the highest-quality label file available for each panel, in order:

1. `panels/panel_N/visual_audit/labels_best_split_v3.npz` — post-fix v3 result
2. `panels/panel_N/visual_audit/labels_best_split_v2.npz` — post-fix v2 result
3. `panels/panel_N/visual_audit/labels_best_split.npz` — workflow best result
4. `panels/panel_N/labels.npz` — raw engine output

Override with `--label-version best_v3|best_v2|best|raw`.

### Step 1: Run Export Script

```bash
PYTHONPATH=src python3 scripts/export_segmentation_txt.py \
  runs/preprocess_newimage_merged \
  --output-dir runs/preprocess_newimage_merged/txt_export \
  --profiles preprocess_newimage_merged_profile_s5e \
             preprocess_newimage_merged_profile_s4d \
             preprocess_newimage_merged_profile_s3c \
             preprocess_newimage_merged_profile_s2b \
             preprocess_newimage_merged_profile_s1a \
  --label-version best_v2 \
  --zip
```

打包后的文件为 `{output_dir}.zip`，例如 `runs/preprocess_newimage_merged/txt_export.zip`。

### Step 2: Verify Outputs

For each profile you should see:

- `{profile}_labels.txt` — `x y label_id` table
- `{profile}_palette.txt` — `label_id r g b` table (median RGB per label)
- `{profile}_reconstructed.jpg` — palette-colored reconstruction
- `{profile}_comparison.jpg` — original | reconstructed side-by-side
- `combined_reconstructed.jpg`
- `combined_comparison.jpg`

## Output Schema

### `{profile}_labels.txt`

```
x y label_id
0 0 4
1 0 4
...
```

### `{profile}_palette.txt`

```
label_id r g b
0 154 4 5
1 209 4 0
...
```

Palette colors are the **median RGB of each label in the original unsmoothed image**.

## Rules

1. **Always use the raw original image** (`01_original.jpg` + bbox) as the reconstruction origin, not a smoothed `panel.png`.
2. **Prefer fixed/best labels** over raw engine output; never downgrade to raw if a `labels_best_split*.npz` exists.
3. **Include label 0 (background)** in the palette so reconstructed images faithfully preserve margins and annotations.
4. **Profile names** should match the figure's panel identifiers (e.g. `s5e`, `s4d`) when known, otherwise use `{figure_id}_panel_NN`.
5. **Report the output directory and file list** after export.

## Integration with Other Skills

- `visual-audit`: Run audit first; export only after `retry_labels` are empty or explicitly accepted.
- `sandbox-segment` / `geo-segment`: Export is the final step of the accepted segmentation.
- `batch-segment`: Run export per figure after batch review is complete.

## Reconstruction Check

You can verify the exported txt files round-trip correctly:

```bash
PYTHONPATH=src python3 scripts/reconstruct_from_txt.py \
  runs/preprocess_newimage_merged/txt_export \
  --output-dir runs/preprocess_newimage_merged/txt_export/reconstruct_check
```

The re-generated reconstructed images should match the exported ones pixel-for-pixel.
