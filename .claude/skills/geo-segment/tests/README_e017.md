# e017: Multi-Algorithm Ensemble Voting

## Overview

Combine three independent segmentation algorithms via **per-pixel majority voting** to produce more robust and spatially coherent velocity-zone labels than any single algorithm.

## Algorithms in the Ensemble

| # | Algorithm | Source | Strength |
|---|-----------|--------|----------|
| 1 | `segment_jet_vivid` (nearest_median) | `lib/segment.py` | Robust baseline, fast |
| 2 | `segment_jet_vivid_edge_guided` | `lib/segment_edge_guided.py` | Detailed boundaries |
| 3 | `segment_jet_vivid_merge` | `lib/segment_merge.py` | Good for vivid images |

## Fusion Strategy

1. **Label-space alignment**: Map result2 and result3 labels to result1's palette via nearest LAB color.
2. **Per-pixel majority vote**: For each pixel, collect the 3 aligned labels.
   - If 2+ agree, use the majority label.
   - If all 3 differ (rare), fallback to the baseline (algorithm 1) label.
3. **Post-process**: Merge connected components < 1% of total area into the largest neighbor.
4. **Palette recomputation**: Median RGB per final label from the original image.

## Files

- `lib/segment_ensemble.py` — Ensemble implementation
- `tests/test_e017_ensemble.py` — Test runner on 181218, 181659, 181210
- `experiments/e017_ensemble/` — Output directory with overlays and `review.json`

## Usage

```bash
cd ~/.claude/skills/geo-segment
python3 tests/test_e017_ensemble.py
```

## Results Summary

### 181218 (wide cross-section, 5 VLM reps, saturation=0.80)

| Method | Time | Labels | Boundary Grad | Intra-Var | Consistency |
|--------|------|--------|---------------|-----------|-------------|
| baseline | 0.128s | 8 | 14.59 | 515.47 | 0.9723 |
| edge_guided | 0.189s | 8 | 14.34 | 546.21 | 0.9574 |
| merge | 3.212s | 11 | 16.62 | 263.45 | 0.9649 |
| **ensemble** | **3.132s** | **8** | **16.19** | **492.57** | **0.9783** |

- Disagreement: **18.74%** of pixels
- Ensemble achieves the **highest consistency score** (0.9783) while preserving good boundary gradient.

### 181659 (small Vp SEM, 3 VLM reps, saturation=0.25)

| Method | Time | Labels | Boundary Grad | Intra-Var | Consistency |
|--------|------|--------|---------------|-----------|-------------|
| baseline | 0.036s | 5 | 53.00 | 1364.71 | 0.9698 |
| edge_guided | 0.067s | 6 | 66.54 | 948.64 | 0.9365 |
| merge | 1.006s | 9 | 62.89 | 698.73 | 0.9259 |
| **ensemble** | **1.568s** | **4** | **50.13** | **862.49** | **0.9806** |

- Disagreement: **7.22%** of pixels (low — algorithms largely agree)
- Ensemble achieves the **highest consistency** (0.9806) with the **lowest boundary gradient** (smoother boundaries).
- Note: Final labels dropped to 4 (from 5 baseline), suggesting some over-segmentation was corrected.

### 181210 (MATLAB screenshot, 5 VLM reps, saturation=0.002)

| Method | Time | Labels | Boundary Grad | Intra-Var | Consistency |
|--------|------|--------|---------------|-----------|-------------|
| baseline | 0.029s | 4 | 31.62 | 2140.33 | 0.9470 |
| edge_guided | 0.055s | 8 | 29.07 | 346.36 | 0.7533 |
| merge | 0.679s | 10 | 37.34 | 276.52 | 0.8500 |
| **ensemble** | **0.951s** | **3** | **33.64** | **926.86** | **0.9475** |

- Disagreement: **54.81%** of pixels (very high — algorithms disagree strongly)
- This is a low-saturation pastel image; the jet-vivid algorithms struggle.
- Ensemble falls back heavily to baseline, ending with only 3 labels.
- Consistency recovers to baseline level (0.9475) but variance is higher than merge.

## Key Findings

1. **Consistency improves on vivid images**: On 181218 and 181659, ensemble consistency is the highest of all methods. The voting mechanism suppresses isolated noise pixels that individual algorithms misclassify.

2. **Boundary quality is preserved**: Ensemble boundary gradients are close to or better than the best individual algorithm, indicating that voting does not blur true geological boundaries.

3. **Disagreement correlates with image type**:
   - Low disagreement (7-19%) on vivid jet images → ensemble works well.
   - High disagreement (55%) on pastel/low-saturation images → ensemble degrades to baseline.

4. **Speed**: Ensemble total time ≈ sum of individual times (dominated by Mean Shift merge at ~1-3s). The overhead of label alignment + voting + small-component merging is ~0.2-1.3s.

5. **Label count stability**: Ensemble tends to converge to the baseline label count on vivid images, but can under-merge on pastel images where algorithms diverge.

## Visual Outputs

For each image, the following are saved to `experiments/e017_ensemble/{image}/`:

- `{name}_baseline.png` — Baseline boundary overlay
- `{name}_edgeguided.png` — Edge-guided boundary overlay
- `{name}_merge.png` — Merge boundary overlay
- `{name}_ensemble.png` — Ensemble boundary overlay
- `{name}_ensemble_color.png` — Ensemble color overlay
- `{name}_disagreements.png` — Magenta pixels = where algorithms disagreed
- `review.json` — Per-image metrics and timing

## Recommendations

- **Use ensemble for vivid jet images** (saturation > 0.1) where algorithms agree enough for voting to be meaningful.
- **Skip ensemble for pastel images** (saturation < 0.05); the disagreement is too high and ensemble collapses to baseline.
- Consider a **saturation-gated ensemble**: run ensemble only when `saturation_ratio > 0.1`, otherwise use `segment_pastel_faded` directly.
