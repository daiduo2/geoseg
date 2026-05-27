# e014: Edge-guided K-means Segmentation

## Objective
Address the problem that standard K-means produces fuzzy boundaries on geological
panels where layer transitions are gradual color ramps.  Edge detection provides
spatial constraints that snap uncertain boundary pixels to actual layer edges.

## Implementation

### File: `lib/segment_edge_guided.py`

**`segment_jet_vivid_edge_guided(panel_rgb, reps, max_auto_k=3, edge_weight=0.3, sigma=4.0)`**

Reuses the full seed-refinement + auto-k pipeline from `segment_jet_vivid`, but
replaces the classifier with an edge-guided K-means:

1. **Edge detection** — Canny on the L channel (sigma=1.0, low=0.02, high=0.1)
   with morphological closing to bridge small gaps.  Produces thin, connected
   boundaries that align with geological layer transitions.

2. **Standard K-means** — Run `scipy.cluster.vq.kmeans2` in LAB space with
   refined seeds as initial centroids.

3. **Selective snapping** — Only pixels that are BOTH near an edge (within
   `sigma` px) AND ambiguous in color (LAB distance to best centroid is within
   `edge_weight` fraction of distance to second-best) are snapped to the dominant
   cluster of their edge-bounded region.  This avoids corrupting confident
   interior pixels.

4. **Post-processing** — `_shape_filter()` merges thin 1-D noise.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `edge_weight` | 0.3 | Ambiguity threshold (0 = pure K-means, no snapping) |
| `sigma` | 4.0 | Distance from edge (px) within which snapping is considered |

### Edge Detection Methods

`_compute_edge_map()` supports two methods:
- **`canny`** (default): Clean thin edges on L channel, best for gradual transitions
- **`sobel`**: Multi-channel LAB gradient, more sensitive but noisier

## Test Results

### 181218 (vivid jet, 200x700)

| Method | Time | Seeds | Auto-k | Edge pixels |
|--------|------|-------|--------|-------------|
| nearest_median | 0.10s | 8 | 3 | — |
| kmeans | 0.10s | 8 | 3 | — |
| edge-guided ew=0.1 | 0.20s | 8 | 3 | 22.1% |
| edge-guided ew=0.3 | 0.20s | 8 | 3 | 22.1% |
| edge-guided ew=0.5 | 0.20s | 8 | 3 | 22.1% |

- **ew=0.3** gives the best visual balance: boundaries snap to layer edges
  (especially the green/blue and blue/purple transitions) without introducing
  artifacts.
- Differences vs baseline K-means: 3.8% (ew=0.1), 5.7% (ew=0.3), 7.6% (ew=0.5).

### 184140 (pastel, 290x690)

| Method | Time | Seeds | Auto-k | Edge pixels |
|--------|------|-------|--------|-------------|
| nearest_median | 0.14s | 6 | 3 | — |
| kmeans | 0.14s | 6 | 3 | — |
| edge-guided ew=0.1 | 0.30s | 6 | 3 | 32.7% |
| edge-guided ew=0.3 | 0.31s | 6 | 3 | 32.7% |
| edge-guided ew=0.5 | 0.31s | 6 | 3 | 32.7% |

- 184140 is a multi-panel figure with text, graphs, and maps. Edge detection
  picks up many non-geological edges (text, plot lines), so the benefit is
  limited.  The method is designed primarily for single cross-section panels.

## Visual Comparison

**181218 — Baseline K-means vs Edge-guided (ew=0.3)**

The edge-guided result shows:
- Sharper green/blue boundary alignment
- Cleaner blue/purple transition along the fault
- Reduced speckle at layer boundaries

## Speed

Edge-guided adds ~0.1s overhead over standard K-means (mainly from Canny edge
detection + connected-component labeling).  Total runtime is ~2x K-means,
still well under 0.5s per panel.

## Conclusions

1. **Edge guidance improves boundary alignment** on single cross-section panels
   with gradual color transitions (e.g. 181218).
2. **Best edge_weight: 0.3** — balances boundary snapping without over-snapping.
3. **Selective snapping is critical** — snapping ALL near-edge pixels creates
   artifacts; only snapping ambiguous pixels preserves interior regions.
4. **Canny on L channel** outperforms Sobel for geological panels because it
   produces thin, connected boundaries that follow layer contours.
5. **Limitation**: On complex multi-panel figures (184140), text and plot edges
   interfere.  The method works best on clean cross-section panels.

## Files

- `lib/segment_edge_guided.py` — Implementation
- `tests/test_e014_edge_guided.py` — Test script
- `experiments/e014_edge_guided/` — Output images and review.json
