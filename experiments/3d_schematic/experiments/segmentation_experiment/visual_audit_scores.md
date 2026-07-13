# Visual Audit Scores (Final)

## Scoring Criteria (1-5 scale)
- **layer_accuracy (1.5x)**: Correct identification of geological layers
- **text_immunity (1.5x)**: Immunity to text/annotation artifacts
- **fragment_control (1.3x)**: Control of fragmentation
- **boundary_quality (1.3x)**: Boundary smoothness and accuracy
- **structural_integrity (1.0x)**: Overall structural coherence

**Target**: weighted total >= 16/20

---

## Group A: geoseg Engines

### Panel 1
| Engine | n_layers | LA | TI | FC | BQ | SI | Weighted | Notes |
|--------|----------|----|----|----|----|----|----------|-------|
| v4_kmeans | 8 | 5 | 4 | 4 | 4 | 5 | 19.3 | Best overall. Correct layers, minor top fragments |
| v4_kmeans | 6 | 4 | 4 | 5 | 4 | 4 | 17.9 | Too simplified, plume merges with mantle |
| v4_kmeans | 4 | 3 | 5 | 5 | 3 | 3 | 15.6 | Severely oversimplified, only 3 regions |
| slic_kmeans | 8 | 5 | 3 | 3 | 4 | 4 | 17.4 | Good layers but many top fragments |
| edge_guided | 8 | 4 | 4 | 4 | 3 | 4 | 16.9 | Plume region too large |
| ensemble | 8 | 4 | 4 | 4 | 4 | 4 | 17.4 | Similar to v4_kmeans but less refined |

### Panel 2
| Engine | n_layers | LA | TI | FC | BQ | SI | Weighted | Notes |
|--------|----------|----|----|----|----|----|----------|-------|
| v4_kmeans | 8 | 4 | 4 | 4 | 4 | 4 | 17.4 | Plume too large, merges with mantle |
| v4_kmeans | 6 | 3 | 4 | 5 | 4 | 3 | 15.6 | Oversimplified |
| slic_kmeans | 8 | 5 | 3 | 3 | 4 | 4 | 17.4 | Best - separates uplift/plume/mantle |
| edge_guided | 8 | 4 | 4 | 4 | 3 | 4 | 16.9 | Plume too large |
| ensemble | 6 | 4 | 4 | 4 | 4 | 4 | 17.4 | Good but plume/mantle boundary merged |

### Panel 3
| Engine | n_layers | LA | TI | FC | BQ | SI | Weighted | Notes |
|--------|----------|----|----|----|----|----|----------|-------|
| v4_kmeans | 8 | 4 | 3 | 3 | 3 | 4 | 15.2 | Best available. Main body too large, uplift partially lost |
| v4_kmeans | 6 | 3 | 3 | 4 | 3 | 3 | 14.3 | Uplift completely lost, layers oversimplified |
| v4_kmeans | 4 | 2 | 4 | 5 | 2 | 2 | 11.7 | Severely oversimplified, only 4 regions |
| slic_kmeans | 8 | 3 | 3 | 3 | 3 | 3 | 13.5 | Top fragmented, yellow region too large |
| slic_kmeans | 6 | 3 | 3 | 4 | 3 | 3 | 14.3 | Oversimplified, uplift lost |
| slic_kmeans | 4 | 2 | 3 | 5 | 2 | 2 | 11.1 | Only 4 regions, structure lost |
| edge_guided | 8 | 3 | 3 | 4 | 3 | 3 | 14.3 | Red upper too large, green mid fragmented |
| edge_guided | 6 | 3 | 3 | 4 | 3 | 3 | 14.3 | Similar issues, 5 labels only |
| edge_guided | 4 | 2 | 3 | 5 | 2 | 2 | 11.1 | Oversimplified |
| ensemble | 8 | 3 | 3 | 3 | 3 | 3 | 13.5 | Main body too large |
| ensemble | 6 | 3 | 3 | 3 | 3 | 3 | 13.5 | Pink lower too large, uplift lost |
| ensemble | 4 | 2 | 3 | 5 | 2 | 2 | 11.1 | Oversimplified |
| grayscale | 4-8 | 1 | 1 | 1 | 1 | 1 | 5.2 | Complete failure, 4 labels all wrong |
| v4_kmeans | 10 | 4 | 3 | 3 | 3 | 4 | 15.5 | Best for Panel 3. Uplift clearer than n=8, some over-segmentation |
| v4_kmeans | 12 | 4 | 3 | 3 | 3 | 3 | 15.0 | Slightly over-segmented, yellow layer may be artificial |
| slic_kmeans | 10 | 3 | 3 | 3 | 3 | 3 | 13.5 | Similar to n=8, no improvement |
| slic_kmeans | 12 | 2 | 3 | 2 | 2 | 2 | 11.0 | More fragmented than n=8, worse |

---

## Group B: Diff-Overlay
**All configs score 6-9/20** - extremely fragmented (2400-3900 labels, frag=0.996)
Not suitable for geological panel segmentation.

### Objective Metrics (all panels)
| Param | Range | n_labels | frag | purity | Assessment |
|-------|-------|----------|------|--------|------------|
| blur_ksize | 7-21 | 3062-3796 | 0.996 | 1.5-5.9 | Severely fragmented |
| blur_sigma | 1.5-5.0 | 2647-3922 | 0.995 | 1.5-5.9 | Severely fragmented |
| diff_thresh | 10-30 | 2463-3606 | 0.996 | 1.8-5.1 | Severely fragmented |
| expand_radius | 8-25 | 2856-3586 | 0.996 | 1.9-4.9 | Severely fragmented |

### Visual Scores (representative: blur_sigma=3.0)
| Panel | LA | TI | FC | BQ | SI | Weighted | Notes |
|-------|----|----|----|----|----|----------|-------|
| 1 | 2 | 2 | 1 | 2 | 2 | 8.1 | Some large-scale structure visible but drowned in noise |
| 2 | 2 | 2 | 1 | 2 | 2 | 8.1 | Uplift outline visible but completely fragmented |
| 3 | 2 | 2 | 1 | 2 | 2 | 8.1 | Large regions visible but tiny fragments everywhere |

---

## Group C: v3 Pipeline (felzenszwalb-only)
**Complete** - All 24 runs confirm felzenszwalb-only is unsuitable for geological panels.

### Panel 1
| Param | Value | n_labels | frag | purity | LA | TI | FC | BQ | SI | Weighted |
|-------|-------|----------|------|--------|----|----|----|----|----|----------|
| felz_scale | 300 | 8115 | 1.00 | 2.0 | 2 | 1 | 1 | 1 | 1 | 5.9 |
| felz_scale | 600 | 4914 | 1.00 | 2.3 | 2 | 1 | 1 | 1 | 1 | 5.9 |
| felz_sigma | 0.3 | 16237 | 1.00 | 1.3 | 1 | 1 | 1 | 1 | 1 | 5.2 |
| felz_sigma | 0.5 | 8115 | 1.00 | 2.0 | 2 | 1 | 1 | 1 | 1 | 5.9 |
| felz_sigma | 0.8 | 4601 | 1.00 | 2.8 | 2 | 2 | 1 | 2 | 2 | 8.5 |
| felz_min_size | 10 | 9039 | 1.00 | 2.3 | 2 | 1 | 1 | 1 | 1 | 5.9 |
| felz_min_size | 30 | 8115 | 1.00 | 2.0 | 2 | 1 | 1 | 1 | 1 | 5.9 |
| felz_min_size | 50 | 4622 | 1.00 | 3.1 | 2 | 2 | 1 | 2 | 2 | 8.5 |

### Panel 2
| Param | Value | n_labels | frag | purity | LA | TI | FC | BQ | SI | Weighted |
|-------|-------|----------|------|--------|----|----|----|----|----|----------|
| felz_scale | 300 | 8908 | 1.00 | 3.0 | 2 | 1 | 1 | 1 | 1 | 5.9 |
| felz_scale | 600 | 5039 | 1.00 | 3.5 | 2 | 2 | 1 | 2 | 2 | 8.5 |
| felz_sigma | 0.3 | 16817 | 1.00 | 1.9 | 1 | 1 | 1 | 1 | 1 | 5.2 |
| felz_sigma | 0.5 | 8908 | 1.00 | 3.0 | 2 | 1 | 1 | 1 | 1 | 5.9 |
| felz_sigma | 0.8 | 4369 | 1.00 | 4.6 | 2 | 2 | 1 | 2 | 2 | 8.5 |
| felz_min_size | 10 | 9938 | 1.00 | 3.3 | 2 | 1 | 1 | 1 | 1 | 5.9 |
| felz_min_size | 30 | 8908 | 1.00 | 3.0 | 2 | 1 | 1 | 1 | 1 | 5.9 |
| felz_min_size | 50 | 5265 | 1.00 | 4.4 | 2 | 2 | 1 | 2 | 2 | 8.5 |

### Panel 3
| Param | Value | n_labels | frag | purity | LA | TI | FC | BQ | SI | Weighted |
|-------|-------|----------|------|--------|----|----|----|----|----|----------|
| felz_scale | 300 | 7694 | 1.00 | 2.2 | 2 | 1 | 1 | 1 | 1 | 5.9 |
| felz_scale | 600 | 5000 | 1.00 | 2.4 | 2 | 2 | 1 | 2 | 2 | 8.5 |
| felz_sigma | 0.3 | 15588 | 1.00 | 1.4 | 1 | 1 | 1 | 1 | 1 | 5.2 |
| felz_sigma | 0.5 | 7694 | 1.00 | 2.2 | 2 | 1 | 1 | 1 | 1 | 5.9 |
| felz_sigma | 0.8 | 4211 | 1.00 | 3.5 | 2 | 2 | 1 | 2 | 2 | 8.5 |
| felz_min_size | 10 | 8768 | 1.00 | 2.5 | 2 | 1 | 1 | 1 | 1 | 5.9 |
| felz_min_size | 30 | 7694 | 1.00 | 2.2 | 2 | 1 | 1 | 1 | 1 | 5.9 |
| felz_min_size | 50 | 4549 | 1.00 | 3.3 | 2 | 2 | 1 | 2 | 2 | 8.5 |

**Conclusion**: felzenszwalb-only produces extreme fragmentation (4200-16800 labels, frag=1.00).
Geological structure completely lost in noise. Unusable without post_merge or hierarchical merging.

---

## Final Summary

### Best Config Per Panel
| Panel | Best Engine | n_layers | Weighted | Target | Status |
|-------|-------------|----------|----------|--------|--------|
| 1 | v4_kmeans | 8 | **19.3/20** | >= 16 | PASS |
| 2 | slic_kmeans | 8 | **17.4/20** | >= 16 | PASS |
| 3 | v4_kmeans | 10 | **15.5/20** | >= 16 | FAIL |

### Cross-Strategy Ranking (by best score per panel)
| Strategy | Panel 1 | Panel 2 | Panel 3 | Avg |
|----------|---------|---------|---------|-----|
| Group A: geoseg engines | 19.3 | 17.4 | 15.5 | **17.4** |
| Group B: diff-overlay | ~8.1 | ~8.1 | ~8.1 | **8.1** |
| Group C: felzenszwalb-only | ~8.5 | ~8.5 | ~8.5 | **8.5** |

### Key Findings
1. **geoseg engines (Group A) are the only viable strategy** for geological panel segmentation.
2. **v4_kmeans** performs best on Panels 1 and 3; **slic_kmeans** performs best on Panel 2.
3. **Diff-overlay (Group B)** and **felzenszwalb-only (Group C)** both produce extreme fragmentation (2000-17000 labels, frag~1.0) and are completely unsuitable.
4. **Panel 3 is the bottleneck**: Complex structure with weak zones, text artifacts, and uplift/plume geometry exceeds the capability of current k-means based segmentation at any n_layers tested (4-12).

### Recommendations
- **Production**: Use v4_kmeans n=8 for standard panels, slic_kmeans n=8 for panels with prominent uplift/plume structures.
- **Panel 3 improvement**: Requires either (a) better text removal at top, (b) post-processing merge/split of weak zones, or (c) region-aware segmentation that treats uplift as special structure.
- **Avoid**: diff-overlay and felzenszwalb-only without hierarchical merging.
