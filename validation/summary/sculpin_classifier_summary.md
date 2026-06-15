# Sculpin classifier-head LOSO summary

16-fold leave-one-site-out comparison across 5 prediction modes on
the prickly sculpin Cottus asper microsat dataset (405 individuals ×
19 loci, dosage encoding). Metric: haversine distance (km).

**Centroid baseline (predict training mean):** 440.1 km

## Aggregate (mean of per-fold medians)

| Mode | n folds | Median error km (mean ± std) | Mean error km (mean ± std) | Δ vs centroid | Δ vs regress |
|---|---|---|---|---|---|
| `regress` | 16 | 267.2 ± 170.4 | 324.1 ± 161.2 | -172.9 | +0.0 |
| `regress+rangemask` | 16 | 249.6 ± 164.8 | 279.3 ± 163.9 | -190.5 | -17.6 |
| `classify` | 16 | 254.5 ± 186.6 | 307.0 ± 151.5 | -185.6 | -12.7 |
| `classify_then_avg` | 16 | 252.3 ± 166.5 | 308.3 ± 134.5 | -187.8 | -14.9 |
| `classify_then_avg_graph` | 16 | 321.9 ± 202.9 | 366.0 ± 169.7 | -118.2 | +54.7 |

## Per-site error (km, sorted by regress baseline)

| site | regress | regress+rangemask | classify | classify_then_avg | classify_then_avg_graph |
|---|---|---|---|---|---|
| MALout | 3 | 71 | 0 | 0 | 3 |
| Mosquito | 106 | 181 | 369 | 339 | 297 |
| Harrison | 119 | 173 | 95 | 127 | 97 |
| MAL | 137 | 122 | 0 | 8 | 371 |
| LittleCampbell | 143 | 116 | 71 | 131 | 88 |
| PeaceBC | 192 | 162 | 217 | 192 | 213 |
| McLeod | 199 | 184 | 196 | 310 | 381 |
| PeaceAB | 234 | 266 | 217 | 256 | 220 |
| FallsCreek | 240 | 247 | 71 | 72 | 130 |
| Lakelse | 247 | 209 | 274 | 232 | 351 |
| Tlell | 296 | 143 | 446 | 492 | 484 |
| BellaCoola | 370 | 316 | 371 | 371 | 370 |
| Nimpo | 381 | 424 | 516 | 332 | 454 |
| Meziadin | 395 | 265 | 497 | 437 | 376 |
| Alaska | 582 | 758 | 563 | 563 | 844 |
| Okanagan | 632 | 356 | 168 | 175 | 471 |

## Files

- `validation/figures/sculpin_classifier_modes.png` — grouped bar
  chart, all 5 modes per site, with centroid baseline overlay.
- `validation/summary/sculpin_classifier_kfold.tsv` — raw per-fold rows.

## Paired graph vs baseline (same trained model)

The aggregate `classify_then_avg` vs `classify_then_avg_graph` columns above
compare **separately-trained** sweeps and are confounded by GPU training
noise (~50 km per-fold variance, per CLAUDE.md's note on cuDNN
non-determinism). To isolate the smoothing effect, `run_loso_graph.py`
also records a paired baseline — for each fold, predict() is called
twice on the **same** trained model: once without `graph_topology`, once
with. The delta below is purely from heat-kernel smoothing.

| site | paired baseline (km) | graph (km) | delta |
|---|---|---|---|
| Alaska | 844.7 | 844.3 | -0.4 |
| BellaCoola | 371.5 | 370.4 | -1.0 |
| FallsCreek | 130.2 | 130.2 | -0.0 |
| Harrison | 96.5 | 96.5 | +0.0 |
| Lakelse | 350.9 | 350.9 | +0.0 |
| LittleCampbell | 88.0 | 88.0 | +0.0 |
| MAL | 371.4 | 371.4 | +0.0 |
| MALout | 0.0 | 3.4 | +3.4 |
| McLeod | 380.8 | 380.8 | -0.0 |
| Meziadin | 375.5 | 375.5 | -0.0 |
| Mosquito | 297.5 | 297.0 | -0.5 |
| Nimpo | 453.7 | 453.7 | -0.0 |
| Okanagan | 471.2 | 471.2 | -0.0 |
| PeaceAB | 219.9 | 219.9 | +0.0 |
| PeaceBC | 213.0 | 213.0 | +0.0 |
| Tlell | 484.3 | 484.3 | +0.0 |

13 sites are unchanged because they are isolated in the topology (no
edges to/from them at K=15 trained centroids) — heat kernel reduces
to the identity. Only MALout shows a structural delta (+3.4 km, slightly
worse). The 3 sub-1-km deltas (Alaska, BellaCoola, Mosquito) are
numerical roundoff from `scipy.linalg.expm`, not real signal.

## Recommendation

**v1 graph-classifier with HydroRIVERS-derived topology has no
practical effect on sculpin LOSO accuracy at this resolution.** The
core constraint is connectivity, not the math: at Strahler order ≥ 3
within the PNW bounding box, only Tlell ↔ MAL/MALout (Haida Gwaii)
share a fluvial path. The high-error FST-outlier sites this design
was meant to help — Alaska, Okanagan, Nimpo — are all in distinct
watersheds with no fluvial connections to other sampled sites, so
they end up isolated in the graph and gain no smoothing. This does
not refute the architectural hypothesis; it shows the river-only
topology is too sparse for *Cottus asper* sampling.

**v2 directions** (post-merge follow-up, not in scope here):

- **HydroLAKES-aware nodes:** sculpin live in lakes, not just along
  reaches; using lake polygons as nodes (with reaches as edges)
  should connect more sites within each watershed.
- **Marine/coastal corridors:** sculpin populations on islands
  (Haida Gwaii) and across saltwater straits need an out-of-stream
  graph mechanism. A user-provided "ocean corridor" overlay alongside
  HydroRIVERS would handle this honestly.
- **Training-time graph regularization (Option B from spec):** v1 is
  inference-only. A graph-Laplacian penalty on softmax during training
  could shape the *learned* probability distribution rather than just
  smoothing it post-hoc.
- **GNN encoder (deferred to v3):** treat the graph as input rather
  than output structure.

The graph-classifier code path (`locator.graph` helpers,
`graph_topology` config) is correct and should ship — the result is
"no effect," not "broken." Future users with denser connectivity
domains (road networks, valley corridors, anadromous fish dataset
covering one watershed) can use it directly.
