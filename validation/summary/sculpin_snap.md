# Sculpin: post-hoc snap-to-nearest-site experiment

Re-scoring the existing unmasked dosage LOSO predictions under two snap
rules to test whether a discrete-site classifier head would meaningfully
outperform the current continuous-regression head.

## Aggregate results (mean of per-site medians, km)

| Variant | Mean(median) km | Δ vs raw | Δ vs centroid |
|---|---|---|---|
| centroid baseline                                | 440.1 | — | — |
| **raw regression (current)**                     | **267.0** | — | -173.0 |
| snap to nearest of 15 others (LOSO classifier)   | 254.3 | -12.7 | -185.8 |
| snap to nearest of all 16 (incl. truth)          | 254.3 | -12.7 | -185.8 |

Snap-correctness rate (fraction of individuals where the regression
prediction's nearest site is in fact the held-out true site):
**0.08** (16-class snap, n=405 individuals).

## Per-site comparison (km)

| site | n | centroid | raw regression | snap-15 | snap-16 | snap16 acc |
|---|---|---|---|---|---|---|
| MALout | 31 | 359 | 3 | 0 | 0 | 0.00 |
| Mosquito | 30 | 378 | 106 | 39 | 39 | 0.00 |
| Harrison | 30 | 579 | 119 | 123 | 123 | 0.00 |
| MAL | 51 | 380 | 137 | 7 | 7 | 0.27 |
| LittleCampbell | 30 | 582 | 143 | 95 | 95 | 0.03 |
| PeaceBC | 30 | 574 | 192 | 217 | 217 | 0.17 |
| McLeod | 13 | 348 | 198 | 196 | 196 | 0.15 |
| PeaceAB | 12 | 737 | 232 | 217 | 217 | 0.00 |
| FallsCreek | 29 | 506 | 240 | 337 | 337 | 0.28 |
| Lakelse | 30 | 185 | 247 | 245 | 245 | 0.03 |
| Tlell | 20 | 345 | 296 | 330 | 330 | 0.20 |
| BellaCoola | 29 | 89 | 370 | 371 | 371 | 0.10 |
| Nimpo | 12 | 158 | 381 | 326 | 326 | 0.00 |
| Meziadin | 30 | 393 | 395 | 429 | 429 | 0.00 |
| Alaska | 14 | 782 | 582 | 567 | 567 | 0.00 |
| Okanagan | 14 | 646 | 632 | 570 | 570 | 0.00 |


## Interpretation

- **snap-15 vs raw regression** is the architecturally honest comparison.
  In LOSO, a 16-class classifier trained on the same data would only have
  15 classes to choose from per fold (the held-out site is by definition
  unknown to the network at training time). If `snap-15 < raw regression`,
  a classifier head would be expected to deliver real gains on this data.
- **snap-16 vs raw regression** is the best-case proxy: it lets the snap
  procedure pick the truth itself when the regression got close enough.
  This is structurally unfair as an LOSO test but useful as an upper
  bound on how much improvement is theoretically possible from snapping.
- The **snap-16 correctness rate** (0.08) is the fraction of
  individuals where the regression's free prediction was already closer
  to the true held-out site than to any other site. High values mean
  "regression is putting predictions in the right neighborhood, just
  imprecisely"; low values mean "regression is in the wrong neighborhood
  entirely."
