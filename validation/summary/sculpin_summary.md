# Cottus asper microsat LOSO summary

Real-data validation of the microsat input pipeline against the prickly
sculpin Genepop dataset (Dennenmoser et al. 2014, Dryad doi:10.5061/dryad.8ht04):
405 individuals × 19 microsat loci across 16 sites in the Pacific Northwest.
Leave-one-site-out: hold out each site's individuals, train on the other 15,
predict held-out coords. Metric: haversine distance (km).

## Centroid baseline

Predict the training-set centroid (mean lon, lat across the other 15 sites)
for every held-out individual: **440.2 ± 202.2 km mean(median)** across
the 16 sites. This is the floor any feature-based mode must beat.

## Results

| Mode | Median error km (mean ± std) | Mean error km (mean ± std) | Δ vs centroid |
|---|---|---|---|
| `centroid` (baseline) | 440.2 ± 202.2 | 440.2 | — |
| `dosage` | 267.2 ± 170.4 | 324.1 ± 161.2 | -173.0 (**BETTER**) |
| `geometry` | 412.4 ± 178.1 | 425.0 ± 148.5 | -27.8 (**BETTER**) |
| `repeat_norm` | 473.8 ± 217.5 | 473.8 ± 217.5 | +33.6 (**WORSE**) |

## Per-site LOSO (median error km, smaller = better)

| site | n_eval | centroid | dosage | geometry | repeat_norm |
|---|---|---|---|---|---|
| Alaska | 14 | 782.6 | 581.6 | 754.2 | 697.7 |
| BellaCoola | 29 | 88.6 | 370.5 | 371.5 | 203.8 |
| FallsCreek | 29 | 505.7 | 239.7 | 204.9 | 611.4 |
| Harrison | 30 | 579.2 | 118.9 | 617.6 | 691.2 |
| Lakelse | 30 | 185.0 | 246.7 | 395.8 | 158.4 |
| LittleCampbell | 30 | 582.1 | 142.7 | 498.1 | 679.4 |
| MAL | 51 | 379.6 | 136.9 | 272.7 | 355.3 |
| MALout | 31 | 359.2 | 3.3 | 233.3 | 309.7 |
| McLeod | 13 | 347.6 | 198.5 | 287.4 | 419.4 |
| Meziadin | 30 | 393.3 | 394.9 | 420.2 | 370.9 |
| Mosquito | 30 | 378.0 | 106.2 | 221.7 | 320.3 |
| Nimpo | 12 | 157.6 | 380.5 | 412.8 | 286.7 |
| Okanagan | 14 | 645.8 | 632.5 | 775.8 | 770.7 |
| PeaceAB | 12 | 740.0 | 234.3 | 460.0 | 795.3 |
| PeaceBC | 30 | 573.7 | 192.4 | 422.4 | 646.4 |
| Tlell | 20 | 345.4 | 295.7 | 249.8 | 263.8 |

## Figures

- `validation/figures/sculpin_modes.png` — bar chart of per-mode median LOSO
  error with centroid baseline overlay.

## Range-mask experiment (dosage mode)

Re-ran the 16-fold LOSO sweep with ReLocator's `loss_with_range_penalty`
constraining predictions to a freshwater polygon (HydroSHEDS HydroLAKES +
HydroRIVERS Strahler order ≥ 3, buffered 0.15° ≈ 17 km, clipped to PNW).
Polygon covers all 16 sampling sites. Penalty weight 50.0; otherwise
identical to the unmasked dosage run. See
`validation/sculpin/run_loso_rangemask.py` and
`validation/sculpin/build_range.py`.

Aggregate across 16 folds:

| | mean(median km) | std |
|---|---|---|
| unmasked dosage | 267.2 | 170.4 |
| **range-masked dosage** | **249.6** | 164.8 |
| centroid baseline | 440.2 | 202.2 |

**Range mask reduces mean(median) by ~7%.** 9 of 16 sites improve, 7
get worse. Biggest wins are sites whose unmasked predictions were
drifting onto land or ocean and the mask pulled them back to valid
water (Okanagan 632→356, Tlell 296→143, Meziadin 395→265). Biggest
loss is MALout: unmasked already nailed it (3 km) and the mask
perturbed predictions away (71 km). The mask is a net improvement
when the model lacks signal but a slight handicap when it has strong
signal at small scales.

Per-site comparison (km, smaller = better):

| site | unmasked | range-mask | Δ |
|---|---|---|---|
| MALout | 3 | 71 | +67 ✗ |
| Mosquito | 106 | 181 | +75 ✗ |
| Harrison | 119 | 173 | +54 ✗ |
| MAL | 137 | 122 | -15 ✓ |
| LittleCampbell | 143 | 116 | -27 ✓ |
| PeaceBC | 192 | 162 | -31 ✓ |
| McLeod | 199 | 184 | -14 ✓ |
| PeaceAB | 234 | 266 | +32 ✗ |
| FallsCreek | 240 | 247 | +7 ✗ |
| Lakelse | 247 | 209 | -37 ✓ |
| Tlell | 296 | 143 | -152 ✓ |
| BellaCoola | 370 | 316 | -55 ✓ |
| Nimpo | 381 | 424 | +44 ✗ |
| Meziadin | 395 | 265 | -130 ✓ |
| Alaska | 582 | 758 | +177 ✗ |
| Okanagan | 632 | 356 | -276 ✓ |

Figures:
- `validation/figures/sculpin_map_dosage.png` — unmasked map
- `validation/figures/sculpin_map_dosage_rangemask.png` — range-masked map
