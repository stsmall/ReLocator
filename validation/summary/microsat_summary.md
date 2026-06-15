# Microsat validation summary

Validation arc for the `microsatellites` branch. SLiM-simulated dataset
(200 individuals × 100 loci, arena coordinates). Metric: Euclidean
distance in arena units. K-fold CV: k=5, 3 seeds, mean ± std reported.

## Centroid baseline

A "predict the training-set mean for every held-out sample" model — i.e. learn nothing from features — produces this error on the same splits:

- **K-fold (k=5, 3 seeds, NaN-truth filtered):** 18.788 ± 1.414 mean(median), 18.516 ± 1.003 mean(mean).
- **Smoke 80/20, seed 42:** 20.792 median, 20.265 mean.

This is the floor any feature-based mode must beat to demonstrate signal.

## Results

| Mode | Median error (mean ± std) | Mean error (mean ± std) | Δ vs centroid k-fold |
|---|---|---|---|
| `centroid` (baseline) | 18.788 ± 1.414 | 18.516 ± 1.003 | — |
| `dosage` | 21.833 ± 2.442 | 22.097 ± 1.471 | **+3.045 (worse)** |
| `geometry` | 21.156 ± 1.567 | 22.039 ± 1.316 | **+2.368 (worse)** |
| `repeat_norm` | 18.795 ± 1.491 | 18.639 ± 1.024 | +0.007 (tied) |
| `polynomial` | 18.729 ± 1.554 | 18.586 ± 1.015 | −0.059 (tied) |

## Smoke (random 80/20, seed 42)

| Mode | n_held_out | n_scored | median_error | mean_error | max_error |
|---|---|---|---|---|---|
| `centroid` (baseline) | 40 | 34 | 20.792 | 20.265 | — |
| `dosage` | 40 | 34 | 23.546 | 21.391 | 46.852 |
| `geometry` | 40 | 34 | 22.571 | 21.708 | 43.086 |
| `repeat_norm` | 40 | 34 | 20.981 | 20.397 | 34.385 |
| `polynomial` | 40 | 34 | 21.087 | 20.050 | 33.098 |

## Recommendation

**No mode meaningfully beats a centroid baseline at n=200 on this dataset.** Earlier numbers suggested `repeat_norm` and `polynomial` were the strongest performers, but explicit comparison against "predict the training-set mean" reveals they are statistically tied with that baseline — the network is collapsing to centroid prediction rather than learning spatial signal. The predicted-vs-actual scatter (`microsat_scatter.png`) confirms this directly: the `repeat_norm` and `polynomial` panels show predictions clustered as flat horizontal bands at the arena centroid, not following the diagonal.

`dosage` and `geometry`, by contrast, *attempt* localization (predictions spread along the diagonal in the scatter) but do so noisily and end up **worse than the centroid baseline** by 2–3 arena units. The network has signal it can use but applies it badly at this sample size.

For the user-facing flag, this means there is **no single-mode default that the validation evidence supports recommending over another**. The combined default (`--features dosage,geometry,repeat_norm`) gives users every encoding the converter offers; downstream model selection (a richer architecture, regularization, or simply more samples) is what determines whether useful signal is recovered, not the encoding.

Three findings worth surfacing:

- **Centroid prediction is the optimal regression-to-the-mean response under weak signal**, and `repeat_norm` / `polynomial` are demonstrating exactly that — not a localization win. The scatter plot is the load-bearing evidence; the bar chart in `microsat_modes.png` is misleading without the centroid bar drawn alongside.
- **Polynomial degree-2 + PCA did not overfit catastrophically** as the spec hypothesized — but only because it converged on the same centroid prediction `repeat_norm` did. The PCA sample-rank cap (199 components from 200 samples) is a binding regularizer here. At larger N this regularizer would weaken and the original overfitting concern may resurface.
- **`dosage` and `geometry` are the modes attempting to do work**, even though they currently lose to the baseline. This is consistent with the categorical / per-individual encodings retaining more identifying information than the population-level z-score in `repeat_norm`. Whether the network can exploit that information depends on N and architecture, neither of which this PR varies.

The honest framing for the PR: **this validation arc demonstrates the converter is correctly wired end-to-end (all 64 folds completed, no errors, output shapes and value ranges sane), but it does not demonstrate that microsat-derived features beat random guessing at n=200**. Real-data validation at larger N, against a published microsat study with stronger spatial structure, is the next step before any encoding-mode recommendation can rest on evidence.

## Known limitations

Validation here used the SLiM-simulated dataset bundled with the original
locator paper. The following microsat-specific concerns are NOT covered by
this arc and require a real-data follow-up:

- Homoplasy (alleles of identical length but different ancestry).
- Null alleles (PCR drop-out producing apparent homozygotes).
- Sizing artefacts (off-by-one stutter calls).
- Real-data ascertainment biases.

See `validation/docs/specs/2026-05-04-microsatellite-input-extension-design.md`
for the full design rationale and the explicit out-of-scope list.

## Figures

- `validation/figures/microsat_modes.png` — bar chart of per-mode median
  error (k-fold mean ± std).
- `validation/figures/microsat_scatter.png` — predicted-vs-actual coords
  for the smoke fold, one panel per mode.
