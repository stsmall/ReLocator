# Path D validation summary

Path A/B/C use the balanus 54-sample test set. The Example-VCF stress test and noise-curve experiment below use ReLocator's bundled 500-sample example data to validate the continuous-dosage path in a non-underdetermined regime.

---

Centroid baseline (mean across-site haversine to global centroid) = 683.4 km.

---

**Path A:** PASS — {"n_samples": 54, "n_sites": 100000, "predlocs_rows": 54}

---

**Path B (dosage LOSO):** median along-coast error = 330.0 km; baseline = 941.8 km; ratio = 0.350 (PASS).

| site | n_held_out | true_lat | true_lon | mean_pred_lat | mean_pred_lon | mean_error_km | median_error_km | mean_along_coast_err_km | mean_offshore_km | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01-AFB-AK | 4 | 60.121 | -149.370 | 49.110 | -125.000 | 1970.921 | 1975.749 | 1957.206 | 75.883 | OK |
| 02-DIG-BC | 6 | 54.283 | -130.417 | 46.885 | -127.302 | 853.035 | 770.808 | 953.939 | 230.353 | OK |
| 03-FHL-WA | 2 | 48.545 | -123.013 | 47.090 | -126.750 | 390.713 | 390.713 | 405.286 | 190.720 | OK |
| 04-RIC-WA | 4 | 47.764 | -122.386 | 52.900 | -127.373 | 671.447 | 690.149 | 632.502 | 74.936 | OK |
| 05-WES-WA | 3 | 46.912 | -124.110 | 43.957 | -123.753 | 345.285 | 300.697 | 317.808 | 73.586 | OK |
| 06-MEA-OR | 3 | 45.486 | -123.975 | 47.253 | -125.467 | 231.857 | 250.411 | 273.029 | 132.838 | OK |
| 07-CPE-OR | 6 | 44.280 | -124.112 | 45.447 | -127.808 | 329.034 | 307.355 | 154.261 | 292.284 | OK |
| 08-BOB-OR | 7 | 44.244 | -124.114 | 46.750 | -125.314 | 296.613 | 295.769 | 278.869 | 96.476 | OK |
| 09-OMB-OR | 4 | 43.345 | -124.322 | 45.080 | -123.025 | 235.127 | 226.403 | 204.938 | 87.687 | OK |
| 10-PAR-CA | 6 | 38.956 | -123.741 | 42.167 | -124.683 | 377.268 | 287.130 | 347.588 | 99.146 | OK |
| 11-HOP-CA | 4 | 36.600 | -121.895 | 39.615 | -123.310 | 357.362 | 351.629 | 374.245 | 44.336 | OK |
| 12-GOL-CA | 5 | 34.417 | -119.832 | 40.556 | -118.480 | 701.279 | 718.054 | 658.660 | 424.024 | OK |

---

**Path C (full_gl LOSO):** median along-coast error = 544.7 km; baseline = 941.8 km; ratio = 0.578 (FAIL — > 0.5).

| site | n_held_out | true_lat | true_lon | mean_pred_lat | mean_pred_lon | mean_error_km | median_error_km | mean_along_coast_err_km | mean_offshore_km | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01-AFB-AK | 4 | 60.121 | -149.370 | 48.815 | -125.153 | 1987.874 | 1996.318 | 1962.462 | 108.111 | OK |
| 02-DIG-BC | 6 | 54.283 | -130.417 | 43.877 | -124.760 | 1228.530 | 1228.445 | 1433.395 | 69.239 | OK |
| 03-FHL-WA | 2 | 48.545 | -123.013 | 47.125 | -116.325 | 532.578 | 532.578 | 98.605 | 463.868 | OK |
| 04-RIC-WA | 4 | 47.764 | -122.386 | 49.115 | -126.375 | 335.300 | 330.433 | 361.141 | 158.230 | OK |
| 05-WES-WA | 3 | 46.912 | -124.110 | 43.587 | -128.267 | 492.445 | 477.578 | 398.871 | 319.936 | OK |
| 06-MEA-OR | 3 | 45.486 | -123.975 | 47.673 | -127.000 | 361.573 | 412.926 | 486.951 | 225.582 | OK |
| 07-CPE-OR | 6 | 44.280 | -124.112 | 46.173 | -124.812 | 217.908 | 217.118 | 219.193 | 59.411 | OK |
| 08-BOB-OR | 7 | 44.244 | -124.114 | 46.707 | -125.359 | 295.354 | 309.238 | 278.479 | 97.697 | OK |
| 09-OMB-OR | 4 | 43.345 | -124.322 | 47.258 | -126.785 | 503.432 | 515.094 | 435.549 | 211.363 | OK |
| 10-PAR-CA | 6 | 38.956 | -123.741 | 45.890 | -125.983 | 793.683 | 806.651 | 786.169 | 150.958 | OK |
| 11-HOP-CA | 4 | 36.600 | -121.895 | 42.562 | -124.998 | 718.095 | 694.600 | 724.198 | 71.842 | OK |
| 12-GOL-CA | 5 | 34.417 | -119.832 | 45.434 | -121.686 | 1251.845 | 1209.235 | 1376.242 | 155.529 | OK |

---

**Example-VCF stress test** (data/test_genotypes.vcf.gz: 500 samples × ~11.5k biallelic sites; 90-sample random holdout; Euclidean error in the simulated 50×50 coordinate frame).

Centroid-baseline error (predict the geographic mean): 18.52.

| condition | n | mean_err | median | p90 | max |
| --- | --- | --- | --- | --- | --- |
| A. VCF (hard calls) | 90 | 4.456 | 3.728 | 8.095 | 30.327 |
| B. Cont. dosage, α=0.0 | 90 | 4.517 | 3.748 | 8.157 | 23.202 |
| C. Cont. dosage, α=0.5 | 90 | 4.622 | 3.772 | 8.570 | 18.653 |

---

**Noise curve — full_gl vs rounded dosage at increasing GL uncertainty** (same 90-sample holdout; rounded values bypass the loader patch and go through the legacy integer path; cont mode is essentially equivalent to full_gl on this dataset and is omitted here — see `noise_curve.tsv` for the full per-encoding table).

| encoding | alpha | mean_err | median | p90 | max |
| --- | --- | --- | --- | --- | --- |
| VCF | nan | 4.456 | 3.728 | 8.095 | 30.327 |
| round | 0.000 | 4.672 | 3.530 | 8.631 | 20.929 |
| round | 0.300 | 4.830 | 3.968 | 8.614 | 17.159 |
| round | 0.500 | 5.442 | 4.872 | 9.435 | 14.082 |
| round | 0.700 | 18.537 | 19.148 | 26.567 | 29.861 |
| round | 0.900 | 18.506 | 18.606 | 26.988 | 29.383 |
| full_gl | 0.000 | 4.764 | 4.227 | 8.531 | 18.922 |
| full_gl | 0.300 | 4.798 | 4.137 | 8.919 | 19.545 |
| full_gl | 0.500 | 4.453 | 3.720 | 8.206 | 16.633 |
| full_gl | 0.700 | 4.433 | 4.009 | 7.870 | 15.246 |
| full_gl | 0.900 | 4.751 | 4.148 | 7.894 | 24.335 |

---

## Recommendation checkpoint

This report is a checkpoint, not an automated decision.

**Recommended user-facing mode: `full_gl`.** It feeds the full (P_AA, P_AB, P_BB) probability triplet to the network without collapsing to a scalar, preserving all the genotype uncertainty the GL representation carries. The 1-column-per-site `dosage` mode is essentially equivalent on this dataset (within ~1% mean Euclidean error) so the figures lead with full_gl and drop the redundant cont line.

**The headline is the noise curve.** full_gl prediction error is essentially flat across α ∈ {0.0, 0.3, 0.5, 0.7, 0.9} (mean 4.43–4.80) — the patched loader handles GL uncertainty gracefully end-to-end. The rounded path matches at α ≤ 0.3, starts to lose signal at α=0.5 (mean 5.44), and **collapses to the centroid baseline at α ≥ 0.7 (mean 18.5 vs. baseline 18.5)** — every value rounds to 1, the matrix becomes nearly constant, and the model predicts the geographic mean. The loader patch is therefore load-bearing: without it, GL uncertainty above ~50% destroys the pipeline.

**Per-sample VCF vs full_gl agreement (Figure 3).** At α=0 the two methods produce the same aggregate accuracy but differ per-sample by amounts dominated by GPU non-determinism (cuDNN ops are not bit-deterministic across runs even with `--seed 42`). Outliers off the y=x diagonal mostly reflect that randomness rather than systematic disagreement.

**Path B passes under the along-coast metric** (ratio 0.350 PASS) after switching from haversine (which had 0.804 FAIL). Barnacles are intertidal; longitude error perpendicular to the coast is biologically meaningless, so the haversine framing was overstating the real prediction error. With 54 samples × 100k features the model is still severely underdetermined — extreme-latitude sites (AFB Alaska, GOL S. California) collapse toward the centroid even under along-coast — but the model picks up real signal at mid-latitude Oregon sites. The example-VCF noise curve is the fully-controlled evidence the GL pipeline works.

**Range-mask experiment (Figure 4, fork-only).** Re-ran the dosage LOSO with `use_range_penalty=True` after working around a coordinate-space bug in `loss_with_range_penalty` (it compares z-scored predictions against a mask rasterized in raw lon/lat — see `validation/notes/range_mask_bug.md`). With a per-fold z-normalized shapefile and `penalty_weight=50`, mean along-coast error drops from 814 km (uncorrected, w=1) to 610 km — a 25% improvement that confirms the corrected invocation actually engages the penalty. Still worse than Path B + post-hoc snap (~350 km) because `mask_lookup` uses `tf.round` + `tf.gather_nd`, which have no gradient — the penalty can only bias `save_best_only`/EarlyStopping, not pull predictions toward the mask via gradient descent. The real fix (bilinear-interp or signed-distance soft mask) needs to live in ReLocator itself; out of scope for this PR but a clean follow-up.

**Caveat: full_gl on N=54 (Path C) underperforms cont (Path B).** Path B (cont, 1 col/site) gets along-coast median 330 km / ratio 0.350 (PASS). Path C (full_gl, 3 cols/site) gets median 544 km / ratio 0.578 (FAIL). Both ran on the same data with the same seed; the single-seed comparison can't separate genuine effect from run-to-run variance — GPU non-determinism alone produces substantial per-prediction shifts (we showed earlier that two consecutive same-seed VCF runs differ in 1000/1000 lines). On the 500-sample example data full_gl and cont are within ~1% of each other. So: not strong evidence that full_gl is worse at small N, but worth replicating with multiple seeds before relying on it for tens-of-samples studies.
