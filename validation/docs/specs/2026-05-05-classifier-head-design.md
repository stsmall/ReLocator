# Classifier Head for ReLocator: Design

Design doc for the `classifier-head` branch. Adds an optional discrete-site
classifier output head to ReLocator's network, with three prediction modes
selectable via a single config flag. The change is decoupled from input
format (works for SNPs, GLs, microsats — anything that produces a continuous
feature matrix) and is scoped to ReLocator core; it does not depend on the
`microsatellites` or `genotype_likelihoods` input pipelines beyond using
their existing data plumbing.

## Context

The sculpin LOSO validation on the `microsat-sculpin` branch documented a
quantitative failure mode of ReLocator's regression head:

- **Snap-correctness rate = 8 %** — fewer than 1 in 12 individuals had
  their regression-mode prediction land closer to the true held-out site
  than to any other training site. The trained model puts predictions in
  the wrong neighborhood the vast majority of the time, even though the
  data has clearly identifiable population structure (mean pairwise FST =
  0.158, with significant clusters in PCA space).
- **Per-site error correlates positively with mean FST** (Pearson r =
  +0.541, p = 0.030) — sites that are genetically distinct from the rest
  of the dataset have *higher* LOSO error, not lower. This is the
  signature of regression-to-the-mean at the geographic edges: the
  network smooths predictions toward the centroid of the training data,
  which systematically penalises FST-outlier sites that happen to also
  be geographic outliers.
- The post-hoc snap-to-nearest-site experiment showed that snapping the
  existing regression predictions to the nearest training-site centroid
  reduces aggregate error from 267 km to 254 km (~5 %). This is a small
  win, and it caps how much improvement is achievable from snap alone:
  the trained features only put predictions in the correct neighborhood
  8 % of the time.

A discrete-site classifier head, trained with a categorical loss directly,
should optimise for site-discriminative features rather than for
continuous-coord smoothness. Two related modes are useful:

- `classify`: predict the most-likely site, return its centroid. Strong
  but committed-to-one-site.
- `classify_then_avg`: weight all training-site centroids by the softmax
  output, return the weighted mean. Continuous output bounded to the
  convex hull of training sites; smoothly handles uncertainty between
  adjacent sites.

## Goals

1. Add a `prediction_mode ∈ {regress, classify, classify_then_avg}` config
   option to ReLocator's `Locator` API, with `regress` as the default and
   the existing behavior unchanged when the option is unset.
2. Implement the classifier head as a final-layer change in
   `locator/models.py` and integrate it into the existing training/
   prediction code paths in `locator/training.py` and
   `locator/prediction.py`.
3. Validate against the sculpin 16-site LOSO with all three new modes
   plus the existing `regress` and `regress + range_mask` baselines (4
   modes total).
4. Produce evidence that the failure mode of the regression head
   (positive error-vs-FST correlation, low snap-correctness rate)
   improves under the classifier modes — or, if it does not improve,
   document that finding clearly.

## Non-goals

- `smm_kernel` microsat encoding (separate follow-up).
- Balanus GL re-validation under the classifier head (separate follow-up
  arc once this lands in GL/main).
- Multi-task hybrid (regression + classification simultaneously). v1
  makes the three modes mutually exclusive.
- Learnable centroids (treating `C ∈ R^{K×2}` as model parameters that
  drift during training). Stretch goal; v1 fixes them at training-set
  centroids.
- Top-K averaging or temperature scaling on the softmax output. Easy to
  add later; not in v1.
- Compatibility with `use_range_penalty=True` under classifier modes.
  The convex hull of training-site centroids is already the natural
  "valid range," and the rasterized-mask loss does not apply to
  probability-vector outputs. v1 errors at config-validation time if
  both flags are set.
- Site-label inference for datasets without coord-grouped samples
  (every individual has unique GPS — the elephant case). v1 errors
  out at training time with a recommendation to use `regress` instead.

## Branch & PR shape

- **Branch:** `classifier-head`, off `microsat-sculpin` HEAD `e39606f`.
- **Commit shape:** the `locator/` core changes are designed to be
  cherry-pickable onto `main` as a standalone PR once validated. The
  `validation/sculpin/` outputs travel with whichever branch carries
  the sculpin work.
- **Final landing strategy:** after sculpin validation passes, the
  `locator/` changes go to `kr-colab/ReLocator:main` as their own PR
  (independent of the GL and microsat input PRs); the validation
  evidence stays on the `microsat-sculpin` lineage.

## Architecture

### `locator/models.py:create_network`

Add an `n_classes: int | None = None` parameter. When `n_classes is None`
(default), the existing two-Dense-layer regression head is preserved
byte-identical. When `n_classes` is set:

- Replace the final two `Dense(2)` layers with a single
  `Dense(n_classes, activation="softmax")`.
- Default loss switches from `euclidean_distance_loss` to
  `tf.keras.losses.CategoricalCrossentropy()`.
- The `loss_fn` parameter still overrides the default, so a caller can
  supply their own loss (used by tests).

The `loss_with_range_penalty` and `rasterize_species_range` helpers do
not need changes; they are simply not callable in classifier modes.

### `locator/training.py:_create_model` and `_build_datasets_and_fit`

Add a `prediction_mode` field to `Locator.config` (validated at the top
of `train`). When `prediction_mode != "regress"`:

1. **Validation gate.** If `use_range_penalty` is also set, raise a
   clear `ValueError`: "range_mask is incompatible with classifier
   modes; predictions are bounded to the convex hull of training-site
   centroids by construction."
2. **Site-label derivation.** During training-fold setup, after the
   existing `normalize_locs` call, call a new helper
   `derive_site_labels(locs)`:

   ```python
   def derive_site_labels(normalized_locs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
       """Group training samples by exact coord equivalence.

       Returns:
         labels:    int array (n,) of site index per training sample
         centroids: float32 array (K, 2) of unique (lon, lat) centroids
                    in normalized coord space, in label order.
       Raises ValueError if K > 0.5 * n (suggesting unique-per-individual
       coords, where classification doesn't apply).
       """
   ```

   The "K > 0.5 * n" threshold is the elephant-case detector. The
   threshold is configurable via `min_samples_per_site` (default 2) for
   users who want stricter or looser site-membership rules.
3. **Label encoding.** One-hot encode the integer labels into shape
   `(n, K)` and pass them to the model as the training target instead
   of the normalized (lat, lon) array.
4. **Centroid storage.** Store the `(K, 2)` centroid matrix on the
   `Locator` instance as `self._site_centroids` for the prediction
   path. Persist it to the saved model's HDF5 metadata so prediction
   from a saved model works.
5. **Pass-through to `_create_model`.** Forward `n_classes=K` and a
   `categorical_crossentropy` loss to `create_network`.

### `locator/prediction.py:predict`

Branch on `self.config["prediction_mode"]`:

- `regress` (default): existing path, unchanged.
- `classify`: model output is `(n, K)` softmax. Compute
  `argmax_k` per sample, look up `self._site_centroids[argmax]`, then
  apply the existing `* sd + mean` denormalization. Result: `(n, 2)`
  array of (lat, lon).
- `classify_then_avg`: model output is `(n, K)` softmax. Compute
  `softmax @ self._site_centroids` (matmul, shape `(n, 2)`), then
  apply the same denormalization. Result: continuous (lat, lon)
  predictions bounded to the convex hull of training centroids.

In addition to the (lat, lon) prediction, classifier modes can return
the full softmax probability vector and the predicted site label as
optional outputs (controlled by an `include_classifier_outputs=True`
flag on `predict()`). The default return type stays the same as
regress mode for backward compatibility.

### Data flow summary

```
input genotypes ─── encoder ──── final layer ─── output
                                       │
                                       ├── regress: Dense(2), MSE loss
                                       │            (lat, lon) z-scored
                                       │
                                       └── classify*: Dense(K, softmax), cross-entropy loss
                                                      P(site_k) for k = 1..K

prediction:
  regress         : raw output → denormalize → (lat, lon)
  classify        : argmax → centroids[idx] → denormalize → (lat, lon)
  classify_then_avg: softmax @ centroids → denormalize → (lat, lon)
```

## Tests (merge zone)

Unit tests in `tests/test_classifier_head.py`:

- `derive_site_labels` correctly groups samples by exact coord equality;
  K matches the number of unique coords; centroids are returned in label
  order.
- `derive_site_labels` raises a clear ValueError when K > 0.5 × n
  (elephant case detection).
- `create_network(n_classes=K)` produces a model whose final layer has
  K units with softmax activation; default loss is categorical
  crossentropy.
- `create_network(n_classes=None)` produces the byte-identical regression
  network as before (backward-compat regression test).
- A small end-to-end fit on synthetic data (50 samples × 4 sites): trains
  for a few epochs without crashing, model summary has the expected shape.
- Prediction-time math: given a known softmax output and centroid matrix,
  `classify` returns `centroids[argmax]` and `classify_then_avg` returns
  `softmax @ centroids` exactly.
- Configuration-validation errors fire when `prediction_mode != "regress"`
  is combined with `use_range_penalty=True`.

Run inside pixi: `pixi run pytest tests/test_classifier_head.py -v`.

## Validation evidence (sculpin LOSO)

Re-run the 16-fold LOSO with all four modes side-by-side:

1. `regress` (already done, 267.0 ± 170.4 km)
2. `regress + range_mask` (already done, 249.6 ± 164.8 km)
3. `classify` (new)
4. `classify_then_avg` (new)

New scripts in `validation/sculpin/`:

- `run_loso_classifier.py` — mirrors `run_loso.py` but invokes ReLocator
  with `prediction_mode={classify, classify_then_avg}`. 16 sites × 2 new
  modes = 32 fits, ~5–10 min wall-clock.
- `summarize_classifier.py` — aggregates per-fold JSONs across all 4
  modes into:
  - `validation/figures/sculpin_classifier_modes.png` — grouped bar
    chart, all 4 modes per site + aggregate.
  - `validation/figures/sculpin_classifier_map.png` — per-individual
    prediction map under each mode (4-panel cartopy figure).
  - `validation/summary/sculpin_classifier_summary.md` — written
    comparison report including:
    - Aggregate mean(median) per mode + Δ vs centroid baseline.
    - Snap-correctness rate per mode (regress = 8 %; classifier modes
      should approach 100 % under classify, since training optimises for
      site assignment).
    - Per-site median error table.
    - Re-computed error-vs-FST correlation per mode (the key
      architectural-question metric: regress has r = +0.541; classifier
      modes should weaken or invert it).

The success criterion is **not** "classifier beats regression by X km."
It is "the classifier modes produce a clear, interpretable improvement
in the quantitative diagnostics that flagged the regression failure
mode (snap-correctness, error-vs-FST correlation)." If aggregate error
drops too, that is a bonus.

## Documentation

- New `docs/prediction_modes.md` explaining the three prediction modes
  (`regress`, `classify`, `classify_then_avg`) with biological rationale
  and a worked example of `classify_then_avg`'s convex-hull behavior.
  Separate file rather than appending to `docs/genotype_likelihoods.md`
  because prediction-mode is an architectural concern decoupled from any
  particular input format.
- Update `validation/docs/specs/2026-05-04-microsatellite-input-extension-design.md`
  TODO list to reference `classifier-head` as the architectural follow-up
  that the sculpin validation arc motivates.

## Out of scope (declared TODOs)

- `smm_kernel` microsat encoding.
- Balanus GL re-validation.
- Learnable centroids.
- Top-K / temperature variants.
- Multi-task hybrid loss.
- Range-mask compatibility with classifier modes.
- Auto-K-means clustering for continuously-sampled (elephant-case)
  datasets.
- Native classifier output type from `predict()` (returning probability
  vectors as the default rather than (lat, lon) for backward-compatibility).
