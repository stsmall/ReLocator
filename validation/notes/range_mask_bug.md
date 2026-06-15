# `loss_with_range_penalty`: coordinate-space mismatch (and zero gradient)

**Status:** out of scope for the genotype-likelihoods PR. Tracked here so we
can open a separate ReLocator PR with the fix.

## Symptom

When training with `use_range_penalty=True`, predictions don't stay on the
species-range mask. Validating on 12 leave-one-site-out folds of Pacific
Northwest *Balanus glandula* (`validation/run_loso_rangemask.py` at
`penalty_weight=1.0`, `resolution=0.05`, coast polygon = 0.1° buffer of
Natural Earth's 10m coastline clipped to the Pacific NW bbox) gave a
mean along-coast error of ~814 km — the model behaved indistinguishably
from `use_range_penalty=False` plus post-hoc snap-to-coast.

## Root cause

Two independent issues compound:

### 1. Coordinate-space mismatch

`locator/training.py::_build_datasets_and_fit` passes `normalized_locs`
(z-scored per-axis from `locator/data/filters.py::normalize_locs`) into
`make_tf_dataset`. So inside the loss:

- `y_true`, `y_pred` are in z-score space, ~ N(0, 1) per axis.
- `mask_tensor`, `transform` come from `rasterize_species_range`, which
  reads `gdf.total_bounds` of the shapefile (raw lon/lat) and builds an
  `Affine.translation(xmin, ymin) * Affine.scale(resolution, resolution)`.

`mask_lookup` then computes:
```
col = clip((lon - transform.c) / resolution, 0, W-1)
```
With `transform.c ≈ -152.0` (raw lon) and `lon ≈ 0` (z-score), this gives
`(0 - -152) / 0.05 ≈ 3040`, far past the mask's ~740 columns. `clip`
saturates every prediction to cell `(0, 0)`. Whether `(0, 0)` is inside
or outside the mask determines the penalty for *every* sample identically;
it has nothing to do with the prediction.

### 2. Zero gradient through the mask lookup

Even if the coordinate spaces matched, `mask_lookup` does:
```python
col = tf.cast(tf.round(col), tf.int32)
row = tf.cast(tf.round(row), tf.int32)
valid = tf.gather_nd(mask_tensor, idx)
```
`tf.round` and `tf.cast(..., tf.int32)` have no registered gradient (they
return zero gradient by default). `tf.gather_nd` with integer indices
produces no gradient with respect to those indices either. So
`∂valid_mask/∂y_pred = 0` exactly, and adding `penalty_weight * (1 -
valid_mask)^2` to the loss does **not** push predictions toward the mask
via gradient descent. The penalty only affects:

- The reported total loss (so EarlyStopping / `save_best_only` can pick
  the epoch where predictions happened to land on valid cells most often)
- Adam's moving averages (indirectly, via the loss scalar)

Net effect: at any sane `penalty_weight`, the gradient-driven training
trajectory is unchanged from `euclidean_distance_loss`. Bumping
`penalty_weight` doesn't help — it just adds a larger constant to the loss.

## Workaround used in this branch

`validation/run_loso_rangemask.py` works around (1) only:

- Per fold, compute `(mean_lon, sd_lon, mean_lat, sd_lat)` from the
  held-out fold's training samples (matches `normalize_locs` exactly).
- Apply the same z-score affine to the coast polygon via
  `shapely.affinity.affine_transform`, write a fold-specific shapefile,
  pass it as `species_range_shapefile`.
- Use `resolution=0.01` in z-score units (~10 km at sd_lat ≈ 10°) and
  `penalty_weight=50.0` so the penalty term is large compared to the
  z-space euclidean term (~1–3 units).

This is enough to make the penalty operate in the right coordinate
space, so EarlyStopping / `save_best_only` see meaningful loss values.
We don't fix (2) — that requires changing `mask_lookup` itself.

## Proper fix (separate PR)

Both issues should be fixed inside `loss_with_range_penalty` so the
existing public API continues to work:

1. **Thread normalization params through the loss.** When
   `use_range_penalty=True`, capture `(meanlong, sdlong, meanlat, sdlat)`
   in the loss closure (same `_create_model` call site) and unnormalize
   `y_pred` before the mask lookup:
   ```python
   y_pred_raw = tf.stack([
       y_pred[:, 0] * sdlong + meanlong,
       y_pred[:, 1] * sdlat + meanlat,
   ], axis=-1)
   valid_mask = mask_lookup(y_pred_raw, mask_tensor, transform, resolution)
   ```
   Or equivalently, transform the raster `transform` into z-score space
   once at construction time.

2. **Restore gradient flow through the mask.** Replace the round/cast/
   gather pipeline with a differentiable approximation. Two reasonable
   options:
   - **Bilinear interpolation:** treat `mask_tensor` as a smooth
     occupancy field and interpolate the four neighboring cells. The
     gradient is then proportional to the local mask gradient.
   - **Soft mask via signed distance:** precompute a signed-distance
     transform of the binary mask (negative inside, positive outside),
     interpolate it bilinearly, and use a soft penalty
     `relu(d)^2 * penalty_weight`. This gives a gradient that points
     directly toward the mask boundary even far outside.
   The signed-distance approach is what most range-penalty implementations
   in the geo-prediction literature use; it's strictly better than a
   binary mask once you've paid the rasterization cost.

3. **Add a regression test.** With the workaround applied (or option 1
   alone), training on a tiny synthetic dataset with a known mask should
   produce predictions inside the mask region at higher rate than
   `euclidean_distance_loss`. Today, no such test exists.

## File pointers

- `locator/models.py:50` — `mask_lookup`
- `locator/models.py:78` — `loss_with_range_penalty`
- `locator/training.py:563` — `_create_model` builds the loss closure
- `locator/training.py:858` — `_build_datasets_and_fit` feeds normalized locs
- `locator/data/filters.py:58` — `normalize_locs` (the per-axis z-score)
- `validation/run_loso_rangemask.py` — workaround harness
