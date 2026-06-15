# Prediction Modes

ReLocator supports three prediction modes via the `prediction_mode`
config option (or `--prediction_mode` on the CLI). They differ only in
the output head of the network and the inference math; the encoder and
training pipeline are otherwise identical.

| Mode | Output head | Loss | Inference |
|---|---|---|---|
| `regress` (default) | `Dense(2) → Dense(2)` | Euclidean distance | network output, denormalized |
| `classify` | `Dense(K, softmax)` | categorical cross-entropy | argmax → site centroid → denormalized |
| `classify_then_avg` | `Dense(K, softmax)` | categorical cross-entropy | softmax @ centroids → denormalized |

`K` is the number of distinct training-site centroids (auto-derived
from exact coord equivalence among training samples).

## When to use each

### `regress` (default)

Use for any data where samples are continuously distributed in space and
no discrete sampling-site grouping is meaningful — for example, vagrant
animals tracked at random points along a migration route, or samples
where every individual has a unique GPS coordinate.

### `classify`

Use when training individuals are clearly grouped at a small number of
known sampling sites (popgen-canonical sampling design) and you want a
hard prediction "this sample came from site X." Returns the centroid of
the most-likely site. Provides a classification accuracy you can compute
directly (correct site or not).

### `classify_then_avg`

Use when training is grouped by sites but you want a continuous (lat,
lon) output that smoothly handles uncertainty between adjacent sites.
The prediction is bounded to the convex hull of training-site centroids
by construction, so predictions cannot land in the middle of nowhere.
For admixed individuals (e.g., genuinely intermediate between two
sites), the prediction lands geometrically between them, weighted by
the classifier's confidence in each.

## Worked example

You sample 405 prickly sculpin (*Cottus asper*) from 16 lakes/rivers
across the Pacific Northwest. Each lake's coordinates are recorded once,
and every individual from that lake gets the same (lon, lat). You train
ReLocator on 15 of the 16 lakes (LOSO test) with `prediction_mode='classify'`.

For each held-out individual:

- **`regress`** would output a continuous (lon, lat) somewhere on a
  smooth surface fit to the 15 training sites. Predictions for outlier
  sites (geographically isolated, like Alaska) tend to drift toward the
  centroid of training data — the regression-to-the-mean failure mode.
- **`classify`** outputs the centroid of one of the 15 training lakes
  (the one with highest softmax probability). Predictions are guaranteed
  to land at one of the trained-on sites.
- **`classify_then_avg`** outputs `Σ_k P(site_k) × centroid_k`. For an
  individual whose genotype is consistent with two training sites at
  similar probability, the prediction lands at the geographic midpoint
  between them, weighted by the relative confidence.

## Caveats

1. **`classify` and `classify_then_avg` are mutually exclusive with
   `use_range_penalty=True`.** The convex hull of training-site
   centroids is already the natural "valid range," and the rasterized
   range mask doesn't apply to softmax outputs. Setting both raises a
   clear `ValueError`.

2. **`derive_site_labels` requires K ≤ ceil(N / min_samples_per_site)
   by default.** If your data has more unique training coords than half
   the number of training samples, the classifier head is degenerate
   and the helper raises a ValueError pointing you at
   `prediction_mode='regress'`. Override with `min_samples_per_site=1`
   if you really want K = N (testing only).

3. **Saved model files (HDF5) include the centroids matrix** so you can
   reload and predict in a new session without re-training. The
   classifier modes round-trip cleanly through `load_model`.

4. **For applying a trained model to a sample from a population that
   wasn't in training**, use `regress` mode if you want the prediction
   to lie outside the trained sites' convex hull. The classifier modes
   are bounded by construction.

## Graph-aware inference (optional)

Both `classify` and `classify_then_avg` can optionally apply a
heat-kernel smoothing over a user-supplied connectivity topology before
the centroid lookup. This is useful when sampling sites are linked by a
known structure that gene flow follows (rivers, roads, valley
corridors) — Euclidean proximity isn't always biologically meaningful.

### How to use

Two new config options, both optional:

| Option | Type | Default | Description |
|---|---|---|---|
| `graph_topology` | path to GeoJSON | None | If set, applies heat-kernel smoothing |
| `graph_kernel_resolution` | float | 1.0 | Diffusion time `t` in `exp(-t * L)`; larger = more smoothing |

Combining `graph_topology` with `prediction_mode='regress'` raises a
clear `ValueError` — the topology has no meaning for continuous
regression output. Use `classify` or `classify_then_avg`.

### GeoJSON schema

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [-132.0, 53.5]},
      "properties": {"name": "MAL"}
    },
    {
      "type": "Feature",
      "geometry": {"type": "LineString",
                   "coordinates": [[-132.0, 53.5], [-131.97, 53.52], [-131.94, 53.55]]},
      "properties": {"name": "MAL_to_Tlell"}
    }
  ]
}
```

- **Point features** are nodes. Their coordinates should match (or be
  within ~10 km of) the training-site centroids; matching uses
  haversine distance via a KD-tree.
- **LineString features** are edges. The geometry traces the actual
  path (river course, road, etc.) and is read end-to-end. Edge weight
  in the graph Laplacian is the haversine sum along the LineString in
  km. LineString endpoints must be within ~1 km of two distinct Point
  features.

### Math

Inference path:

```
network → (n, K) softmax P
       → P' = P @ H  where H = exp(-graph_kernel_resolution * L)
                     and L is the K×K weighted graph Laplacian
       → classify          : argmax(P') → centroids[idx] → denormalize
       → classify_then_avg : P' @ centroids → denormalize
```

`graph_kernel_resolution = 0` recovers the un-smoothed classifier
exactly. Larger values smooth probability mass more aggressively along
edges. Suggested starting value: `1.0`. Tune per dataset.

### LOSO-safety

For LOSO-style validation, the topology spans the FULL site set; each
fold's training subset is automatically detected from the trained
model's `_site_centroids`. Edges involving held-out sites are dropped
before the Laplacian is built. The user provides the topology once;
the per-fold subset is invisible.
