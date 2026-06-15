# Graph-Aware Inference for ReLocator's Classifier Head: Design

Design doc for the `graph-classifier` branch. Adds optional inference-time
graph awareness to ReLocator's classifier output head: load a user-supplied
GeoJSON describing site topology (Point features = nodes, LineString
features = edges/connection paths), apply a heat-kernel diffusion to the
softmax output before the centroid average. No training-time changes;
augments the existing `classify` and `classify_then_avg` modes via a
`--graph_topology` flag. Decoupled from input format (works for SNPs, GLs,
microsats — anything that flows through the existing classifier head).

## Context

The flat classifier head (`classifier-head` branch, `classify` and
`classify_then_avg` modes) demonstrably extracts spatial signal from real
microsat data on the sculpin LOSO arc — both new modes beat regress by
12–15 km (252–254 km vs 267 km). But the classifier head treats sites as
unrelated K-way categories. Sculpin populations, like many freshwater
fish, are connected by waterways: two lakes sharing a river drainage are
genetically closer than two equidistant lakes that aren't.

The sculpin genetic-structure analysis (`sculpin_genetic_structure.md`)
flagged the regression head's regression-to-the-mean failure at FST-outlier
sites (Alaska, Okanagan, Nimpo). The flat classifier doesn't fully fix this
— per-site analysis shows the high-FST sites still get high error because
the classifier has no prior that "Mosquito and MAL are connected; Alaska
and Okanagan are not."

A graph-aware inference step exploits this connectivity prior. It
generalizes to non-freshwater domains: roads connecting cities, valley
corridors connecting alpine populations, coastal currents connecting
intertidal sites. The user supplies the topology; ReLocator core stays
generic.

## Goals

1. Add `graph_topology` and `graph_kernel_resolution` (also called
   `graph_kernel_t`) config options that, when set, apply a heat-kernel
   smoothing to the classifier's softmax output before the centroid
   average. Compatible with `classify` and `classify_then_avg`; ignored
   in `regress` mode.
2. Define and document a GeoJSON schema with Point features (nodes) and
   LineString features (edges, with the geometry tracing the connection
   path). Edge weight = haversine length of the LineString.
3. Provide a small builder helper for the sculpin case
   (`validation/sculpin/build_topology.py`) that auto-derives a topology
   GeoJSON from HydroSHEDS HydroLAKES + HydroRIVERS, demonstrating the
   pattern users would adapt for other data sources.
4. Validate against the sculpin 16-fold LOSO with the existing 4-mode
   classifier baseline plus the new graph-aware variant — same trained
   classifier-head models, no retraining needed.
5. Produce evidence specifically targeting the FST-outlier sites
   (Alaska, Okanagan, Nimpo) — the sculpin sites where the flat
   classifier currently fails most.

## Non-goals

- **Training-time graph-regularized loss.** Deferred to a v2 follow-up
  that builds on the topology infrastructure shipped here.
- **GNN-style graph-aware encoder.** Deferred to v3. Would change the
  layers BEFORE the output head, not just the post-processing.
- **Random-walk smoothing or distance-weighted reweighting alternatives**
  to the heat kernel. Easy to add later if heat kernel underperforms.
- **Auto-derivation of topology for non-HydroSHEDS data sources** (road
  networks, mountain corridors). The pattern in
  `validation/sculpin/build_topology.py` is the template; users adapt it
  per domain.
- **Per-individual graphs.** The topology is over sampling sites, not
  over individuals. (Per-individual graphs would be a different problem
  — kinship inference, family-aware models — not in scope.)
- **CLI flag exposure beyond `--graph_topology` and
  `--graph_kernel_resolution`.** The Python API receives both via the
  existing `Locator(config=...)` dict.
- **Compatibility with `prediction_mode='regress'`.** The graph
  topology has no meaningful interpretation for continuous regression
  output. Setting both raises a clear `ValueError` at config validation.

## Branch & PR shape

- **Branch:** `graph-classifier`, off `classifier-head` HEAD `df064ea`.
- **Stacked PR:** depends on the classifier-head PR landing first. Cannot
  merge until classifier-head merges, since it depends on
  `_site_centroids` and the classifier prediction path.
- **Final landing:** the `locator/graph.py` + `locator/prediction.py`
  modifications go to `kr-colab/ReLocator:main` as a standalone PR after
  classifier-head lands. The `validation/sculpin/build_topology.py` +
  `sculpin_topology.geojson` + updated sculpin summary travel with the
  sculpin lineage.

## Architecture

### Mode shape

`graph_topology` is **a flag, not a new mode**. When set with
`prediction_mode='classify'` or `prediction_mode='classify_then_avg'`,
the same prediction code path applies a heat-kernel smoothing step.
When unset, the classifier modes behave exactly as on the
classifier-head branch. With `prediction_mode='regress'`, setting
`graph_topology` is a config error.

Concretely:

```
predict() flow:
  1. network forward pass → (n, K) softmax P
  2. if graph_topology set and prediction_mode in {classify, classify_then_avg}:
        P' = H @ P    where H = exp(-graph_kernel_resolution × L)
                      and L is the K×K Laplacian of the loaded topology
        (otherwise P' = P)
  3. classify         : argmax(P') → centroid lookup → denormalize
     classify_then_avg: P' @ centroids → denormalize
```

### Components

Three new files plus small modifications to two existing ones:

#### `locator/graph.py` (new, ~150 lines)

Module-level pure functions for graph operations:

- `load_topology(path: Path) -> dict` — reads a GeoJSON via geopandas;
  validates that it contains both Point features (nodes) and LineString
  features (edges); returns a dict with keys `nodes` (`(K_topo, 2)`
  float64 lon/lat array), `edges` (list of `(i, j, weight_km)` tuples
  where `i`, `j` index into `nodes` and `weight_km` is the haversine
  length of the LineString), and `node_names` (list of strings from
  Point properties, default empty if absent).
- `match_centroids_to_nodes(centroids: np.ndarray, topology: dict, tol_km: float = 10.0) -> np.ndarray` —
  for each row in `centroids` (the trained model's `_site_centroids`,
  in normalized space → caller denormalizes first), find the nearest
  topology node within `tol_km` haversine kilometers. Returns a
  `(K_train,)` int array mapping each centroid to its node index.
  Centroids with no node within tolerance get -1 and trigger a warning.
- `subset_edges_to_centroids(edges: list, centroid_to_node: np.ndarray) -> list` —
  given the full topology's edge list and the per-fold centroid→node
  mapping, return the subset of edges where both endpoints map to a
  trained centroid (with indices renumbered to `(0..K_train-1)`). Drops
  edges that involve LOSO-held-out sites. LOSO-safe.
- `heat_kernel(edges: list, k: int, t: float) -> np.ndarray` — build the
  `(K, K)` graph Laplacian from the edge list (with edge weights
  contributing to the off-diagonal and degree to the diagonal), return
  `scipy.linalg.expm(-t * L)`. For `K ≤ a few hundred` this is
  millisecond-fast.

#### `locator/prediction.py` (small modifications)

In `predict()`, after the network forward pass and before the
`mode == "classify"` / `"classify_then_avg"` branches, insert a
graph-smoothing step:

```python
if (config.get("graph_topology") is not None
        and mode in ("classify", "classify_then_avg")):
    if not hasattr(self, "_heat_kernel"):
        # Lazy load: build the kernel once on first predict.
        topology = load_topology(Path(config["graph_topology"]))
        # Denormalize centroids for matching against raw lon/lat.
        centroids_raw = self._site_centroids.copy()
        centroids_raw[:, 0] = centroids_raw[:, 0] * self.sdlong + self.meanlong
        centroids_raw[:, 1] = centroids_raw[:, 1] * self.sdlat + self.meanlat
        c_to_n = match_centroids_to_nodes(centroids_raw, topology)
        edges = subset_edges_to_centroids(topology["edges"], c_to_n)
        self._heat_kernel = heat_kernel(
            edges, k=self._n_classes,
            t=config.get("graph_kernel_resolution", 1.0),
        )
    # Apply (n, K) @ (K, K) → (n, K)
    predictions = predictions @ self._heat_kernel
```

Caching: kernel is built once and stored on `self._heat_kernel`. Cleared
when the model is reloaded (in `load_model` if needed).

#### `locator/training.py` (small validation addition)

Extend `_validate_prediction_mode_config` to reject
`prediction_mode='regress'` combined with `graph_topology`:

```python
if config.get("graph_topology") is not None and mode == "regress":
    raise ValueError(
        "graph_topology is only meaningful with classifier modes; "
        "set prediction_mode='classify' or 'classify_then_avg', "
        "or remove graph_topology."
    )
```

#### `validation/sculpin/build_topology.py` (new, validation-zone)

Domain-specific builder for the sculpin case. Uses the existing HydroSHEDS
infrastructure from `validation/sculpin/build_range.py`:

1. Load `validation/sculpin/sites.tsv` (16 sampling sites with lat/lon).
2. Load HydroLAKES + HydroRIVERS; build a connectivity graph by
   identifying river reaches between sampled lakes, OR by buffering
   each site and unioning overlapping freshwater geometry.
3. For each pair of connected sites, find the shortest path through the
   river network and emit a LineString tracing that path.
4. Write a GeoJSON `validation/sculpin/sculpin_topology.geojson` with
   Point features for each site and LineString features for each
   connection.

The graph-construction algorithm choice (river-network shortest path vs
freshwater overlap) is an implementation detail captured in the plan
document, not the spec. Users who care about exact connectivity can
hand-edit the resulting GeoJSON.

#### `validation/sculpin/sculpin_topology.geojson` (committed artifact)

The output of `build_topology.py`. Committed to the branch so the
validation runner doesn't need HydroSHEDS to reproduce results.

### Hyperparameters

Two new config options, both optional:

- **`graph_topology`** (str path): GeoJSON file. When None (default),
  graph-aware inference is disabled.
- **`graph_kernel_resolution`** (float, default `1.0`): also called
  diffusion time `t` in the heat kernel `H = exp(-t L)`. Larger values
  smooth probability mass more aggressively across the graph. `t = 0`
  recovers the flat classifier exactly. The user may need to tune per
  dataset; the validation runner sweeps a small set of values
  (`{0.1, 0.5, 1.0, 2.0, 5.0}`).

### GeoJSON schema (user-facing contract)

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
      "geometry": {"type": "Point", "coordinates": [-131.94, 53.55]},
      "properties": {"name": "Tlell"}
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

- **Point features = nodes.** The `coordinates` field is `[lon, lat]`.
  Properties are read but only `name` is used.
- **LineString features = edges.** The geometry traces the actual path
  of the connection (river course, road, etc.). Edge weight = haversine
  length of the path in km. Endpoints are matched to the nearest Point
  feature within a small tolerance (default 1 km — different from the
  larger centroid-matching tolerance) to determine which two nodes the
  edge connects.

The format is permissive: extra properties are ignored, the order of
features doesn't matter, and Point features without explicit names get
auto-named by index.

### Data flow summary

```
training:    unchanged (classifier-head infrastructure)

inference:   model output (n, K) softmax  P
                                            │
                                            ▼ (if graph_topology + classify* mode)
             load GeoJSON → match_centroids_to_nodes(_site_centroids)
                          → subset_edges_to_centroids(edges, c_to_n)
                          → heat_kernel(edges, K, t)
                          → cache as self._heat_kernel
                                            │
                                            ▼
             P' = P @ self._heat_kernel  (per-prediction matmul)
                                            │
                                            ▼
             classify         : argmax(P') → centroids[idx] → denormalize
             classify_then_avg: P' @ centroids → denormalize
             regress          : (config error caught earlier)
```

## Tests (merge zone)

`tests/test_graph_classifier.py`:

- `test_load_topology_basic` — hand-built GeoJSON with 4 Points + 3
  LineStrings; verify nodes/edges/weights match expected.
- `test_load_topology_haversine_weight` — LineString with known
  geometry; weight should be the haversine sum of segment lengths.
- `test_load_topology_rejects_malformed` — missing geometry, empty
  feature list, or no Point features → ValueError.
- `test_match_centroids_to_nodes_within_tolerance` — centroid at
  `(0, 0)` matches a Point at `(0.05, 0.05)` (within 10 km tolerance);
  centroid at `(10, 10)` doesn't match.
- `test_match_centroids_to_nodes_no_match_within_tolerance` —
  centroid with no node within tol → -1 + warning.
- `test_subset_edges_to_centroids_drops_held_out_sites` — full
  topology has 4 nodes; centroid mapping covers only 3; edges involving
  the missing node are dropped from the result.
- `test_heat_kernel_t_zero_is_identity` — `heat_kernel(edges, K, t=0)`
  returns `np.eye(K)` exactly.
- `test_heat_kernel_symmetric` — `H == H.T` to floating-point
  tolerance (Laplacian is symmetric, so `expm(-t L)` is too).
- `test_heat_kernel_row_sums_to_one_under_normalized_laplacian` — if
  we use the random-walk normalized Laplacian variant; otherwise this
  test asserts the conservation property of the standard Laplacian.
- `test_predict_classify_with_graph_smoothing_changes_argmax` — train
  a classifier on synthetic 4-site data; without graph, argmax is site
  A; with a topology that strongly connects site A to site B, argmax
  shifts (or probability mass redistributes). Verifies the smoothing
  has an observable effect on predictions.
- `test_predict_regress_with_graph_topology_errors` — config error
  fires early.

Run inside pixi: `pixi run pytest tests/test_graph_classifier.py -v`.

## Validation evidence (sculpin LOSO)

Re-run the sculpin 16-fold LOSO with all five modes side-by-side, using
the **same trained classifier-head models** (no retraining needed —
this is inference-time only):

1. `regress` (existing, 267.2 km baseline)
2. `regress + range_mask` (existing, 249.6 km)
3. `classify` (existing, 254.5 km)
4. `classify_then_avg` (existing, 252.3 km)
5. **`classify_then_avg` + graph topology** (new, several values of `t`)

Per-site comparison expected to win at FST-outlier sites where the flat
classifier currently fails:

- **Alaska** (581 km flat-regress, 567 km flat-classify): the topology
  isolates Alaska from the BC mainland; graph-aware predictions should
  refuse to assign Alaska samples to BC sites.
- **Okanagan** (632 / 570 km): topology connects Okanagan to the
  Columbia River system; graph predictions should respect that.
- **Nimpo** (380 / 325 km): topology shows it's an isolated interior
  lake; graph predictions should down-weight far-away coastal sites.

Generate an updated `sculpin_classifier_modes.png` with the 5th set of
bars. New summary: `validation/summary/sculpin_graph_summary.md`.

The success criterion is **not** "graph mode beats every other mode by
X km on aggregate." It's: **at the high-error FST-outlier sites, the
graph mode's median error must be lower than the flat classifier's**.
If the topology isolates Alaska, the graph mode shouldn't put Alaska
predictions in the BC interior. Aggregate improvements are bonus.

## Documentation

- New section in `docs/prediction_modes.md` (the user-facing classifier
  guide shipped on classifier-head): "Graph-aware inference" subsection
  describing the GeoJSON schema, the `--graph_topology` /
  `--graph_kernel_resolution` flags, and a worked example with the
  sculpin topology file.
- `validation/sculpin/build_topology.py` is itself documentation by
  example — users can copy its structure for road-network or
  mountain-corridor analogues.

## Out of scope (declared TODOs)

- Training-time graph-regularized loss (v2 follow-up).
- GNN encoder (v3 follow-up — significantly more work).
- Random-walk smoothing variants of the kernel.
- Distance-weighted reweighting (alternative to heat kernel).
- Auto-derive helpers for non-HydroSHEDS data (roads, mountain
  corridors) — pattern is the same; users adapt.
- CLI flags beyond `--graph_topology` and `--graph_kernel_resolution`.
- Per-fold caching of the heat kernel beyond the lazy first-call cache.
- Edge-weight normalization variants (currently raw haversine length;
  could try inverse, log, drainage-flow-direction-aware, etc.).
- Allowing K_topo > K_train (topology covers more sites than the
  trained model knows about). For LOSO this is the typical case;
  the `subset_edges_to_centroids` helper handles it.
- Per-edge typing (e.g., distinguishing "river" from "ocean" edges in
  a mixed-domain topology). Could be added as edge `properties.type`
  in v2.
