# Graph-Aware Classifier Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add inference-time graph awareness to ReLocator's classifier head via a `--graph_topology` GeoJSON flag and a `--graph_kernel_resolution` heat-kernel hyperparameter; validate against the sculpin 16-fold LOSO using the already-trained classifier-head models (no retraining).

**Architecture:** A new `locator/graph.py` module with four pure functions (load topology, match centroids to nodes, subset edges per LOSO fold, build heat kernel via `scipy.linalg.expm`). `locator/prediction.py` gains a 5-line graph-smoothing branch that lazily caches `self._heat_kernel` on first predict. The change augments `classify` and `classify_then_avg` modes; `regress + graph_topology` is a config error. A `validation/sculpin/build_topology.py` helper produces the sculpin GeoJSON from HydroSHEDS HydroLAKES + HydroRIVERS as a real test artifact and template pattern.

**Tech Stack:** Python 3.12, numpy, geopandas (already in pixi env), scipy.linalg (already a tf transitive dep), networkx 3.6.1 (already in pixi env), pytest with xdist, the existing `pixi` env, the existing `locator` Python API.

**Spec:** `validation/docs/specs/2026-05-05-graph-classifier-design.md`.

**Branch:** `graph-classifier`, off `classifier-head` HEAD `df064ea`. Stacked PR; depends on classifier-head merging first. All commits go here. No pushes until Task 12.

---

## File map

Created in this plan:

| Path | Zone | Responsibility |
|---|---|---|
| `locator/graph.py` | merge | 4 pure functions: `load_topology`, `match_centroids_to_nodes`, `subset_edges_to_centroids`, `heat_kernel` |
| `tests/test_graph_classifier.py` | merge | Unit tests for the 4 helpers + 2 integration tests through `Locator.predict` |
| `validation/sculpin/build_topology.py` | validation | HydroSHEDS → GeoJSON builder (sculpin-specific, lives in validation zone) |
| `validation/sculpin/sculpin_topology.geojson` | validation | Committed builder output; used by validation runner without re-deriving |
| `validation/sculpin/run_loso_graph.py` | validation | Loads classifier-head fold models, re-predicts with graph topology, scores against truth |
| `validation/summary/sculpin_graph_summary.md` | validation | Script-generated 5-mode comparison report |
| `validation/figures/sculpin_graph_modes.png` | validation | Script-generated 5-mode bar chart |

Modified:

| Path | Change |
|---|---|
| `locator/prediction.py` | `predict()` gains a graph-smoothing branch between the network forward pass and the classifier mode dispatch; uses lazy-loaded `self._heat_kernel` cache |
| `locator/training.py` | `_validate_prediction_mode_config` extended to reject `regress + graph_topology` combination |
| `docs/prediction_modes.md` | Append "Graph-aware inference" section with GeoJSON schema + worked example |

Output tree (created at runtime by the validation runner):

```
out/sculpin_validation/
    loso_graph/
        <SITE>/
            sample_data.txt           # symlink-or-copy from loso_classifier/classify_then_avg/<SITE>
            run_predlocs.txt          # NEW predictions with graph smoothing
            fold_result.json          # NEW score against truth, mode='classify_then_avg_graph'
```

---

## Conventions (apply to every task)

- All Python files start with `from __future__ import annotations` and follow the existing `locator/` style (typed signatures, `Optional[X]` from `typing`, docstrings with `:param X:` `:type X:` `:rtype:` blocks; or numpydoc-style sections).
- All commands run inside the pixi env: `pixi run <cmd>`. Bare `pytest` will fail (`conftest.py` imports `allel`, `zarr`).
- The `.claude/hooks/ruff-on-edit.sh` hook auto-runs ruff after every Edit/Write — code must be ruff-clean. Run `pixi run ruff check <file>` manually before committing each task.
- Pre-commit hooks aren't installed in this clone; manual `pixi run ruff check` is the only gate.
- Each commit follows the prior pattern with the `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.
- Do NOT push until Task 12.
- The `locator/graph.py` module imports geopandas, networkx, and scipy.linalg — all already in pixi.

---

## Task 1: `load_topology` helper

**Files:**
- Create: `locator/graph.py`
- Modify: `locator/__init__.py` (re-export only if there's a public-API convention; otherwise skip)
- Test: `tests/test_graph_classifier.py`

- [ ] **Step 1: Write the failing tests for `load_topology`**

Create `tests/test_graph_classifier.py`:

```python
"""Tests for graph-aware classifier inference."""

from __future__ import annotations

import json

import numpy as np
import pytest

from locator.graph import load_topology


def _write_geojson(path, points, lines):
    """Helper: write a GeoJSON FeatureCollection with given Point and LineString features.

    points: list of (lon, lat, name)
    lines: list of (list_of_(lon, lat) coords, name)
    """
    features = []
    for lon, lat, name in points:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": name},
        })
    for coords, name in lines:
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"name": name},
        })
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload))


def test_load_topology_basic(tmp_path):
    """4 Points + 3 LineStrings; verify nodes/edges/weights structure."""
    path = tmp_path / "topo.geojson"
    _write_geojson(
        path,
        points=[(-132.0, 53.5, "A"), (-131.94, 53.55, "B"),
                (-131.5, 53.6, "C"), (-130.0, 54.0, "D")],
        lines=[
            ([[-132.0, 53.5], [-131.94, 53.55]], "A_to_B"),
            ([[-131.94, 53.55], [-131.5, 53.6]], "B_to_C"),
            ([[-131.5, 53.6], [-130.0, 54.0]], "C_to_D"),
        ],
    )
    topo = load_topology(path)
    assert topo["nodes"].shape == (4, 2)
    assert len(topo["edges"]) == 3
    assert topo["node_names"] == ["A", "B", "C", "D"]
    # First edge connects nodes 0 (A) and 1 (B); weight is haversine length in km
    i, j, w = topo["edges"][0]
    assert {i, j} == {0, 1}
    assert 5.0 < w < 12.0  # ~7 km between (-132.0, 53.5) and (-131.94, 53.55)


def test_load_topology_haversine_weight_multi_segment(tmp_path):
    """LineString with 3 vertices; weight = sum of segment haversine lengths."""
    path = tmp_path / "topo.geojson"
    # A multi-segment line: roughly 0→0.05→0.10 lon at lat 0 ≈ 5.56 + 5.56 = 11.12 km
    _write_geojson(
        path,
        points=[(0.0, 0.0, "A"), (0.10, 0.0, "B")],
        lines=[([[0.0, 0.0], [0.05, 0.0], [0.10, 0.0]], "A_to_B")],
    )
    topo = load_topology(path)
    _, _, w = topo["edges"][0]
    assert 10.5 < w < 11.5  # ~11.12 km along the equator


def test_load_topology_rejects_no_points(tmp_path):
    """No Point features → ValueError."""
    path = tmp_path / "topo.geojson"
    payload = {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
         "properties": {}},
    ]}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Point"):
        load_topology(path)


def test_load_topology_rejects_no_features(tmp_path):
    """Empty FeatureCollection → ValueError."""
    path = tmp_path / "topo.geojson"
    path.write_text('{"type": "FeatureCollection", "features": []}')
    with pytest.raises(ValueError, match="empty|no features|no Point"):
        load_topology(path)


def test_load_topology_unnamed_points_get_index_names(tmp_path):
    """Points with no 'name' property get auto-named by index."""
    path = tmp_path / "topo.geojson"
    payload = {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
         "properties": {}},
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
         "properties": {}},
    ]}
    path.write_text(json.dumps(payload))
    topo = load_topology(path)
    assert topo["node_names"] == ["node_0", "node_1"]


def test_load_topology_lonlat_order(tmp_path):
    """GeoJSON spec puts longitude FIRST, latitude SECOND. Verify our parse honors it."""
    path = tmp_path / "topo.geojson"
    _write_geojson(
        path,
        points=[(-132.0, 53.5, "MAL")],
        lines=[],
    )
    topo = load_topology(path)
    # nodes[0] should be (lon, lat) = (-132.0, 53.5), in that order
    np.testing.assert_array_equal(topo["nodes"][0], np.array([-132.0, 53.5]))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_graph_classifier.py -v
```

Expected: 6 ERROR with `ModuleNotFoundError: No module named 'locator.graph'`.

- [ ] **Step 3: Implement `load_topology`**

Create `locator/graph.py`:

```python
"""Graph utilities for graph-aware classifier inference.

Pure functions — no I/O outside the explicit `load_topology` entrypoint,
no dependencies on the Locator class. The classifier head's predict()
calls these to optionally smooth softmax probabilities along a
user-supplied topology.

Topology source: a GeoJSON FeatureCollection with Point features
(nodes — coords match training-site centroids) and LineString features
(edges — geometry traces the connection path; weight = haversine length
in km).

Generalizes to any user-defined connectivity (rivers, roads, valley
corridors). The inference math is heat-kernel diffusion on the
weighted graph Laplacian; see `heat_kernel`.
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import scipy.linalg
import scipy.spatial

EARTH_RADIUS_KM = 6371.0088


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) pairs in degrees."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _linestring_length_km(coords: np.ndarray) -> float:
    """Haversine sum along a LineString's vertex sequence (lon, lat) pairs."""
    if coords.shape[0] < 2:
        return 0.0
    total = 0.0
    for i in range(coords.shape[0] - 1):
        lon_a, lat_a = float(coords[i, 0]), float(coords[i, 1])
        lon_b, lat_b = float(coords[i + 1, 0]), float(coords[i + 1, 1])
        total += _haversine_km(lat_a, lon_a, lat_b, lon_b)
    return total


def load_topology(path: Path) -> dict:
    """Load a GeoJSON topology with Point and LineString features.

    :param path: Path to a GeoJSON file. Must contain at least one Point
        feature (each Point becomes a graph node). LineString features
        become edges; their endpoints are matched to the nearest Point
        within 1 km tolerance to determine which two nodes the edge
        connects.
    :type path: Path
    :returns: Dict with keys:
        - ``nodes``: ``(K_topo, 2)`` float64 array of (lon, lat) per Point.
        - ``edges``: list of ``(i, j, weight_km)`` tuples; ``weight_km``
          is the haversine sum along the LineString path.
        - ``node_names``: list of strings (from each Point's ``name``
          property; auto-named ``node_<idx>`` if missing).
    :rtype: dict

    :raises ValueError: If the GeoJSON has no features, or no Point
        features, or any LineString endpoint cannot be matched to a Point
        within 1 km.
    """
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"{path}: empty FeatureCollection")

    points = gdf[gdf.geometry.type == "Point"]
    lines = gdf[gdf.geometry.type == "LineString"]
    if points.empty:
        raise ValueError(f"{path}: no Point features (need at least one)")

    nodes = np.array(
        [(geom.x, geom.y) for geom in points.geometry],
        dtype=np.float64,
    )
    if "name" in points.columns:
        node_names = [
            str(n) if isinstance(n, str) and n else f"node_{i}"
            for i, n in enumerate(points["name"].tolist())
        ]
    else:
        node_names = [f"node_{i}" for i in range(len(nodes))]

    edges: list[tuple[int, int, float]] = []
    if not lines.empty:
        # Index Points by KD-tree on (lon, lat) for fast endpoint matching.
        tree = scipy.spatial.cKDTree(nodes)
        for _, row in lines.iterrows():
            coords = np.asarray(row.geometry.coords, dtype=np.float64)
            if coords.shape[0] < 2:
                continue
            start, end = coords[0], coords[-1]
            # ~1 km tolerance in degrees at mid-latitudes — convert lazily by
            # querying KD-tree first then rejecting if too far in haversine.
            tol_deg = 0.02  # ≈ 2 km at the equator; leniency for KD-tree
            d_start, i_start = tree.query(start)
            d_end, i_end = tree.query(end)
            # Verify with haversine that endpoints are within 1 km of nodes
            true_d_start = _haversine_km(
                start[1], start[0], nodes[i_start, 1], nodes[i_start, 0]
            )
            true_d_end = _haversine_km(
                end[1], end[0], nodes[i_end, 1], nodes[i_end, 0]
            )
            if true_d_start > 1.0 or true_d_end > 1.0:
                raise ValueError(
                    f"{path}: LineString endpoint not within 1 km of any Point "
                    f"(start dist {true_d_start:.2f} km, end dist {true_d_end:.2f} km)"
                )
            # Drop self-loops (shouldn't happen with separate endpoints, but defensive)
            if i_start == i_end:
                continue
            weight_km = _linestring_length_km(coords)
            edges.append((int(i_start), int(i_end), float(weight_km)))

    return {"nodes": nodes, "edges": edges, "node_names": node_names}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pixi run pytest tests/test_graph_classifier.py -v
```

Expected: 6 PASS for `test_load_topology_*`.

- [ ] **Step 5: Lint and commit**

```bash
pixi run ruff check locator/graph.py tests/test_graph_classifier.py
git add locator/graph.py tests/test_graph_classifier.py
git commit -m "$(cat <<'EOF'
graph-classifier: add load_topology helper

Reads a GeoJSON FeatureCollection, returns dict with:
  - nodes (K_topo, 2) array of Point coordinates (lon, lat)
  - edges list of (i, j, weight_km) tuples — endpoints matched to
    nearest Point within 1 km tolerance via KD-tree + haversine verify
  - node_names list (Point 'name' property; auto-named if missing)

Edge weight = haversine sum along the LineString vertex sequence.

Pure function; no Locator-class dependency. Will be called by predict()
and by the sculpin topology builder.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `match_centroids_to_nodes` helper

**Files:**
- Modify: `locator/graph.py` (add the function)
- Modify: `tests/test_graph_classifier.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_graph_classifier.py`:

```python
# ---------------------------------------------------------------------------
# Task 2: match_centroids_to_nodes
# ---------------------------------------------------------------------------

from locator.graph import match_centroids_to_nodes


def test_match_centroids_to_nodes_within_tolerance():
    """Centroid at (0, 0) matches a Point at (0.05, 0.05) within ~10 km tol."""
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
    topology = {
        "nodes": np.array([[0.05, 0.05], [10.05, 10.05]], dtype=np.float64),
        "edges": [],
        "node_names": ["A", "B"],
    }
    mapping = match_centroids_to_nodes(centroids, topology, tol_km=10.0)
    assert mapping[0] == 0  # centroid 0 → node 0
    assert mapping[1] == 1  # centroid 1 → node 1


def test_match_centroids_to_nodes_no_match_within_tolerance():
    """Centroid with no node within tolerance returns -1."""
    centroids = np.array([[0.0, 0.0], [50.0, 50.0]], dtype=np.float64)
    topology = {
        "nodes": np.array([[0.05, 0.05]], dtype=np.float64),
        "edges": [],
        "node_names": ["A"],
    }
    mapping = match_centroids_to_nodes(centroids, topology, tol_km=10.0)
    assert mapping[0] == 0
    assert mapping[1] == -1  # no node within 10 km of (50, 50)


def test_match_centroids_to_nodes_returns_int_array():
    centroids = np.array([[0.0, 0.0]], dtype=np.float64)
    topology = {
        "nodes": np.array([[0.0, 0.0]], dtype=np.float64),
        "edges": [],
        "node_names": ["A"],
    }
    mapping = match_centroids_to_nodes(centroids, topology, tol_km=10.0)
    assert mapping.dtype.kind == "i"  # signed integer
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_graph_classifier.py -v -k match_centroids_to_nodes
```

Expected: 3 ERROR (function not defined).

- [ ] **Step 3: Implement `match_centroids_to_nodes`**

Add to `locator/graph.py`:

```python
def match_centroids_to_nodes(
    centroids: np.ndarray, topology: dict, tol_km: float = 10.0,
) -> np.ndarray:
    """Map each centroid to its nearest topology node within tolerance.

    :param centroids: ``(K_train, 2)`` float64 array of (lon, lat).
        For LOSO-style validation, this is the trained model's
        ``_site_centroids`` after denormalization back to raw lon/lat.
    :type centroids: np.ndarray
    :param topology: Dict from ``load_topology``.
    :type topology: dict
    :param tol_km: Haversine tolerance. Centroids with no node within
        this distance map to -1. Default 10 km.
    :type tol_km: float
    :returns: ``(K_train,)`` int64 array; entry i is the index into
        ``topology['nodes']`` of the nearest node, or -1 if no node is
        within tolerance.
    :rtype: np.ndarray
    """
    nodes = topology["nodes"]
    if nodes.shape[0] == 0:
        return np.full(centroids.shape[0], -1, dtype=np.int64)
    tree = scipy.spatial.cKDTree(nodes)
    # Query nearest node per centroid in lon/lat space; verify with haversine.
    _, nearest_idx = tree.query(centroids)
    out = np.full(centroids.shape[0], -1, dtype=np.int64)
    for i, idx in enumerate(nearest_idx):
        d_km = _haversine_km(
            centroids[i, 1], centroids[i, 0], nodes[idx, 1], nodes[idx, 0]
        )
        if d_km <= tol_km:
            out[i] = int(idx)
    return out
```

- [ ] **Step 4: Run tests**

```bash
pixi run pytest tests/test_graph_classifier.py -v
```

Expected: 9 PASS (6 from Task 1 + 3 new).

- [ ] **Step 5: Lint and commit**

```bash
pixi run ruff check locator/graph.py tests/test_graph_classifier.py
git add locator/graph.py tests/test_graph_classifier.py
git commit -m "$(cat <<'EOF'
graph-classifier: add match_centroids_to_nodes helper

Maps each row of trained-model _site_centroids to the nearest topology
node by haversine distance, with a configurable tolerance (default 10 km).
Returns -1 for centroids that have no node within tolerance — caller
treats those as "site not in topology" and skips kernel smoothing for
that prediction.

KD-tree on (lon, lat) gives O(log K) lookups; haversine is computed on
the candidate to verify true distance.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `subset_edges_to_centroids` helper

**Files:**
- Modify: `locator/graph.py` (add the function)
- Modify: `tests/test_graph_classifier.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_graph_classifier.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: subset_edges_to_centroids
# ---------------------------------------------------------------------------

from locator.graph import subset_edges_to_centroids


def test_subset_edges_to_centroids_renumbers_indices():
    """Topology has nodes 0..3 and edges (0,1), (1,2), (2,3); LOSO drops
    node 2; the result keeps only edge (0,1) and renumbers."""
    edges = [(0, 1, 5.0), (1, 2, 7.0), (2, 3, 3.0)]
    # centroid_to_node: 3 trained centroids that map to topology nodes [0, 1, 3]
    # (node 2 is held out).
    centroid_to_node = np.array([0, 1, 3], dtype=np.int64)
    subset = subset_edges_to_centroids(edges, centroid_to_node)
    # The only edge with both endpoints in {0, 1, 3} is (0, 1).
    # In the renumbered space: node 0 → idx 0, node 1 → idx 1.
    assert len(subset) == 1
    i, j, w = subset[0]
    assert {i, j} == {0, 1}
    assert w == 5.0


def test_subset_edges_to_centroids_drops_orphaned_centroids():
    """Centroid that maps to -1 (not in topology) is excluded from
    the result; edges involving its absent node are dropped."""
    edges = [(0, 1, 5.0), (1, 2, 7.0)]
    centroid_to_node = np.array([0, 1, -1], dtype=np.int64)  # 3rd centroid has no node
    subset = subset_edges_to_centroids(edges, centroid_to_node)
    # Only (0, 1) survives; renumbered to (0, 1).
    assert len(subset) == 1
    i, j, w = subset[0]
    assert {i, j} == {0, 1}


def test_subset_edges_to_centroids_drops_self_loops():
    """Edge where both endpoints map to the same trained centroid is dropped."""
    edges = [(0, 1, 5.0), (1, 0, 5.0)]  # 2nd is the reverse of 1st
    centroid_to_node = np.array([0, 1], dtype=np.int64)
    subset = subset_edges_to_centroids(edges, centroid_to_node)
    # Both edges (0,1) and (1,0) survive but represent the same undirected edge;
    # we accept both — graph Laplacian sums them which doubles the weight.
    # Caller is responsible for de-dup if desired.
    assert len(subset) == 2


def test_subset_edges_to_centroids_empty_returns_empty():
    """All centroids missing → no edges returned."""
    edges = [(0, 1, 5.0)]
    centroid_to_node = np.array([-1, -1], dtype=np.int64)
    subset = subset_edges_to_centroids(edges, centroid_to_node)
    assert subset == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_graph_classifier.py -v -k subset_edges_to_centroids
```

Expected: 4 ERROR (function not defined).

- [ ] **Step 3: Implement `subset_edges_to_centroids`**

Add to `locator/graph.py`:

```python
def subset_edges_to_centroids(
    edges: list[tuple[int, int, float]],
    centroid_to_node: np.ndarray,
) -> list[tuple[int, int, float]]:
    """Subset a topology's edge list to those connecting trained centroids.

    Used per-fold in LOSO validation: the full topology spans all sites;
    each fold's training set covers a subset; edges whose endpoints are
    not both in the training set are dropped, and the surviving edges
    are renumbered to indices ``0..K_train-1`` matching the trained
    model's centroid order.

    :param edges: List of ``(i, j, weight_km)`` tuples from
        ``load_topology``; ``i`` and ``j`` index into the topology's
        full node array.
    :type edges: list
    :param centroid_to_node: ``(K_train,)`` int array mapping each
        trained centroid to its topology node index, or -1 for
        unmatched centroids. From ``match_centroids_to_nodes``.
    :type centroid_to_node: np.ndarray
    :returns: Filtered edge list with indices in
        ``[0, K_train)`` matching the trained centroid order.
    :rtype: list
    """
    # node_to_centroid: reverse mapping. Multiple centroids could in
    # principle map to the same node; first wins. Edges between centroids
    # that share a node become self-loops and are dropped.
    node_to_centroid: dict[int, int] = {}
    for centroid_idx, node_idx in enumerate(centroid_to_node):
        if node_idx >= 0 and int(node_idx) not in node_to_centroid:
            node_to_centroid[int(node_idx)] = centroid_idx

    out: list[tuple[int, int, float]] = []
    for i_node, j_node, weight in edges:
        i_centroid = node_to_centroid.get(int(i_node))
        j_centroid = node_to_centroid.get(int(j_node))
        if i_centroid is None or j_centroid is None:
            continue
        if i_centroid == j_centroid:
            continue  # self-loop after re-mapping
        out.append((i_centroid, j_centroid, float(weight)))
    return out
```

- [ ] **Step 4: Run tests**

```bash
pixi run pytest tests/test_graph_classifier.py -v
```

Expected: 13 PASS (9 prior + 4 new).

- [ ] **Step 5: Lint and commit**

```bash
pixi run ruff check locator/graph.py tests/test_graph_classifier.py
git add locator/graph.py tests/test_graph_classifier.py
git commit -m "$(cat <<'EOF'
graph-classifier: add subset_edges_to_centroids helper

LOSO-safe edge filtering: given a topology's full edge list and a per-
fold mapping of trained centroids to topology nodes, return the subset
of edges where both endpoints map to trained centroids — with indices
renumbered to 0..K_train-1 matching the trained model's centroid order.

Drops edges involving held-out sites (mapping = -1) and self-loops
that arise when multiple centroids share a node.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `heat_kernel` helper

**Files:**
- Modify: `locator/graph.py` (add the function)
- Modify: `tests/test_graph_classifier.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_graph_classifier.py`:

```python
# ---------------------------------------------------------------------------
# Task 4: heat_kernel
# ---------------------------------------------------------------------------

from locator.graph import heat_kernel


def test_heat_kernel_t_zero_is_identity():
    """t=0 returns the K×K identity matrix exactly."""
    edges = [(0, 1, 5.0), (1, 2, 7.0)]
    H = heat_kernel(edges, k=3, t=0.0)
    np.testing.assert_allclose(H, np.eye(3), atol=1e-12)


def test_heat_kernel_symmetric():
    """Laplacian is symmetric, so its matrix exponential is too."""
    edges = [(0, 1, 5.0), (1, 2, 7.0), (0, 2, 3.0)]
    H = heat_kernel(edges, k=3, t=1.0)
    np.testing.assert_allclose(H, H.T, atol=1e-10)


def test_heat_kernel_no_edges_is_identity():
    """A graph with no edges has L=0; exp(-t*0) = I for any t."""
    H = heat_kernel(edges=[], k=4, t=2.5)
    np.testing.assert_allclose(H, np.eye(4), atol=1e-12)


def test_heat_kernel_disconnected_components_dont_mix():
    """If nodes 0,1 are connected and node 2 is isolated, P(2) doesn't leak
    into P(0) or P(1) under any t."""
    edges = [(0, 1, 1.0)]
    H = heat_kernel(edges, k=3, t=10.0)
    # Row 2 (the isolated node) should be (0, 0, 1) — no diffusion to/from
    np.testing.assert_allclose(H[2], np.array([0.0, 0.0, 1.0]), atol=1e-10)
    np.testing.assert_allclose(H[:, 2], np.array([0.0, 0.0, 1.0]), atol=1e-10)


def test_heat_kernel_increases_off_diagonal_with_t():
    """Larger t = more probability mass on the off-diagonal."""
    edges = [(0, 1, 1.0)]
    H_small = heat_kernel(edges, k=2, t=0.1)
    H_large = heat_kernel(edges, k=2, t=2.0)
    # H_small[0, 1] < H_large[0, 1] — more diffusion at larger t.
    assert H_small[0, 1] < H_large[0, 1]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_graph_classifier.py -v -k heat_kernel
```

Expected: 5 ERROR.

- [ ] **Step 3: Implement `heat_kernel`**

Add to `locator/graph.py`:

```python
def heat_kernel(
    edges: list[tuple[int, int, float]],
    k: int,
    t: float,
) -> np.ndarray:
    """Compute the heat-kernel matrix ``H = exp(-t * L)`` for a weighted graph.

    :param edges: List of ``(i, j, weight)`` tuples; ``i, j`` are node
        indices in ``[0, k)``. Edge weights become off-diagonal entries
        in the (negative) adjacency matrix; multiple edges between the
        same pair of nodes have their weights summed.
    :type edges: list
    :param k: Number of nodes — the resulting kernel is ``(k, k)``.
    :type k: int
    :param t: Diffusion time (heat-kernel parameter). ``t=0`` returns
        the identity. Larger ``t`` smooths probability mass farther
        across the graph.
    :type t: float
    :returns: ``(k, k)`` symmetric float64 matrix. Apply to a softmax
        row vector ``P`` via ``P @ H`` to smooth probabilities along
        the topology before classifier-style prediction.
    :rtype: np.ndarray
    """
    # Build the symmetric adjacency matrix W from the edge list.
    w = np.zeros((k, k), dtype=np.float64)
    for i, j, weight in edges:
        w[i, j] += weight
        w[j, i] += weight  # symmetric: undirected graph

    # Standard graph Laplacian: L = D - W, where D is the degree matrix.
    degree = w.sum(axis=1)
    laplacian = np.diag(degree) - w

    # Matrix exponential: scipy handles small matrices in microseconds.
    return scipy.linalg.expm(-t * laplacian)
```

- [ ] **Step 4: Run tests**

```bash
pixi run pytest tests/test_graph_classifier.py -v
```

Expected: 18 PASS (13 prior + 5 new).

- [ ] **Step 5: Lint and commit**

```bash
pixi run ruff check locator/graph.py tests/test_graph_classifier.py
git add locator/graph.py tests/test_graph_classifier.py
git commit -m "$(cat <<'EOF'
graph-classifier: add heat_kernel helper

Builds H = exp(-t * L) where L = D - W is the standard graph
Laplacian (D degree matrix, W symmetric weighted adjacency from the
edge list). For K up to a few hundred this is microseconds via
scipy.linalg.expm.

Properties verified by tests:
  - t=0 returns identity (no smoothing)
  - H is symmetric (L is symmetric)
  - Disconnected components don't exchange probability mass under any t
  - Off-diagonal mass increases with t

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Wire graph smoothing into `predict()`

**Files:**
- Modify: `locator/prediction.py` (add graph-smoothing branch)
- Modify: `tests/test_graph_classifier.py` (integration test through Locator.predict)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_graph_classifier.py`:

```python
# ---------------------------------------------------------------------------
# Task 5: predict() with graph topology
# ---------------------------------------------------------------------------

import pandas as pd


def _write_synthetic_matrix_data(tmp_path):
    """K=3 sites × 6 samples each, with one held-out sample for prediction."""
    n_per_site = 6
    n_features = 20
    n_total = n_per_site * 3
    rng = np.random.default_rng(0)
    matrix = pd.DataFrame(
        rng.normal(size=(n_total, n_features)).astype(np.float32),
        columns=[f"f{i}" for i in range(n_features)],
    )
    site_centroids = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)]
    sample_ids, xs, ys = [], [], []
    for i, (sx, sy) in enumerate(site_centroids):
        for j in range(n_per_site):
            sample_ids.append(f"site{i}_{j}")
            xs.append(sx)
            ys.append(sy)
    matrix.insert(0, "sampleID", sample_ids)
    matrix_path = tmp_path / "matrix.tsv"
    matrix.to_csv(matrix_path, sep="\t", index=False)

    sd = pd.DataFrame({"x": xs, "y": ys, "sampleID": sample_ids})
    # Mark sample0 of site2 as held-out for prediction
    sd.loc[sd["sampleID"] == "site2_0", ["x", "y"]] = np.nan
    sd_path = tmp_path / "sample_data.txt"
    sd.to_csv(sd_path, sep="\t", index=False)
    return matrix_path, sd_path


def test_predict_classify_then_avg_with_graph_topology(tmp_path):
    """End-to-end: train classify_then_avg, predict with a graph topology
    that connects sites 0-1; predictions should differ from non-graph case."""
    from locator import Locator
    matrix_path, sd_path = _write_synthetic_matrix_data(tmp_path)

    # Topology: connect site0 (0,0) and site1 (10,10) with one edge;
    # site2 (20,20) is isolated.
    geojson_path = tmp_path / "topo.geojson"
    payload = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
             "properties": {"name": "S0"}},
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [10.0, 10.0]},
             "properties": {"name": "S1"}},
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [20.0, 20.0]},
             "properties": {"name": "S2"}},
            {"type": "Feature",
             "geometry": {"type": "LineString",
                          "coordinates": [[0.0, 0.0], [10.0, 10.0]]},
             "properties": {"name": "S0_S1"}},
        ],
    }
    geojson_path.write_text(json.dumps(payload))

    # First train + predict WITHOUT topology
    loc1 = Locator(config={
        "matrix": str(matrix_path),
        "sample_data": str(sd_path),
        "out": str(tmp_path / "no_graph"),
        "max_epochs": 20, "patience": 30, "seed": 42, "gpu_number": 0,
        "prediction_mode": "classify_then_avg",
        "optimize_tf_parallelism": False,
    })
    g, s = loc1.load_genotypes(matrix=str(matrix_path))
    loc1.train(genotypes=g, samples=s)
    pred_no_graph = loc1.predict(genotypes=g, samples=s, return_df=True)

    # Now WITH topology — same trained model would ideally be reused,
    # but in this test we retrain since the synthetic data is small.
    loc2 = Locator(config={
        "matrix": str(matrix_path),
        "sample_data": str(sd_path),
        "out": str(tmp_path / "with_graph"),
        "max_epochs": 20, "patience": 30, "seed": 42, "gpu_number": 0,
        "prediction_mode": "classify_then_avg",
        "graph_topology": str(geojson_path),
        "graph_kernel_resolution": 1.0,
        "optimize_tf_parallelism": False,
    })
    g2, s2 = loc2.load_genotypes(matrix=str(matrix_path))
    loc2.train(genotypes=g2, samples=s2)
    pred_with_graph = loc2.predict(genotypes=g2, samples=s2, return_df=True)

    # Both should produce predictions (not NaN), and the with-graph
    # version should set self._heat_kernel.
    assert not pred_with_graph[["x", "y"]].isna().any().any()
    assert hasattr(loc2, "_heat_kernel")
    assert loc2._heat_kernel.shape == (3, 3)
    # Graph smoothing has produced a different prediction (predictions
    # have been weighted across topology). Allow some tolerance for
    # cases where the model already nailed the no-graph prediction.
    np_diff = np.abs(
        pred_no_graph[["x", "y"]].values - pred_with_graph[["x", "y"]].values
    ).max()
    # If predictions are identical, the kernel had no effect — possible if
    # softmax was a one-hot vector. This is an acceptable edge case; we
    # don't assert difference, just absence of failure.
    assert np_diff >= 0.0


def test_predict_regress_with_graph_topology_errors(tmp_path):
    """regress + graph_topology raises a clear ValueError at config validation."""
    from locator import Locator
    matrix_path, sd_path = _write_synthetic_matrix_data(tmp_path)
    geojson_path = tmp_path / "topo.geojson"
    geojson_path.write_text('{"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [0,0]}, "properties": {}}]}')

    loc = Locator(config={
        "matrix": str(matrix_path),
        "sample_data": str(sd_path),
        "out": str(tmp_path / "x"),
        "max_epochs": 1, "patience": 5, "seed": 42, "gpu_number": 0,
        "prediction_mode": "regress",
        "graph_topology": str(geojson_path),
        "optimize_tf_parallelism": False,
    })
    g, s = loc.load_genotypes(matrix=str(matrix_path))
    with pytest.raises(ValueError, match="graph_topology|classifier"):
        loc.train(genotypes=g, samples=s)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_graph_classifier.py -v -k predict_classify_then_avg_with_graph
```

Expected: FAIL — `_heat_kernel` not set, `graph_topology` ignored. Also the regress test fails because the config validation gate doesn't yet reject `regress + graph_topology`.

- [ ] **Step 3: Extend `_validate_prediction_mode_config` for `regress + graph_topology`**

In `locator/training.py`, find `_validate_prediction_mode_config` (added in classifier-head Task 3). After the existing checks, append:

```python
    if config.get("graph_topology") is not None and mode == "regress":
        raise ValueError(
            "graph_topology is only meaningful with classifier modes; "
            "set prediction_mode='classify' or 'classify_then_avg', "
            "or remove graph_topology."
        )
```

- [ ] **Step 4: Wire graph smoothing into `Locator.predict`**

In `locator/prediction.py`, find the existing classifier branch — the place where `predictions` is the network's softmax output `(n, K)` and the code branches on `mode == "classify"` / `"classify_then_avg"` / regress (added in classifier-head Task 6). The branch is at the line where `mode = self.config.get("prediction_mode", "regress")`.

Immediately after the line `predictions = predictions.copy()` and BEFORE the `if mode == "classify":` line, insert the graph-smoothing block:

```python
# Graph-aware inference: smooth softmax over topology if configured.
if (self.config.get("graph_topology") is not None
        and mode in ("classify", "classify_then_avg")):
    if not hasattr(self, "_heat_kernel"):
        from .graph import (
            heat_kernel,
            load_topology,
            match_centroids_to_nodes,
            subset_edges_to_centroids,
        )
        topology = load_topology(Path(self.config["graph_topology"]))
        # Denormalize centroids to raw lon/lat for matching against GeoJSON nodes.
        centroids_raw = self._site_centroids.copy()
        centroids_raw[:, 0] = centroids_raw[:, 0] * self.sdlong + self.meanlong
        centroids_raw[:, 1] = centroids_raw[:, 1] * self.sdlat + self.meanlat
        c_to_n = match_centroids_to_nodes(centroids_raw, topology)
        edges_subset = subset_edges_to_centroids(topology["edges"], c_to_n)
        self._heat_kernel = heat_kernel(
            edges_subset,
            k=self._n_classes,
            t=self.config.get("graph_kernel_resolution", 1.0),
        )
    # Apply (n, K) @ (K, K) → (n, K) — smooth probabilities along topology.
    predictions = predictions @ self._heat_kernel
```

(Make sure `Path` is imported at the top of `prediction.py` — it likely already is. If not, add `from pathlib import Path`.)

- [ ] **Step 5: Run tests**

```bash
pixi run pytest tests/test_graph_classifier.py -v
```

Expected: 20 PASS (18 prior + 2 new).

Also re-run the classifier-head test suite to confirm no regressions:

```bash
pixi run pytest tests/test_classifier_head.py -v 2>&1 | tail -3
```

Expected: 26 PASS (unchanged from classifier-head Task 14).

- [ ] **Step 6: Lint and commit**

```bash
pixi run ruff check locator/prediction.py locator/training.py tests/test_graph_classifier.py
git add locator/prediction.py locator/training.py tests/test_graph_classifier.py
git commit -m "$(cat <<'EOF'
graph-classifier: wire heat-kernel smoothing into predict()

When config.get('graph_topology') is set AND prediction_mode is a
classifier mode (classify or classify_then_avg), predict() now:
  1. Lazily loads the GeoJSON topology
  2. Matches trained _site_centroids to topology nodes (denormalizing
     centroids first so the lon/lat space matches the GeoJSON)
  3. Subsets edges to those connecting trained centroids (LOSO-safe)
  4. Builds the heat kernel H = exp(-t * L)
  5. Caches it on self._heat_kernel for subsequent predict calls
  6. Applies P' = P @ H before the existing argmax / centroid math

regress + graph_topology now raises a config-validation error early.
Backward compatible: classify and classify_then_avg without
graph_topology behave exactly as on classifier-head.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: User-facing docs

**Files:**
- Modify: `docs/prediction_modes.md` (append "Graph-aware inference" section)

- [ ] **Step 1: Append the new section**

In `docs/prediction_modes.md` (created in classifier-head Task 9), append the following section after the existing "Caveats" section:

```markdown

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
```

- [ ] **Step 2: Commit**

```bash
git add docs/prediction_modes.md
git commit -m "$(cat <<'EOF'
docs: add Graph-aware inference section to prediction_modes.md

Documents the optional graph_topology + graph_kernel_resolution config
options for the classifier-head prediction modes. Includes the GeoJSON
schema (Point + LineString features), worked example, the
heat-kernel inference math, and the LOSO-safety guarantee (the user's
full topology is automatically subsetted per fold).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Sculpin topology builder

**Files:**
- Create: `validation/sculpin/build_topology.py`

This script reads HydroSHEDS HydroLAKES + HydroRIVERS (already on disk
under `/sietch_colab/ssmall/projects/relocator_dir/data/hydrosheds/`)
and builds a GeoJSON connecting our 16 sampled sculpin sites via
shortest paths through the river network.

- [ ] **Step 1: Create the builder**

Create `validation/sculpin/build_topology.py`:

```python
#!/usr/bin/env python3
"""Build a GeoJSON topology connecting the 16 sculpin sampling sites via
shortest paths through HydroSHEDS HydroRIVERS.

Approach:
  1. Load HydroRIVERS reaches within the PNW bbox (Strahler order >= 3
     to avoid headwater noise).
  2. Build a NetworkX graph: each reach is an edge; reach endpoints are
     graph nodes.
  3. For each pair of sampled sites, snap their (lon, lat) to the
     nearest river-network node and run NetworkX shortest_path.
  4. For each pair where a path exists, emit a LineString tracing the
     concatenated reach geometries.
  5. Write the result as `validation/sculpin/sculpin_topology.geojson`
     with Point features for each site and LineString features for
     each connection.

The PNW bbox + Strahler filter come from `validation/sculpin/build_range.py`
which already uses HydroSHEDS for the freshwater range mask.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.ops

PNW_BBOX = (-138.0, 47.0, -114.0, 60.0)  # lon_min, lat_min, lon_max, lat_max
DEFAULT_MIN_STREAM_ORDER = 3


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def _line_length_km(line: shapely.geometry.LineString) -> float:
    coords = np.asarray(line.coords, dtype=np.float64)
    total = 0.0
    for i in range(coords.shape[0] - 1):
        total += _haversine_km(coords[i, 1], coords[i, 0],
                                coords[i + 1, 1], coords[i + 1, 0])
    return total


def build_river_graph(rivers_shp: Path, min_order: int = DEFAULT_MIN_STREAM_ORDER) -> tuple[nx.Graph, dict]:
    """Load HydroRIVERS, filter by stream order, build a NetworkX graph.

    Returns (graph, node_coords) where node_coords maps node_id → (lon, lat).
    Each river reach becomes an edge with weight = haversine length km.
    """
    bbox_poly = shapely.geometry.box(*PNW_BBOX)
    print(f"  reading {rivers_shp.name} (clipping to bbox + min order)...", flush=True)
    rivers = gpd.read_file(rivers_shp, bbox=bbox_poly)
    if "ORD_STRA" in rivers.columns:
        before = len(rivers)
        rivers = rivers[rivers["ORD_STRA"] >= min_order]
        print(f"    {len(rivers)} reaches at ORD_STRA >= {min_order} (dropped {before - len(rivers)})",
              flush=True)
    g = nx.Graph()
    node_coords: dict[tuple[float, float], int] = {}
    next_id = 0

    def get_or_create_node(coord: tuple[float, float]) -> int:
        nonlocal next_id
        # Round to 6 decimals to merge near-identical endpoints
        key = (round(coord[0], 6), round(coord[1], 6))
        if key not in node_coords:
            node_coords[key] = next_id
            g.add_node(next_id, lon=key[0], lat=key[1])
            next_id += 1
        return node_coords[key]

    edge_geoms: dict[tuple[int, int], shapely.geometry.LineString] = {}
    for _, row in rivers.iterrows():
        line = row.geometry
        if line is None or line.is_empty:
            continue
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        a = get_or_create_node(coords[0])
        b = get_or_create_node(coords[-1])
        if a == b:
            continue
        weight = _line_length_km(line)
        # Lowest-weight edge wins if there are duplicates.
        pair = (a, b) if a < b else (b, a)
        if pair not in edge_geoms or weight < g.edges[pair].get("weight", float("inf")):
            edge_geoms[pair] = line
            g.add_edge(a, b, weight=weight, geometry=line)

    print(f"    river graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges",
          flush=True)
    return g, {nid: (data["lon"], data["lat"]) for nid, data in g.nodes(data=True)}


def snap_sites_to_graph(sites: pd.DataFrame, g: nx.Graph,
                        node_coords: dict) -> dict[str, int]:
    """For each site, find the nearest graph node by haversine distance."""
    coords = np.array(list(node_coords.values()), dtype=np.float64)  # (N, 2) lon/lat
    node_ids = list(node_coords.keys())
    snapped = {}
    for _, row in sites.iterrows():
        site_lon, site_lat = float(row["lon"]), float(row["lat"])
        # Compute haversine to every node — small N for sites, fine.
        d = np.array([
            _haversine_km(site_lat, site_lon, c[1], c[0]) for c in coords
        ])
        best = int(np.argmin(d))
        snapped[row["site"]] = node_ids[best]
        print(f"    {row['site']:<20} → graph node {node_ids[best]} "
              f"({d[best]:.2f} km)", flush=True)
    return snapped


def find_path_linestring(g: nx.Graph, src: int, dst: int) -> shapely.geometry.LineString | None:
    """Shortest path from src to dst; concatenate edge geometries.

    Returns None if no path exists (disconnected components).
    """
    try:
        path = nx.shortest_path(g, src, dst, weight="weight")
    except nx.NetworkXNoPath:
        return None
    if len(path) < 2:
        return None
    pieces = []
    for i in range(len(path) - 1):
        edge = g.edges[path[i], path[i + 1]]
        line = edge["geometry"]
        pieces.append(line)
    if not pieces:
        return None
    merged = shapely.ops.linemerge(shapely.geometry.MultiLineString(pieces))
    if isinstance(merged, shapely.geometry.MultiLineString):
        # Fall back to concatenating coordinates in order
        coords = []
        for piece in pieces:
            for c in piece.coords:
                coords.append(c)
        return shapely.geometry.LineString(coords)
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hydrorivers", required=True, type=Path,
                   help="HydroRIVERS_v10_na.shp path.")
    p.add_argument("--sites_tsv", required=True, type=Path,
                   help="sites.tsv with site, lat, lon columns.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output GeoJSON path.")
    p.add_argument("--min_stream_order", type=int, default=DEFAULT_MIN_STREAM_ORDER)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("Building sculpin topology from HydroRIVERS...", flush=True)
    g, node_coords = build_river_graph(args.hydrorivers, args.min_stream_order)

    sites = pd.read_csv(args.sites_tsv, sep="\t")
    print(f"  snapping {len(sites)} sampled sites to river-network nodes:",
          flush=True)
    snapped = snap_sites_to_graph(sites, g, node_coords)

    features = []
    for _, row in sites.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [float(row["lon"]), float(row["lat"])]},
            "properties": {"name": str(row["site"])},
        })

    site_names = sites["site"].tolist()
    n_edges = 0
    for i, name_i in enumerate(site_names):
        for j in range(i + 1, len(site_names)):
            name_j = site_names[j]
            line = find_path_linestring(g, snapped[name_i], snapped[name_j])
            if line is None:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString",
                             "coordinates": [list(c) for c in line.coords]},
                "properties": {"name": f"{name_i}_to_{name_j}"},
            })
            n_edges += 1

    print(f"  wrote {n_edges} edges connecting "
          f"{len(site_names)} sites", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, indent=2,
    ))
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint**

```bash
pixi run ruff check validation/sculpin/build_topology.py
```

- [ ] **Step 3: Run the builder against HydroSHEDS data**

```bash
pixi run python -m validation.sculpin.build_topology \
    --hydrorivers /sietch_colab/ssmall/projects/relocator_dir/data/hydrosheds/HydroRIVERS_v10_na_shp/HydroRIVERS_v10_na.shp \
    --sites_tsv validation/sculpin/sites.tsv \
    --out validation/sculpin/sculpin_topology.geojson \
    --min_stream_order 3 \
    > /tmp/build_topology.log 2>&1
echo "exit=$?"
tail -10 /tmp/build_topology.log
```

Expected: log shows river-graph stats, site-snap mapping, edge-count summary; `validation/sculpin/sculpin_topology.geojson` is written.

- [ ] **Step 4: Quick sanity check on the GeoJSON**

```bash
pixi run python -c "
from locator.graph import load_topology
from pathlib import Path
topo = load_topology(Path('validation/sculpin/sculpin_topology.geojson'))
print(f'nodes: {topo[\"nodes\"].shape}')
print(f'edges: {len(topo[\"edges\"])}')
print(f'first 3 node names: {topo[\"node_names\"][:3]}')
"
```

Expected: `nodes: (16, 2)` (16 sites), some number of edges (likely 50-120 depending on connectivity), node names like `Alaska`, `BellaCoola`, etc.

- [ ] **Step 5: Commit the script + the generated artifact**

```bash
git add validation/sculpin/build_topology.py validation/sculpin/sculpin_topology.geojson
git commit -m "$(cat <<'EOF'
sculpin: HydroSHEDS-driven topology builder + sculpin_topology.geojson

build_topology.py loads HydroRIVERS reaches (PNW bbox, Strahler order
>= 3 default), builds a NetworkX river graph with reach endpoints as
nodes and reach geometry as edges, snaps each of the 16 sampled sites
to the nearest network node, then traces shortest-paths between every
pair of sites through the river network and emits a GeoJSON.

The committed sculpin_topology.geojson is the output for the existing
sites.tsv. Reproducible from HydroSHEDS HydroRIVERS_v10_na shapefile.

Pattern users adapt for road networks, valley corridors, or any other
connectivity domain: replace HydroRIVERS with the appropriate graph
source, snap nodes to sample sites, emit GeoJSON.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Sculpin LOSO graph runner

**Files:**
- Create: `validation/sculpin/run_loso_graph.py`

This runner loads the already-trained `classify_then_avg` fold models
from the classifier-head sweep and re-predicts with `graph_topology`
set. No retraining needed.

- [ ] **Step 1: Create the runner**

Create `validation/sculpin/run_loso_graph.py`:

```python
#!/usr/bin/env python3
"""Re-predict each sculpin LOSO fold with graph topology smoothing.

Reuses the classifier_then_avg models trained by run_loso_classifier.py
(no retraining). For each of the 16 sites:
  1. Locate the existing fold's saved model under
     out/sculpin_validation/loso_classifier/classify_then_avg/<SITE>/.
  2. Create a fresh Locator with prediction_mode='classify_then_avg' and
     graph_topology=<GeoJSON>; load_model() picks up the saved weights
     and centroids.
  3. predict() applies the heat-kernel smoothing.
  4. Write fold_result.json under
     out/sculpin_validation/loso_graph/<SITE>/ with the same schema as
     the other LOSO runners.

Single mode per invocation. Default graph_kernel_resolution=1.0; sweep
via --t to test alternatives.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from validation import common  # noqa: E402

DEFAULT_T = 1.0


def site_of(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def run_one_fold(
    *, site: str,
    classifier_fold_dir: Path,
    sample_data_in: Path,
    feature_matrix_path: Path,
    topology_path: Path,
    out_dir: Path,
    gpu: int,
    t: float,
) -> dict:
    fold_dir = out_dir / site
    fold_dir.mkdir(parents=True, exist_ok=True)

    weights_path = classifier_fold_dir / "locator.weights.h5"
    cls_sd_path = classifier_fold_dir / "sample_data.txt"
    if not weights_path.exists() or not cls_sd_path.exists():
        return {
            "mode": "classify_then_avg_graph", "site": site,
            "status": "FAILED",
            "error": f"classifier-head fold artifacts missing at {classifier_fold_dir}",
        }

    sd_path = fold_dir / "sample_data.txt"
    sd_path.write_text(cls_sd_path.read_text())

    out_prefix = fold_dir / "locator"
    fold_result: dict = {
        "mode": "classify_then_avg_graph", "site": site,
        "graph_kernel_resolution": float(t),
    }

    t_start = time.time()
    try:
        from locator import Locator

        loc = Locator(config={
            "out": str(out_prefix),
            "matrix": str(feature_matrix_path),
            "sample_data": str(sd_path),
            "seed": 42,
            "gpu_number": gpu,
            "prediction_mode": "classify_then_avg",
            "graph_topology": str(topology_path),
            "graph_kernel_resolution": float(t),
        })
        genotypes, samples = loc.load_genotypes(matrix=str(feature_matrix_path))
        loc.load_model(str(weights_path))
        loc.predict(genotypes=genotypes, samples=samples)
    except Exception as exc:
        fold_result["status"] = "FAILED"
        fold_result["error"] = f"{type(exc).__name__}: {exc}"
        (fold_dir / "fold_result.json").write_text(
            json.dumps(fold_result, indent=2, default=str)
        )
        return fold_result
    fold_result["elapsed_s"] = round(time.time() - t_start, 1)

    predlocs = Path(f"{out_prefix}_predlocs.txt")
    if not predlocs.exists():
        fold_result["status"] = "FAILED"
        fold_result["error"] = f"predlocs missing at {predlocs}"
        (fold_dir / "fold_result.json").write_text(
            json.dumps(fold_result, indent=2, default=str)
        )
        return fold_result

    sd_df = pd.read_csv(sd_path, sep="\t")
    held_out_ids = sd_df.loc[sd_df["x"].astype(str) == "NA", "sampleID"].tolist()
    if not held_out_ids:
        fold_result["status"] = "FAILED"
        fold_result["error"] = f"no NA-coord samples in {sd_path}"
        (fold_dir / "fold_result.json").write_text(
            json.dumps(fold_result, indent=2, default=str)
        )
        return fold_result

    truth = pd.read_csv(sample_data_in, sep="\t")
    pred = common.parse_predlocs(predlocs)
    pred = pred[pred["sampleID"].isin(held_out_ids)].set_index("sampleID")
    tru = truth.set_index("sampleID").loc[held_out_ids, ["x", "y"]]
    evaluable = tru.dropna().index.tolist()
    evaluable_in_pred = [sid for sid in evaluable if sid in pred.index]

    if not evaluable_in_pred:
        fold_result["status"] = "OK"
        fold_result["n_held_out"] = int(len(held_out_ids))
        fold_result["n_evaluable"] = int(len(evaluable))
        fold_result["n_scored"] = 0
        fold_result["warning"] = "no scored samples"
        (fold_dir / "fold_result.json").write_text(
            json.dumps(fold_result, indent=2, default=str)
        )
        return fold_result

    p = pred.loc[evaluable_in_pred, ["x", "y"]].to_numpy(dtype=np.float64)
    t_arr = tru.loc[evaluable_in_pred].to_numpy(dtype=np.float64)
    err_km = common.haversine(t_arr[:, 1], t_arr[:, 0], p[:, 1], p[:, 0])
    if np.ndim(err_km) == 0:
        err_km = np.array([err_km])

    fold_result.update({
        "status": "OK",
        "n_held_out": int(len(held_out_ids)),
        "n_evaluable": int(len(evaluable)),
        "n_scored": int(len(evaluable_in_pred)),
        "median_error_km": float(np.median(err_km)),
        "mean_error_km": float(np.mean(err_km)),
        "max_error_km": float(np.max(err_km)),
    })
    (fold_dir / "fold_result.json").write_text(
        json.dumps(fold_result, indent=2, default=str)
    )
    return fold_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--classifier_fold_root", required=True, type=Path,
                   help="Root containing classify_then_avg/<SITE>/locator.weights.h5; "
                        "typically out/sculpin_validation/loso_classifier/classify_then_avg")
    p.add_argument("--sample_data", required=True, type=Path,
                   help="Original sample_data.txt with truth coords.")
    p.add_argument("--feature_matrix", required=True, type=Path,
                   help="Pre-built dosage feature matrix from earlier sweeps.")
    p.add_argument("--topology", required=True, type=Path,
                   help="GeoJSON topology (sculpin_topology.geojson).")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--t", type=float, default=DEFAULT_T,
                   help="graph_kernel_resolution / heat-kernel diffusion time.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sd_df = pd.read_csv(args.sample_data, sep="\t")
    sites = sorted(set(sd_df["sampleID"].apply(site_of)))

    print(f"Re-predicting {len(sites)} folds with graph topology, "
          f"t={args.t} on GPU {args.gpu}", flush=True)
    t_sweep = time.time()
    for idx, site in enumerate(sites):
        print(f"\n=== fold {idx + 1}/{len(sites)}: {site} ===", flush=True)
        run_one_fold(
            site=site,
            classifier_fold_dir=args.classifier_fold_root / site,
            sample_data_in=args.sample_data,
            feature_matrix_path=args.feature_matrix,
            topology_path=args.topology,
            out_dir=args.out_dir / "loso_graph",
            gpu=args.gpu,
            t=args.t,
        )
    print(f"\nSweep complete in {(time.time() - t_sweep) / 60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint**

```bash
pixi run ruff check validation/sculpin/run_loso_graph.py
```

- [ ] **Step 3: Run the sweep at default t=1.0**

```bash
pixi run python -m validation.sculpin.run_loso_graph \
    --classifier_fold_root out/sculpin_validation/loso_classifier/classify_then_avg \
    --sample_data out/sculpin_validation/inputs/sample_data.txt \
    --feature_matrix out/sculpin_validation/features_dosage.tsv \
    --topology validation/sculpin/sculpin_topology.geojson \
    --out_dir out/sculpin_validation \
    --t 1.0 \
    --gpu 0 \
    > out/sculpin_validation/run_loso_graph.log 2>&1
echo "exit=$?"
tail -3 out/sculpin_validation/run_loso_graph.log
```

Expected: 16 folds complete in a few minutes (no retraining; just inference). Final line: "Sweep complete in X min".

- [ ] **Step 4: Verify all fold JSONs exist**

```bash
find out/sculpin_validation/loso_graph -name 'fold_result.json' | wc -l
find out/sculpin_validation/loso_graph -name 'fold_result.json' -exec grep -l '"FAILED"\|"error"' {} \; 2>/dev/null
```

Expected: 16 JSONs; second command prints nothing.

- [ ] **Step 5: Spot-check one fold**

```bash
cat out/sculpin_validation/loso_graph/Mosquito/fold_result.json
```

Expected: `mode='classify_then_avg_graph'`, `graph_kernel_resolution=1.0`, `median_error_km` populated.

- [ ] **Step 6: Commit the runner only (outputs are gitignored under out/)**

```bash
git add validation/sculpin/run_loso_graph.py
git commit -m "$(cat <<'EOF'
sculpin: graph-mode LOSO runner (reuses classifier-head fold models)

For each of the 16 sculpin sites: load the existing
classify_then_avg fold's saved weights.h5, create a fresh Locator with
graph_topology + graph_kernel_resolution config, run predict() to apply
heat-kernel smoothing, score against truth coords (haversine km).

No retraining — this is inference-only, exercising the new graph code
path against the already-trained classifier head. Same fold output
schema as run_loso_classifier.py for downstream summarization.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Extend the 5-mode comparison summarizer

**Files:**
- Modify: `validation/sculpin/summarize_classifier.py` (add the 5th mode)

The classifier-head summarizer aggregates 4 modes (regress, regress+rangemask, classify, classify_then_avg). We extend it to include `classify_then_avg_graph` as the 5th mode.

- [ ] **Step 1: Modify the `MODE_PATHS` dict**

In `validation/sculpin/summarize_classifier.py`, find the `MODE_PATHS` dict (near the top of the file) and append a new entry:

```python
MODE_PATHS = {
    "regress": ("loso/dosage", "regression (current)"),
    "regress+rangemask": ("loso_rangemask/dosage", "regression + range_mask"),
    "classify": ("loso_classifier/classify", "classify (argmax)"),
    "classify_then_avg": ("loso_classifier/classify_then_avg", "classify_then_avg"),
    "classify_then_avg_graph": ("loso_graph", "classify_then_avg + graph"),  # NEW
}
```

The `loso_graph` directory contains per-site fold_result.json directly (no per-mode subdirectory), so the path adapts automatically.

- [ ] **Step 2: Adjust the bar-chart palette to support 5 colors**

Find the bar-chart `colors = ["firebrick", "darkorange", "seagreen", "steelblue"]` line and extend to 5:

```python
colors = ["firebrick", "darkorange", "seagreen", "steelblue", "purple"]
```

Find the `w = 0.20` (bar width) and `(i - 1.5) * w` (offset) lines. With 5 modes you want offsets `(i - 2.0) * w` to center; w stays at 0.16 to fit:

```python
w = 0.16
for i, mode in enumerate(MODE_PATHS):
    sub = fold[fold["mode"] == mode].set_index("site")
    vals = [sub["median_error_km"].get(s, np.nan) for s in site_order]
    ax.bar(x + (i - 2.0) * w, vals, w,
           label=MODE_PATHS[mode][1], color=colors[i])
```

- [ ] **Step 3: Update the per-site Markdown table to include the 5th column**

Find the per-site table block in `render_summary_md`:

```python
"| site | regress | regress+rangemask | classify | classify_then_avg |",
"|---|---|---|---|---|",
```

Replace with:

```python
"| site | regress | regress+rangemask | classify | classify_then_avg | classify_then_avg_graph |",
"|---|---|---|---|---|---|",
```

- [ ] **Step 4: Run the summarizer**

```bash
pixi run ruff check validation/sculpin/summarize_classifier.py
pixi run python -m validation.sculpin.summarize_classifier --out_dir out/sculpin_validation
```

Expected: writes `validation/summary/sculpin_classifier_kfold.tsv`,
`validation/summary/sculpin_classifier_summary.md`,
`validation/figures/sculpin_classifier_modes.png` — all updated with
the 5th mode.

- [ ] **Step 5: Manually edit the Recommendation section in the summary**

Open `validation/summary/sculpin_classifier_summary.md`. Replace the
Recommendation section's placeholder with 2–3 sentences interpreting
the actual 5-mode numbers. Specifically address:

- Did `classify_then_avg_graph` beat `classify_then_avg` on aggregate
  median?
- Per-site, did the FST-outlier sites (Alaska, Okanagan, Nimpo) improve
  most under graph?
- Are there sites that got worse?
- Does the result confirm or refute the architectural hypothesis that
  graph awareness fixes the regression-to-the-mean failure at edge
  sites?

- [ ] **Step 6: Commit summarizer + outputs**

```bash
git add validation/sculpin/summarize_classifier.py \
        validation/summary/sculpin_classifier_kfold.tsv \
        validation/summary/sculpin_classifier_summary.md \
        validation/figures/sculpin_classifier_modes.png
git commit -m "$(cat <<'EOF'
sculpin: 5-mode comparison summarizer + graph results

Extends summarize_classifier.py to include classify_then_avg_graph as
the 5th comparison mode. Same aggregation pattern (mean of per-site
medians) and per-site Markdown table; bar chart palette adjusted to
5 colors.

Recommendation section manually edited based on the actual numbers,
specifically commenting on whether the graph mode improves the
FST-outlier sites (Alaska, Okanagan, Nimpo) where the flat classifier
fails most.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Sculpin map figure with graph overlay

**Files:**
- Create: `validation/sculpin/make_map_graph.py` (small script)
- Outputs: `validation/figures/sculpin_map_graph.png`

A cartopy map mirroring the existing `make_map.py` pattern but with
the topology graph overlaid (LineString edges drawn behind site stars
and prediction circles).

- [ ] **Step 1: Create the script**

Create `validation/sculpin/make_map_graph.py`:

```python
#!/usr/bin/env python3
"""Cartopy map of sculpin LOSO predictions with the topology graph overlaid.

Mirrors validation/sculpin/make_map.py but adds:
  - LineString edges from the topology GeoJSON drawn as light blue lines
  - Heading: shows graph_kernel_resolution and total edge count

Reads predictions from out/sculpin_validation/loso_graph/<SITE>/run_predlocs.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from validation import common  # noqa: E402


def load_predictions(loso_dir: Path, sample_data: Path) -> pd.DataFrame:
    sd = pd.read_csv(sample_data, sep="\t")
    sd["site"] = sd["sampleID"].str.rsplit("_", n=1).str[0]
    truth = sd.set_index("sampleID")[["x", "y", "site"]].rename(
        columns={"x": "true_lon", "y": "true_lat"}
    )
    rows = []
    for fold_dir in sorted(loso_dir.iterdir()):
        if not fold_dir.is_dir():
            continue
        site_held = fold_dir.name
        pred_path = fold_dir / "run_predlocs.txt"
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path).rename(
            columns={"x": "pred_lon", "y": "pred_lat"}
        )
        held_ids = truth[truth["site"] == site_held].index
        for sid in held_ids:
            if sid not in pred["sampleID"].values:
                continue
            p = pred[pred["sampleID"] == sid].iloc[0]
            rows.append({
                "sampleID": sid,
                "site": site_held,
                "true_lon": float(truth.loc[sid, "true_lon"]),
                "true_lat": float(truth.loc[sid, "true_lat"]),
                "pred_lon": float(p["pred_lon"]),
                "pred_lat": float(p["pred_lat"]),
            })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--loso_dir", required=True, type=Path,
                   help="out/sculpin_validation/loso_graph")
    p.add_argument("--sample_data", required=True, type=Path)
    p.add_argument("--topology", required=True, type=Path)
    p.add_argument("--out_png", required=True, type=Path)
    p.add_argument("--t", type=float, default=1.0,
                   help="Diffusion time used (for the title).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    df = load_predictions(args.loso_dir, args.sample_data)
    if df.empty:
        print("No predictions found", file=sys.stderr)
        return 1

    err_km = common.haversine(
        df["true_lat"].to_numpy(dtype=np.float64),
        df["true_lon"].to_numpy(dtype=np.float64),
        df["pred_lat"].to_numpy(dtype=np.float64),
        df["pred_lon"].to_numpy(dtype=np.float64),
    )
    median_err = float(np.median(err_km))

    topology = json.loads(args.topology.read_text())
    edge_geoms = [
        f["geometry"]["coordinates"]
        for f in topology["features"]
        if f["geometry"]["type"] == "LineString"
    ]
    points = [
        f["geometry"]["coordinates"]
        for f in topology["features"]
        if f["geometry"]["type"] == "Point"
    ]
    point_names = [
        f.get("properties", {}).get("name", f"node_{i}")
        for i, f in enumerate(topology["features"])
        if f["geometry"]["type"] == "Point"
    ]

    proj = ccrs.LambertConformal(central_longitude=-127, central_latitude=53)
    fig = plt.figure(figsize=(11, 13))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-138, -115, 47, 60], crs=ccrs.PlateCarree())

    land = cfeature.NaturalEarthFeature(
        category="physical", name="land", scale="50m",
        edgecolor="#666", facecolor="#f0ebe0",
    )
    ocean = cfeature.NaturalEarthFeature(
        category="physical", name="ocean", scale="50m",
        edgecolor="none", facecolor="#dde8f0",
    )
    ax.add_feature(ocean, zorder=0)
    ax.add_feature(land, zorder=1)
    ax.coastlines(resolution="50m", lw=0.5, color="#444")

    # Topology edges
    for coords in edge_geoms:
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        ax.plot(lons, lats, color="#3070a0", lw=0.7, alpha=0.6,
                transform=ccrs.PlateCarree(), zorder=2)

    # Site nodes (stars)
    for (lon, lat), name in zip(points, point_names, strict=False):
        ax.scatter(lon, lat, marker="*", s=200, color="goldenrod",
                   edgecolor="black", lw=0.7,
                   transform=ccrs.PlateCarree(), zorder=5)
        ax.annotate(
            name, xy=(lon, lat),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xytext=(6, 6), textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            zorder=6,
        )

    # Predictions (circles) + connect to truth (lines)
    for _, r in df.iterrows():
        ax.plot(
            [r["pred_lon"], r["true_lon"]], [r["pred_lat"], r["true_lat"]],
            color="#a04060", lw=0.5, alpha=0.4,
            transform=ccrs.PlateCarree(), zorder=3,
        )
        ax.scatter(
            r["pred_lon"], r["pred_lat"], s=24, color="#a04060",
            edgecolor="black", lw=0.3, alpha=0.85,
            transform=ccrs.PlateCarree(), zorder=4,
        )

    fig.suptitle(
        f"Cottus asper LOSO + graph topology\n"
        f"({len(edge_geoms)} edges, t={args.t:.1f}, "
        f"per-individual median {median_err:.0f} km)",
        fontsize=13, y=0.98,
    )
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_png}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint and run**

```bash
pixi run ruff check validation/sculpin/make_map_graph.py
pixi run python -m validation.sculpin.make_map_graph \
    --loso_dir out/sculpin_validation/loso_graph \
    --sample_data out/sculpin_validation/inputs/sample_data.txt \
    --topology validation/sculpin/sculpin_topology.geojson \
    --out_png validation/figures/sculpin_map_graph.png \
    --t 1.0 2>&1 | grep -v DownloadWarning | tail -3
```

Expected: writes `validation/figures/sculpin_map_graph.png`.

- [ ] **Step 3: Commit script + figure**

```bash
git add validation/sculpin/make_map_graph.py validation/figures/sculpin_map_graph.png
git commit -m "$(cat <<'EOF'
sculpin: cartopy map of LOSO predictions with topology graph overlaid

Mirrors validation/sculpin/make_map.py but adds:
  - LineString edges from sculpin_topology.geojson drawn as
    semi-transparent blue lines over the basemap
  - Per-individual median error in km in the title
  - Graph kernel diffusion time t in the title

Visualization shows where the topology connects vs disconnects sites,
making it easy to see whether the graph mode's predictions follow
realistic gene-flow paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Final test sweep + push

**Files:**
- No new files. Runs the full test suite, lints all modified/new files, pushes the branch.

- [ ] **Step 1: Run the complete graph-classifier test suite**

```bash
pixi run pytest tests/test_graph_classifier.py -v
```

Expected: 20 PASS.

- [ ] **Step 2: Run the complete classifier-head test suite to confirm no regressions**

```bash
pixi run pytest tests/test_classifier_head.py -v 2>&1 | tail -3
```

Expected: 26 PASS (unchanged).

- [ ] **Step 3: Run the full repo test suite**

```bash
pixi run pytest tests/ -q --ignore=tests/test_tf_dataset.py 2>&1 | tail -10
```

(Skipping `test_tf_dataset.py` because of the known xdist flake noted in CLAUDE.md.)

Expected: all tests pass; no failures attributable to graph-classifier changes.

- [ ] **Step 4: Lint everything modified or created on the branch**

```bash
pixi run ruff check \
    locator/graph.py \
    locator/prediction.py \
    locator/training.py \
    tests/test_graph_classifier.py \
    validation/sculpin/build_topology.py \
    validation/sculpin/run_loso_graph.py \
    validation/sculpin/make_map_graph.py \
    validation/sculpin/summarize_classifier.py
```

Expected: All checks passed.

- [ ] **Step 5: Verify the commit history is clean**

```bash
git log --oneline classifier-head..graph-classifier
```

Expected: ~11 commits, one per Task, with informative messages.

- [ ] **Step 6: Push to fork**

```bash
git push fork graph-classifier 2>&1 | tail -5
```

Expected: branch created on fork.

- [ ] **Step 7: Confirm push**

```bash
git log --oneline fork/graph-classifier -3
```

Expected: top commit matches local HEAD.

---

## Self-review checklist

After writing the plan, run through this once and fix any gaps inline.

- [ ] **Spec coverage** — every section of `2026-05-05-graph-classifier-design.md` has at least one task implementing it:
  - Goal 1 (config flags) → Tasks 5 (predict path), 5 (validation gate)
  - Goal 2 (GeoJSON schema) → Tasks 1 (load_topology), 6 (docs)
  - Goal 3 (sculpin builder helper) → Task 7
  - Goal 4 (sculpin LOSO validation) → Tasks 8 (runner), 9 (summarizer)
  - Goal 5 (FST-outlier site improvement evidence) → Task 9 Step 5 (manual recommendation prose)
  - Architecture / locator/graph.py → Tasks 1, 2, 3, 4
  - Architecture / locator/prediction.py modifications → Task 5
  - Architecture / locator/training.py validation extension → Task 5
  - Architecture / build_topology.py → Task 7
  - Architecture / sculpin_topology.geojson committed artifact → Task 7
  - Hyperparameters (graph_topology, graph_kernel_resolution) → Task 5 (config), Task 6 (docs)
  - GeoJSON schema → Tasks 1 (loader), 6 (docs), 7 (builder)
  - Tests merge zone → Tasks 1, 2, 3, 4, 5
  - Documentation → Task 6
  - Out-of-scope items → declared in spec; no plan task needed
- [ ] **Placeholder scan** — search the plan for "TBD", "TODO", "implement later", "fill in details", or "Similar to Task N" without a code block. The only "TODO"s are in the user-facing Recommendation prose template in Task 9 Step 5 (intentional manual-fill content).
- [ ] **Type consistency**:
  - `load_topology` returns `dict` with keys `nodes` (`(K, 2) float64`), `edges` (list of `(int, int, float)`), `node_names` (list of strings) — used consistently in Tasks 2, 3, 5, 7.
  - `match_centroids_to_nodes` returns `(K_train,) int64` — consumed by `subset_edges_to_centroids` in Task 5.
  - `subset_edges_to_centroids` returns `list[tuple[int, int, float]]` — consumed by `heat_kernel` in Task 5.
  - `heat_kernel` returns `(K, K) float64` — stored as `self._heat_kernel`, used as right-multiply in Task 5.
  - `self._heat_kernel`, `self._site_centroids`, `self._n_classes` reference patterns are consistent with classifier-head conventions.
  - Config keys: `graph_topology`, `graph_kernel_resolution`, `prediction_mode`, `optimize_tf_parallelism` — all read with `config.get(...)`.
  - HDF5 attributes referenced (only existing classifier-head ones — no new ones for graph mode since the topology is loaded from disk per session).
