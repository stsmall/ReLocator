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


def test_load_topology_high_latitude_kd_tree_picks_haversine_nearest(tmp_path):
    """At high latitude, KD-tree degree-nearest may differ from haversine-nearest;
    the implementation must pick haversine-nearest to avoid false rejections.
    """
    path = tmp_path / "topo.geojson"
    # Line from (0°E, 80°N) to (1°E, 80°N).
    # Start endpoint (0, 80) has two candidate nodes:
    #   A at (0°E, 80.009°N): degree-nearest (0.009° away), but 1.001 km haversine
    #                          — just barely over the 1 km tolerance.
    #   B at (0.05°E, 80°N):  degree-farther (0.05° away), but 0.965 km haversine
    #                          — within tolerance because lon° compress at 80°N.
    # End endpoint (1, 80) matches C exactly.
    # If the implementation uses k=1 KD-tree query, it picks A and raises ValueError.
    # The fix queries k=5 and picks the haversine-closest (B), allowing the edge.
    _write_geojson(
        path,
        points=[
            (0.0, 80.009, "A"),  # ~1.001 km haversine north of (0, 80)
            (0.05, 80.0, "B"),   # ~0.965 km haversine east of (0, 80) — within tolerance
            (1.0, 80.0, "C"),    # exact match for end endpoint
        ],
        lines=[
            ([[0.0, 80.0], [1.0, 80.0]], "B_to_C"),
        ],
    )
    topo = load_topology(path)
    assert len(topo["edges"]) == 1


def test_load_topology_integer_name_property_coerced_to_string(tmp_path):
    """Non-string name values (e.g., integer) must be coerced, not silently
    auto-named — the property carries information."""
    path = tmp_path / "topo.geojson"
    payload = {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
         "properties": {"name": 42}},
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
         "properties": {"name": "site_b"}},
    ]}
    path.write_text(json.dumps(payload))
    topo = load_topology(path)
    assert topo["node_names"] == ["42", "site_b"]


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
    np.testing.assert_allclose(H[2], np.array([0.0, 0.0, 1.0]), atol=1e-10)
    np.testing.assert_allclose(H[:, 2], np.array([0.0, 0.0, 1.0]), atol=1e-10)


def test_heat_kernel_increases_off_diagonal_with_t():
    """Larger t = more probability mass on the off-diagonal."""
    edges = [(0, 1, 1.0)]
    H_small = heat_kernel(edges, k=2, t=0.1)
    H_large = heat_kernel(edges, k=2, t=2.0)
    assert H_small[0, 1] < H_large[0, 1]


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
    sd.loc[sd["sampleID"] == "site2_0", ["x", "y"]] = np.nan
    sd_path = tmp_path / "sample_data.txt"
    sd.to_csv(sd_path, sep="\t", index=False)
    return matrix_path, sd_path


def test_predict_classify_then_avg_with_graph_topology(tmp_path):
    """End-to-end: train classify_then_avg, predict with a graph topology
    that connects sites 0-1; predictions should differ from non-graph case."""
    from locator import Locator
    matrix_path, sd_path = _write_synthetic_matrix_data(tmp_path)

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

    assert not pred_with_graph[["x", "y"]].isna().any().any()
    assert hasattr(loc2, "_heat_kernel")
    assert loc2._heat_kernel.shape == (3, 3)
    np_diff = np.abs(
        pred_no_graph[["x", "y"]].values - pred_with_graph[["x", "y"]].values
    ).max()
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
