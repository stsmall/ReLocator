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


def _nearest_node_haversine(
    point: np.ndarray, tree: scipy.spatial.cKDTree,
    nodes: np.ndarray, k: int,
) -> tuple[int, float]:
    """Find the haversine-closest node among the k degree-nearest candidates.

    Two-step lookup: KD-tree gives O(log K) candidates by degree distance;
    haversine gives the authoritative km distance. Necessary at high
    latitudes where 1° of longitude is much less than 1° of latitude in km.
    """
    _, cands = tree.query(point, k=k)
    cands = np.atleast_1d(cands)
    hav = np.array([
        _haversine_km(point[1], point[0], nodes[c, 1], nodes[c, 0])
        for c in cands
    ])
    best = int(np.argmin(hav))
    return int(cands[best]), float(hav[best])


def _name_or_default(value, idx: int) -> str:
    if value is None:
        return f"node_{idx}"
    s = str(value).strip()
    return s if s else f"node_{idx}"


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
            _name_or_default(n, i)
            for i, n in enumerate(points["name"].tolist())
        ]
    else:
        node_names = [f"node_{i}" for i in range(len(nodes))]

    edges: list[tuple[int, int, float]] = []
    if not lines.empty:
        # Index Points by KD-tree on (lon, lat) for fast endpoint matching.
        tree = scipy.spatial.cKDTree(nodes)
        k_query = min(len(nodes), 5)
        for _, row in lines.iterrows():
            coords = np.asarray(row.geometry.coords, dtype=np.float64)
            if coords.shape[0] < 2:
                continue
            start, end = coords[0], coords[-1]
            # Query k candidates by degree distance, then pick the
            # haversine-closest. The two-step approach is necessary at
            # high latitudes where 1 degree of longitude can be much
            # less than 1 degree of latitude in km.
            i_start, true_d_start = _nearest_node_haversine(
                start, tree, nodes, k_query
            )
            i_end, true_d_end = _nearest_node_haversine(
                end, tree, nodes, k_query
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
    :param k: Number of nodes — the resulting kernel is ``(k, k)``.
    :param t: Diffusion time. ``t=0`` returns the identity. Larger ``t``
        smooths probability mass farther across the graph.
    :returns: ``(k, k)`` symmetric float64 matrix. Apply to a softmax
        row vector ``P`` via ``P @ H`` to smooth probabilities along
        the topology before classifier-style prediction.
    """
    w = np.zeros((k, k), dtype=np.float64)
    for i, j, weight in edges:
        w[i, j] += weight
        w[j, i] += weight  # symmetric: undirected graph

    degree = w.sum(axis=1)
    laplacian = np.diag(degree) - w

    return scipy.linalg.expm(-t * laplacian)
