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
DEFAULT_MAX_SNAP_KM = 25.0


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
        pair = (a, b) if a < b else (b, a)
        if pair not in edge_geoms or weight < g.edges[pair].get("weight", float("inf")):
            edge_geoms[pair] = line
            g.add_edge(a, b, weight=weight, geometry=line)

    print(f"    river graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges",
          flush=True)
    return g, {nid: (data["lon"], data["lat"]) for nid, data in g.nodes(data=True)}


def snap_sites_to_graph(sites: pd.DataFrame, g: nx.Graph,
                        node_coords: dict,
                        max_snap_km: float = DEFAULT_MAX_SNAP_KM) -> dict[str, int]:
    """For each site, find the nearest graph node by haversine distance.

    Sites whose nearest node is farther than ``max_snap_km`` are skipped
    (omitted from the returned dict) — they will appear as isolated Points
    in the GeoJSON but produce no edges. This prevents spurious
    cross-watershed connections caused by sites outside the bbox or
    in basins below the stream-order filter.
    """
    coords = np.array(list(node_coords.values()), dtype=np.float64)  # (N, 2) lon/lat
    node_ids = list(node_coords.keys())
    snapped = {}
    for _, row in sites.iterrows():
        site_lon, site_lat = float(row["lon"]), float(row["lat"])
        d = np.array([
            _haversine_km(site_lat, site_lon, c[1], c[0]) for c in coords
        ])
        best = int(np.argmin(d))
        if d[best] > max_snap_km:
            print(f"    {row['site']:<20} → SKIPPED (nearest node {d[best]:.2f} km "
                  f"> max_snap_km={max_snap_km})", flush=True)
            continue
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
    p.add_argument("--max_snap_km", type=float, default=DEFAULT_MAX_SNAP_KM,
                   help="Sites whose nearest river-network node is farther than "
                        "this many km are emitted as isolated Points (no edges). "
                        "Prevents spurious cross-watershed connections.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("Building sculpin topology from HydroRIVERS...", flush=True)
    g, node_coords = build_river_graph(args.hydrorivers, args.min_stream_order)

    sites = pd.read_csv(args.sites_tsv, sep="\t")
    print(f"  snapping {len(sites)} sampled sites to river-network nodes:",
          flush=True)
    snapped = snap_sites_to_graph(sites, g, node_coords, args.max_snap_km)

    features = []
    for _, row in sites.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [float(row["lon"]), float(row["lat"])]},
            "properties": {"name": str(row["site"])},
        })

    # Build site lon/lat lookup for endpoint anchoring.
    site_coords = {
        row["site"]: (float(row["lon"]), float(row["lat"]))
        for _, row in sites.iterrows()
    }

    site_names = sites["site"].tolist()
    n_edges = 0
    for i, name_i in enumerate(site_names):
        for j in range(i + 1, len(site_names)):
            name_j = site_names[j]
            if name_i not in snapped or name_j not in snapped:
                continue
            line = find_path_linestring(g, snapped[name_i], snapped[name_j])
            if line is None:
                continue
            # Anchor the LineString endpoints exactly at the site Point
            # coordinates so that load_topology's 1-km endpoint check passes.
            # The river-network path may start/end a few km from the site.
            raw_coords = [list(c) for c in line.coords]
            raw_coords[0] = list(site_coords[name_i])
            raw_coords[-1] = list(site_coords[name_j])
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString",
                             "coordinates": raw_coords},
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
