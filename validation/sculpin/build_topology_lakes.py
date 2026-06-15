#!/usr/bin/env python3
"""Build a denser sculpin GeoJSON topology using HydroLAKES polygons as lake
super-nodes that absorb HydroRIVERS reach endpoints.

The v1 builder (build_topology.py) used river-segment endpoints as graph
nodes at Strahler order ≥ 3 and produced only 2 edges across 16 sculpin
sites — most sites snapped tens to hundreds of km from any node. This
builder produces a richer graph by:

  1. Loading HydroLAKES polygons (filtered by area) within the bbox.
  2. Loading HydroRIVERS reaches without stream-order filter.
  3. Building a NetworkX graph where each river endpoint becomes a node,
     EXCEPT endpoints that fall inside a lake polygon — those collapse
     into a single super-node per lake.
  4. Snapping each site to a lake (point-in-polygon) or, failing that,
     to the nearest graph node by haversine.
  5. Tracing shortest paths between every pair of site-nodes through
     the unified lake+river graph.

The resulting GeoJSON has the same schema as v1 (Point features for
sites, LineString features for site-to-site connections), so the
existing locator/graph.py loader needs no changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.ops
from scipy.spatial import cKDTree

PNW_BBOX = (-138.0, 47.0, -110.0, 60.0)  # extended east to catch Mackenzie/Peace
DEFAULT_MIN_LAKE_AREA_KM2 = 0.5
DEFAULT_MAX_SNAP_KM = 25.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def load_lakes(lakes_shp: Path, bbox_poly: shapely.geometry.Polygon,
               min_area_km2: float) -> gpd.GeoDataFrame:
    print(f"  reading {lakes_shp.name}...", flush=True)
    lakes = gpd.read_file(lakes_shp, bbox=bbox_poly).to_crs("EPSG:4326")
    before = len(lakes)
    lakes = lakes[lakes["Lake_area"] >= min_area_km2].reset_index(drop=True)
    print(f"    {len(lakes)} lakes ≥ {min_area_km2} km² (dropped {before - len(lakes)})",
          flush=True)
    return lakes


def build_lake_river_graph(
    lakes: gpd.GeoDataFrame,
    rivers_shp: Path,
    bbox_poly: shapely.geometry.Polygon,
) -> tuple[nx.Graph, dict, dict]:
    """Build a NetworkX graph using HydroRIVERS' NEXT_DOWN topology + HydroLAKES.

    Each river reach becomes a node identified by its ``HYRIV_ID``; reaches
    are connected to their downstream neighbor via the ``NEXT_DOWN`` field
    (the canonical HydroSHEDS topology — reach geometries don't reliably
    share endpoint coordinates, so geometry-based stitching gives a graph
    full of orphan singletons).

    Each lake polygon becomes an additional node ``('lake', Hylak_id)``.
    Lake nodes are connected to every river reach whose centroid falls
    inside (or is covered by) the lake polygon — this lets shortest-path
    queries traverse rivers → through lakes → back to rivers.

    Returns ``(g, lake_centroids, reach_centroids)`` for KD-tree snapping.
    """
    print(f"  reading {rivers_shp.name}...", flush=True)
    rivers = gpd.read_file(rivers_shp, bbox=bbox_poly).to_crs("EPSG:4326")
    rivers = rivers.reset_index(drop=True)
    print(f"    {len(rivers)} reaches in bbox", flush=True)

    g = nx.Graph()

    # Lake nodes (always present, even if isolated).
    for _, lake in lakes.iterrows():
        cen = lake.geometry.centroid
        g.add_node(("lake", int(lake["Hylak_id"])),
                   lon=float(cen.x), lat=float(cen.y),
                   lake_name=str(lake["Lake_name"]) if lake["Lake_name"] else "",
                   lake_area=float(lake["Lake_area"]))

    # Reach nodes — keyed by HYRIV_ID (int).
    valid_reach_ids = set(rivers["HYRIV_ID"].astype(int).tolist())
    for _, r in rivers.iterrows():
        cen = r.geometry.centroid
        rid = int(r["HYRIV_ID"])
        g.add_node(("reach", rid), lon=float(cen.x), lat=float(cen.y),
                   length_km=float(r["LENGTH_KM"]))

    # NEXT_DOWN edges (reach → downstream reach).
    for _, r in rivers.iterrows():
        rid = int(r["HYRIV_ID"])
        nd = int(r["NEXT_DOWN"])
        if nd == 0 or nd not in valid_reach_ids:
            continue  # terminal or downstream is outside bbox
        g.add_edge(("reach", rid), ("reach", nd),
                   weight=float(r["LENGTH_KM"]),
                   geometry=r.geometry)

    # Lake ↔ reach edges, two sources:
    # 1) Geometry-intersects: any reach line that touches the lake polygon.
    # 2) Pour-point: for each lake, connect to the closest reach to the
    #    HydroLAKES-provided pour point (HydroRIVERS coverage is incomplete
    #    around small lakes — outflow streams below their drainage-area
    #    threshold are missing — so geometry alone strands many big lakes
    #    like McLeod / PeaceBC as singletons).
    print("  joining reaches to lakes (intersect + pour-point)...", flush=True)
    lakes_sindex = lakes.sindex
    lake_geoms = lakes.geometry.values
    lake_ids = lakes["Hylak_id"].astype(int).values
    lake_river_edges = 0
    for _, r in rivers.iterrows():
        rid = int(r["HYRIV_ID"])
        for ci in lakes_sindex.intersection(r.geometry.bounds):
            if lake_geoms[ci].intersects(r.geometry):
                lake_node = ("lake", int(lake_ids[ci]))
                if not g.has_edge(lake_node, ("reach", rid)):
                    g.add_edge(lake_node, ("reach", rid),
                               weight=0.0,
                               geometry=r.geometry)
                    lake_river_edges += 1

    # Pour-point fallback: KD-tree of reach centroids, snap each lake's
    # Pour_long/Pour_lat to the nearest reach within MAX_POUR_SNAP_KM.
    pour_snap_km = 10.0
    reach_nodes = [n for n in g.nodes if n[0] == "reach"]
    reach_coord_arr = np.array(
        [(g.nodes[n]["lon"], g.nodes[n]["lat"]) for n in reach_nodes],
        dtype=np.float64,
    )
    reach_tree = cKDTree(reach_coord_arr)
    pour_added = 0
    for _, lake in lakes.iterrows():
        plon = float(lake["Pour_long"])
        plat = float(lake["Pour_lat"])
        if not (np.isfinite(plon) and np.isfinite(plat)):
            continue
        lake_node = ("lake", int(lake["Hylak_id"]))
        if any(n[0] == "reach" for n in g.neighbors(lake_node)):
            continue  # already connected to the river network
        _, cand_ix = reach_tree.query([plon, plat], k=min(20, len(reach_nodes)))
        if np.isscalar(cand_ix):
            cand_ix = [int(cand_ix)]
        else:
            cand_ix = [int(i) for i in cand_ix]
        best_d, best_node = float("inf"), None
        for ci in cand_ix:
            n = reach_nodes[ci]
            d = _haversine_km(plat, plon, reach_coord_arr[ci, 1], reach_coord_arr[ci, 0])
            if d < best_d:
                best_d, best_node = d, n
        if best_node is not None and best_d <= pour_snap_km:
            g.add_edge(lake_node, best_node, weight=best_d,
                       geometry=shapely.geometry.LineString([
                           (plon, plat),
                           (g.nodes[best_node]["lon"], g.nodes[best_node]["lat"]),
                       ]))
            lake_river_edges += 1
            pour_added += 1
    print(f"    {pour_added} lakes attached via pour-point", flush=True)

    n_lake_nodes = sum(1 for n in g.nodes if n[0] == "lake")
    n_reach_nodes = sum(1 for n in g.nodes if n[0] == "reach")
    print(f"    graph: {g.number_of_nodes()} nodes "
          f"({n_lake_nodes} lakes, {n_reach_nodes} reaches), "
          f"{g.number_of_edges()} edges "
          f"({lake_river_edges} lake↔reach)", flush=True)

    ccs = list(nx.connected_components(g))
    sizes = sorted([len(cc) for cc in ccs], reverse=True)
    print(f"    connected components: {len(ccs)}; largest 5: {sizes[:5]}",
          flush=True)

    lake_centroids = {n: (g.nodes[n]["lon"], g.nodes[n]["lat"])
                      for n in g.nodes if n[0] == "lake"}
    reach_centroids = {n: (g.nodes[n]["lon"], g.nodes[n]["lat"])
                       for n in g.nodes if n[0] == "reach"}
    return g, lake_centroids, reach_centroids


def snap_sites_to_graph(
    sites: pd.DataFrame,
    g: nx.Graph,
    lakes: gpd.GeoDataFrame,
    lake_centroids: dict,
    reach_centroids: dict,
    max_snap_km: float,
) -> dict:
    """For each site, prefer point-in-lake; else nearest graph node by haversine."""
    lakes_sindex = lakes.sindex
    lake_geoms = lakes.geometry.values
    lake_ids = lakes["Hylak_id"].astype(int).values

    node_list = list(g.nodes)
    coord_arr = np.array([(g.nodes[n]["lon"], g.nodes[n]["lat"]) for n in node_list],
                         dtype=np.float64)
    tree = cKDTree(coord_arr)

    snapped: dict[str, tuple] = {}
    for _, row in sites.iterrows():
        site_lon, site_lat = float(row["lon"]), float(row["lat"])
        # Step 1: point-in-polygon for any lake.
        pt = shapely.geometry.Point(site_lon, site_lat)
        in_lake = None
        for ci in lakes_sindex.intersection(pt.bounds):
            if lake_geoms[ci].covers(pt):
                in_lake = ("lake", int(lake_ids[ci]))
                break
        if in_lake is not None and in_lake in g.nodes:
            snapped[row["site"]] = in_lake
            cen = lake_centroids[in_lake]
            print(f"    {row['site']:<15} → IN_LAKE {in_lake[1]} "
                  f"@ ({cen[0]:.3f}, {cen[1]:.3f})", flush=True)
            continue
        # Step 2: nearest node within max_snap_km (top-K KD-tree + haversine refine).
        k = min(len(node_list), 20)
        _, cand_ix = tree.query([site_lon, site_lat], k=k)
        if np.isscalar(cand_ix):
            cand_ix = [int(cand_ix)]
        else:
            cand_ix = [int(i) for i in cand_ix]
        best_d, best_node = float("inf"), None
        for ci in cand_ix:
            n = node_list[ci]
            nlon, nlat = coord_arr[ci]
            d = _haversine_km(site_lat, site_lon, nlat, nlon)
            if d < best_d:
                best_d, best_node = d, n
        if best_node is None or best_d > max_snap_km:
            print(f"    {row['site']:<15} → SKIPPED ({best_d:.2f} km > "
                  f"max_snap_km={max_snap_km})", flush=True)
            continue
        snapped[row["site"]] = best_node
        kind = "LAKE" if best_node[0] == "lake" else "reach"
        print(f"    {row['site']:<15} → {kind} id={best_node[1]} "
              f"({best_d:.2f} km)", flush=True)
    return snapped


def find_path_linestring(
    g: nx.Graph, src: tuple, dst: tuple,
) -> shapely.geometry.LineString | None:
    try:
        path = nx.shortest_path(g, src, dst, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    if len(path) < 2:
        return None
    pieces = []
    for i in range(len(path) - 1):
        edge = g.edges.get((path[i], path[i + 1]))
        if edge is None or "geometry" not in edge:
            continue
        pieces.append(edge["geometry"])
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
    p.add_argument("--hydrolakes", required=True, type=Path)
    p.add_argument("--hydrorivers", required=True, type=Path)
    p.add_argument("--sites_tsv", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--min_lake_area_km2", type=float, default=DEFAULT_MIN_LAKE_AREA_KM2)
    p.add_argument("--max_snap_km", type=float, default=DEFAULT_MAX_SNAP_KM)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bbox_poly = shapely.geometry.box(*PNW_BBOX)

    print("Loading HydroLAKES...", flush=True)
    lakes = load_lakes(args.hydrolakes, bbox_poly, args.min_lake_area_km2)

    print("Building lake+river graph...", flush=True)
    g, lake_centroids, reach_centroids = build_lake_river_graph(
        lakes, args.hydrorivers, bbox_poly,
    )

    print("Snapping sites...", flush=True)
    sites = pd.read_csv(args.sites_tsv, sep="\t")
    snapped = snap_sites_to_graph(
        sites, g, lakes, lake_centroids, reach_centroids, args.max_snap_km,
    )

    print(f"\nSites snapped: {len(snapped)}/{len(sites)}", flush=True)

    print("\nFinding shortest paths...", flush=True)
    site_names = sites["site"].tolist()
    features = []
    for _, row in sites.iterrows():
        # Use the snapped node's centroid coord if we have one (for in-lake sites
        # the GeoJSON Point matches the lake centroid; otherwise the original).
        if row["site"] in snapped:
            n = snapped[row["site"]]
            lon, lat = g.nodes[n]["lon"], g.nodes[n]["lat"]
        else:
            lon, lat = float(row["lon"]), float(row["lat"])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": str(row["site"])},
        })

    n_edges = 0
    for i, name_i in enumerate(site_names):
        for j in range(i + 1, len(site_names)):
            name_j = site_names[j]
            if name_i not in snapped or name_j not in snapped:
                continue
            line = find_path_linestring(g, snapped[name_i], snapped[name_j])
            if line is None:
                continue
            raw_coords = [list(c) for c in line.coords]
            # Anchor LineString endpoints at the GeoJSON Point coords so the
            # 1-km endpoint check in load_topology passes cleanly. We keep the
            # interior path geometry so users can see the actual route.
            features_pi = next(f for f in features
                               if f["properties"]["name"] == name_i)
            features_pj = next(f for f in features
                               if f["properties"]["name"] == name_j)
            raw_coords[0] = list(features_pi["geometry"]["coordinates"])
            raw_coords[-1] = list(features_pj["geometry"]["coordinates"])
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": raw_coords},
                "properties": {"name": f"{name_i}_to_{name_j}"},
            })
            n_edges += 1

    print(f"  wrote {n_edges} edges connecting "
          f"{len(snapped)} of {len(site_names)} sites", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, indent=2,
    ))
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
