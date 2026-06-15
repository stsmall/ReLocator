#!/usr/bin/env python3
"""Build the freshwater range polygon for the sculpin range-mask experiment.

Source: HydroSHEDS HydroLAKES (global lake polygons, ≥ 10 ha) and HydroRIVERS
(North America river network). Both are vastly more comprehensive than
NaturalEarth at the inland-PNW scale we care about — the NE 10m product
covers only ~15 lakes + 25 river segments in the entire PNW bbox, missing
most of the sculpin sampling sites (Mosquito, Martins, Tlell, Bella Coola,
Nimpo, Meziadin all > 1° from the nearest NE feature).

Inputs are expected at:
  --hydrolakes  HydroLAKES_polys_v10.shp (from HydroLAKES_polys_v10_shp.zip)
  --hydrorivers HydroRIVERS_v10_na.shp   (from HydroRIVERS_v10_na_shp.zip)

Both shapefiles live outside the repo (e.g. /sietch_colab/.../data/hydrosheds/)
because of size: ~1.5 GB and ~257 MB extracted.

Pipeline:
  1. Read HydroLAKES, filter to bbox (only lake polygons that touch PNW).
  2. Read HydroRIVERS, filter to bbox AND drop tiny streams (ORD_STRA <= 2,
     keep order ≥ 3 strahler stream order — sculpin habitat is in larger
     drainages, and including every headwater would make the buffered
     polygon meaninglessly large).
  3. Buffer rivers by ``--buffer_deg`` to convert lines to ribbons.
  4. Buffer lakes by ``--buffer_deg`` (sculpin within a few km of a lake's
     edge are plausibly from that lake).
  5. Union everything, clip to PNW bbox, write a single shapefile.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import shapely.geometry
import shapely.ops

PNW_BBOX = (-138.0, 47.0, -114.0, 60.0)
DEFAULT_BUFFER_DEG = 0.05  # ~5 km at the latitudes we care about
DEFAULT_MIN_STREAM_ORDER = 3  # 1=headwater, 8+=major rivers; 3 keeps medium drainages


def build_freshwater_polygon(
    hydrolakes_shp: Path,
    hydrorivers_shp: Path,
    bbox: tuple[float, float, float, float] = PNW_BBOX,
    buffer_deg: float = DEFAULT_BUFFER_DEG,
    min_stream_order: int = DEFAULT_MIN_STREAM_ORDER,
) -> shapely.geometry.base.BaseGeometry:
    bbox_poly = shapely.geometry.box(*bbox)

    print(f"  reading {hydrolakes_shp.name} (clipping to bbox)...", flush=True)
    lakes_gdf = gpd.read_file(hydrolakes_shp, bbox=bbox)
    print(f"    {len(lakes_gdf)} lake polygons within bbox", flush=True)
    if len(lakes_gdf) == 0:
        lakes_buffered = []
    else:
        lakes_union = shapely.ops.unary_union(lakes_gdf.geometry.tolist())
        lakes_buffered = [lakes_union.buffer(buffer_deg)]

    print(f"  reading {hydrorivers_shp.name} (clipping to bbox + min order)...",
          flush=True)
    rivers_gdf = gpd.read_file(hydrorivers_shp, bbox=bbox)
    print(f"    {len(rivers_gdf)} river reaches within bbox", flush=True)
    # ORD_STRA is the Strahler stream order column in HydroRIVERS.
    if "ORD_STRA" in rivers_gdf.columns:
        before = len(rivers_gdf)
        rivers_gdf = rivers_gdf[rivers_gdf["ORD_STRA"] >= min_stream_order]
        print(f"    after ORD_STRA >= {min_stream_order} filter: "
              f"{len(rivers_gdf)} (dropped {before - len(rivers_gdf)})", flush=True)
    if len(rivers_gdf) == 0:
        rivers_buffered = []
    else:
        rivers_union = shapely.ops.unary_union(rivers_gdf.geometry.tolist())
        rivers_buffered = [rivers_union.buffer(buffer_deg)]

    full = shapely.ops.unary_union(lakes_buffered + rivers_buffered)
    return full.intersection(bbox_poly)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hydrolakes", required=True, type=Path,
                   help="HydroLAKES_polys_v10.shp path.")
    p.add_argument("--hydrorivers", required=True, type=Path,
                   help="HydroRIVERS_v10_na.shp path.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--buffer_deg", type=float, default=DEFAULT_BUFFER_DEG)
    p.add_argument("--min_stream_order", type=int, default=DEFAULT_MIN_STREAM_ORDER,
                   help="Drop river reaches below this Strahler order. "
                        "1=every headwater (huge); 3=medium drainages; "
                        "5=large rivers only.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Building freshwater range polygon (buffer={args.buffer_deg}°, "
          f"min_stream_order={args.min_stream_order})...", flush=True)
    poly = build_freshwater_polygon(
        hydrolakes_shp=args.hydrolakes,
        hydrorivers_shp=args.hydrorivers,
        bbox=PNW_BBOX,
        buffer_deg=args.buffer_deg,
        min_stream_order=args.min_stream_order,
    )
    print(f"  polygon area ≈ {poly.area:.2f} sq.deg "
          f"(~{int(poly.area * 111 * 111)} sq.km)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {"name": ["freshwater"]}, geometry=[poly], crs="EPSG:4326",
    )
    gdf.to_file(args.out, driver="ESRI Shapefile")
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
