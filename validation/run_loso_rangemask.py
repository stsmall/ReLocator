#!/usr/bin/env python3
"""Path B re-run with ReLocator's range-mask training penalty.

Uses ReLocator's existing `loss_with_range_penalty` (locator/models.py)
to train the network with a coast-polygon constraint, so predictions
stay on/near the Pacific coast natively rather than being snapped after
the fact (validation/summarize.py:_snap_to_coast). No merge-zone code
changes — uses the Locator Python API since the CLI doesn't expose the
range-mask flags.

The species-range polygon is built once at startup by buffering Natural
Earth's 10m coastline by ~10 km (≈0.1° at these latitudes) and clipping
to the Pacific NW bbox. Saved alongside fold outputs so the run is
reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.affinity
import shapely.geometry
import shapely.ops

from validation import common
from validation.coastline import (
    BARNACLE_COAST,
    build_coast_index,
    project_to_coast,
)

_COAST_CUM, _COAST_PTS = build_coast_index()
_SITE_TO_INDEX = {site: i for i, (site, _, _) in enumerate(BARNACLE_COAST)}

PACIFIC_BBOX = (-152.0, 30.0, -115.0, 62.0)
COAST_BUFFER_DEG = 0.1

# ReLocator's loss_with_range_penalty has a coordinate-space mismatch:
# y_pred is z-scored (locator/training.py:_build_datasets_and_fit feeds
# normalized_locs to the loss), but the rasterized mask is built in raw
# lon/lat from the shapefile bounds. We work around by writing a fold-
# specific shapefile whose polygon is transformed into the same z-score
# space ReLocator uses, computed from the held-out training samples.
# See validation/notes/range_mask_bug.md for the full write-up.
NORMALIZED_RESOLUTION = 0.01  # z-score units; ~10 km at sd_lat ≈ 10°
DEFAULT_PENALTY_WEIGHT = 50.0  # z-space euclidean ≈ 1-3, so >>1 to dominate


def build_coast_range_polygon(
    bbox: tuple[float, float, float, float] = PACIFIC_BBOX,
    buffer_deg: float = COAST_BUFFER_DEG,
) -> shapely.geometry.base.BaseGeometry:
    """Buffer Natural Earth's 10m coastline and clip to bbox. Returns geometry."""
    from cartopy.io.shapereader import Reader, natural_earth

    shp_in = natural_earth(category="physical", name="coastline", resolution="10m")
    bbox_poly = shapely.geometry.box(*bbox)
    geoms = []
    for geom in Reader(shp_in).geometries():
        if geom.intersects(bbox_poly):
            clipped = geom.intersection(bbox_poly)
            if not clipped.is_empty:
                geoms.append(clipped)
    if not geoms:
        raise RuntimeError(f"no coastline geometries intersect {bbox}")
    coast_lines = shapely.ops.unary_union(geoms)
    coast_polygon = coast_lines.buffer(buffer_deg)
    return coast_polygon.intersection(bbox_poly)


def write_polygon_shapefile(
    polygon: shapely.geometry.base.BaseGeometry,
    out_path: Path,
    name: str = "range",
    crs: str = "EPSG:4326",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame({"name": [name]}, geometry=[polygon], crs=crs)
    gdf.to_file(out_path, driver="ESRI Shapefile")
    return out_path


def normalize_polygon(
    polygon: shapely.geometry.base.BaseGeometry,
    mean_lon: float, sd_lon: float,
    mean_lat: float, sd_lat: float,
) -> shapely.geometry.base.BaseGeometry:
    """Apply z-score transform x'=(x-mean)/sd to polygon coords.

    Matches `locator.data.filters.normalize_locs`: lon is index 0, lat is 1.
    Shapely affine matrix is [a, b, d, e, xoff, yoff] for
    x' = a*x + b*y + xoff,  y' = d*x + e*y + yoff.
    """
    return shapely.affinity.affine_transform(
        polygon,
        [1.0 / sd_lon, 0.0, 0.0, 1.0 / sd_lat,
         -mean_lon / sd_lon, -mean_lat / sd_lat],
    )


def fold_normalization_params(
    held_sd: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """Reproduce ReLocator's nanmean/nanstd of x (lon) and y (lat)."""
    locs = held_sd[["x", "y"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    return (
        float(np.nanmean(locs[:, 0])),
        float(np.nanstd(locs[:, 0])),
        float(np.nanmean(locs[:, 1])),
        float(np.nanstd(locs[:, 1])),
    )


def blank_holdout_locations(
    sample_data: pd.DataFrame, sample_table: pd.DataFrame, target_site: str
) -> pd.DataFrame:
    held = sample_table.loc[sample_table["site"] == target_site, "sample_name"].tolist()
    out = sample_data.copy()
    out["x"] = out["x"].astype(object)
    out["y"] = out["y"].astype(object)
    mask = out["sampleID"].isin(held)
    out.loc[mask, "x"] = "NA"
    out.loc[mask, "y"] = "NA"
    return out


def _run_one_fold_inproc(
    *,
    site_code: str,
    site_lat: float,
    site_lon: float,
    feature_matrix_path: Path,
    sample_data_full: pd.DataFrame,
    sample_table: pd.DataFrame,
    coast_polygon: shapely.geometry.base.BaseGeometry,
    out_dir: Path,
    max_epochs: int,
    gpu_number: int,
    penalty_weight: float = DEFAULT_PENALTY_WEIGHT,
) -> dict:
    """Train one LOSO fold via the Locator Python API with range penalty.

    Writes a fold-specific shapefile whose polygon is z-score-transformed
    using the held-out fold's training-sample lon/lat statistics, so the
    rasterized mask lives in the same coordinate space as the model's
    normalized predictions. See module-docstring note about the upstream
    coordinate-space mismatch.
    """
    fold_dir = out_dir / site_code
    fold_dir.mkdir(parents=True, exist_ok=True)

    held_sd = blank_holdout_locations(sample_data_full, sample_table, target_site=site_code)
    sd_path = fold_dir / "sample_data.txt"
    held_sd.to_csv(sd_path, sep="\t", index=False)

    mean_lon, sd_lon, mean_lat, sd_lat = fold_normalization_params(held_sd)
    coast_norm = normalize_polygon(
        coast_polygon, mean_lon, sd_lon, mean_lat, sd_lat
    )
    fold_shp = fold_dir / "range_normalized.shp"
    write_polygon_shapefile(coast_norm, fold_shp, name="coast_z")

    out_prefix = fold_dir / "locator"
    fold_result: dict = {
        "site": site_code,
        "true_lat": site_lat,
        "true_lon": site_lon,
        "gl_mode": "dosage",
        "use_range_penalty": True,
        "penalty_weight": penalty_weight,
        "norm_mean_lon": mean_lon,
        "norm_sd_lon": sd_lon,
        "norm_mean_lat": mean_lat,
        "norm_sd_lat": sd_lat,
    }
    t0 = time.time()
    try:
        from locator import Locator

        config = {
            "out": str(out_prefix),
            "matrix": str(feature_matrix_path),
            "sample_data": str(sd_path),
            "max_epochs": max_epochs,
            "patience": 30,
            "seed": 42,
            "gpu_number": gpu_number,
            "use_range_penalty": True,
            "species_range_shapefile": str(fold_shp),
            "resolution": NORMALIZED_RESOLUTION,
            "penalty_weight": penalty_weight,
        }
        loc = Locator(config=config)
        genotypes, samples = loc.load_genotypes(matrix=str(feature_matrix_path))
        loc.train(genotypes=genotypes, samples=samples)
        loc.predict(genotypes=genotypes, samples=samples)
    except Exception as exc:
        fold_result["status"] = "FAILED"
        fold_result["reason"] = f"{type(exc).__name__}: {exc}"
        (fold_dir / "fold_result.json").write_text(
            json.dumps(fold_result, indent=2, default=str)
        )
        return fold_result

    elapsed = time.time() - t0
    fold_result["elapsed_s"] = round(elapsed, 1)

    predlocs = Path(f"{out_prefix}_predlocs.txt")
    if not predlocs.exists():
        fold_result["status"] = "FAILED"
        fold_result["reason"] = f"predlocs missing at {predlocs}"
        (fold_dir / "fold_result.json").write_text(
            json.dumps(fold_result, indent=2, default=str)
        )
        return fold_result

    pred = common.parse_predlocs(predlocs)
    held_ids = sample_table.loc[
        sample_table["site"] == site_code, "sample_name"
    ].tolist()
    held_pred = pred[pred["sampleID"].isin(held_ids)].copy()
    held_pred["error_km"] = common.haversine(
        held_pred["y"].values, held_pred["x"].values, site_lat, site_lon
    )
    fold_result["status"] = "OK"
    fold_result["n_held_out"] = int(len(held_pred))
    fold_result["mean_pred_lat"] = float(held_pred["y"].mean())
    fold_result["mean_pred_lon"] = float(held_pred["x"].mean())
    fold_result["mean_error_km"] = float(held_pred["error_km"].mean())
    fold_result["median_error_km"] = float(held_pred["error_km"].median())

    true_idx = _SITE_TO_INDEX.get(site_code)
    if true_idx is not None:
        coast_results = [
            project_to_coast(lat, lon, _COAST_CUM, _COAST_PTS)
            for lat, lon in zip(
                held_pred["y"].values, held_pred["x"].values, strict=True
            )
        ]
        coast_pos = np.array([cp for cp, _ in coast_results], dtype=np.float64)
        offshore = np.array([off for _, off in coast_results], dtype=np.float64)
        along_coast_err = np.abs(coast_pos - _COAST_CUM[true_idx])
        held_pred["coast_pos_km"] = coast_pos
        held_pred["offshore_km"] = offshore
        held_pred["along_coast_err_km"] = along_coast_err
        fold_result["mean_along_coast_err_km"] = float(along_coast_err.mean())
        fold_result["median_along_coast_err_km"] = float(np.median(along_coast_err))
        fold_result["mean_offshore_km"] = float(offshore.mean())

    fold_result["per_sample"] = held_pred.to_dict(orient="records")
    (fold_dir / "fold_result.json").write_text(
        json.dumps(fold_result, indent=2, default=str)
    )
    return fold_result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True, type=Path)
    p.add_argument("--samples-locations", required=True, type=Path)
    p.add_argument("--sample-table", required=True, type=Path)
    p.add_argument("--feature-matrix", required=True, type=Path,
                   help="path to gl_dosage.txt produced by gl_to_locator.py")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--gpu-number", type=int, default=0)
    p.add_argument("--penalty-weight", type=float, default=DEFAULT_PENALTY_WEIGHT,
                   help="per-sample weight on the out-of-range penalty term")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    coast_polygon = build_coast_range_polygon()
    raw_shp = args.out_dir / "pacific_coast_range.shp"
    write_polygon_shapefile(coast_polygon, raw_shp, name="pacific_coast_buffer")
    print(f"run_loso_rangemask: built coast polygon "
          f"(area ≈ {coast_polygon.area:.2f} sq.deg, "
          f"~{int(coast_polygon.area * 111 * 111)} sq.km), "
          f"raw shapefile at {raw_shp}", flush=True)

    sample_data_full = pd.read_csv(args.inputs_dir / "sample_data.txt", sep="\t")
    sample_table = pd.read_csv(args.sample_table, sep="\t")
    samples_locations = pd.read_csv(args.samples_locations, sep="\t")
    site_rows = list(samples_locations.itertuples(index=False))

    print(f"run_loso_rangemask: starting {len(site_rows)} folds with "
          f"use_range_penalty=True on GPU {args.gpu_number}", flush=True)

    t_sweep = time.time()
    for idx, row in enumerate(site_rows):
        print(f"\n=== fold {idx+1}/{len(site_rows)}: {row.SiteCode} ===", flush=True)
        res = _run_one_fold_inproc(
            site_code=row.SiteCode,
            site_lat=float(row.Lat),
            site_lon=float(row.Lon),
            feature_matrix_path=args.feature_matrix,
            sample_data_full=sample_data_full,
            sample_table=sample_table,
            coast_polygon=coast_polygon,
            out_dir=args.out_dir,
            max_epochs=args.max_epochs,
            gpu_number=args.gpu_number,
            penalty_weight=args.penalty_weight,
        )
        extra = ""
        if res["status"] == "OK":
            extra = (
                f" along_coast={res.get('mean_along_coast_err_km', float('nan')):.1f} km "
                f"offshore={res.get('mean_offshore_km', float('nan')):.1f} km "
                f"({res.get('elapsed_s', '?')}s)"
            )
        else:
            extra = f" reason={res.get('reason', '?')}"
        print(f"run_loso_rangemask: fold {res['site']} -> {res['status']}{extra}",
              flush=True)

    sweep_elapsed = time.time() - t_sweep
    print(f"\nrun_loso_rangemask: sweep done in {sweep_elapsed:.1f}s "
          f"({sweep_elapsed / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
