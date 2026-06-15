#!/usr/bin/env python3
"""LOSO sweep with ReLocator's range-mask training penalty (sculpin, dosage mode).

Mirrors validation/run_loso_rangemask.py from the GL branch but adapted for
freshwater (2D polygon mask) instead of coastline (1D + buffer):

  - Polygon source: HydroSHEDS HydroLAKES + HydroRIVERS, prepared by
    ``validation.sculpin.build_range`` and stored as
    ``validation/sculpin/freshwater_range.shp`` (committed, ~280 KB).
  - Per fold: blank held-out site's coords, compute fold normalization
    params (μ, σ for x/y over the training samples), z-score-transform the
    polygon, save a fold-specific shapefile so the loss reads the mask in
    the same coordinate space as the model's normalized predictions.
  - Use ReLocator's Python API (``Locator(config=...)``) with
    ``use_range_penalty=True`` and a fold-specific
    ``species_range_shapefile`` — the CLI doesn't expose these flags.

Loads microsat genotypes through ReLocator's native ``--microsat`` path
(PR #46): one call to ``loc.load_genotypes(microsat=...)`` per fold, no
intermediate feature-matrix TSV. Dosage encoding is the only supported
mode here (geometry / repeat_norm modes were ruled out for merge by the
sculpin LOSO comparison and live on the ``microsatellites`` branch as
branch-only experimental record).

See ``validation/notes/range_mask_bug.md`` for why the z-score-space
transformation is necessary (upstream coord-space bug in
``loss_with_range_penalty``).
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

from validation import common  # noqa: E402

NORMALIZED_RESOLUTION = 0.01
DEFAULT_PENALTY_WEIGHT = 50.0


def site_of(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def fold_normalization_params(
    held_sd: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """Reproduce ReLocator's nanmean/nanstd of x (lon) and y (lat) over training samples."""
    locs = held_sd[["x", "y"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    return (
        float(np.nanmean(locs[:, 0])),
        float(np.nanstd(locs[:, 0])),
        float(np.nanmean(locs[:, 1])),
        float(np.nanstd(locs[:, 1])),
    )


def normalize_polygon(
    polygon: shapely.geometry.base.BaseGeometry,
    mean_lon: float,
    sd_lon: float,
    mean_lat: float,
    sd_lat: float,
) -> shapely.geometry.base.BaseGeometry:
    """Apply z-score transform x' = (x - mean) / sd to polygon coords.

    Matches `locator.data.filters.normalize_locs`: lon is index 0, lat is 1.
    Shapely affine matrix is [a, b, d, e, xoff, yoff] for
    x' = a*x + b*y + xoff,  y' = d*x + e*y + yoff.
    """
    return shapely.affinity.affine_transform(
        polygon,
        [1.0 / sd_lon, 0.0, 0.0, 1.0 / sd_lat, -mean_lon / sd_lon, -mean_lat / sd_lat],
    )


def write_polygon_shapefile(
    polygon: shapely.geometry.base.BaseGeometry,
    out_path: Path,
    name: str,
    crs: str = "EPSG:4326",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame({"name": [name]}, geometry=[polygon], crs=crs)
    gdf.to_file(out_path, driver="ESRI Shapefile")
    return out_path


def write_holdout_sample_data(
    sample_data_in: Path, holdout_site: str, out_path: Path
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Write a copy of sample_data with holdout-site coords blanked.

    Returns ``(truth, held_out_ids, held_sd)``: the unchanged truth DataFrame,
    the list of held-out sampleIDs, and the modified DataFrame written to disk.
    """
    df = pd.read_csv(sample_data_in, sep="\t")
    sites = df["sampleID"].apply(site_of)
    holdout_mask = sites == holdout_site
    held_out_ids = df.loc[holdout_mask, "sampleID"].tolist()

    out = df.copy()
    out["x"] = out["x"].astype(object)
    out["y"] = out["y"].astype(object)
    out.loc[holdout_mask, "x"] = "NA"
    out.loc[holdout_mask, "y"] = "NA"
    out.to_csv(out_path, sep="\t", index=False)
    return df, held_out_ids, out


def _run_one_fold_inproc(
    *,
    site: str,
    microsat_path: Path,
    sample_data_in: Path,
    range_polygon: shapely.geometry.base.BaseGeometry,
    out_dir: Path,
    gpu_number: int,
    seed: int,
    max_epochs: int,
    penalty_weight: float,
) -> dict:
    fold_dir = out_dir / site
    fold_dir.mkdir(parents=True, exist_ok=True)

    sd_path = fold_dir / "sample_data.txt"
    truth, held_out_ids, held_sd = write_holdout_sample_data(
        sample_data_in,
        site,
        sd_path,
    )

    mean_lon, sd_lon, mean_lat, sd_lat = fold_normalization_params(held_sd)
    range_norm = normalize_polygon(range_polygon, mean_lon, sd_lon, mean_lat, sd_lat)
    fold_shp = fold_dir / "range_normalized.shp"
    write_polygon_shapefile(range_norm, fold_shp, name="freshwater_z")

    out_prefix = fold_dir / "locator"
    fold_result: dict = {
        "site": site,
        "mode": "dosage",
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
            "microsat": str(microsat_path),
            "sample_data": str(sd_path),
            "max_epochs": max_epochs,
            "patience": 30,
            "seed": seed,
            "gpu_number": gpu_number,
            "use_range_penalty": True,
            "species_range_shapefile": str(fold_shp),
            "resolution": NORMALIZED_RESOLUTION,
            "penalty_weight": penalty_weight,
        }
        loc = Locator(config=config)
        genotypes, samples = loc.load_genotypes(microsat=str(microsat_path))
        loc.train(genotypes=genotypes, samples=samples)
        loc.predict(genotypes=genotypes, samples=samples)
    except Exception as exc:
        fold_result["status"] = "FAILED"
        fold_result["error"] = f"{type(exc).__name__}: {exc}"
        fold_result["n_held_out"] = int(len(held_out_ids))
        (fold_dir / "fold_result.json").write_text(
            json.dumps(fold_result, indent=2, default=str)
        )
        return fold_result

    elapsed = time.time() - t0
    fold_result["elapsed_s"] = round(elapsed, 1)

    predlocs = Path(f"{out_prefix}_predlocs.txt")
    if not predlocs.exists():
        fold_result["status"] = "FAILED"
        fold_result["error"] = f"predlocs missing at {predlocs}"
        fold_result["n_held_out"] = int(len(held_out_ids))
        (fold_dir / "fold_result.json").write_text(
            json.dumps(fold_result, indent=2, default=str)
        )
        return fold_result

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
        fold_result["warning"] = "no evaluable samples"
        (fold_dir / "fold_result.json").write_text(json.dumps(fold_result, indent=2))
        return fold_result

    p = pred.loc[evaluable_in_pred, ["x", "y"]].to_numpy(dtype=np.float64)
    t = tru.loc[evaluable_in_pred].to_numpy(dtype=np.float64)
    err_km = common.haversine(t[:, 1], t[:, 0], p[:, 1], p[:, 0])
    if np.ndim(err_km) == 0:
        err_km = np.array([err_km])

    fold_result["status"] = "OK"
    fold_result["n_held_out"] = int(len(held_out_ids))
    fold_result["n_evaluable"] = int(len(evaluable))
    fold_result["n_scored"] = int(len(evaluable_in_pred))
    fold_result["median_error_km"] = float(np.median(err_km))
    fold_result["mean_error_km"] = float(np.mean(err_km))
    fold_result["max_error_km"] = float(np.max(err_km))

    (fold_dir / "fold_result.json").write_text(
        json.dumps(fold_result, indent=2, default=str)
    )
    return fold_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--microsat",
        required=True,
        type=Path,
        help="Pair-format microsat TSV (output of parse_genepop.py). "
        "Loaded per fold via loc.load_genotypes(microsat=...).",
    )
    p.add_argument(
        "--sample_data",
        required=True,
        type=Path,
        help="ReLocator sample_data.txt (truth coords).",
    )
    p.add_argument(
        "--range_shp",
        required=True,
        type=Path,
        help="Freshwater range polygon (output of build_range.py).",
    )
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--penalty_weight", type=float, default=DEFAULT_PENALTY_WEIGHT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading range polygon from {args.range_shp}", flush=True)
    range_gdf = gpd.read_file(args.range_shp)
    range_polygon = range_gdf.geometry.iloc[0]
    print(f"  polygon area ≈ {range_polygon.area:.2f} sq.deg", flush=True)

    sd_df = pd.read_csv(args.sample_data, sep="\t")
    sites = sorted(set(sd_df["sampleID"].apply(site_of)))
    print(f"Sites: {len(sites)} → {sites}", flush=True)

    rangemask_dir = args.out_dir / "loso_rangemask" / "dosage"

    print(
        f"Starting {len(sites)} folds with use_range_penalty=True "
        f"(weight={args.penalty_weight}, mode=dosage, gpu={args.gpu})",
        flush=True,
    )
    t_sweep = time.time()
    for idx, site in enumerate(sites):
        print(f"\n=== fold {idx + 1}/{len(sites)}: {site} ===", flush=True)
        _run_one_fold_inproc(
            site=site,
            microsat_path=args.microsat,
            sample_data_in=args.sample_data,
            range_polygon=range_polygon,
            out_dir=rangemask_dir,
            gpu_number=args.gpu,
            seed=args.seed,
            max_epochs=args.max_epochs,
            penalty_weight=args.penalty_weight,
        )

    elapsed = time.time() - t_sweep
    print(f"\nSweep complete in {elapsed / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
