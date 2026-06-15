#!/usr/bin/env python3
"""Cartopy map of KNP elephant LOSO predictions.

The KNP dataset is per-individual (no sampling sites), so the map is also
per-individual:

  ★ at each true GPS coord (one star per held-out elephant)
  ● at each predicted coord (one circle per held-out elephant prediction)
  faint line connecting predicted → true

Aggregates over k-fold runs: each individual ends up held out 5 × n_seeds
times (k=5, default 3 seeds → 15 predictions per individual). We take the
median predicted (lon, lat) per individual.

Default extent is Kibale-area (~30 km × 60 km centered on the data); pass
--extent to override.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

USER_MODES = ("dosage", "geometry", "repeat_norm")


def aggregate_predictions(out_dir: Path, mode: str, sample_data: Path) -> pd.DataFrame:
    """Median predicted (lon, lat) per individual across all kfold folds + smoke.

    Returns a DataFrame with columns: sampleID, true_lon, true_lat, pred_lon, pred_lat.
    """
    sd = pd.read_csv(sample_data, sep="\t").set_index("sampleID")
    sd = sd.rename(columns={"x": "true_lon", "y": "true_lat"})

    pred_rows = []
    # K-fold predlocs
    for fold_dir in sorted((out_dir / "kfold" / mode).glob("seed*_fold*")):
        pred_path = fold_dir / "run_predlocs.txt"
        if not pred_path.exists():
            continue
        sd_path = fold_dir / "sample_data.txt"
        if not sd_path.exists():
            continue
        pred = pd.read_csv(pred_path).rename(columns={"x": "pred_lon", "y": "pred_lat"})
        # Held-out individuals are those whose x is "NA" in the fold's sample_data.
        fold_sd = pd.read_csv(sd_path, sep="\t", dtype=str, keep_default_na=False)
        held = set(fold_sd.loc[fold_sd["x"] == "NA", "sampleID"])
        pred = pred[pred["sampleID"].isin(held)]
        pred_rows.append(pred)

    # Smoke fold
    smoke_pred = out_dir / "smoke" / mode / "run_predlocs.txt"
    smoke_sd = out_dir / "smoke" / mode / "sample_data.txt"
    if smoke_pred.exists() and smoke_sd.exists():
        pred = pd.read_csv(smoke_pred).rename(columns={"x": "pred_lon", "y": "pred_lat"})
        fold_sd = pd.read_csv(smoke_sd, sep="\t", dtype=str)
        held = set(fold_sd.loc[fold_sd["x"] == "NA", "sampleID"])
        pred = pred[pred["sampleID"].isin(held)]
        pred_rows.append(pred)

    if not pred_rows:
        return pd.DataFrame()

    all_preds = pd.concat(pred_rows, ignore_index=True)
    median = all_preds.groupby("sampleID")[["pred_lon", "pred_lat"]].median()
    out = sd.join(median, how="inner")
    out = out.reset_index()
    return out[["sampleID", "true_lon", "true_lat", "pred_lon", "pred_lat"]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True, type=Path,
                   help="K-fold runner output directory.")
    p.add_argument("--sample_data", required=True,
                   help="sample_data.txt with x=lon, y=lat, sampleID.")
    p.add_argument("--out_png", required=True, type=Path)
    p.add_argument("--mode", default="dosage", choices=USER_MODES)
    p.add_argument("--extent", default=None,
                   help="Map extent: lon_min,lon_max,lat_min,lat_max. "
                        "Default: data bbox + 0.05° padding.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = aggregate_predictions(Path(args.out_dir), args.mode, Path(args.sample_data))
    if df.empty:
        print("No predictions found.", file=sys.stderr)
        return 1

    # Per-individual error in degrees and km (rough — Euclidean lon/lat ≈ km at 0°N)
    err_deg = np.sqrt(
        (df["pred_lon"] - df["true_lon"]) ** 2
        + (df["pred_lat"] - df["true_lat"]) ** 2
    )
    median_deg = float(np.median(err_deg))
    median_km = median_deg * 111  # 1° ≈ 111 km at the equator

    if args.extent is None:
        pad = 0.04
        lon_min = float(min(df["true_lon"].min(), df["pred_lon"].min())) - pad
        lon_max = float(max(df["true_lon"].max(), df["pred_lon"].max())) + pad
        lat_min = float(min(df["true_lat"].min(), df["pred_lat"].min())) - pad
        lat_max = float(max(df["true_lat"].max(), df["pred_lat"].max())) + pad
        extent = [lon_min, lon_max, lat_min, lat_max]
    else:
        extent = [float(v) for v in args.extent.split(",")]

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    land = cfeature.NaturalEarthFeature(
        category="physical", name="land", scale="10m",
        edgecolor="#666", facecolor="#f0ebe0",
    )
    rivers = cfeature.NaturalEarthFeature(
        category="physical", name="rivers_lake_centerlines", scale="10m",
        edgecolor="#88a", facecolor="none", lw=0.5,
    )
    ax.add_feature(land, zorder=0)
    ax.add_feature(rivers, zorder=1)

    # Lines: predicted → true
    for _, r in df.iterrows():
        ax.plot([r["pred_lon"], r["true_lon"]],
                [r["pred_lat"], r["true_lat"]],
                color="#333", lw=0.4, alpha=0.4,
                transform=ccrs.PlateCarree(), zorder=2)

    # Truth: stars (filled, slightly larger)
    ax.scatter(df["true_lon"], df["true_lat"],
               marker="*", s=80, color="#1f77b4",
               edgecolor="black", lw=0.5,
               transform=ccrs.PlateCarree(), zorder=4, label="true GPS")
    # Prediction: filled circles
    ax.scatter(df["pred_lon"], df["pred_lat"],
               s=22, color="#d62728",
               edgecolor="black", lw=0.3, alpha=0.85,
               transform=ccrs.PlateCarree(), zorder=3,
               label=f"predicted ({args.mode})")

    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.set_title(
        f"KNP elephants — ReLocator k-fold prediction vs truth\n"
        f"({args.mode} mode, n={len(df)}, per-individual median "
        f"{median_deg:.3f}° ≈ {median_km:.1f} km)"
    )
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_png}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
