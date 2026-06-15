#!/usr/bin/env python3
"""Cartopy map of sculpin LOSO predictions.

For each held-out site:
  - star at the true site coords
  - filled circle at each individual's predicted coords
  - line from each prediction back to the true site

Default: shows the `dosage` mode (the empirical winner). Pass --mode to
switch. Pass --modes dosage,geometry,repeat_norm for a 3-panel comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib
import shapely.geometry

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from validation import common  # noqa: E402

USER_MODES = ("dosage", "geometry", "repeat_norm")


def load_predictions(out_dir: Path, mode: str, sample_data: Path) -> pd.DataFrame:
    """Return a DataFrame with one row per held-out individual.

    Columns:
      sampleID, site, true_lon, true_lat, pred_lon, pred_lat
    """
    sd = pd.read_csv(sample_data, sep="\t")
    sd["site"] = sd["sampleID"].str.rsplit("_", n=1).str[0]
    truth = sd.set_index("sampleID")[["x", "y", "site"]]
    truth = truth.rename(columns={"x": "true_lon", "y": "true_lat"})

    rows = []
    for fold_dir in sorted((out_dir / "loso" / mode).iterdir()):
        if not fold_dir.is_dir():
            continue
        site_held = fold_dir.name
        # Naming convention varies: unmasked runner writes run_predlocs.txt,
        # range-mask runner writes locator_predlocs.txt. Accept either.
        pred_path = fold_dir / "run_predlocs.txt"
        if not pred_path.exists():
            pred_path = fold_dir / "locator_predlocs.txt"
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path)
        pred = pred.rename(columns={"x": "pred_lon", "y": "pred_lat"})
        pred = pred.set_index("sampleID")
        # Only individuals from the held-out site (their coords were blanked)
        held_ids = truth[truth["site"] == site_held].index
        present = [sid for sid in held_ids if sid in pred.index]
        for sid in present:
            row = {"sampleID": sid, "site": site_held}
            row.update(truth.loc[sid].to_dict())
            row["pred_lon"] = pred.loc[sid, "pred_lon"]
            row["pred_lat"] = pred.loc[sid, "pred_lat"]
            rows.append(row)
    return pd.DataFrame(rows)


def site_centroids(sample_data: Path) -> pd.DataFrame:
    sd = pd.read_csv(sample_data, sep="\t")
    sd["site"] = sd["sampleID"].str.rsplit("_", n=1).str[0]
    return sd.groupby("site").agg(true_lon=("x", "mean"), true_lat=("y", "mean")).reset_index()


def draw_panel(ax, df: pd.DataFrame, sites: pd.DataFrame, mode: str,
               title: str | None = None,
               range_polygon: shapely.geometry.base.BaseGeometry | None = None) -> None:
    land = cfeature.NaturalEarthFeature(
        category="physical", name="land", scale="50m",
        edgecolor="#666", facecolor="#f0ebe0",
    )
    ocean = cfeature.NaturalEarthFeature(
        category="physical", name="ocean", scale="50m",
        edgecolor="none", facecolor="#dde8f0",
    )
    lakes = cfeature.NaturalEarthFeature(
        category="physical", name="lakes", scale="50m",
        edgecolor="#88a", facecolor="#dde8f0", lw=0.3,
    )
    rivers = cfeature.NaturalEarthFeature(
        category="physical", name="rivers_lake_centerlines", scale="50m",
        edgecolor="#88a", facecolor="none", lw=0.3,
    )
    ax.add_feature(ocean, zorder=0)
    ax.add_feature(land, zorder=1)
    # Freshwater range polygon overlay (where the model is allowed to
    # predict under range mask, OR a reference of "valid sculpin habitat"
    # for the unmasked runs). Drawn between land and the coarse NE lakes
    # so it's visible but doesn't obscure coastlines.
    if range_polygon is not None:
        ax.add_geometries(
            [range_polygon], crs=ccrs.PlateCarree(),
            facecolor="#5e8fb8", alpha=0.32, edgecolor="#3e6f98",
            linewidth=0.4, zorder=1.3,
        )
    ax.add_feature(lakes, zorder=1.5)
    ax.add_feature(rivers, zorder=1.5)
    ax.coastlines(resolution="50m", lw=0.5, color="#444")

    # Lines: prediction → true site, color by site
    sites_in_data = df["site"].unique()
    palette = plt.cm.tab20(np.linspace(0, 1, len(sites_in_data)))
    site_color = dict(zip(sites_in_data, palette, strict=False))

    for _, r in df.iterrows():
        c = site_color[r["site"]]
        ax.plot(
            [r["pred_lon"], r["true_lon"]],
            [r["pred_lat"], r["true_lat"]],
            color=c, lw=0.5, alpha=0.35, transform=ccrs.PlateCarree(),
            zorder=2,
        )
        ax.scatter(
            r["pred_lon"], r["pred_lat"],
            s=24, color=c, edgecolor="black", lw=0.3, alpha=0.85,
            transform=ccrs.PlateCarree(), zorder=3,
        )

    # Stars: true site centroids, with site name labels
    for _, r in sites.iterrows():
        c = site_color.get(r["site"], "k")
        ax.scatter(
            r["true_lon"], r["true_lat"],
            marker="*", s=260, color=c, edgecolor="black", lw=0.9,
            transform=ccrs.PlateCarree(), zorder=5,
        )
        ax.annotate(
            r["site"],
            xy=(r["true_lon"], r["true_lat"]), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xytext=(6, 6), textcoords="offset points",
            fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            zorder=6,
        )

    if title is not None:
        ax.set_title(title, fontsize=11)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True, type=Path,
                   help="LOSO run output dir (parent of loso/<mode>/<site>/).")
    p.add_argument("--sample_data", required=True,
                   help="sample_data.txt with x,y,sampleID columns (true coords).")
    p.add_argument("--out_png", required=True, type=Path,
                   help="Output PNG path.")
    p.add_argument("--modes", default="dosage",
                   help="Comma-separated subset of dosage,geometry,repeat_norm. "
                        "If multiple, panels are drawn side-by-side.")
    p.add_argument("--extent", default="-138,-115,46,60",
                   help="Map extent: lon_min,lon_max,lat_min,lat_max (degrees).")
    p.add_argument("--range_shp", default=None,
                   help="Optional shapefile overlay (e.g. freshwater_range.shp) "
                        "drawn as a translucent blue fill so the user can see "
                        "where valid habitat actually is at high resolution.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in USER_MODES]
    if bad:
        print(f"Unknown modes: {bad}. Valid: {USER_MODES}", file=sys.stderr)
        return 2

    extent = [float(v) for v in args.extent.split(",")]
    sites = site_centroids(Path(args.sample_data))
    range_polygon = None
    if args.range_shp is not None:
        range_gdf = gpd.read_file(args.range_shp)
        range_polygon = range_gdf.geometry.iloc[0]

    proj = ccrs.LambertConformal(central_longitude=-127, central_latitude=53)
    if len(modes) == 1:
        fig = plt.figure(figsize=(11, 13))
        axes = [fig.add_subplot(1, 1, 1, projection=proj)]
    else:
        fig = plt.figure(figsize=(9 * len(modes), 13))
        axes = [fig.add_subplot(1, len(modes), i + 1, projection=proj)
                for i in range(len(modes))]

    for ax, mode in zip(axes, modes, strict=False):
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        df = load_predictions(Path(args.out_dir), mode, Path(args.sample_data))
        # Median LOSO error in km (haversine, matches summary metric)
        if len(df) > 0:
            err_km = common.haversine(
                df["true_lat"].to_numpy(dtype=np.float64),
                df["true_lon"].to_numpy(dtype=np.float64),
                df["pred_lat"].to_numpy(dtype=np.float64),
                df["pred_lon"].to_numpy(dtype=np.float64),
            )
            err_label = f" — per-individual median {np.median(err_km):.0f} km"
        else:
            err_label = ""
        draw_panel(ax, df, sites, mode, title=f"{mode}{err_label}",
                   range_polygon=range_polygon)

    fig.suptitle("Cottus asper microsat LOSO: predictions ★→● by site",
                 fontsize=13, y=0.98)
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_png}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
