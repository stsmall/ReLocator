#!/usr/bin/env python3
"""Cartopy map of sculpin LOSO predictions with the topology graph overlaid.

Mirrors validation/sculpin/make_map.py but adds:
  - LineString edges from the topology GeoJSON drawn as light blue lines
  - Heading: shows graph_kernel_resolution and total edge count

Reads predictions from out/sculpin_validation/loso_graph/<SITE>/locator_predlocs.txt
(the graph-smoothed predictions; locator_predlocs_paired_baseline.txt also
exists in each fold dir but the map is for the graph mode).
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
        pred_path = fold_dir / "locator_predlocs.txt"
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

    df = df.dropna(subset=["true_lon", "true_lat"])
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
