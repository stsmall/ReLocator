#!/usr/bin/env python3
"""Post-hoc snap-to-nearest-site experiment for the sculpin LOSO predictions.

Tests the hypothesis that a discrete-site classifier head would beat the
current continuous-regression head at LOSO, by re-scoring the existing
unmasked LOSO predictions under two snap rules:

  (A) snap each prediction to the nearest of all 16 training-site centroids.
      Cheats slightly — the held-out site is in the snap target list, so a
      "perfect" prediction (closer to truth than any other site) snaps to
      0 km error. Use this as the optimistic / best-case classifier proxy.

  (B) snap each prediction to the nearest of the 15 *other* sites
      (excluding the held-out site). LOSO-faithful — a real classifier
      trained under the same hold-out scheme could only choose among 15
      classes per fold. Use this as the strict, comparable-to-the-network
      classifier proxy.

Outputs:
  --out_csv            per-site comparison table (unsnapped, A, B)
  --out_png            grouped bar chart vs centroid baseline
  --out_summary_md     written report with the deltas

The snap is computed in haversine km on the actual (lat, lon) sphere, so
the result is directly comparable to the existing summary numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from validation import common  # noqa: E402


def load_per_individual_predictions(
    loso_dir: Path, sample_data: Path, mode: str = "dosage"
) -> pd.DataFrame:
    """Per-individual: held-out site, true (lon, lat), predicted (lon, lat).

    Predictions come from each fold's run_predlocs.txt for individuals
    whose coords were blanked (i.e., the held-out site's individuals).
    """
    sd = pd.read_csv(sample_data, sep="\t")
    sd["site"] = sd["sampleID"].str.rsplit("_", n=1).str[0]
    truth = sd.set_index("sampleID")[["x", "y", "site"]].rename(
        columns={"x": "true_lon", "y": "true_lat"}
    )

    rows = []
    for fold_dir in sorted((loso_dir / mode).iterdir()):
        if not fold_dir.is_dir():
            continue
        site_held = fold_dir.name
        pred_path = fold_dir / "run_predlocs.txt"
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


def site_centroids(sites_tsv: Path) -> pd.DataFrame:
    return pd.read_csv(sites_tsv, sep="\t").set_index("site")[["lat", "lon"]]


def snap_to_nearest(
    pred_lon: float, pred_lat: float,
    candidates: pd.DataFrame,
) -> tuple[str, float, float]:
    """Return (snapped_site, snapped_lon, snapped_lat) — nearest candidate by haversine."""
    lats = candidates["lat"].to_numpy(dtype=np.float64)
    lons = candidates["lon"].to_numpy(dtype=np.float64)
    d = common.haversine(
        np.full(len(lats), pred_lat), np.full(len(lons), pred_lon),
        lats, lons,
    )
    if np.ndim(d) == 0:
        d = np.array([d])
    idx = int(np.argmin(d))
    site = candidates.index[idx]
    return site, float(lons[idx]), float(lats[idx])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--loso_dir", required=True, type=Path)
    p.add_argument("--sample_data", required=True, type=Path)
    p.add_argument("--sites_tsv", required=True, type=Path)
    p.add_argument("--mode", default="dosage")
    p.add_argument("--out_csv", required=True, type=Path)
    p.add_argument("--out_png", required=True, type=Path)
    p.add_argument("--out_summary_md", required=True, type=Path)
    args = p.parse_args()

    pred_df = load_per_individual_predictions(
        args.loso_dir, args.sample_data, args.mode
    )
    sites = site_centroids(args.sites_tsv)
    print(f"Loaded {len(pred_df)} held-out predictions across "
          f"{pred_df['site'].nunique()} sites", flush=True)

    rows = []
    for _, r in pred_df.iterrows():
        # Unsnapped
        d_raw = float(common.haversine(
            r["true_lat"], r["true_lon"], r["pred_lat"], r["pred_lon"]
        ))
        # (A) snap to nearest of all 16 sites
        snap_a_site, snap_a_lon, snap_a_lat = snap_to_nearest(
            r["pred_lon"], r["pred_lat"], sites
        )
        d_a = float(common.haversine(
            r["true_lat"], r["true_lon"], snap_a_lat, snap_a_lon
        ))
        # (B) snap to nearest of 15 others (exclude held-out site)
        others = sites.drop(index=r["site"])
        snap_b_site, snap_b_lon, snap_b_lat = snap_to_nearest(
            r["pred_lon"], r["pred_lat"], others
        )
        d_b = float(common.haversine(
            r["true_lat"], r["true_lon"], snap_b_lat, snap_b_lon
        ))
        rows.append({
            "sampleID": r["sampleID"],
            "site": r["site"],
            "raw_err_km": d_raw,
            "snap16_site": snap_a_site,
            "snap16_correct": int(snap_a_site == r["site"]),
            "snap16_err_km": d_a,
            "snap15_site": snap_b_site,
            "snap15_err_km": d_b,
        })
    out = pd.DataFrame(rows)

    # Per-site aggregation
    per_site = (
        out.groupby("site")
        .agg(n=("sampleID", "count"),
             raw_med=("raw_err_km", "median"),
             snap16_med=("snap16_err_km", "median"),
             snap15_med=("snap15_err_km", "median"),
             snap16_acc=("snap16_correct", "mean"))
        .round(2)
    )

    # Centroid baseline (per-site, computed from the spatial sample_data
    # against the sample's own held-out site as truth — same as
    # summarize.py's centroid baseline).
    sd = pd.read_csv(args.sample_data, sep="\t").dropna(subset=["x", "y"])
    sd["site"] = sd["sampleID"].str.rsplit("_", n=1).str[0]
    sites_in_data = sorted(sd["site"].unique())
    centroid_per_site = {}
    for hold in sites_in_data:
        train = sd[sd["site"] != hold]
        test = sd[sd["site"] == hold]
        if train.empty or test.empty:
            continue
        cx = train["x"].mean()
        cy = train["y"].mean()
        d = common.haversine(
            test["y"].to_numpy(dtype=np.float64),
            test["x"].to_numpy(dtype=np.float64),
            np.full(len(test), cy),
            np.full(len(test), cx),
        )
        if np.ndim(d) == 0:
            d = np.array([d])
        centroid_per_site[hold] = float(np.median(d))
    per_site["centroid_med"] = pd.Series(centroid_per_site)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    per_site.to_csv(args.out_csv, sep="\t")
    print(f"Wrote {args.out_csv}", flush=True)

    # Aggregate (mean of medians)
    agg_centroid = per_site["centroid_med"].mean()
    agg_raw = per_site["raw_med"].mean()
    agg_snap16 = per_site["snap16_med"].mean()
    agg_snap15 = per_site["snap15_med"].mean()
    snap16_acc = per_site["snap16_acc"].mean()
    print()
    print("=== aggregate (mean of per-site medians) ===")
    print(f"  centroid baseline:        {agg_centroid:.1f} km")
    print(f"  unsnapped (raw dosage):   {agg_raw:.1f} km")
    print(f"  snap-to-16 (incl. truth): {agg_snap16:.1f} km   "
          f"(snap-correctness rate: {snap16_acc:.2f})")
    print(f"  snap-to-15 (LOSO-faithful): {agg_snap15:.1f} km")
    print()
    print("=== per-site comparison ===")
    print(per_site.to_string())

    # Plot: bar chart per site
    fig, ax = plt.subplots(figsize=(13, 6))
    sites_ordered = per_site.sort_values("raw_med").index
    x = np.arange(len(sites_ordered))
    w = 0.22
    ax.bar(x - 1.5 * w, per_site.loc[sites_ordered, "centroid_med"], w,
           label="centroid baseline", color="firebrick", alpha=0.8)
    ax.bar(x - 0.5 * w, per_site.loc[sites_ordered, "raw_med"], w,
           label="raw regression (current)", color="steelblue")
    ax.bar(x + 0.5 * w, per_site.loc[sites_ordered, "snap15_med"], w,
           label="snap to nearest of 15 others (LOSO classifier proxy)",
           color="seagreen")
    ax.bar(x + 1.5 * w, per_site.loc[sites_ordered, "snap16_med"], w,
           label="snap to nearest of 16 (best-case)",
           color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(sites_ordered, rotation=60, ha="right")
    ax.set_ylabel("Median LOSO error (km)")
    ax.set_title(
        f"Sculpin: snap-to-nearest-site experiment\n"
        f"raw {agg_raw:.0f} km   |   snap-15 {agg_snap15:.0f} km   |   "
        f"snap-16 {agg_snap16:.0f} km   |   centroid {agg_centroid:.0f} km"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_png}", flush=True)

    # Markdown summary
    args.out_summary_md.parent.mkdir(parents=True, exist_ok=True)
    md = f"""# Sculpin: post-hoc snap-to-nearest-site experiment

Re-scoring the existing unmasked dosage LOSO predictions under two snap
rules to test whether a discrete-site classifier head would meaningfully
outperform the current continuous-regression head.

## Aggregate results (mean of per-site medians, km)

| Variant | Mean(median) km | Δ vs raw | Δ vs centroid |
|---|---|---|---|
| centroid baseline                                | {agg_centroid:.1f} | — | — |
| **raw regression (current)**                     | **{agg_raw:.1f}** | — | {agg_raw - agg_centroid:+.1f} |
| snap to nearest of 15 others (LOSO classifier)   | {agg_snap15:.1f} | {agg_snap15 - agg_raw:+.1f} | {agg_snap15 - agg_centroid:+.1f} |
| snap to nearest of all 16 (incl. truth)          | {agg_snap16:.1f} | {agg_snap16 - agg_raw:+.1f} | {agg_snap16 - agg_centroid:+.1f} |

Snap-correctness rate (fraction of individuals where the regression
prediction's nearest site is in fact the held-out true site):
**{snap16_acc:.2f}** (16-class snap, n={len(out)} individuals).

## Per-site comparison (km)

| site | n | centroid | raw regression | snap-15 | snap-16 | snap16 acc |
|---|---|---|---|---|---|---|
"""
    for site in sites_ordered:
        r = per_site.loc[site]
        md += (
            f"| {site} | {int(r['n'])} | {r['centroid_med']:.0f} | "
            f"{r['raw_med']:.0f} | {r['snap15_med']:.0f} | "
            f"{r['snap16_med']:.0f} | {r['snap16_acc']:.2f} |\n"
        )

    md += f"""

## Interpretation

- **snap-15 vs raw regression** is the architecturally honest comparison.
  In LOSO, a 16-class classifier trained on the same data would only have
  15 classes to choose from per fold (the held-out site is by definition
  unknown to the network at training time). If `snap-15 < raw regression`,
  a classifier head would be expected to deliver real gains on this data.
- **snap-16 vs raw regression** is the best-case proxy: it lets the snap
  procedure pick the truth itself when the regression got close enough.
  This is structurally unfair as an LOSO test but useful as an upper
  bound on how much improvement is theoretically possible from snapping.
- The **snap-16 correctness rate** ({snap16_acc:.2f}) is the fraction of
  individuals where the regression's free prediction was already closer
  to the true held-out site than to any other site. High values mean
  "regression is putting predictions in the right neighborhood, just
  imprecisely"; low values mean "regression is in the wrong neighborhood
  entirely."
"""
    args.out_summary_md.write_text(md)
    print(f"Wrote {args.out_summary_md}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
