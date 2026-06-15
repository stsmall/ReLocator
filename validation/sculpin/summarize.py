#!/usr/bin/env python3
"""Aggregate Cottus asper LOSO fold JSONs → kfold tsv, summary md, figure.

Reports per-mode mean ± std median error in km, plus a centroid baseline:
"predict the training-set sites' centroid for every held-out individual."

Per-fold JSON schema (from run_loso.py):
  success: {mode, site, n_held_out, n_evaluable, n_scored,
            median_error_km, mean_error_km, max_error_km, [warning]}
  failure: {mode, site, n_held_out, error}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from validation import common  # noqa: E402

USER_MODES = ("dosage", "geometry", "repeat_norm")


def load_fold_jsons(out_dir: Path, mode: str) -> pd.DataFrame:
    rows = []
    for p in sorted((out_dir / "loso" / mode).glob("*/fold_result.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["mode"] = mode
    return df


def compute_centroid_baseline(sample_data_path: Path) -> dict:
    """Per-fold baseline: predict the training-set centroid for every held-out sample.

    "Training set" = all sites except the held-out one. Centroid is the mean
    of (lon, lat) over training individuals' coords (same shape as the runner
    actually trains on).
    """
    sd = pd.read_csv(sample_data_path, sep="\t").dropna(subset=["x", "y"])
    sd["site"] = sd["sampleID"].str.rsplit("_", n=1).str[0]
    sites = sorted(sd["site"].unique())

    rows = []
    for hold_site in sites:
        train = sd[sd["site"] != hold_site]
        test = sd[sd["site"] == hold_site]
        if train.empty or test.empty:
            continue
        cx = train["x"].mean()
        cy = train["y"].mean()
        # cx = lon centroid, cy = lat centroid; haversine takes (lat, lon).
        err = common.haversine(test["y"].to_numpy(dtype=np.float64),
                               test["x"].to_numpy(dtype=np.float64),
                               np.full(len(test), cy),
                               np.full(len(test), cx))
        if np.ndim(err) == 0:
            err = np.array([err])
        rows.append({
            "site": hold_site,
            "n_evaluable": int(len(test)),
            "median_error_km": float(np.median(err)),
            "mean_error_km": float(np.mean(err)),
        })
    df = pd.DataFrame(rows)
    return {
        "per_site": df,
        "loso_median_mean": float(df["median_error_km"].mean()),
        "loso_median_std": float(df["median_error_km"].std(ddof=1)),
        "loso_mean_mean": float(df["mean_error_km"].mean()),
    }


def write_kfold_tsv(out_dir: Path, summary_dir: Path) -> pd.DataFrame:
    frames = [load_fold_jsons(out_dir, mode) for mode in USER_MODES]
    frames = [f for f in frames if not f.empty]
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(summary_dir / "sculpin_loso.tsv", sep="\t", index=False)
    return full


def make_figure(full: pd.DataFrame, fig_path: Path, centroid: dict) -> None:
    if full.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "no fold data available", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        return
    if "median_error_km" not in full.columns:
        full["median_error_km"] = np.nan
    fold = full.dropna(subset=["median_error_km"])
    agg = (
        fold.groupby("mode")["median_error_km"]
        .agg(["mean", "std"])
        .reindex(USER_MODES)
        .dropna(subset=["mean"])
    )
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if not agg.empty:
        x = np.arange(len(agg))
        ax.bar(x, agg["mean"], yerr=agg["std"], capsize=5, color="steelblue",
               label="encoding mode")
        ax.set_xticks(x)
        ax.set_xticklabels(agg.index)
        ax.set_ylabel("Median LOSO error (km, haversine)")
        cmean = centroid["loso_median_mean"]
        cstd = centroid["loso_median_std"]
        ax.axhline(cmean, color="firebrick", linestyle="--", lw=1.5,
                   label=f"centroid baseline ({cmean:.0f} ± {cstd:.0f} km)")
        ax.axhspan(cmean - cstd, cmean + cstd, color="firebrick", alpha=0.10)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_title("Cottus asper microsat LOSO\n"
                     "(16-site leave-one-out, dashed = predict-training-centroid baseline)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def render_md(full: pd.DataFrame, centroid: dict, summary_md: Path) -> None:
    if full.empty or "median_error_km" not in full.columns:
        summary_md.write_text("# Sculpin LOSO summary\n\nNo fold results found.\n")
        return

    fold = full.dropna(subset=["median_error_km"])
    agg = (
        fold.groupby("mode")
        .agg(med_mean=("median_error_km", "mean"),
             med_std=("median_error_km", "std"),
             mn_mean=("mean_error_km", "mean"),
             mn_std=("mean_error_km", "std"))
        .reindex(USER_MODES)
        .dropna(subset=["med_mean"])
    )
    n_failed = full["error"].notna().sum() if "error" in full.columns else 0
    cmean = centroid["loso_median_mean"]
    cstd = centroid["loso_median_std"]

    lines = [
        "# Cottus asper microsat LOSO summary",
        "",
        "Real-data validation of the microsat input pipeline against the prickly",
        "sculpin Genepop dataset (Dennenmoser et al. 2014, Dryad doi:10.5061/dryad.8ht04):",
        "405 individuals × 19 microsat loci across 16 sites in the Pacific Northwest.",
        "Leave-one-site-out: hold out each site's individuals, train on the other 15,",
        "predict held-out coords. Metric: haversine distance (km).",
        "",
    ]
    if n_failed > 0:
        lines.append(f"**Note:** {n_failed} fold(s) failed and are excluded from aggregation.")
        lines.append("")

    lines += [
        "## Centroid baseline",
        "",
        "Predict the training-set centroid (mean lon, lat across the other 15 sites)",
        f"for every held-out individual: **{cmean:.1f} ± {cstd:.1f} km mean(median)** across",
        "the 16 sites. This is the floor any feature-based mode must beat.",
        "",
        "## Results",
        "",
        "| Mode | Median error km (mean ± std) | Mean error km (mean ± std) | Δ vs centroid |",
        "|---|---|---|---|",
        f"| `centroid` (baseline) | {cmean:.1f} ± {cstd:.1f} | "
        f"{centroid['loso_mean_mean']:.1f} | — |",
    ]
    for mode in agg.index:
        r = agg.loc[mode]
        delta = r["med_mean"] - cmean
        verdict = "**WORSE**" if delta > 1 else ("tied" if abs(delta) <= 1 else "**BETTER**")
        lines.append(
            f"| `{mode}` | {r['med_mean']:.1f} ± {r['med_std']:.1f} | "
            f"{r['mn_mean']:.1f} ± {r['mn_std']:.1f} | "
            f"{delta:+.1f} ({verdict}) |"
        )

    lines += [
        "",
        "## Per-site LOSO (median error km, smaller = better)",
        "",
    ]
    if not fold.empty:
        pivot = fold.pivot_table(index="site", columns="mode",
                                  values="median_error_km", aggfunc="first")
        pivot = pivot.reindex(columns=[m for m in USER_MODES if m in pivot.columns])
        # Add baseline column from per_site dataframe
        base = centroid["per_site"].set_index("site")["median_error_km"]
        pivot.insert(0, "centroid", base.reindex(pivot.index))
        lines.append("| site | n_eval | " + " | ".join(pivot.columns) + " |")
        lines.append("|---" * (len(pivot.columns) + 2) + "|")
        n_per_site = fold.groupby("site")["n_evaluable"].first()
        for site, row in pivot.iterrows():
            n = int(n_per_site.get(site, 0)) if pd.notna(n_per_site.get(site, np.nan)) else "—"
            cells = " | ".join(
                f"{v:.1f}" if pd.notna(v) else "—" for v in row.values
            )
            lines.append(f"| {site} | {n} | {cells} |")
        lines.append("")

    lines += [
        "## Figures",
        "",
        "- `validation/figures/sculpin_modes.png` — bar chart of per-mode median LOSO",
        "  error with centroid baseline overlay.",
        "",
    ]
    summary_md.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--sample_data", required=True,
                   help="sample_data.txt used in the LOSO runs (for centroid baseline).")
    p.add_argument("--summary_dir", type=Path, default=Path("validation/summary"))
    p.add_argument("--figures_dir", type=Path, default=Path("validation/figures"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    full = write_kfold_tsv(args.out_dir, args.summary_dir)
    centroid = compute_centroid_baseline(Path(args.sample_data))
    make_figure(full, args.figures_dir / "sculpin_modes.png", centroid)
    render_md(full, centroid, args.summary_dir / "sculpin_summary.md")
    print(f"Wrote summary, loso tsv, and figure under {args.summary_dir} and {args.figures_dir}",
          flush=True)
    print(f"Centroid baseline: {centroid['loso_median_mean']:.1f} "
          f"± {centroid['loso_median_std']:.1f} km mean(median) across 16 sites", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
