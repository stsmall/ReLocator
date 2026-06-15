#!/usr/bin/env python3
"""Aggregate microsat validation per-fold JSONs → kfold.tsv, summary.md, figures.

Handles two JSON schemas: success (with median_error/mean_error/max_error/n_held_out/
n_evaluable/n_scored/[warning]) and failure (with error key). Failure JSONs are
recorded as rows with NaN metrics in the kfold TSV; the summary skips them.
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

ALL_MODES = ("dosage", "geometry", "repeat_norm", "polynomial")


def load_fold_jsons(out_dir: Path, mode: str) -> pd.DataFrame:
    """Load all per-fold JSONs for a mode (kfold + smoke). Adds 'mode' column."""
    rows = []
    for p in sorted((out_dir / "kfold" / mode).glob("seed*_fold*/fold_result.json")):
        rows.append(json.loads(p.read_text()))
    smoke_p = out_dir / "smoke" / mode / "fold_result.json"
    if smoke_p.exists():
        rows.append(json.loads(smoke_p.read_text()))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["mode"] = mode
    return df


def write_kfold_tsv(out_dir: Path, summary_dir: Path) -> pd.DataFrame:
    frames = []
    for mode in ALL_MODES:
        df = load_fold_jsons(out_dir, mode)
        if not df.empty:
            frames.append(df)
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(summary_dir / "microsat_kfold.tsv", sep="\t", index=False)
    return full


def compute_centroid_baseline(spatial_path: Path, kfold_k: int = 5,
                              kfold_seeds: tuple = (1, 2, 3),
                              smoke_seed: int = 42, smoke_frac: float = 0.8) -> dict:
    """Predict-the-training-mean baseline using the same splits as the runner."""
    spatial = pd.read_csv(spatial_path, sep="\t")
    sample_ids = spatial["sampleID"].tolist()
    n = len(sample_ids)

    def kfold(n_, k, seed):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_)
        folds = np.array_split(perm, k)
        return [(np.concatenate([folds[j] for j in range(k) if j != i]).tolist(),
                 folds[i].tolist()) for i in range(k)]

    kf_meds, kf_means = [], []
    for seed in kfold_seeds:
        for train_idx, test_idx in kfold(n, kfold_k, seed):
            tr_ids = [sample_ids[i] for i in train_idx]
            te_ids = [sample_ids[i] for i in test_idx]
            tr = spatial[spatial["sampleID"].isin(tr_ids)].dropna(subset=["x", "y"])
            te = spatial[spatial["sampleID"].isin(te_ids)].dropna(subset=["x", "y"])
            if tr.empty or te.empty:
                continue
            cx, cy = tr["x"].mean(), tr["y"].mean()
            d = np.sqrt((te["x"] - cx) ** 2 + (te["y"] - cy) ** 2)
            kf_meds.append(d.median())
            kf_means.append(d.mean())

    rng = np.random.default_rng(smoke_seed)
    perm = rng.permutation(n)
    cut = int(round(n * smoke_frac))
    tr_ids = [sample_ids[i] for i in perm[:cut]]
    te_ids = [sample_ids[i] for i in perm[cut:]]
    tr = spatial[spatial["sampleID"].isin(tr_ids)].dropna(subset=["x", "y"])
    te = spatial[spatial["sampleID"].isin(te_ids)].dropna(subset=["x", "y"])
    cx, cy = tr["x"].mean(), tr["y"].mean()
    d_smoke = np.sqrt((te["x"] - cx) ** 2 + (te["y"] - cy) ** 2)

    return {
        "kfold_median_mean": float(np.mean(kf_meds)),
        "kfold_median_std": float(np.std(kf_meds, ddof=1)) if len(kf_meds) > 1 else 0.0,
        "kfold_mean_mean": float(np.mean(kf_means)),
        "smoke_median": float(d_smoke.median()),
        "smoke_mean": float(d_smoke.mean()),
    }


def make_modes_figure(full: pd.DataFrame, fig_path: Path,
                      centroid: dict | None = None) -> None:
    """Bar chart: median per-fold error per mode (k-fold mean ± std).

    If centroid baseline is provided, overlays a horizontal reference line
    at the centroid's k-fold median error so readers can see which modes
    actually beat "predict the training-set mean."
    """
    if full.empty or "split" not in full.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "no fold data available", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        return
    kfold = full[full["split"] == "kfold"].copy()
    if "median_error" not in kfold.columns:
        kfold["median_error"] = np.nan
    kfold = kfold.dropna(subset=["median_error"])
    agg = (
        kfold.groupby("mode")["median_error"]
        .agg(["mean", "std"])
        .reindex(ALL_MODES)
        .dropna(subset=["mean"])
    )
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if agg.empty:
        ax.text(0.5, 0.5, "no kfold data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(agg))
        ax.bar(x, agg["mean"], yerr=agg["std"], capsize=5, color="steelblue",
               label="encoding mode")
        ax.set_xticks(x)
        ax.set_xticklabels(agg.index, rotation=15)
        ax.set_ylabel("Median Euclidean error (arena units)")
        if centroid is not None:
            cmean = centroid["kfold_median_mean"]
            cstd = centroid["kfold_median_std"]
            ax.axhline(cmean, color="firebrick", linestyle="--", lw=1.5,
                       label=f"centroid baseline ({cmean:.2f} ± {cstd:.2f})")
            ax.axhspan(cmean - cstd, cmean + cstd, color="firebrick", alpha=0.10)
            ax.legend(loc="lower right", fontsize=9)
        ax.set_title("Microsat: median per-fold error by encoding mode\n"
                     "(k=5, 3 seeds, mean ± std; dashed = predict-training-mean baseline)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def make_scatter_figure(out_dir: Path, spatial_path: Path, fig_path: Path) -> None:
    """Predicted-vs-actual scatter (smoke fold), one panel per mode."""
    spatial = pd.read_csv(spatial_path, sep="\t").set_index("sampleID")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, mode in zip(axes, ALL_MODES, strict=False):
        smoke_dir = out_dir / "smoke" / mode
        pred_path = smoke_dir / "run_predlocs.txt"
        if not pred_path.exists():
            ax.set_title(f"{mode}: no predlocs")
            ax.set_axis_off()
            continue
        pred = pd.read_csv(pred_path).set_index("sampleID")
        # Filter to evaluable (non-NaN truth) and present-in-pred samples
        common = sorted(set(pred.index) & set(spatial.dropna(subset=["x", "y"]).index))
        if not common:
            ax.set_title(f"{mode}: no overlap")
            ax.set_axis_off()
            continue
        ax.scatter(spatial.loc[common, "x"], pred.loc[common, "x"],
                   s=14, alpha=0.7, label="x")
        ax.scatter(spatial.loc[common, "y"], pred.loc[common, "y"],
                   s=14, alpha=0.7, label="y", marker="^")
        lo = float(min(
            spatial.loc[common, ["x", "y"]].min().min(),
            pred.loc[common, ["x", "y"]].min().min(),
        ))
        hi = float(max(
            spatial.loc[common, ["x", "y"]].max().max(),
            pred.loc[common, ["x", "y"]].max().max(),
        ))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.set_title(f"{mode} (smoke 80/20)")
        ax.set_xlabel("true coord")
        ax.set_ylabel("predicted coord")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def render_summary_md(full: pd.DataFrame, summary_md: Path) -> None:
    if full.empty:
        summary_md.write_text("# Microsat validation summary\n\nNo fold results found.\n")
        return

    # Drop rows missing median_error (failure folds) for aggregation
    if "median_error" not in full.columns:
        full["median_error"] = np.nan
    if "mean_error" not in full.columns:
        full["mean_error"] = np.nan

    kfold = full[full["split"] == "kfold"].dropna(subset=["median_error"])
    smoke = full[full["split"] == "smoke80_20"]

    if not kfold.empty:
        agg = (
            kfold.groupby("mode")
            .agg(median_mean=("median_error", "mean"),
                 median_std=("median_error", "std"),
                 mean_mean=("mean_error", "mean"),
                 mean_std=("mean_error", "std"))
            .reindex(ALL_MODES)
            .dropna(subset=["median_mean"])
        )
    else:
        agg = pd.DataFrame()

    n_failed = full["error"].notna().sum() if "error" in full.columns else 0

    lines = [
        "# Microsat validation summary",
        "",
        "Validation arc for the `microsatellites` branch. SLiM-simulated dataset",
        "(200 individuals × 100 loci, arena coordinates). Metric: Euclidean",
        "distance in arena units. K-fold CV: k=5, 3 seeds, mean ± std reported.",
        "",
    ]
    if n_failed > 0:
        lines.append(f"**Note:** {n_failed} fold(s) failed and are excluded from aggregation.")
        lines.append("")

    if not agg.empty:
        lines += [
            "## Results",
            "",
            "| Mode | Median error (mean ± std) | Mean error (mean ± std) |",
            "|---|---|---|",
        ]
        for mode in agg.index:
            r = agg.loc[mode]
            lines.append(
                f"| `{mode}` | {r['median_mean']:.3f} ± {r['median_std']:.3f} | "
                f"{r['mean_mean']:.3f} ± {r['mean_std']:.3f} |"
            )
        lines.append("")
    else:
        lines += ["## Results", "", "No k-fold results to aggregate.", ""]

    lines += [
        "## Smoke (random 80/20, seed 42)",
        "",
        "| Mode | n_held_out | n_scored | median_error | mean_error | max_error |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in smoke.iterrows():
        if "error" in r and pd.notna(r.get("error", None)):
            lines.append(
                f"| `{r['mode']}` | — | — | — | — | (failed: {r['error']}) |"
            )
            continue
        n_scored = r.get("n_scored", r.get("n_evaluable", r.get("n_held_out", "—")))
        med = r.get("median_error", np.nan)
        me = r.get("mean_error", np.nan)
        mx = r.get("max_error", np.nan)
        lines.append(
            f"| `{r['mode']}` | {int(r['n_held_out']) if pd.notna(r.get('n_held_out', np.nan)) else '—'} | "
            f"{int(n_scored) if pd.notna(n_scored) else '—'} | "
            f"{med:.3f} | {me:.3f} | {mx:.3f} |"
        )

    lines += [
        "",
        "## Recommendation",
        "",
        "_TODO: fill in after the run — pick the lowest-error mode for the recommendation."
        " The TODO marker is intentional template content; replace it with 2-3 sentences"
        " summarizing the finding before committing._",
        "",
        "## Known limitations",
        "",
        "Validation here used the SLiM-simulated dataset bundled with the original",
        "locator paper. The following microsat-specific concerns are NOT covered by",
        "this arc and require a real-data follow-up:",
        "",
        "- Homoplasy (alleles of identical length but different ancestry).",
        "- Null alleles (PCR drop-out producing apparent homozygotes).",
        "- Sizing artefacts (off-by-one stutter calls).",
        "- Real-data ascertainment biases.",
        "",
        "See `CLAUDE.md` 'Microsats (active)' TODO list and",
        "`validation/docs/specs/2026-05-04-microsatellite-input-extension-design.md`",
        "for the full design rationale.",
        "",
        "## Figures",
        "",
        "- `validation/figures/microsat_modes.png` — bar chart of per-mode median",
        "  error (k-fold mean ± std).",
        "- `validation/figures/microsat_scatter.png` — predicted-vs-actual coords",
        "  for the smoke fold, one panel per mode.",
        "",
    ]
    summary_md.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True, type=Path,
                   help="Directory containing smoke/<mode>/ and kfold/<mode>/seed*_fold*/ JSONs.")
    p.add_argument("--spatial", required=True,
                   help="Path to microsat_spatial_location.txt for scatter figure.")
    p.add_argument("--summary_dir", type=Path,
                   default=Path("validation/summary"),
                   help="Where to write microsat_kfold.tsv and microsat_summary.md.")
    p.add_argument("--figures_dir", type=Path,
                   default=Path("validation/figures"),
                   help="Where to write microsat_modes.png and microsat_scatter.png.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    full = write_kfold_tsv(args.out_dir, args.summary_dir)
    centroid = compute_centroid_baseline(Path(args.spatial))
    make_modes_figure(full, args.figures_dir / "microsat_modes.png", centroid=centroid)
    make_scatter_figure(args.out_dir, Path(args.spatial),
                        args.figures_dir / "microsat_scatter.png")
    render_summary_md(full, args.summary_dir / "microsat_summary.md")
    print(f"Wrote summary, kfold tsv, and figures under {args.summary_dir} and {args.figures_dir}",
          flush=True)
    print(f"Centroid baseline (k-fold): {centroid['kfold_median_mean']:.3f} "
          f"± {centroid['kfold_median_std']:.3f} mean(median); "
          f"smoke median {centroid['smoke_median']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
