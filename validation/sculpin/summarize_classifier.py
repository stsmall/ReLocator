#!/usr/bin/env python3
"""Aggregate sculpin LOSO results across all 5 modes for the graph-classifier experiment.

Modes:
  - regress                 : out/sculpin_validation/loso/dosage/<site>/fold_result.json
  - regress+rangemask       : out/sculpin_validation/loso_rangemask/dosage/<site>/fold_result.json
  - classify                : out/sculpin_validation/loso_classifier/classify/<site>/fold_result.json
  - classify_then_avg       : out/sculpin_validation/loso_classifier/classify_then_avg/<site>/fold_result.json
  - classify_then_avg_graph : out/sculpin_validation/loso_graph/<site>/fold_result.json

Outputs:
  validation/summary/sculpin_classifier_kfold.tsv  — all per-fold rows tagged by mode
  validation/summary/sculpin_classifier_summary.md — written report
  validation/figures/sculpin_classifier_modes.png  — grouped bar chart
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

MODE_PATHS = {
    "regress": ("loso/dosage", "regression (current)"),
    "regress+rangemask": ("loso_rangemask/dosage", "regression + range_mask"),
    "classify": ("loso_classifier/classify", "classify (argmax)"),
    "classify_then_avg": ("loso_classifier/classify_then_avg", "classify_then_avg"),
    "classify_then_avg_graph": ("loso_graph", "classify_then_avg + graph"),
}


def load_mode_folds(out_dir: Path, mode_subdir: str) -> pd.DataFrame:
    rows = []
    for p in sorted((out_dir / mode_subdir).glob("*/fold_result.json")):
        d = json.loads(p.read_text())
        rows.append(d)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def write_aggregate_tsv(out_dir: Path, summary_dir: Path) -> pd.DataFrame:
    frames = []
    for label, (subdir, _) in MODE_PATHS.items():
        df = load_mode_folds(out_dir, subdir)
        if not df.empty:
            df["mode"] = label
            frames.append(df)
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(summary_dir / "sculpin_classifier_kfold.tsv", sep="\t", index=False)
    return full


def aggregate(full: pd.DataFrame) -> pd.DataFrame:
    if full.empty:
        return pd.DataFrame()
    fold = full.dropna(subset=["median_error_km"])
    return (
        fold.groupby("mode")
        .agg(
            n_folds=("site", "count"),
            med_mean=("median_error_km", "mean"),
            med_std=("median_error_km", "std"),
            mn_mean=("mean_error_km", "mean"),
            mn_std=("mean_error_km", "std"),
        )
        .reindex(list(MODE_PATHS.keys()))
        .dropna(subset=["med_mean"])
    )


def make_bar_figure(full: pd.DataFrame, fig_path: Path,
                    centroid_baseline: float = 440.1) -> None:
    if full.empty:
        return
    fold = full.dropna(subset=["median_error_km"])
    site_order = (
        fold[fold["mode"] == "regress"]
        .sort_values("median_error_km")["site"].tolist()
    )
    if not site_order:
        site_order = sorted(fold["site"].unique())

    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(site_order))
    w = 0.16
    colors = ["firebrick", "darkorange", "seagreen", "steelblue", "purple"]
    for i, mode in enumerate(MODE_PATHS):
        sub = fold[fold["mode"] == mode].set_index("site")
        vals = [sub["median_error_km"].get(s, np.nan) for s in site_order]
        ax.bar(x + (i - 2.0) * w, vals, w,
               label=MODE_PATHS[mode][1], color=colors[i])
    ax.axhline(centroid_baseline, color="gray", linestyle="--", lw=1.2,
               label=f"centroid baseline ({centroid_baseline:.0f} km)", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(site_order, rotation=60, ha="right")
    ax.set_ylabel("Median LOSO error (km)")
    ax.set_title("Sculpin LOSO: 5-mode comparison (per site)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_summary_md(full: pd.DataFrame, agg: pd.DataFrame,
                      summary_md: Path, centroid_baseline: float = 440.1) -> None:
    if agg.empty:
        summary_md.write_text(
            "# Sculpin classifier-head summary\n\nNo fold results found.\n"
        )
        return
    lines = [
        "# Sculpin classifier-head LOSO summary",
        "",
        "16-fold leave-one-site-out comparison across 5 prediction modes on",
        "the prickly sculpin Cottus asper microsat dataset (405 individuals ×",
        "19 loci, dosage encoding). Metric: haversine distance (km).",
        "",
        f"**Centroid baseline (predict training mean):** {centroid_baseline:.1f} km",
        "",
        "## Aggregate (mean of per-fold medians)",
        "",
        "| Mode | n folds | Median error km (mean ± std) | Mean error km (mean ± std) | Δ vs centroid | Δ vs regress |",
        "|---|---|---|---|---|---|",
    ]
    regress_med = float(agg.loc["regress", "med_mean"]) if "regress" in agg.index else float("nan")
    for mode in agg.index:
        r = agg.loc[mode]
        d_centroid = r["med_mean"] - centroid_baseline
        d_regress = r["med_mean"] - regress_med if not np.isnan(regress_med) else float("nan")
        lines.append(
            f"| `{mode}` | {int(r['n_folds'])} | {r['med_mean']:.1f} ± {r['med_std']:.1f} | "
            f"{r['mn_mean']:.1f} ± {r['mn_std']:.1f} | {d_centroid:+.1f} | {d_regress:+.1f} |"
        )
    lines += [
        "",
        "## Per-site error (km, sorted by regress baseline)",
        "",
        "| site | regress | regress+rangemask | classify | classify_then_avg | classify_then_avg_graph |",
        "|---|---|---|---|---|---|",
    ]
    fold = full.dropna(subset=["median_error_km"])
    site_order = fold[fold["mode"] == "regress"].sort_values("median_error_km")["site"].tolist()
    for site in site_order:
        cells = []
        for mode in MODE_PATHS:
            v = fold[(fold["mode"] == mode) & (fold["site"] == site)]
            cells.append(f"{v['median_error_km'].iloc[0]:.0f}" if not v.empty else "—")
        lines.append(f"| {site} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Files",
        "",
        "- `validation/figures/sculpin_classifier_modes.png` — grouped bar",
        "  chart, all 5 modes per site, with centroid baseline overlay.",
        "- `validation/summary/sculpin_classifier_kfold.tsv` — raw per-fold rows.",
        "",
        "## Recommendation",
        "",
        "_Manually edited after the run — replace with 2–3 sentence interpretation._",
        "",
    ]
    summary_md.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True, type=Path,
                   help="Sculpin validation root (parent of loso/, loso_rangemask/, loso_classifier/).")
    p.add_argument("--summary_dir", type=Path,
                   default=Path("validation/summary"))
    p.add_argument("--figures_dir", type=Path,
                   default=Path("validation/figures"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    full = write_aggregate_tsv(args.out_dir, args.summary_dir)
    agg = aggregate(full)
    make_bar_figure(full, args.figures_dir / "sculpin_classifier_modes.png")
    render_summary_md(full, agg, args.summary_dir / "sculpin_classifier_summary.md")
    print(f"Wrote summary, kfold tsv, and figures under {args.summary_dir} and {args.figures_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
