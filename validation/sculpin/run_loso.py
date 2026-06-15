#!/usr/bin/env python3
"""Leave-one-site-out validation runner for the Cottus asper microsat dataset.

For each user-facing mode in scripts/microsat_to_locator.py (dosage, geometry,
repeat_norm), generates the feature matrix once on the full sculpin dataset,
then iterates over the 16 sampling sites; for each site it blanks all that
site's individuals' coords in sample_data.txt, runs locator on the remaining
15 sites, predicts the held-out individuals, and writes fold_result.json.

Metric: haversine distance (km) between predicted and true (lat, lon). The
prickly sculpin range spans ~25 degrees of latitude, so haversine is the right
metric — Euclidean on (lat, lon) would systematically distort.

Per-fold JSON schema:
  {
    mode, site, n_held_out, n_evaluable, n_scored,
    median_error_km, mean_error_km, max_error_km,
    [warning],
    [error]   # populated on locator failure
  }

The summarizer aggregates these into a per-mode k-fold table plus a centroid
baseline computed from the same splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from validation import common  # noqa: E402

USER_MODES = ("dosage", "geometry", "repeat_norm")


def build_features(microsat_in: Path, out_tsv: Path, mode: str) -> int:
    """Run scripts/microsat_to_locator.py once to produce a feature matrix."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent.parent / "scripts/microsat_to_locator.py"),
        "--microsat", str(microsat_in),
        "--out", str(out_tsv),
        "--features", mode,
    ]
    rc, _ = common.stream_run(cmd, out_tsv.with_suffix(".buildlog"), stage=f"build_{mode}")
    return rc


def site_of(sample_id: str) -> str:
    """Extract the site label from ``<site>_<NNN>`` sampleIDs written by parse_genepop."""
    return sample_id.rsplit("_", 1)[0]


def write_holdout_sample_data(
    sample_data_in: Path, holdout_site: str, out_path: Path
) -> tuple[pd.DataFrame, list[str]]:
    """Write a copy of sample_data with the holdout site's coords blanked.

    Returns the unchanged truth DataFrame and the list of held-out sampleIDs.
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
    return df, held_out_ids


def run_locator_fold(
    *,
    features_tsv: Path,
    sample_data: Path,
    out_dir: Path,
    fold_label: str,
    gpu: int,
    seed: int,
) -> tuple[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / "run"
    cmd = [
        "locator",
        "--matrix", str(features_tsv),
        "--sample_data", str(sample_data),
        "--out", str(out_prefix),
        "--gpu_number", str(gpu),
        "--seed", str(seed),
    ]
    rc, _ = common.stream_run(cmd, out_dir / "locator.log", stage=f"locator_{fold_label}")
    return rc, Path(str(out_prefix) + "_predlocs.txt")


def evaluate_fold(
    *, predlocs: Path, truth: pd.DataFrame, holdout_ids: list[str]
) -> dict:
    """Score predicted vs true coords for the held-out site, in km."""
    pred = common.parse_predlocs(predlocs)
    pred = pred[pred["sampleID"].isin(holdout_ids)].set_index("sampleID")
    tru = truth.set_index("sampleID").loc[holdout_ids, ["x", "y"]]
    evaluable = tru.dropna().index.tolist()
    evaluable_in_pred = [sid for sid in evaluable if sid in pred.index]
    missing_from_pred = [sid for sid in evaluable if sid not in pred.index]

    if not evaluable_in_pred:
        return {
            "n_held_out": int(len(holdout_ids)),
            "n_evaluable": int(len(evaluable)),
            "n_scored": 0,
            "median_error_km": None,
            "mean_error_km": None,
            "max_error_km": None,
            "warning": f"no scored samples (held_out={len(holdout_ids)}, "
                       f"evaluable={len(evaluable)}, missing_from_pred={len(missing_from_pred)})",
        }

    p = pred.loc[evaluable_in_pred, ["x", "y"]].to_numpy(dtype=np.float64)
    t = tru.loc[evaluable_in_pred].to_numpy(dtype=np.float64)
    # x = lon, y = lat (matches parse_genepop output convention).
    err_km = common.haversine(t[:, 1], t[:, 0], p[:, 1], p[:, 0])
    if np.ndim(err_km) == 0:
        err_km = np.array([err_km])
    result = {
        "n_held_out": int(len(holdout_ids)),
        "n_evaluable": int(len(evaluable)),
        "n_scored": int(len(evaluable_in_pred)),
        "median_error_km": float(np.median(err_km)),
        "mean_error_km": float(np.mean(err_km)),
        "max_error_km": float(np.max(err_km)),
    }
    if missing_from_pred:
        result["warning"] = f"{len(missing_from_pred)} evaluable sample(s) missing from predlocs output"
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--microsat", required=True,
                   help="Pair-format microsat TSV (output of parse_genepop.py).")
    p.add_argument("--sample_data", required=True,
                   help="ReLocator sample_data.txt (output of parse_genepop.py).")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--modes", default=",".join(USER_MODES))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42,
                   help="locator seed for both data shuffle and net init.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample_data_in = Path(args.sample_data)
    sd_df = pd.read_csv(sample_data_in, sep="\t")
    sites = sorted(set(sd_df["sampleID"].apply(site_of)))
    print(f"Sites: {len(sites)} → {sites}", flush=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in USER_MODES]
    if bad:
        print(f"Unknown modes: {bad}. Valid: {USER_MODES}", file=sys.stderr)
        return 2

    for mode in modes:
        feat = args.out_dir / f"features_{mode}.tsv"
        if not feat.exists():
            rc = build_features(Path(args.microsat), feat, mode)
            if rc != 0:
                print(f"build_features failed for mode={mode} rc={rc}", file=sys.stderr)
                return rc

        for site in sites:
            fold_dir = args.out_dir / "loso" / mode / site
            fold_dir.mkdir(parents=True, exist_ok=True)
            sd = fold_dir / "sample_data.txt"
            truth, held = write_holdout_sample_data(sample_data_in, site, sd)
            rc, pred = run_locator_fold(
                features_tsv=feat, sample_data=sd, out_dir=fold_dir,
                fold_label=f"loso_{mode}_{site}", gpu=args.gpu, seed=args.seed,
            )
            if rc == 0 and pred.exists():
                metrics = evaluate_fold(predlocs=pred, truth=truth, holdout_ids=held)
                (fold_dir / "fold_result.json").write_text(json.dumps(
                    {"mode": mode, "site": site, **metrics}
                ))
            else:
                (fold_dir / "fold_result.json").write_text(json.dumps(
                    {"mode": mode, "site": site,
                     "n_held_out": int(len(held)),
                     "error": f"rc={rc}, predlocs_exists={pred.exists()}"}
                ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
