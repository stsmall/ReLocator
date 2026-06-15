#!/usr/bin/env python3
"""LOSO sweep for sculpin under classifier modes (classify, classify_then_avg).

For each of the 16 sculpin sampling sites: hold out the site, train a
classifier (K = 15 remaining sites) on dosage features via the Locator
Python API with prediction_mode={classify, classify_then_avg}, predict
held-out individuals, score against truth coords with haversine distance.

Output schema (fold_result.json) matches run_loso.py for downstream
summarization compatibility:
  {mode, site, n_held_out, n_evaluable, n_scored,
   median_error_km, mean_error_km, max_error_km, [warning], [error]}

Single mode per invocation (--mode {classify,classify_then_avg}); the
sculpin dosage feature matrix is reused for both modes (and for the
existing regress / regress+rangemask runs).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from validation import common  # noqa: E402

VALID_MODES = ("classify", "classify_then_avg")


def site_of(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def write_holdout_sample_data(
    sample_data_in: Path, holdout_site: str, out_path: Path
) -> tuple[pd.DataFrame, list[str]]:
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


def run_one_fold(
    *, mode: str, site: str, feature_matrix_path: Path,
    sample_data_in: Path, out_dir: Path, gpu: int, seed: int,
    max_epochs: int,
) -> dict:
    fold_dir = out_dir / mode / site
    fold_dir.mkdir(parents=True, exist_ok=True)
    sd_path = fold_dir / "sample_data.txt"
    truth, held_out_ids = write_holdout_sample_data(sample_data_in, site, sd_path)

    out_prefix = fold_dir / "locator"
    fold_result: dict = {"mode": mode, "site": site}
    t0 = time.time()
    try:
        from locator import Locator

        loc = Locator(config={
            "out": str(out_prefix),
            "matrix": str(feature_matrix_path),
            "sample_data": str(sd_path),
            "max_epochs": max_epochs,
            "patience": 30,
            "seed": seed,
            "gpu_number": gpu,
            "prediction_mode": mode,
        })
        genotypes, samples = loc.load_genotypes(matrix=str(feature_matrix_path))
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
    fold_result["elapsed_s"] = round(time.time() - t0, 1)

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
        fold_result["warning"] = "no scored samples"
        (fold_dir / "fold_result.json").write_text(
            json.dumps(fold_result, indent=2, default=str)
        )
        return fold_result

    p = pred.loc[evaluable_in_pred, ["x", "y"]].to_numpy(dtype=np.float64)
    t = tru.loc[evaluable_in_pred].to_numpy(dtype=np.float64)
    err_km = common.haversine(t[:, 1], t[:, 0], p[:, 1], p[:, 0])
    if np.ndim(err_km) == 0:
        err_km = np.array([err_km])

    fold_result.update({
        "status": "OK",
        "n_held_out": int(len(held_out_ids)),
        "n_evaluable": int(len(evaluable)),
        "n_scored": int(len(evaluable_in_pred)),
        "median_error_km": float(np.median(err_km)),
        "mean_error_km": float(np.mean(err_km)),
        "max_error_km": float(np.max(err_km)),
    })
    (fold_dir / "fold_result.json").write_text(
        json.dumps(fold_result, indent=2, default=str)
    )
    return fold_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--microsat", required=True, type=Path,
                   help="Pair-format microsat TSV (parse_genepop output).")
    p.add_argument("--sample_data", required=True, type=Path,
                   help="ReLocator sample_data.txt (truth coords).")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--mode", required=True, choices=VALID_MODES,
                   help="Single classifier mode per invocation.")
    p.add_argument("--feature_matrix", default=None, type=Path,
                   help="Pre-built dosage feature matrix; if not given, builds from --microsat via microsat_to_locator.py.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_epochs", type=int, default=500)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_matrix is None:
        feat = args.out_dir / "features_dosage.tsv"
        if not feat.exists():
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent.parent.parent
                    / "scripts/microsat_to_locator.py"),
                "--microsat", str(args.microsat),
                "--out", str(feat),
                "--features", "dosage",
            ]
            rc, _ = common.stream_run(
                cmd, feat.with_suffix(".buildlog"), stage="build_dosage"
            )
            if rc != 0:
                return rc
    else:
        feat = args.feature_matrix

    sd_df = pd.read_csv(args.sample_data, sep="\t")
    sites = sorted(set(sd_df["sampleID"].apply(site_of)))

    print(f"Starting {len(sites)} folds with prediction_mode={args.mode} "
          f"on GPU {args.gpu}", flush=True)
    t_sweep = time.time()
    for idx, site in enumerate(sites):
        print(f"\n=== fold {idx + 1}/{len(sites)}: {site} ===", flush=True)
        run_one_fold(
            mode=args.mode, site=site,
            feature_matrix_path=feat,
            sample_data_in=args.sample_data,
            out_dir=args.out_dir / "loso_classifier",
            gpu=args.gpu, seed=args.seed,
            max_epochs=args.max_epochs,
        )
    print(f"\nSweep complete in {(time.time() - t_sweep) / 60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
