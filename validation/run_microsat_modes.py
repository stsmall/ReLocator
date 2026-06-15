#!/usr/bin/env python3
"""Random 80/20 and k-fold CV runner for the microsat validation arc.

For each user-facing mode (dosage, geometry, repeat_norm), generates the
feature matrix once on the full SLiM dataset, then runs:
    - Step 1: a random 80/20 split with a single seed (smoke).
    - Step 2: 5-fold CV with 3 seeds.

For each fold, blanks the held-out samples' coordinates in sample_data.txt,
invokes the locator CLI, parses predicted vs true coords, and writes
fold_result.json. The summarize_microsat.py script aggregates these.

Polynomial mode is handled by run_microsat_polynomial.py (separate script
to keep sklearn imports out of the user-facing modes).
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


def euclidean(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.sqrt(((p - q) ** 2).sum(axis=1))


def build_features(microsat_in: Path, out_tsv: Path, mode: str) -> int:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "scripts/microsat_to_locator.py"),
        "--microsat", str(microsat_in),
        "--out", str(out_tsv),
        "--features", mode,
    ]
    rc, _ = common.stream_run(cmd, out_tsv.with_suffix(".buildlog"), stage=f"build_{mode}")
    return rc


def write_sample_data(spatial_in: Path, holdout_ids: list[str], out_path: Path) -> pd.DataFrame:
    df = pd.read_csv(spatial_in, sep="\t")
    sub = df[["x", "y", "sampleID"]].copy()
    sub["x"] = sub["x"].astype(object)
    sub["y"] = sub["y"].astype(object)
    mask = sub["sampleID"].isin(holdout_ids)
    sub.loc[mask, "x"] = "NA"
    sub.loc[mask, "y"] = "NA"
    sub.to_csv(out_path, sep="\t", index=False)
    return df


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
    *,
    predlocs: Path,
    truth: pd.DataFrame,
    holdout_ids: list[str],
) -> dict:
    pred = common.parse_predlocs(predlocs)
    pred = pred[pred["sampleID"].isin(holdout_ids)].set_index("sampleID")
    tru = truth.set_index("sampleID").loc[holdout_ids, ["x", "y"]]
    evaluable = tru.dropna().index.tolist()

    # Filter to IDs actually present in predlocs output (locator can drop samples)
    evaluable_in_pred = [sid for sid in evaluable if sid in pred.index]
    missing_from_pred = [sid for sid in evaluable if sid not in pred.index]

    if not evaluable_in_pred:
        return {
            "n_held_out": int(len(holdout_ids)),
            "n_evaluable": int(len(evaluable)),
            "n_scored": 0,
            "median_error": None,
            "mean_error": None,
            "max_error": None,
            "warning": (
                f"no scored samples (held_out={len(holdout_ids)}, "
                f"evaluable={len(evaluable)}, missing_from_pred={len(missing_from_pred)})"
            ),
        }

    p = pred.loc[evaluable_in_pred, ["x", "y"]].to_numpy(dtype=np.float64)
    t = tru.loc[evaluable_in_pred].to_numpy(dtype=np.float64)
    err = euclidean(p, t)
    result = {
        "n_held_out": int(len(holdout_ids)),
        "n_evaluable": int(len(evaluable)),
        "n_scored": int(len(evaluable_in_pred)),
        "median_error": float(np.median(err)),
        "mean_error": float(err.mean()),
        "max_error": float(err.max()),
    }
    if missing_from_pred:
        result["warning"] = (
            f"{len(missing_from_pred)} evaluable sample(s) missing from predlocs output"
        )
    return result


def random_split_indices(n: int, frac_train: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(n * frac_train))
    return perm[:cut].tolist(), perm[cut:].tolist()


def kfold_indices(n: int, k: int, seed: int) -> list[tuple[list[int], list[int]]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out = []
    for i in range(k):
        test = folds[i].tolist()
        train = np.concatenate([folds[j] for j in range(k) if j != i]).tolist()
        out.append((train, test))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Microsat random-split + k-fold CV runner.")
    p.add_argument("--microsat", required=True, help="SLiM microsat TSV.")
    p.add_argument("--spatial", required=True, help="microsat_spatial_location.txt path.")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--modes", default=",".join(USER_MODES))
    p.add_argument("--smoke_seed", type=int, default=42)
    p.add_argument("--kfold_k", type=int, default=5)
    p.add_argument("--kfold_seeds", default="1,2,3")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spatial = pd.read_csv(args.spatial, sep="\t")
    sample_ids = spatial["sampleID"].tolist()
    n = len(sample_ids)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in USER_MODES]
    if bad:
        print(f"Unknown modes: {bad}. Valid: {USER_MODES}", file=sys.stderr)
        return 2
    seeds = [int(s) for s in args.kfold_seeds.split(",")]

    for mode in modes:
        feat = args.out_dir / f"features_{mode}.tsv"
        if not feat.exists():
            rc = build_features(Path(args.microsat), feat, mode)
            if rc != 0:
                print(f"build_features failed for mode={mode} rc={rc}", file=sys.stderr)
                return rc

        # ---- Smoke: single-seed 80/20 ----
        smoke_dir = args.out_dir / "smoke" / mode
        train, test = random_split_indices(n, 0.8, args.smoke_seed)
        held = [sample_ids[i] for i in test]
        sd = smoke_dir / "sample_data.txt"
        sd.parent.mkdir(parents=True, exist_ok=True)
        truth = write_sample_data(Path(args.spatial), held, sd)
        rc, pred = run_locator_fold(
            features_tsv=feat, sample_data=sd, out_dir=smoke_dir,
            fold_label=f"smoke_{mode}", gpu=args.gpu, seed=args.smoke_seed,
        )
        if rc == 0 and pred.exists():
            metrics = evaluate_fold(predlocs=pred, truth=truth, holdout_ids=held)
            (smoke_dir / "fold_result.json").write_text(json.dumps(
                {"mode": mode, "split": "smoke80_20", "seed": args.smoke_seed, **metrics}
            ))
        else:
            (smoke_dir / "fold_result.json").write_text(json.dumps(
                {"mode": mode, "split": "smoke80_20", "seed": args.smoke_seed,
                 "error": f"rc={rc}, predlocs_exists={pred.exists()}"}
            ))

        # ---- K-fold CV ----
        for seed in seeds:
            for fold_idx, (_, test_idx) in enumerate(kfold_indices(n, args.kfold_k, seed)):
                held = [sample_ids[i] for i in test_idx]
                kdir = args.out_dir / "kfold" / mode / f"seed{seed}_fold{fold_idx}"
                kdir.mkdir(parents=True, exist_ok=True)
                sd = kdir / "sample_data.txt"
                truth = write_sample_data(Path(args.spatial), held, sd)
                rc, pred = run_locator_fold(
                    features_tsv=feat, sample_data=sd, out_dir=kdir,
                    fold_label=f"kfold_{mode}_s{seed}_f{fold_idx}", gpu=args.gpu, seed=seed,
                )
                if rc == 0 and pred.exists():
                    metrics = evaluate_fold(predlocs=pred, truth=truth, holdout_ids=held)
                    (kdir / "fold_result.json").write_text(json.dumps(
                        {"mode": mode, "split": "kfold", "seed": seed,
                         "fold": fold_idx, **metrics}
                    ))
                else:
                    (kdir / "fold_result.json").write_text(json.dumps(
                        {"mode": mode, "split": "kfold", "seed": seed, "fold": fold_idx,
                         "error": f"rc={rc}, predlocs_exists={pred.exists()}"}
                    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
