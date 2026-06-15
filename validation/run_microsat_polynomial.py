#!/usr/bin/env python3
"""Polynomial-degree-2 experimental run for the microsat validation arc.

Validation-zone only. NOT in the user-facing scripts/microsat_to_locator.py
--features flag because n=200 makes degree-2 + PCA severely under-determined
in expectation; this runner produces empirical evidence to either confirm
or contradict that prediction.

Pipeline:
  scripts/microsat_to_locator.py --features dosage  →  dosage matrix
  sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
  sklearn.decomposition.PCA(n_components=200)
  → write polynomial features TSV
  → re-use run_microsat_modes harness for splits and locator runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

from validation.run_microsat_modes import (
    build_features,
    evaluate_fold,
    kfold_indices,
    random_split_indices,
    run_locator_fold,
    write_sample_data,
)


def expand_polynomial(dosage_tsv: Path, out_tsv: Path, n_components: int = 200) -> None:
    df = pd.read_csv(dosage_tsv, sep="\t")
    sids = df["sampleID"].values
    X = df.drop(columns=["sampleID"]).to_numpy(dtype=np.float64)

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    Xp = poly.fit_transform(X)
    print(f"polynomial expansion: {X.shape[1]} → {Xp.shape[1]}", flush=True)

    n_pca = min(n_components, Xp.shape[0] - 1, Xp.shape[1])
    pca = PCA(n_components=n_pca)
    Xr = pca.fit_transform(Xp)
    print(
        f"PCA: {Xp.shape[1]} → {Xr.shape[1]} components, "
        f"cumvar={pca.explained_variance_ratio_.sum():.4f}",
        flush=True,
    )

    cols = [f"poly_pc{i}" for i in range(Xr.shape[1])]
    out = pd.DataFrame(Xr, columns=cols)
    out.insert(0, "sampleID", sids)
    out.to_csv(out_tsv, sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--microsat", required=True)
    p.add_argument("--spatial", required=True)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--n_components", type=int, default=200)
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

    # 1. Build dosage matrix (re-using the merge-zone script).
    dosage_tsv = args.out_dir / "features_dosage.tsv"
    if not dosage_tsv.exists():
        rc = build_features(Path(args.microsat), dosage_tsv, "dosage")
        if rc != 0:
            return rc

    # 2. Expand to polynomial + PCA.
    poly_tsv = args.out_dir / "features_polynomial.tsv"
    if not poly_tsv.exists():
        expand_polynomial(dosage_tsv, poly_tsv, args.n_components)

    seeds = [int(s) for s in args.kfold_seeds.split(",")]

    # 3. Smoke 80/20.
    smoke_dir = args.out_dir / "smoke" / "polynomial"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    _, test = random_split_indices(n, 0.8, args.smoke_seed)
    held = [sample_ids[i] for i in test]
    sd = smoke_dir / "sample_data.txt"
    truth = write_sample_data(Path(args.spatial), held, sd)
    rc, pred = run_locator_fold(
        features_tsv=poly_tsv,
        sample_data=sd,
        out_dir=smoke_dir,
        fold_label="smoke_polynomial",
        gpu=args.gpu,
        seed=args.smoke_seed,
    )
    if rc == 0 and pred.exists():
        m = evaluate_fold(predlocs=pred, truth=truth, holdout_ids=held)
        (smoke_dir / "fold_result.json").write_text(
            json.dumps(
                {"mode": "polynomial", "split": "smoke80_20", "seed": args.smoke_seed, **m}
            )
        )
    else:
        (smoke_dir / "fold_result.json").write_text(
            json.dumps(
                {
                    "mode": "polynomial",
                    "split": "smoke80_20",
                    "seed": args.smoke_seed,
                    "error": f"rc={rc}, predlocs_exists={pred.exists()}",
                }
            )
        )

    # 4. K-fold CV.
    for seed in seeds:
        for fold_idx, (_, test_idx) in enumerate(kfold_indices(n, args.kfold_k, seed)):
            held = [sample_ids[i] for i in test_idx]
            kdir = args.out_dir / "kfold" / "polynomial" / f"seed{seed}_fold{fold_idx}"
            kdir.mkdir(parents=True, exist_ok=True)
            sd = kdir / "sample_data.txt"
            truth = write_sample_data(Path(args.spatial), held, sd)
            rc, pred = run_locator_fold(
                features_tsv=poly_tsv,
                sample_data=sd,
                out_dir=kdir,
                fold_label=f"kfold_poly_s{seed}_f{fold_idx}",
                gpu=args.gpu,
                seed=seed,
            )
            if rc == 0 and pred.exists():
                m = evaluate_fold(predlocs=pred, truth=truth, holdout_ids=held)
                (kdir / "fold_result.json").write_text(
                    json.dumps(
                        {
                            "mode": "polynomial",
                            "split": "kfold",
                            "seed": seed,
                            "fold": fold_idx,
                            **m,
                        }
                    )
                )
            else:
                (kdir / "fold_result.json").write_text(
                    json.dumps(
                        {
                            "mode": "polynomial",
                            "split": "kfold",
                            "seed": seed,
                            "fold": fold_idx,
                            "error": f"rc={rc}, predlocs_exists={pred.exists()}",
                        }
                    )
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
