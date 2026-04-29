#!/usr/bin/env python3
"""Verify a ReLocator --geno TSV against a sample_data.txt file.

Catches the most common silent-failure modes: sampleID mismatch, malformed
shape, out-of-range values, and missing coordinates. See SKILL.md for usage.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geno", required=True, help="Feature TSV with 'sampleID' first column.")
    p.add_argument("--sample_data", required=True, help="ReLocator sample_data.txt (sampleID, x, y).")
    p.add_argument("--mode", choices=("dosage", "full_gl"), default="dosage",
                   help="Expected value-range encoding. Default: dosage.")
    p.add_argument("--triplet_tol", type=float, default=1e-2,
                   help="Tolerance for full_gl triplet sum check. Default: 1e-2.")
    return p.parse_args()


def check_geno(path, mode, triplet_tol):
    findings = []
    df = pd.read_csv(path, sep="\t")
    if "sampleID" not in df.columns:
        return df, [f"{path}: missing 'sampleID' column."]

    df = df.set_index("sampleID")
    n_samples, n_features = df.shape
    if n_features == 0:
        findings.append(f"{path}: zero feature columns.")
    if n_samples == 0:
        findings.append(f"{path}: zero sample rows.")

    arr = df.values
    if not np.issubdtype(arr.dtype, np.number):
        findings.append(f"{path}: non-numeric feature values present.")
        return df, findings

    nan_count = int(np.isnan(arr).sum())
    if nan_count:
        findings.append(f"{path}: {nan_count} NaN values in feature matrix.")

    if mode == "dosage":
        lo, hi = arr.min(), arr.max()
        if lo < -1e-3 or hi > 2 + 1e-3:
            findings.append(
                f"{path}: dosage values out of [0, 2] (min={lo:.4f}, max={hi:.4f})."
            )
    else:  # full_gl
        if n_features % 3 != 0:
            findings.append(
                f"{path}: full_gl mode expects 3*N columns, got {n_features}."
            )
        else:
            cols = list(df.columns)
            suffixes = [c.rsplit("_", 1)[-1] for c in cols]
            expected = ["AA", "AB", "BB"] * (n_features // 3)
            if suffixes != expected:
                findings.append(
                    f"{path}: column suffixes are not strictly _AA/_AB/_BB triplets."
                )
            else:
                triplets = arr.reshape(n_samples, n_features // 3, 3)
                sums = triplets.sum(axis=2)
                bad = np.abs(sums - 1.0) > triplet_tol
                if bad.any():
                    n_bad = int(bad.sum())
                    findings.append(
                        f"{path}: {n_bad} GL triplets do not sum to ~1 "
                        f"(tol={triplet_tol})."
                    )

    return df, findings


def check_sample_data(path):
    findings = []
    df = pd.read_csv(path, sep="\t")
    needed = {"sampleID", "x", "y"}
    missing = needed - set(df.columns)
    if missing:
        findings.append(f"{path}: missing required columns: {sorted(missing)}")
        return df, findings

    df["sampleID"] = df["sampleID"].astype(str)
    n_with_loc = df.dropna(subset=["x", "y"]).shape[0]
    if n_with_loc == 0:
        findings.append(f"{path}: no samples have non-NA (x, y); nothing to train on.")

    return df, findings


def check_overlap(geno_df, sd_df, geno_path, sd_path):
    findings = []
    geno_ids = {str(s) for s in geno_df.index}
    sd_ids = set(sd_df["sampleID"])
    only_geno = geno_ids - sd_ids
    only_sd = sd_ids - geno_ids
    if only_geno:
        findings.append(
            f"{len(only_geno)} sampleID(s) in {geno_path} but not in {sd_path}: "
            + ", ".join(sorted(list(only_geno))[:10])
            + (" ..." if len(only_geno) > 10 else "")
        )
    if only_sd:
        findings.append(
            f"{len(only_sd)} sampleID(s) in {sd_path} but not in {geno_path}: "
            + ", ".join(sorted(list(only_sd))[:10])
            + (" ..." if len(only_sd) > 10 else "")
        )
    return findings


def main():
    args = parse_args()
    findings = []

    geno_df, f1 = check_geno(args.geno, args.mode, args.triplet_tol)
    findings.extend(f1)

    sd_df, f2 = check_sample_data(args.sample_data)
    findings.extend(f2)

    if "sampleID" in geno_df.index.name or geno_df.index.name == "sampleID":
        if "sampleID" in sd_df.columns:
            findings.extend(check_overlap(geno_df, sd_df, args.geno, args.sample_data))

    if findings:
        print("FAIL — verification findings:", file=sys.stderr)
        for i, f in enumerate(findings, 1):
            print(f"  {i}. {f}", file=sys.stderr)
        sys.exit(1)

    n_samples = geno_df.shape[0]
    n_features = geno_df.shape[1]
    n_with_loc = sd_df.dropna(subset=["x", "y"]).shape[0]
    print(
        f"OK — {Path(args.geno).name}: {n_samples} samples × {n_features} features; "
        f"{Path(args.sample_data).name}: {n_with_loc} samples with coordinates."
    )


if __name__ == "__main__":
    main()
