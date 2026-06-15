#!/usr/bin/env python3
"""Fabricate ANGSD-style beagle.gz from a VCF's hard genotype calls.

For each diploid genotype (0/0, 0/1, 1/1) emit a GL triplet that is a linear
blend of the deterministic one-hot encoding and the uniform distribution:

    triplet = (1 - alpha) * one_hot(true_call) + alpha * (1/3, 1/3, 1/3)

`alpha=0` reproduces the hard call exactly; `alpha=1` is fully uninformative.
This is the simplest "noisy GL" model that lets us study how continuous
dosage vs. hard-call rounding behave at increasing genotype uncertainty,
without the complexity of a full coverage-stratified read simulator.

Output is a beagle.gz compatible with ANGSD's `-doGlf 2` format and with
this repo's `gl_to_locator.py`.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import allel
import numpy as np

UNIFORM = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
ONE_HOT = np.eye(3, dtype=np.float64)  # rows: 0=AA, 1=AB, 2=BB


def _calls_to_dosage(call_array: np.ndarray) -> np.ndarray:
    """Sum the two haplotype indices per call to get dosage in {0, 1, 2}.

    Missing calls (any -1 entry) are mapped to dosage 0 here; gl_to_locator's
    own missing-detection logic re-flags them as missing via the flat-GL test.
    """
    dosage = call_array.sum(axis=2)
    dosage[(call_array < 0).any(axis=2)] = 0
    return dosage.astype(np.int8)


def _dosage_to_gl(dosage: np.ndarray, alpha: float) -> np.ndarray:
    """Map an (n_sites, n_samples) dosage matrix to (n_sites, n_samples, 3).

    Each triplet = (1 - alpha) * one-hot + alpha * uniform. Output values are
    raw probabilities; gl_to_locator treats them as flat-prior posteriors.
    """
    triplets = ONE_HOT[dosage]  # (n_sites, n_samples, 3)
    return (1.0 - alpha) * triplets + alpha * UNIFORM


def write_beagle(out_path: Path, gl: np.ndarray, chroms: np.ndarray,
                 positions: np.ndarray, ref: np.ndarray, alt: np.ndarray) -> None:
    n_sites, n_samples, _ = gl.shape
    flat = gl.reshape(n_sites, n_samples * 3)
    header = ["marker", "allele1", "allele2"]
    for i in range(n_samples):
        header += [f"Ind{i}", f"Ind{i}", f"Ind{i}"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as fh:
        fh.write("\t".join(header) + "\n")
        for s in range(n_sites):
            marker = f"{chroms[s]}_{positions[s]}"
            row = [marker, str(ref[s]), str(alt[s])]
            row += [f"{v:.6f}" for v in flat[s]]
            fh.write("\t".join(row) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vcf", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--alpha", type=float, default=0.0,
                   help="Noise level in [0, 1]; 0 is deterministic, 1 is uniform.")
    args = p.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        print(f"synth_gls: --alpha must be in [0, 1], got {args.alpha}",
              file=sys.stderr)
        return 1

    print(f"synth_gls: reading {args.vcf}", flush=True)
    vcf = allel.read_vcf(str(args.vcf))
    if vcf is None:
        print(f"synth_gls: failed to read VCF: {args.vcf}", file=sys.stderr)
        return 1
    calls = vcf["calldata/GT"]  # (n_sites, n_samples, 2)
    chroms = vcf["variants/CHROM"]
    positions = vcf["variants/POS"]
    ref = vcf["variants/REF"]
    alt_full = vcf["variants/ALT"]  # (n_sites, max_alts)
    n_sites_raw, n_samples = calls.shape[:2]

    # Keep only biallelic sites: at most one ALT allele AND no genotype call
    # references allele index >= 2.
    biallelic = (alt_full[:, 1:] == "").all(axis=1)
    high_allele = (calls >= 2).any(axis=(1, 2))
    keep = biallelic & ~high_allele
    n_dropped = int((~keep).sum())

    calls = calls[keep]
    chroms = chroms[keep]
    positions = positions[keep]
    ref = ref[keep]
    alt = alt_full[keep, 0]
    n_sites = calls.shape[0]
    print(
        f"synth_gls: {n_sites:,} biallelic sites x {n_samples} samples "
        f"(dropped {n_dropped:,} multi-allelic); alpha={args.alpha}",
        flush=True,
    )

    dosage = _calls_to_dosage(calls)
    gl = _dosage_to_gl(dosage, args.alpha).astype(np.float32)

    write_beagle(args.out, gl, chroms, positions, ref, alt)
    print(f"synth_gls: wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
