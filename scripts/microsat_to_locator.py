#!/usr/bin/env python3
"""
microsat_to_locator.py

Convert tab-delimited microsatellite genotype data to a continuous float
matrix compatible with ReLocator's --matrix loader (patched on the
genotype_likelihoods branch to accept float dosage input).

Three composable encoding modes are supported via --features:

  dosage
      One-hot allele counts (0/1/2) per (locus, allele) pair after the
      --min_allele_freq filter. Equivalent to treating microsats as
      multi-allelic SNPs.

  geometry
      Three continuous summaries per locus:
          {locus}_mean_repeat   (a1 + a2) / 2
          {locus}_allele_diff   abs(a1 - a2)
          {locus}_het           int(a1 != a2)

  repeat_norm
      Per-locus mean-repeat, z-scored across all input samples. Under SMM
      at mutation-drift equilibrium, within-population repeat distributions
      are approximately Gaussian, so per-locus z-scoring is the correct
      transformation.

See docs/microsatellites.md for biological rationale and full option
descriptions; see validation/docs/specs/2026-05-04-microsatellite-input-extension-design.md
for design context.

Output is tab-delimited with 'sampleID' as the first column. Rows are
samples; columns are mode-prefixed feature names. Combined-mode output
concatenates blocks in order: dosage, geometry, repeat_norm.

Exit codes:
  0  success
  1  invalid --features argument
  2  input missing 'sampleID' column
  3  no loci have alleles after MAF filtering
  4  duplicate sampleIDs in input
  5  input format error (e.g. odd locus column count for two-column format)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd

MISSING_STRINGS = {"NA", "NAN", ".", "", "0,0", "0/0"}


def parse_genotype(cell: str) -> tuple[int | None, int | None]:
    """Parse a diploid genotype cell.

    Accepts: '12,14'  '12/14'  '12 14'  '12|14'  '14'  'NA'  '.'  ''
    Single value (no separator) is interpreted as homozygote shorthand.
    Returns ``(int, int)`` or ``(None, None)`` for missing.
    """
    cell = str(cell).strip()
    if cell.upper() in MISSING_STRINGS:
        return (None, None)

    for sep in (",", "/", " ", "|"):
        if sep in cell:
            parts = cell.split(sep, 1)
            try:
                return (int(parts[0].strip()), int(parts[1].strip()))
            except (ValueError, IndexError):
                return (None, None)

    try:
        v = int(cell)
        return (v, v)
    except ValueError:
        return (None, None)


def build_allele_catalog(
    df: pd.DataFrame,
    loci: list[str],
    min_allele_freq: float,
    max_locus_missing: float,
) -> dict[str, list[int]]:
    """For each locus, return the sorted list of alleles passing MAF.

    Per-locus missing rate is reported on stderr; loci above
    ``max_locus_missing`` get a warning but the alleles are still returned
    (caller decides whether to drop the locus).
    """
    n_samples = len(df)
    catalog: dict[str, list[int]] = {}

    for locus in loci:
        allele_counts: dict[int, int] = defaultdict(int)
        total_alleles = 0
        n_missing = 0

        for val in df[locus]:
            a1, a2 = parse_genotype(val)
            if a1 is None:
                n_missing += 1
            else:
                allele_counts[a1] += 1
                allele_counts[a2] += 1
                total_alleles += 2

        missing_frac = n_missing / n_samples if n_samples > 0 else 0.0

        if missing_frac > max_locus_missing:
            print(
                f"  WARNING: locus {locus} missing rate {100 * missing_frac:.1f}% "
                f"({n_missing}/{n_samples}); the locus will be included — consider dropping it manually if needed.",
                file=sys.stderr,
            )

        if total_alleles == 0:
            catalog[locus] = []
            continue

        catalog[locus] = sorted(
            a
            for a, cnt in allele_counts.items()
            if cnt / total_alleles >= min_allele_freq
        )

    return catalog


def encode_dosage_block(
    df: pd.DataFrame,
    active_loci: list[str],
    catalog: dict[str, list[int]],
) -> tuple[np.ndarray, list[str]]:
    """Build the (n_samples, K) dosage block. Missing genotypes get site-mean imputation.

    Returns ``(matrix, column_names)`` where each column is named
    ``dosage_<locus>_<allele>``. K = sum(len(catalog[l]) for l in active_loci).
    Alleles observed in df but not in ``catalog[l]`` (e.g. dropped by MAF) are
    silently ignored — the dosage at that allele's column simply stays 0.
    """
    allele_index = {
        loc: {a: i for i, a in enumerate(catalog[loc])} for loc in active_loci if catalog[loc]
    }
    col_names: list[str] = []
    col_groups: list[tuple[int, int]] = []  # (start, end) per locus
    n_features = 0
    for loc in active_loci:
        n_alleles = len(catalog[loc])
        col_groups.append((n_features, n_features + n_alleles))
        for a in catalog[loc]:
            col_names.append(f"dosage_{loc}_{a}")
        n_features += n_alleles

    n_samples = len(df)
    matrix = np.zeros((n_samples, n_features), dtype=np.float32)
    missing = np.zeros((n_samples, len(active_loci)), dtype=bool)

    for i, sid in enumerate(df.index):
        for li, locus in enumerate(active_loci):
            if not catalog[locus]:
                continue
            a1, a2 = parse_genotype(df.loc[sid, locus])
            if a1 is None:
                missing[i, li] = True
                continue
            idx = allele_index[locus]
            offset = col_groups[li][0]
            for a in (a1, a2):
                if a in idx:
                    matrix[i, offset + idx[a]] += 1.0

    # Site-mean imputation for missing samples per locus
    for li, (start, end) in enumerate(col_groups):
        if start == end:
            continue
        m = missing[:, li]
        if not m.any():
            continue
        present = ~m
        if not present.any():
            continue
        site_mean = matrix[present, start:end].mean(axis=0)
        matrix[m, start:end] = site_mean

    return matrix, col_names


def encode_geometry_block(
    df: pd.DataFrame, active_loci: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Build the (n_samples, 3 * n_loci) geometry block.

    Per-locus columns in this fixed order: ``mean_repeat``, ``allele_diff``,
    ``het``. Missing genotypes are imputed: per-locus mean for ``mean_repeat``
    and ``allele_diff``; per-locus heterozygosity rate for ``het``.
    """
    n_samples = len(df)
    n_loci = len(active_loci)
    mean_repeat = np.zeros((n_samples, n_loci), dtype=np.float32)
    allele_diff = np.zeros((n_samples, n_loci), dtype=np.float32)
    het = np.zeros((n_samples, n_loci), dtype=np.float32)
    missing = np.zeros((n_samples, n_loci), dtype=bool)

    for j, locus in enumerate(active_loci):
        col = df[locus].values
        for i in range(n_samples):
            a1, a2 = parse_genotype(col[i])
            if a1 is None:
                missing[i, j] = True
                continue
            mean_repeat[i, j] = (a1 + a2) / 2.0
            allele_diff[i, j] = abs(a1 - a2)
            het[i, j] = 1.0 if a1 != a2 else 0.0

    for j in range(n_loci):
        m = missing[:, j]
        if not m.any():
            continue
        present = ~m
        if not present.any():
            mean_repeat[:, j] = np.nan
            allele_diff[:, j] = np.nan
            het[:, j] = np.nan
            continue
        mean_repeat[m, j] = mean_repeat[present, j].mean()
        allele_diff[m, j] = allele_diff[present, j].mean()
        het[m, j] = het[present, j].mean()

    # Interleave per-locus blocks: [L1_mean, L1_diff, L1_het, L2_mean, ...]
    matrix = np.empty((n_samples, 3 * n_loci), dtype=np.float32)
    col_names: list[str] = []
    for j, locus in enumerate(active_loci):
        matrix[:, 3 * j] = mean_repeat[:, j]
        matrix[:, 3 * j + 1] = allele_diff[:, j]
        matrix[:, 3 * j + 2] = het[:, j]
        col_names.append(f"geom_{locus}_mean_repeat")
        col_names.append(f"geom_{locus}_allele_diff")
        col_names.append(f"geom_{locus}_het")
    return matrix, col_names


def encode_repeat_norm_block(
    df: pd.DataFrame, active_loci: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Build the (n_samples, n_loci) z-scored mean-repeat block.

    For each locus, compute the per-individual mean repeat ``(a1 + a2) / 2``,
    then z-score across all samples. Missing genotypes are imputed with
    per-locus mean (which becomes 0 after z-scoring). Loci with zero
    variance produce an all-zeros column (no information).
    """
    n_samples = len(df)
    n_loci = len(active_loci)
    mean_repeat = np.zeros((n_samples, n_loci), dtype=np.float32)
    missing = np.zeros((n_samples, n_loci), dtype=bool)

    for j, locus in enumerate(active_loci):
        col = df[locus].values
        for i in range(n_samples):
            a1, a2 = parse_genotype(col[i])
            if a1 is None:
                missing[i, j] = True
                continue
            mean_repeat[i, j] = (a1 + a2) / 2.0

    for j in range(n_loci):
        m = missing[:, j]
        present = ~m
        if not present.any():
            continue
        if m.any():
            mean_repeat[m, j] = mean_repeat[present, j].mean()

    mu = mean_repeat.mean(axis=0)
    sigma = mean_repeat.std(axis=0, ddof=0)
    safe_sigma = np.where(sigma > 0, sigma, 1.0)
    matrix = ((mean_repeat - mu) / safe_sigma).astype(np.float32)
    matrix[:, sigma == 0] = 0.0  # explicit no-information for zero-variance loci

    col_names = [f"rnorm_{locus}" for locus in active_loci]
    return matrix, col_names


# Space is intentionally excluded from PAIR_SEPARATORS: CSV parsing can
# introduce leading/trailing whitespace on numeric cells, which would
# trigger false pair-format detection. parse_genotype() does accept " "
# as a within-cell separator (e.g. "10 11"), so single-cell space-separated
# pair format still works downstream — just not for auto-detection.
PAIR_SEPARATORS = (",", "/", "|")


def detect_format(df: pd.DataFrame) -> Literal["pair", "two_column"]:
    """Return ``"pair"`` or ``"two_column"`` based on cell content.

    Pair format: cells contain a separator (``,`` ``/`` ``|``).
    Two-column format: no cells contain pair separators; locus pairs are
    reconstructed from consecutive columns. Raises ``ValueError`` if
    two-column format has an odd number of locus columns.
    """
    locus_cols = [c for c in df.columns if c != "sampleID"]
    has_separator = False
    for c in locus_cols:
        # Note: regex must stay in sync with PAIR_SEPARATORS.
        if df[c].astype(str).str.contains(r"[,/|]", regex=True, na=False).any():
            has_separator = True
            break

    if has_separator:
        return "pair"

    if len(locus_cols) % 2 != 0:
        raise ValueError(
            f"Two-column format detected but locus column count ({len(locus_cols)}) "
            f"is odd; cannot reconstruct diploid pairs."
        )
    return "two_column"


def convert_two_column_to_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a two-column-format DataFrame to pair format.

    Consecutive locus columns are merged: ``variant_0``/``variant_1`` →
    ``locus_0`` with cell values ``"a1,a2"``. Locus names use a generic
    ``locus_<i>`` scheme (the original column names are not preserved).
    """
    locus_cols = [c for c in df.columns if c != "sampleID"]
    if len(locus_cols) % 2 != 0:
        raise ValueError(
            f"Cannot convert: odd number of locus columns ({len(locus_cols)}); "
            f"columns must form consecutive diploid pairs."
        )

    out = pd.DataFrame({"sampleID": df["sampleID"].values})
    for i in range(0, len(locus_cols), 2):
        c1, c2 = locus_cols[i], locus_cols[i + 1]
        out[f"locus_{i // 2}"] = df[c1].astype(str) + "," + df[c2].astype(str)
    return out


VALID_FEATURES = ("dosage", "geometry", "repeat_norm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert microsatellite genotypes to a continuous feature matrix for ReLocator."
    )
    p.add_argument("--microsat", required=True, help="Input TSV (pair or two-column format).")
    p.add_argument("--out", required=True, help="Output TSV for ReLocator --matrix.")
    p.add_argument(
        "--features",
        default="dosage,geometry,repeat_norm",
        help=(
            f"Comma-separated subset of {VALID_FEATURES}. Default: all three. "
            "Output blocks are always emitted in the canonical order "
            "(dosage, geometry, repeat_norm) regardless of input order."
        ),
    )
    p.add_argument("--min_allele_freq", type=float, default=0.01,
                   help="Drop alleles with frequency below this threshold (dosage mode only). Default: 0.01.")
    p.add_argument("--max_locus_missing", type=float, default=1.0,
                   help="Warn (but keep) loci with missing fraction above this. Default: 1.0 (never warn).")
    p.add_argument("--report_encoding", default=None,
                   help="Optional column→source mapping TSV.")
    return p.parse_args()


def parse_feature_list(spec: str) -> list[str]:
    requested = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [f for f in requested if f not in VALID_FEATURES]
    if bad:
        raise ValueError(f"Unknown feature(s): {bad}. Valid: {VALID_FEATURES}")
    if not requested:
        raise ValueError("--features must request at least one mode.")
    seen: set[str] = set()
    ordered: list[str] = []
    for f in requested:
        if f not in seen:
            ordered.append(f)
            seen.add(f)
    return ordered


def main() -> int:
    args = parse_args()

    try:
        features = parse_feature_list(args.features)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    df = pd.read_csv(args.microsat, sep="\t", dtype=str)
    if "sampleID" not in df.columns:
        print("Input must have a 'sampleID' column.", file=sys.stderr)
        return 2

    if df["sampleID"].duplicated().any():
        dups = df.loc[df["sampleID"].duplicated(), "sampleID"].unique().tolist()
        print(
            f"Duplicate sampleIDs in input: {dups}. Each sample must appear once.",
            file=sys.stderr,
        )
        return 4

    try:
        fmt = detect_format(df)
    except ValueError as e:
        print(f"Input format error: {e}", file=sys.stderr)
        return 5
    print(f"Detected input format: {fmt}", flush=True)
    if fmt == "two_column":
        df = convert_two_column_to_pair(df)

    df = df.set_index("sampleID")
    loci = list(df.columns)
    print(f"Samples: {len(df)}, Loci: {len(loci)}", flush=True)
    print(f"Requested features: {features}", flush=True)

    catalog = build_allele_catalog(df, loci, args.min_allele_freq, args.max_locus_missing)
    active_loci = [locus for locus in loci if catalog[locus]]
    if not active_loci:
        print("No loci have any alleles after filtering.", file=sys.stderr)
        return 3

    blocks: list[np.ndarray] = []
    column_names: list[str] = []
    encoding_records: list[dict] = []

    # Build encoding records at construction time so locus names containing
    # underscores (e.g. auto-generated 'locus_0' from two-column conversion,
    # or user names like 'Locus_1') don't break the parse.
    if "dosage" in features:
        m, cols = encode_dosage_block(df, active_loci, catalog)
        blocks.append(m)
        column_names.extend(cols)
        for locus in active_loci:
            for allele in catalog[locus]:
                encoding_records.append({
                    "feature_type": "dosage",
                    "locus": locus,
                    "detail": str(allele),
                    "column_name": f"dosage_{locus}_{allele}",
                })

    if "geometry" in features:
        m, cols = encode_geometry_block(df, active_loci)
        blocks.append(m)
        column_names.extend(cols)
        for locus in active_loci:
            for kind in ("mean_repeat", "allele_diff", "het"):
                encoding_records.append({
                    "feature_type": "geometry",
                    "locus": locus,
                    "detail": kind,
                    "column_name": f"geom_{locus}_{kind}",
                })

    if "repeat_norm" in features:
        m, cols = encode_repeat_norm_block(df, active_loci)
        blocks.append(m)
        column_names.extend(cols)
        for locus in active_loci:
            encoding_records.append({
                "feature_type": "repeat_norm",
                "locus": locus,
                "detail": "",
                "column_name": f"rnorm_{locus}",
            })

    matrix = np.concatenate(blocks, axis=1) if blocks else np.empty((len(df), 0), dtype=np.float32)
    out_df = pd.DataFrame(matrix, columns=column_names, index=df.index).reset_index()
    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}: {out_df.shape[0]} samples × {len(column_names)} features", flush=True)

    if args.report_encoding is not None:
        pd.DataFrame(encoding_records).to_csv(args.report_encoding, sep="\t", index=False)
        print(f"Wrote {args.report_encoding}: {len(encoding_records)} rows", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
