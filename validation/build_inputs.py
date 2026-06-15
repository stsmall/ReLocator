#!/usr/bin/env python3
"""Build bam.filelist and sample_data.txt for the path D validation pipeline.

Joins sample_table.tsv (sample_name, bam, site) with samples_locations.tsv
(SiteCode, Lat, Lon) and emits two files consumed by gl_to_locator.py and
locator. Hard-fails on missing BAMs, NaN coordinates, beagle column mismatch,
or unmatched site codes.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import pandas as pd


def _die(msg: str) -> None:
    print(f"build_inputs: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _read_beagle_header(beagle_path: Path) -> list[str]:
    with gzip.open(beagle_path, "rt") as fh:
        first = fh.readline().rstrip("\n")
    return first.split("\t")


def _validate_join(sample_table: pd.DataFrame, samples_locations: pd.DataFrame) -> pd.DataFrame:
    merged = sample_table.merge(
        samples_locations,
        left_on="site",
        right_on="SiteCode",
        how="left",
        validate="many_to_one",
    )
    unmatched = merged[merged["SiteCode"].isna()]["site"].tolist()
    if unmatched:
        _die(f"unmatched/unjoinable site codes: {sorted(set(unmatched))}")
    if merged["Lat"].isna().any() or merged["Lon"].isna().any():
        _die("NaN Lat/Lon after join — check samples_locations.tsv")
    return merged


def _validate_bams(merged: pd.DataFrame) -> None:
    missing = [b for b in merged["bam"] if not Path(b).exists()]
    if missing:
        _die(f"missing bam files (first 3 of {len(missing)}): {missing[:3]}")


def _validate_beagle_columns(beagle_path: Path, n_samples: int) -> None:
    header = _read_beagle_header(beagle_path)
    expected = 3 + 3 * n_samples
    if len(header) != expected:
        _die(
            f"beagle column count mismatch: header has {len(header)} cols, "
            f"expected {expected} ({n_samples} samples * 3 + 3 leading cols)"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-table", required=True, type=Path)
    p.add_argument("--samples-locations", required=True, type=Path)
    p.add_argument("--beagle", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    sample_table = pd.read_csv(args.sample_table, sep="\t")
    samples_locations = pd.read_csv(args.samples_locations, sep="\t")

    required_st = {"sample_name", "bam", "site"}
    if not required_st.issubset(sample_table.columns):
        _die(f"sample_table missing columns; need {required_st}")
    required_sl = {"SiteCode", "Lat", "Lon"}
    if not required_sl.issubset(samples_locations.columns):
        _die(f"samples_locations missing columns; need {required_sl}")

    merged = _validate_join(sample_table, samples_locations)
    _validate_bams(merged)
    _validate_beagle_columns(args.beagle, n_samples=len(merged))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    bam_filelist = args.out_dir / "bam.filelist"
    bam_filelist.write_text("\n".join(merged["bam"].tolist()) + "\n")

    sample_data = pd.DataFrame(
        {
            "x": merged["Lon"].astype(float),
            "y": merged["Lat"].astype(float),
            "sampleID": merged["sample_name"],
        }
    )
    sample_data.to_csv(args.out_dir / "sample_data.txt", sep="\t", index=False)

    log = args.out_dir / "build_log.txt"
    log.write_text(
        f"n_samples = {len(merged)}\n"
        f"n_sites = {merged['SiteCode'].nunique()}\n"
        f"sample_id_order = {merged['sample_name'].tolist()}\n"
        "ASSUMPTION: ANGSD was invoked with BAMs in sample_table row order. "
        "Beagle column count was verified but column-to-sample identity was not; "
        "see spec section 'Open assumptions / load-bearing risks'.\n"
    )

    print(f"build_inputs: OK — wrote {bam_filelist}, sample_data.txt, build_log.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
