#!/usr/bin/env python3
"""Convert the KNP elephant xlsx to pair-format TSV + ReLocator sample_data.txt.

Source: KNP_ConsensusGenotypes.xlsx — 124 elephants (mostly forest×savanna
hybrids) sampled in Kibale National Park, Uganda. 14 microsatellite loci
recorded as paired ``<locus>.1`` / ``<locus>.2`` columns.

Two important data-quality notes:

1. **Column labels are swapped.** ``GPS_long_DD`` actually contains
   latitudes (~0.4°N, equator) and ``GPS_lat_DD`` contains longitudes
   (~30.4°E, eastern Uganda). This script unswaps them before writing
   ReLocator's ``x = lon, y = lat`` convention.

2. **Per-individual GPS** — every elephant has its own coords, so the
   harness uses random 80/20 + k-fold splits (no site centroids). The
   geographic spread is tight: lat 0.22°-0.68° (~50 km), lon 30.26°-30.54°
   (~30 km). Centroid baseline error in this regime is ~16-18 km — beating
   it requires the network to detect within-park structure.

Outputs:
  --out_microsat       pair-format TSV the converter consumes
  --out_sample_data    x,y,sampleID for the K-fold runner

Both files include all 124 individuals; metadata columns (sex, mtDNA clade,
STRUCTURE classification) are dropped from the parsed output but the
original xlsx is small enough to re-read for those fields if needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _gather_loci(columns: list[str]) -> list[str]:
    """Return ordered locus names with both ``.1`` and ``.2`` partner columns present."""
    seen: set[str] = set()
    loci: list[str] = []
    for c in columns:
        if c.endswith(".1"):
            base = c[:-2]
            if base in seen:
                continue
            if f"{base}.2" in columns:
                seen.add(base)
                loci.append(base)
    return loci


def parse_xlsx(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read xlsx; return (microsat_pair_df, sample_data_df).

    microsat_pair_df: sampleID + one column per locus, cells like ``"75,95"``.
    sample_data_df:   x (lon), y (lat), sampleID — matches ReLocator's expectation.
    """
    df = pd.read_excel(xlsx_path)
    if "Indiv" not in df.columns:
        raise ValueError(f"{xlsx_path}: missing 'Indiv' column")
    if "GPS_long_DD" not in df.columns or "GPS_lat_DD" not in df.columns:
        raise ValueError(f"{xlsx_path}: missing GPS columns")

    loci = _gather_loci(list(df.columns))
    if not loci:
        raise ValueError(f"{xlsx_path}: no <locus>.1/<locus>.2 paired columns found")

    sids = df["Indiv"].astype(str).tolist()

    pair_rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        rec = {}
        for locus in loci:
            a1 = row[f"{locus}.1"]
            a2 = row[f"{locus}.2"]
            if pd.isna(a1) or pd.isna(a2):
                rec[locus] = "NA"
            else:
                rec[locus] = f"{int(a1)},{int(a2)}"
        pair_rows.append(rec)
    pair_df = pd.DataFrame(pair_rows)
    pair_df.insert(0, "sampleID", sids)

    # Unswap the GPS columns: source labels them backwards.
    sample_data = pd.DataFrame({
        "x": df["GPS_lat_DD"].astype(float).values,    # actual longitude
        "y": df["GPS_long_DD"].astype(float).values,   # actual latitude
        "sampleID": sids,
    })

    return pair_df, sample_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KNP elephant xlsx → ReLocator inputs.")
    p.add_argument("--xlsx", required=True, type=Path,
                   help="Input KNP_ConsensusGenotypes.xlsx (or compatible).")
    p.add_argument("--out_microsat", required=True, type=Path,
                   help="Output pair-format microsat TSV.")
    p.add_argument("--out_sample_data", required=True, type=Path,
                   help="Output sample_data.txt (x=lon, y=lat, sampleID).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pair_df, sample_data = parse_xlsx(args.xlsx)
    args.out_microsat.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(args.out_microsat, sep="\t", index=False)
    print(f"Wrote {args.out_microsat}: {pair_df.shape[0]} samples × "
          f"{pair_df.shape[1] - 1} loci", flush=True)

    args.out_sample_data.parent.mkdir(parents=True, exist_ok=True)
    sample_data.to_csv(args.out_sample_data, sep="\t", index=False)
    print(f"Wrote {args.out_sample_data}: {len(sample_data)} samples; "
          f"lon range [{sample_data.x.min():.4f}, {sample_data.x.max():.4f}], "
          f"lat range [{sample_data.y.min():.4f}, {sample_data.y.max():.4f}]",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
