#!/usr/bin/env python3
"""Convert a Genepop file to pair-format microsat TSV + sample_data.txt.

Genepop format reminder (https://genepop.curtin.edu.au/help_input.html):
  Line 1:   free-text title.
  Lines 2..k+1: one locus name per line (or comma-separated on one line).
  Then alternating ``POP`` markers and population blocks. Each individual line
  reads ``<id_or_pop_label>, <genotype1> <genotype2> ...`` where each genotype
  is a string of digits — 2 or 3 digits per allele depending on file format.
  ``0000`` (or ``000000``) means missing.

This script:

- Auto-detects 2-vs-3 digits-per-allele from the first non-zero genotype.
- Splits each genotype into ``(allele1, allele2)`` integers; emits
  ``allele1,allele2`` cells (the pair format that ``microsat_to_locator.py``
  consumes).
- Replaces the per-population identifier with a sequence-numbered sampleID
  (the Genepop convention puts the population label in front of the comma; we
  preserve it as ``site``) so downstream tools can join coordinates.
- Joins each sample to a (lat, lon) using ``--sites <tsv>`` (4-col schema:
  ``site\tlat\tlon\t...``) and writes ``sample_data.txt`` in ReLocator's
  ``x\ty\tsampleID`` schema (lon → x, lat → y, matching the convention used
  in the GL branch's barnacle data).

Outputs:
  --out_microsat  pair-format TSV: sampleID + locus columns with ``a1,a2`` cells
  --out_sample_data ReLocator sample_data.txt: x,y,sampleID columns
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _detect_allele_width(token: str) -> int:
    """Pick 2 or 3 digits per allele based on token length (4 → 2, 6 → 3)."""
    n = len(token)
    if n == 4:
        return 2
    if n == 6:
        return 3
    raise ValueError(
        f"Cannot determine allele width from genotype token {token!r}; "
        "expected 4 or 6 digits."
    )


def _parse_genotype_token(token: str, width: int) -> str:
    """Convert a Genepop genotype string (e.g. '0505' or '012345') to 'a1,a2'.

    Returns ``"NA"`` if either allele is the missing-value sentinel ``0``.
    """
    a1 = int(token[:width])
    a2 = int(token[width:])
    if a1 == 0 or a2 == 0:
        return "NA"
    return f"{a1},{a2}"


def parse_genepop(path: Path) -> tuple[list[str], list[str], list[list[str]], list[str]]:
    """Return (loci, sampleIDs, genotype_rows_in_pair_format, site_labels).

    ``genotype_rows_in_pair_format`` is a list of lists; each inner list has
    len(loci) entries, each ``"a1,a2"`` or ``"NA"``.
    """
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")

    # Locus header. Genepop allows either one-locus-per-line or comma-separated
    # on a single line. Detect by scanning until "POP".
    loci: list[str] = []
    idx = 1  # skip title
    while idx < len(lines) and lines[idx].strip().upper() != "POP":
        line = lines[idx].strip()
        if "," in line and not any(ch.isdigit() for ch in line.split(",")[-1].strip()):
            loci.extend(t.strip() for t in line.split(",") if t.strip())
        else:
            loci.append(line)
        idx += 1
    if not loci:
        raise ValueError(f"{path}: no loci parsed before first POP marker")

    # Now iterate POP/individual blocks.
    sample_ids: list[str] = []
    site_labels: list[str] = []
    rows: list[list[str]] = []
    width: int | None = None
    site_counters: dict[str, int] = {}
    current_site = None

    for line in lines[idx:]:
        s = line.strip()
        if not s:
            continue
        if s.upper() == "POP":
            current_site = None
            continue
        if "," not in s:
            raise ValueError(f"Unexpected non-individual line: {line!r}")
        head, _, rest = s.partition(",")
        site_label = head.strip()
        tokens = rest.strip().split()
        if width is None:
            width = _detect_allele_width(tokens[0])
        if current_site is None:
            current_site = site_label
        # Use the first individual's site label for the whole block.
        sid_idx = site_counters.get(current_site, 0) + 1
        site_counters[current_site] = sid_idx
        sample_ids.append(f"{current_site}_{sid_idx:03d}")
        site_labels.append(current_site)
        if len(tokens) != len(loci):
            raise ValueError(
                f"Genotype count mismatch: {len(tokens)} tokens vs {len(loci)} loci "
                f"in line {line!r}"
            )
        rows.append([_parse_genotype_token(t, width) for t in tokens])

    return loci, sample_ids, rows, site_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genepop → ReLocator pair-format TSV.")
    p.add_argument("--genepop", required=True, type=Path)
    p.add_argument("--sites", required=True, type=Path,
                   help="TSV with columns: site, lat, lon (additional cols ignored).")
    p.add_argument("--out_microsat", required=True, type=Path,
                   help="Output pair-format microsat TSV (consumed by microsat_to_locator.py).")
    p.add_argument("--out_sample_data", required=True, type=Path,
                   help="Output sample_data.txt (x, y, sampleID columns).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    loci, sids, rows, sites = parse_genepop(args.genepop)
    print(f"Parsed: {len(sids)} individuals, {len(loci)} loci, "
          f"{len(set(sites))} populations", flush=True)

    sites_df = pd.read_csv(args.sites, sep="\t").set_index("site")
    missing = [s for s in set(sites) if s not in sites_df.index]
    if missing:
        print(f"Sites in genepop without coords in sites.tsv: {missing}", file=sys.stderr)
        return 2

    # Pair-format TSV
    args.out_microsat.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=loci)
    df.insert(0, "sampleID", sids)
    df.to_csv(args.out_microsat, sep="\t", index=False)
    print(f"Wrote {args.out_microsat}: {df.shape[0]} samples × {df.shape[1] - 1} loci",
          flush=True)

    # sample_data.txt: x = lon, y = lat (matches GL branch convention).
    args.out_sample_data.parent.mkdir(parents=True, exist_ok=True)
    sample_data = pd.DataFrame({
        "x": [sites_df.loc[s, "lon"] for s in sites],
        "y": [sites_df.loc[s, "lat"] for s in sites],
        "sampleID": sids,
    })
    sample_data.to_csv(args.out_sample_data, sep="\t", index=False)
    print(f"Wrote {args.out_sample_data}: {len(sample_data)} samples", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
