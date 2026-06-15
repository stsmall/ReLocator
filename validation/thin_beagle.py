#!/usr/bin/env python3
"""Thin a beagle.gz to the K sites with the least missing data.

A "missing" sample at a site is one whose GL triplet is near-uniform
(max(GL) < 0.4) — same threshold used by gl_to_locator.py. We score
every site by the number of missing samples, print the distribution,
then write the K sites with the lowest scores (most informative) to
a new beagle.gz.

The full balanus combined.beagle.gz has ~34M sites; ReLocator's stock
MLP cannot meaningfully train on that wide an input. Path D's smoke
and LOSO runs use this filtered subset for the pipeline-plumbing tests.
"""

from __future__ import annotations

import argparse
import gzip
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MISSING_THRESHOLD = 0.4
SCAN_CHUNK = 100_000


def _peek_n_samples(beagle_path: Path) -> int:
    with gzip.open(beagle_path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    n_cols = len(header)
    if n_cols < 6 or (n_cols - 3) % 3 != 0:
        raise ValueError(
            f"Beagle header has {n_cols} columns; expected 3 + 3*n_samples"
        )
    return (n_cols - 3) // 3


def score_missingness(beagle_path: Path, n_samples: int) -> np.ndarray:
    """Return a 1D int32 array: per-row count of samples whose GL triplet is flat."""
    scores: list[np.ndarray] = []
    n_seen = 0
    t0 = time.time()
    reader = pd.read_csv(beagle_path, sep="\t", chunksize=SCAN_CHUNK)
    for chunk in reader:
        triplets = chunk.iloc[:, 3:].to_numpy(dtype=np.float32, copy=False)
        # (chunk_size, n_samples, 3)
        triplets = triplets.reshape(len(chunk), n_samples, 3)
        max_per_sample = triplets.max(axis=2)
        missing = (max_per_sample < MISSING_THRESHOLD).sum(axis=1).astype(np.int32)
        scores.append(missing)
        n_seen += len(chunk)
        elapsed = time.time() - t0
        print(
            f"thin_beagle: scanned {n_seen:>10,} rows (elapsed={elapsed:.0f}s)",
            file=sys.stderr,
            flush=True,
        )
    return np.concatenate(scores) if scores else np.zeros(0, dtype=np.int32)


def print_distribution(scores: np.ndarray, n_samples: int) -> None:
    """Print a histogram of per-site missing-sample counts."""
    n_total = len(scores)
    print(
        f"\nMissingness distribution across {n_total:,} sites "
        f"(n_samples={n_samples}):",
        flush=True,
    )
    counts = np.bincount(scores, minlength=n_samples + 1)
    cumulative = 0
    print(f"  {'n_missing':>9}  {'sites':>12}  {'cumulative':>12}  {'pct_cum':>7}")
    for n_miss in range(n_samples + 1):
        c = int(counts[n_miss])
        if c == 0 and n_miss > 0 and cumulative == n_total:
            break
        cumulative += c
        if c == 0:
            continue
        print(
            f"  {n_miss:>9}  {c:>12,}  {cumulative:>12,}  "
            f"{100.0 * cumulative / n_total:>6.2f}%"
        )


def select_lowest(scores: np.ndarray, target: int) -> np.ndarray:
    """Return sorted row indices of the `target` lowest-score rows.

    Ties are broken by row order (earlier rows kept).
    """
    if target >= len(scores):
        return np.arange(len(scores))
    # argpartition: indices of `target` smallest, in arbitrary order, then sort.
    idx = np.argpartition(scores, target)[:target]
    return np.sort(idx)


def write_selected(beagle_in: Path, beagle_out: Path, selected_idx: np.ndarray) -> int:
    """Stream `beagle_in` and write only the rows with index in `selected_idx`.

    Returns the number of rows written.
    """
    selected_set = set(selected_idx.tolist())
    n_written = 0
    t0 = time.time()
    with gzip.open(beagle_in, "rt") as fin, gzip.open(beagle_out, "wt") as fout:
        header = fin.readline()
        fout.write(header)
        for idx, line in enumerate(fin):
            if idx in selected_set:
                fout.write(line)
                n_written += 1
            if (idx + 1) % 5_000_000 == 0:
                elapsed = time.time() - t0
                print(
                    f"thin_beagle: write pass at row {idx + 1:,} "
                    f"(written={n_written:,}, elapsed={elapsed:.0f}s)",
                    file=sys.stderr,
                    flush=True,
                )
    return n_written


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--beagle", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-sites", type=int, default=100_000,
                   help="number of best-missingness sites to keep")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_samples = _peek_n_samples(args.beagle)
    print(f"thin_beagle: n_samples={n_samples}", flush=True)

    print("\n--- Pass 1: score missingness ---", flush=True)
    scores = score_missingness(args.beagle, n_samples)
    print_distribution(scores, n_samples)

    n_total = len(scores)
    target = min(args.n_sites, n_total)
    print(
        f"\nthin_beagle: selecting {target:,} sites with lowest missingness "
        f"(out of {n_total:,})",
        flush=True,
    )
    selected_idx = select_lowest(scores, target)
    selected_scores = scores[selected_idx]
    print(
        f"thin_beagle: selected score range: min={selected_scores.min()} "
        f"missing/{n_samples}, max={selected_scores.max()} missing/{n_samples}, "
        f"median={int(np.median(selected_scores))}",
        flush=True,
    )

    print("\n--- Pass 2: write selected rows ---", flush=True)
    n_written = write_selected(args.beagle, args.out, selected_idx)
    print(
        f"\nthin_beagle: wrote {n_written:,} sites to {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
