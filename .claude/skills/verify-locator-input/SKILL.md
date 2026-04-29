---
name: verify-locator-input
description: Validate that a feature TSV (from gl_to_locator.py, microsat_to_locator.py, microsat_features.py, or microsat_poly.py) is structurally consistent with a sample_data.txt file before running ReLocator. Use when the user has just generated a --geno TSV and wants a sanity check, or when a ReLocator run produces unexpected/empty results.
disable-model-invocation: true
---

# Verify ReLocator Input

ReLocator joins genotypes to coordinates by `sampleID`. If IDs in the `--geno`
TSV do not exactly match `sample_data.txt`, the join silently drops or misaligns
samples — predictions will be nonsense and there's no error message. This skill
runs `verify.py` against the two files and reports any structural problem.

## Usage

```bash
.claude/skills/verify-locator-input/verify.py \
    --geno path/to/feature_matrix.tsv \
    --sample_data path/to/sample_data.txt
```

Optional:
- `--mode {dosage,full_gl}` (default `dosage`) — controls the value-range checks
  for the feature matrix.

## What the script checks

1. **Sample-ID alignment** — every sampleID in the geno file appears in
   sample_data.txt and vice versa; reports IDs in one but not the other.
2. **Shape sanity** — header column count, sample row count.
3. **Value-range plausibility**:
   - `dosage` mode: feature values lie in `[0, 2]` (with small tolerance).
   - `full_gl` mode: column count is divisible by 3, suffixes are
     `{_AA, _AB, _BB}`, and each triplet sums to ≈1.
4. **Coordinate column presence** — `sample_data.txt` has `sampleID, x, y`,
   and at least one sample has non-NA coordinates (otherwise nothing to train
   on).

## Output

Exits 0 on success with a one-line summary. Exits 1 with a numbered list of
specific findings on failure. Use this as the last step before
`locator --geno ... --sample_data ...`.
