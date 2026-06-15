# Microsatellite Input Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-script microsat-genotype preprocessor (`scripts/microsat_to_locator.py`) with three composable feature modes (`dosage` / `geometry` / `repeat_norm`), validate it end-to-end on the SLiM-simulated dataset bundled with the original locator paper, and produce a stacked PR with the same merge-zone / validation-zone discipline as the GL branch.

**Architecture:** One converter script with `--features` flag → continuous float matrix → ReLocator's already-patched `--matrix` loader. No changes to ReLocator core. Validation harness mirrors `validation/run_loso.py` but uses random 80/20 + k-fold CV (no sites in arena data). Most encoding logic is re-extracted from the dropped `microsat_to_locator.py` / `microsat_features.py` (commit `bb47ced^`) with modifications: combined into one script, new `repeat_norm` mode added, two-column-format auto-detection added.

**Tech Stack:** Python 3.12, numpy, pandas, scikit-learn (`PolynomialFeatures`, `PCA` — validation-zone only), matplotlib, the existing `pixi` env (TF 2.19.1, 3× A100 80GB), `pytest` via `pixi run pytest`, the existing `locator` CLI from the editable install.

**Spec:** `validation/docs/specs/2026-05-04-microsatellite-input-extension-design.md`.

**Branch:** `microsatellites`, off `genotype_likelihoods` HEAD `2ee683d`. All commits go here. No pushes until Task 14.

---

## File map

Created in this plan:

| Path | Zone | Responsibility |
|---|---|---|
| `scripts/microsat_to_locator.py` | merge | Single converter script with `--features dosage,geometry,repeat_norm` flag |
| `tests/test_microsat_input.py` | merge | Synthetic-fixture unit tests for the converter |
| `docs/microsatellites.md` | merge | User-facing guide: input formats, modes, examples |
| `validation/run_microsat_modes.py` | validation | Random 80/20 + k-fold CV runner across the 3 user-facing modes |
| `validation/run_microsat_polynomial.py` | validation | Polynomial degree-2 + PCA experimental run |
| `validation/summarize_microsat.py` | validation | Aggregate per-fold JSONs → comparison figure + summary.md |
| `validation/summary/microsat_smoke.txt` | validation | Output of step 2 (random 80/20 single seed) |
| `validation/summary/microsat_kfold.tsv` | validation | Output of step 3 (k-fold CV per mode) |
| `validation/summary/microsat_summary.md` | validation | Step 6 markdown summary |
| `validation/figures/microsat_modes.png` | validation | Bar chart: median Euclidean error per mode |
| `validation/figures/microsat_scatter.png` | validation | Predicted-vs-actual scatter, one panel per mode |

Modified:

| Path | Change |
|---|---|
| `CLAUDE.md` | Replace "Microsats (deferred)" with "Microsats (active)"; add `microsat_to_locator.py` row to Scripts Summary; update Branch state to note stacked PR shape; add operational notes |

Output tree (created at runtime):

```
out/microsat_validation/
    inputs/{microsat.tsv, sample_data.txt}
    smoke/{features_<mode>.tsv, locator_run/, result.json}
    kfold/{<mode>/<fold>/{features.tsv, locator_run/, fold_result.json}}
    polynomial/{<fold>/{features.tsv, locator_run/, fold_result.json}}
```

---

## Conventions (apply to every task)

- All Python files start with `from __future__ import annotations` and follow the existing `scripts/` style (argparse-based CLI with a `main()` entrypoint guarded by `if __name__ == "__main__":`).
- All commands run inside the pixi env: `pixi run <cmd>`. Bare `pytest` will fail (`conftest.py` imports `allel`, `zarr`).
- The `.claude/hooks/ruff-on-edit.sh` hook auto-runs ruff after every Edit/Write — code must be ruff-clean. Run `pixi run ruff check <file>` manually before committing each task.
- The `.pre-commit-config.yaml` exists but isn't installed in this clone; pre-commit does not run on commit. Manual `pixi run ruff check` is the only gate.
- `pixi.lock` is untracked and the `.claude/hooks/block-pixi-lock.sh` hook prevents direct edits — leave it alone.
- The active branch is `microsatellites`. All commits go there. No pushes until Task 14.
- When importing inside `scripts/microsat_to_locator.py`, do NOT import anything from `validation/` — the merge zone must be self-contained. The validation runners in `validation/` may import from `validation/common.py` (already exists).
- `subprocess.run(...)` calls to the `locator` CLI use the `validation.common.stream_run` helper (already exists) to keep stdout streaming and capture rc + tail. See `validation/run_loso.py` for the reference pattern.
- Prefer `np.float32` for feature matrices (matches what ReLocator's loader produces).
- Use `pd.DataFrame.to_csv(..., sep="\t", index=False)` for all TSV outputs. The first column is always `sampleID`.
- Test fixture sizes stay tiny (≤ 8 samples, ≤ 5 loci) to keep the suite fast.

---

## Task 1: Genotype parsing primitives

**Files:**
- Create: `scripts/microsat_to_locator.py` (file header + parsing functions only — full CLI added in Task 7)
- Create: `tests/test_microsat_input.py` (test scaffolding + parsing tests)

- [ ] **Step 1: Write the failing tests for `parse_genotype`**

Create `tests/test_microsat_input.py`:

```python
"""Tests for scripts/microsat_to_locator.py.

Helper functions are tested by direct import; full-script behavior is tested
through subprocess invocations against synthetic fixtures (matches
test_input_extensions.py pattern for the GL converter).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "microsat_to_locator.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("microsat_to_locator", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def msl():
    return _load_module()


def run_script(*args):
    cmd = [sys.executable, str(SCRIPT_PATH), *map(str, args)]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


# ---------------------------------------------------------------------------
# Task 1: parse_genotype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cell, expected",
    [
        ("12,14", (12, 14)),
        ("12/14", (12, 14)),
        ("12 14", (12, 14)),
        ("12|14", (12, 14)),
        (" 12 , 14 ", (12, 14)),
        ("14", (14, 14)),
        ("NA", (None, None)),
        ("nan", (None, None)),
        (".", (None, None)),
        ("", (None, None)),
        ("0,0", (None, None)),
        ("not_a_number", (None, None)),
        ("12,not_a_number", (None, None)),
    ],
)
def test_parse_genotype_variants(msl, cell, expected):
    assert msl.parse_genotype(cell) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_microsat_input.py -v
```

Expected: ERROR `FileNotFoundError` or `ImportError` because `scripts/microsat_to_locator.py` does not yet exist.

- [ ] **Step 3: Create the script with file header and parsing primitives**

Create `scripts/microsat_to_locator.py`:

```python
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
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

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


def main() -> int:
    raise SystemExit("microsat_to_locator: CLI not yet wired (see Task 7).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pixi run pytest tests/test_microsat_input.py -v
```

Expected: 13 PASS for `test_parse_genotype_variants[*]`.

- [ ] **Step 5: Lint and commit**

```bash
pixi run ruff check scripts/microsat_to_locator.py tests/test_microsat_input.py
git add scripts/microsat_to_locator.py tests/test_microsat_input.py
git commit -m "$(cat <<'EOF'
microsat: add parse_genotype primitive + tests

Scaffolds scripts/microsat_to_locator.py with the file docstring and the
parse_genotype function (re-extracted from the dropped bb47ced^ scripts
with no logic changes). main() raises a clear "not yet wired" stub so
incomplete invocations don't silently no-op. Adds tests/test_microsat_input.py
with 13 parametrized cases covering all separators and missing-value tokens.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Allele catalog (MAF + max_locus_missing filters)

**Files:**
- Modify: `scripts/microsat_to_locator.py` (add `build_allele_catalog`)
- Modify: `tests/test_microsat_input.py` (add catalog tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_microsat_input.py`:

```python
# ---------------------------------------------------------------------------
# Task 2: build_allele_catalog
# ---------------------------------------------------------------------------

def _df_pairs(rows: list[dict]) -> pd.DataFrame:
    """Helper: build a microsat DataFrame from a list of {sampleID, locus: 'a,b'} dicts."""
    df = pd.DataFrame(rows).set_index("sampleID")
    return df


def test_catalog_keeps_all_alleles_above_maf(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,11"},
        {"sampleID": "s2", "L1": "10,12"},
        {"sampleID": "s3", "L1": "11,12"},
        {"sampleID": "s4", "L1": "10,11"},
    ])
    catalog = msl.build_allele_catalog(df, ["L1"], min_allele_freq=0.0, max_locus_missing=1.0)
    assert catalog["L1"] == [10, 11, 12]


def test_catalog_drops_rare_alleles(msl):
    rows = [{"sampleID": f"s{i}", "L1": "10,11"} for i in range(99)]
    rows.append({"sampleID": "s99", "L1": "10,99"})
    df = _df_pairs(rows)
    catalog = msl.build_allele_catalog(df, ["L1"], min_allele_freq=0.05, max_locus_missing=1.0)
    assert 99 not in catalog["L1"]
    assert 10 in catalog["L1"] and 11 in catalog["L1"]


def test_catalog_handles_all_missing_locus(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "NA"},
        {"sampleID": "s2", "L1": "NA"},
    ])
    catalog = msl.build_allele_catalog(df, ["L1"], min_allele_freq=0.0, max_locus_missing=1.0)
    assert catalog["L1"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_microsat_input.py -v -k catalog
```

Expected: 3 ERROR with `AttributeError: module 'microsat_to_locator' has no attribute 'build_allele_catalog'`.

- [ ] **Step 3: Implement `build_allele_catalog`**

Add to `scripts/microsat_to_locator.py` (after `parse_genotype`, before `main`):

```python
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

        if missing_frac >= max_locus_missing:
            print(
                f"  WARNING: locus {locus} missing rate {100 * missing_frac:.1f}% "
                f"({n_missing}/{n_samples}); see CLAUDE.md TODO: null allele handling.",
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pixi run pytest tests/test_microsat_input.py -v -k catalog
```

Expected: 3 PASS.

- [ ] **Step 5: Lint and commit**

```bash
pixi run ruff check scripts/microsat_to_locator.py tests/test_microsat_input.py
git add scripts/microsat_to_locator.py tests/test_microsat_input.py
git commit -m "$(cat <<'EOF'
microsat: add build_allele_catalog with MAF + missing-rate reporting

Re-extracts the catalog builder from the dropped bb47ced^ script. Slight
simplification: max_locus_missing only warns; the caller decides whether
to drop the locus (was previously bundled in the same function).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Dosage encoding block

**Files:**
- Modify: `scripts/microsat_to_locator.py` (add `encode_dosage_block`)
- Modify: `tests/test_microsat_input.py` (add dosage-block tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_microsat_input.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: encode_dosage_block
# ---------------------------------------------------------------------------

def test_dosage_block_basic(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,11"},
        {"sampleID": "s2", "L1": "10,10"},
        {"sampleID": "s3", "L1": "11,11"},
    ])
    catalog = {"L1": [10, 11]}
    matrix, col_names = msl.encode_dosage_block(df, ["L1"], catalog)
    assert col_names == ["dosage_L1_10", "dosage_L1_11"]
    np.testing.assert_array_equal(matrix, np.array([[1, 1], [2, 0], [0, 2]], dtype=np.float32))


def test_dosage_block_multi_locus(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,11", "L2": "20,21"},
        {"sampleID": "s2", "L1": "11,11", "L2": "20,20"},
    ])
    catalog = {"L1": [10, 11], "L2": [20, 21]}
    matrix, col_names = msl.encode_dosage_block(df, ["L1", "L2"], catalog)
    assert col_names == ["dosage_L1_10", "dosage_L1_11", "dosage_L2_20", "dosage_L2_21"]
    expected = np.array([[1, 1, 1, 1], [0, 2, 2, 0]], dtype=np.float32)
    np.testing.assert_array_equal(matrix, expected)


def test_dosage_block_imputes_missing_with_site_mean(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,10"},
        {"sampleID": "s2", "L1": "10,11"},
        {"sampleID": "s3", "L1": "NA"},
    ])
    catalog = {"L1": [10, 11]}
    matrix, _ = msl.encode_dosage_block(df, ["L1"], catalog)
    # site mean across non-missing: col 0 = (2+1)/2 = 1.5; col 1 = (0+1)/2 = 0.5
    np.testing.assert_allclose(matrix[2], np.array([1.5, 0.5], dtype=np.float32))


def test_dosage_block_drops_alleles_outside_catalog(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,99"},  # 99 is not in catalog
        {"sampleID": "s2", "L1": "10,10"},
    ])
    catalog = {"L1": [10]}
    matrix, col_names = msl.encode_dosage_block(df, ["L1"], catalog)
    assert col_names == ["dosage_L1_10"]
    # s1 has only one in-catalog allele (10), so dosage = 1
    np.testing.assert_allclose(matrix, np.array([[1.0], [2.0]], dtype=np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_microsat_input.py -v -k dosage
```

Expected: 4 ERROR — `encode_dosage_block` not defined.

- [ ] **Step 3: Implement `encode_dosage_block`**

Add to `scripts/microsat_to_locator.py`:

```python
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
        l: {a: i for i, a in enumerate(catalog[l])} for l in active_loci if catalog[l]
    }
    col_names: list[str] = []
    col_groups: list[tuple[int, int]] = []  # (start, end) per locus
    n_features = 0
    for l in active_loci:
        n_alleles = len(catalog[l])
        col_groups.append((n_features, n_features + n_alleles))
        for a in catalog[l]:
            col_names.append(f"dosage_{l}_{a}")
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pixi run pytest tests/test_microsat_input.py -v -k dosage
```

Expected: 4 PASS.

- [ ] **Step 5: Lint and commit**

```bash
pixi run ruff check scripts/microsat_to_locator.py tests/test_microsat_input.py
git add scripts/microsat_to_locator.py tests/test_microsat_input.py
git commit -m "$(cat <<'EOF'
microsat: add encode_dosage_block with site-mean imputation

Returns (matrix, column_names) with deterministic column ordering
(dosage_<locus>_<allele>). Re-extracted from bb47ced^ with column-name
prefix changed from {locus}_{allele} to dosage_{locus}_{allele} to
support combined-mode output where blocks are concatenated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Geometry encoding block

**Files:**
- Modify: `scripts/microsat_to_locator.py` (add `encode_geometry_block`)
- Modify: `tests/test_microsat_input.py` (add geometry tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_microsat_input.py`:

```python
# ---------------------------------------------------------------------------
# Task 4: encode_geometry_block
# ---------------------------------------------------------------------------

def test_geometry_block_hand_computation(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,12"},  # mean=11, diff=2, het=1
        {"sampleID": "s2", "L1": "14,14"},  # mean=14, diff=0, het=0
    ])
    matrix, col_names = msl.encode_geometry_block(df, ["L1"])
    assert col_names == ["geom_L1_mean_repeat", "geom_L1_allele_diff", "geom_L1_het"]
    expected = np.array([[11.0, 2.0, 1.0], [14.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(matrix, expected)


def test_geometry_block_imputes_missing(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,10"},   # mean=10, diff=0, het=0
        {"sampleID": "s2", "L1": "12,14"},   # mean=13, diff=2, het=1
        {"sampleID": "s3", "L1": "NA"},
    ])
    matrix, _ = msl.encode_geometry_block(df, ["L1"])
    # imputed: mean_repeat = (10+13)/2 = 11.5, allele_diff = (0+2)/2 = 1.0, het rate = 0.5
    np.testing.assert_allclose(matrix[2], np.array([11.5, 1.0, 0.5], dtype=np.float32))


def test_geometry_block_multi_locus_column_order(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,12", "L2": "20,22"},
    ])
    _, col_names = msl.encode_geometry_block(df, ["L1", "L2"])
    assert col_names == [
        "geom_L1_mean_repeat", "geom_L1_allele_diff", "geom_L1_het",
        "geom_L2_mean_repeat", "geom_L2_allele_diff", "geom_L2_het",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_microsat_input.py -v -k geometry
```

Expected: 3 ERROR — `encode_geometry_block` not defined.

- [ ] **Step 3: Implement `encode_geometry_block`**

Add to `scripts/microsat_to_locator.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pixi run pytest tests/test_microsat_input.py -v -k geometry
```

Expected: 3 PASS.

- [ ] **Step 5: Lint and commit**

```bash
pixi run ruff check scripts/microsat_to_locator.py tests/test_microsat_input.py
git add scripts/microsat_to_locator.py tests/test_microsat_input.py
git commit -m "$(cat <<'EOF'
microsat: add encode_geometry_block (mean_repeat / allele_diff / het)

Output columns interleaved per locus and prefixed with geom_. Missing
genotypes imputed per-locus: mean for mean_repeat and allele_diff,
heterozygosity rate for het. Re-extracted from bb47ced^ scripts/microsat_features.py
with column naming changed to support combined-mode block concatenation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Repeat-norm encoding block

**Files:**
- Modify: `scripts/microsat_to_locator.py` (add `encode_repeat_norm_block`)
- Modify: `tests/test_microsat_input.py` (add repeat_norm tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_microsat_input.py`:

```python
# ---------------------------------------------------------------------------
# Task 5: encode_repeat_norm_block
# ---------------------------------------------------------------------------

def test_repeat_norm_zscore_basic(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,10"},  # mean = 10
        {"sampleID": "s2", "L1": "12,12"},  # mean = 12
        {"sampleID": "s3", "L1": "14,14"},  # mean = 14
    ])
    matrix, col_names = msl.encode_repeat_norm_block(df, ["L1"])
    assert col_names == ["rnorm_L1"]
    # mean=12, std=sqrt(((10-12)^2 + 0 + (14-12)^2)/3) = sqrt(8/3) ≈ 1.633
    expected_std = np.sqrt(8 / 3)
    expected = np.array(
        [[(10 - 12) / expected_std], [0.0], [(14 - 12) / expected_std]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(matrix, expected, atol=1e-5)


def test_repeat_norm_zero_variance_locus(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,10"},
        {"sampleID": "s2", "L1": "10,10"},
    ])
    matrix, _ = msl.encode_repeat_norm_block(df, ["L1"])
    # std = 0; we set z-score to 0 for all samples (no information)
    np.testing.assert_array_equal(matrix, np.zeros((2, 1), dtype=np.float32))


def test_repeat_norm_imputes_missing_to_zero(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,10"},
        {"sampleID": "s2", "L1": "14,14"},
        {"sampleID": "s3", "L1": "NA"},
    ])
    matrix, _ = msl.encode_repeat_norm_block(df, ["L1"])
    # Missing → per-locus mean → 0 after z-scoring
    assert matrix[2, 0] == pytest.approx(0.0, abs=1e-5)


def test_repeat_norm_multi_locus_column_order(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,12", "L2": "20,22"},
        {"sampleID": "s2", "L1": "14,16", "L2": "30,32"},
    ])
    _, col_names = msl.encode_repeat_norm_block(df, ["L1", "L2"])
    assert col_names == ["rnorm_L1", "rnorm_L2"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_microsat_input.py -v -k repeat_norm
```

Expected: 4 ERROR — `encode_repeat_norm_block` not defined.

- [ ] **Step 3: Implement `encode_repeat_norm_block`**

Add to `scripts/microsat_to_locator.py`:

```python
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
    sigma = mean_repeat.std(axis=0)
    safe_sigma = np.where(sigma > 0, sigma, 1.0)
    matrix = ((mean_repeat - mu) / safe_sigma).astype(np.float32)
    matrix[:, sigma == 0] = 0.0  # explicit no-information for zero-variance loci

    col_names = [f"rnorm_{l}" for l in active_loci]
    return matrix, col_names
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pixi run pytest tests/test_microsat_input.py -v -k repeat_norm
```

Expected: 4 PASS.

- [ ] **Step 5: Lint and commit**

```bash
pixi run ruff check scripts/microsat_to_locator.py tests/test_microsat_input.py
git add scripts/microsat_to_locator.py tests/test_microsat_input.py
git commit -m "$(cat <<'EOF'
microsat: add encode_repeat_norm_block (SMM-aware z-score)

New encoding (not present in the dropped scripts): per-locus mean repeat
count, z-scored across all samples. Output dimension is n_loci (one column
per locus), the most compact encoding of the three modes. Zero-variance
loci produce an all-zeros column. Missing genotypes imputed with per-locus
mean — becomes 0 after z-scoring.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Two-column-format auto-detection

**Files:**
- Modify: `scripts/microsat_to_locator.py` (add `detect_format` and `convert_two_column_to_pair`)
- Modify: `tests/test_microsat_input.py` (add detection tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_microsat_input.py`:

```python
# ---------------------------------------------------------------------------
# Task 6: format detection
# ---------------------------------------------------------------------------

PAIR_SEPARATORS = ",/| "  # not testing space-as-separator detection


def test_detect_format_pair(msl):
    df = pd.DataFrame({
        "sampleID": ["s1", "s2"],
        "L1": ["10,11", "12,13"],
    })
    assert msl.detect_format(df) == "pair"


def test_detect_format_two_column(msl):
    df = pd.DataFrame({
        "sampleID": ["s1", "s2"],
        "variant_0": ["10", "12"],
        "variant_1": ["11", "13"],
    })
    assert msl.detect_format(df) == "two_column"


def test_detect_format_two_column_odd_columns_raises(msl):
    df = pd.DataFrame({
        "sampleID": ["s1"],
        "variant_0": ["10"],
        "variant_1": ["11"],
        "variant_2": ["12"],  # odd
    })
    with pytest.raises(ValueError, match="odd"):
        msl.detect_format(df)


def test_convert_two_column_pairs_alleles(msl):
    df = pd.DataFrame({
        "sampleID": ["s1", "s2"],
        "variant_0": ["10", "14"],
        "variant_1": ["11", "16"],
        "variant_2": ["20", "22"],
        "variant_3": ["21", "24"],
    })
    out = msl.convert_two_column_to_pair(df)
    assert list(out.columns) == ["sampleID", "locus_0", "locus_1"]
    assert list(out["locus_0"]) == ["10,11", "14,16"]
    assert list(out["locus_1"]) == ["20,21", "22,24"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_microsat_input.py -v -k "format or two_column or detect_format"
```

Expected: 4 ERROR — `detect_format` / `convert_two_column_to_pair` not defined.

- [ ] **Step 3: Implement detection and conversion**

Add to `scripts/microsat_to_locator.py`:

```python
PAIR_SEPARATORS = (",", "/", "|")


def detect_format(df: pd.DataFrame) -> str:
    """Return ``"pair"`` or ``"two_column"`` based on cell content.

    Pair format: cells contain a separator (``,`` ``/`` ``|``).
    Two-column format: no cells contain pair separators; locus pairs are
    reconstructed from consecutive columns. Raises ``ValueError`` if
    two-column format has an odd number of locus columns.
    """
    locus_cols = [c for c in df.columns if c != "sampleID"]
    has_separator = False
    for c in locus_cols:
        for v in df[c].astype(str):
            if any(s in v for s in PAIR_SEPARATORS):
                has_separator = True
                break
        if has_separator:
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
        raise ValueError("Cannot convert: odd number of locus columns.")

    out = pd.DataFrame({"sampleID": df["sampleID"].values})
    for i in range(0, len(locus_cols), 2):
        c1, c2 = locus_cols[i], locus_cols[i + 1]
        out[f"locus_{i // 2}"] = df[c1].astype(str) + "," + df[c2].astype(str)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pixi run pytest tests/test_microsat_input.py -v -k "format or two_column or detect_format"
```

Expected: 4 PASS.

- [ ] **Step 5: Lint and commit**

```bash
pixi run ruff check scripts/microsat_to_locator.py tests/test_microsat_input.py
git add scripts/microsat_to_locator.py tests/test_microsat_input.py
git commit -m "$(cat <<'EOF'
microsat: add two-column-format auto-detection

detect_format() checks whether any cell contains a pair separator; if not,
treats consecutive columns as diploid pairs (the SLiM dataset's shape).
convert_two_column_to_pair() merges them into pair format with
locus_<i> names. Odd column count raises a clear ValueError.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: CLI + main() composition

**Files:**
- Modify: `scripts/microsat_to_locator.py` (replace stub `main` with full CLI)
- Modify: `tests/test_microsat_input.py` (add end-to-end subprocess tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_microsat_input.py`:

```python
# ---------------------------------------------------------------------------
# Task 7: end-to-end CLI
# ---------------------------------------------------------------------------

def _write_pair_tsv(path: Path) -> int:
    """Write a small pair-format input. Returns n_samples."""
    rows = [
        ["sampleID", "L1", "L2"],
        ["s1", "10,11", "20,22"],
        ["s2", "10,10", "20,20"],
        ["s3", "11,12", "22,24"],
        ["s4", "10,12", "NA"],
    ]
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    return len(rows) - 1


def test_cli_default_features_combines_all_three_modes(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    _write_pair_tsv(inp)
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out)
    assert proc.returncode == 0, proc.stderr
    df = pd.read_csv(out, sep="\t")
    cols = list(df.columns)
    assert cols[0] == "sampleID"
    assert any(c.startswith("dosage_") for c in cols)
    assert any(c.startswith("geom_") for c in cols)
    assert any(c.startswith("rnorm_") for c in cols)


def test_cli_features_subset_dosage_only(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    _write_pair_tsv(inp)
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out, "--features", "dosage")
    assert proc.returncode == 0, proc.stderr
    df = pd.read_csv(out, sep="\t")
    cols = list(df.columns)
    assert cols[0] == "sampleID"
    for c in cols[1:]:
        assert c.startswith("dosage_"), c


def test_cli_unknown_feature_errors(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    _write_pair_tsv(inp)
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out, "--features", "bogus")
    assert proc.returncode != 0
    assert "bogus" in (proc.stderr + proc.stdout).lower()


def test_cli_two_column_format_works(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    rows = [
        ["sampleID", "variant_0", "variant_1", "variant_2", "variant_3"],
        ["s1", "10", "11", "20", "22"],
        ["s2", "10", "10", "20", "20"],
        ["s3", "11", "12", "22", "24"],
    ]
    inp.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out, "--features", "geometry")
    assert proc.returncode == 0, proc.stderr
    df = pd.read_csv(out, sep="\t")
    assert len(df) == 3
    # Two reconstructed loci × 3 geometry columns each = 6 feature cols + sampleID
    assert df.shape[1] == 7


def test_cli_combined_mode_column_order_is_dosage_then_geom_then_rnorm(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    _write_pair_tsv(inp)
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out)
    assert proc.returncode == 0, proc.stderr
    df = pd.read_csv(out, sep="\t")
    cols = [c for c in df.columns if c != "sampleID"]
    prefixes = [c.split("_")[0] for c in cols]
    last_dosage = max(i for i, p in enumerate(prefixes) if p == "dosage")
    first_geom = min(i for i, p in enumerate(prefixes) if p == "geom")
    last_geom = max(i for i, p in enumerate(prefixes) if p == "geom")
    first_rnorm = min(i for i, p in enumerate(prefixes) if p == "rnorm")
    assert last_dosage < first_geom < last_geom < first_rnorm


def test_cli_writes_encoding_report(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    _write_pair_tsv(inp)
    out = tmp_path / "feat.tsv"
    enc = tmp_path / "encoding.tsv"
    proc = run_script(
        "--microsat", inp, "--out", out,
        "--features", "dosage,geometry",
        "--report_encoding", enc,
    )
    assert proc.returncode == 0, proc.stderr
    df = pd.read_csv(enc, sep="\t")
    assert {"feature_type", "locus", "column_name"}.issubset(df.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_microsat_input.py -v -k cli_
```

Expected: 6 FAIL — `main()` raises `SystemExit("CLI not yet wired ...")`.

- [ ] **Step 3: Implement the CLI**

Replace the stub `main()` in `scripts/microsat_to_locator.py` with:

```python
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
        help=f"Comma-separated subset of {VALID_FEATURES}. Default: all three.",
    )
    p.add_argument("--min_allele_freq", type=float, default=0.01)
    p.add_argument("--max_locus_missing", type=float, default=1.0)
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
    features = parse_feature_list(args.features)

    df = pd.read_csv(args.microsat, sep="\t", dtype=str)
    if "sampleID" not in df.columns:
        print("Input must have a 'sampleID' column.", file=sys.stderr)
        return 2

    fmt = detect_format(df)
    print(f"Detected input format: {fmt}", flush=True)
    if fmt == "two_column":
        df = convert_two_column_to_pair(df)

    df = df.set_index("sampleID")
    loci = list(df.columns)
    print(f"Samples: {len(df)}, Loci: {len(loci)}", flush=True)
    print(f"Requested features: {features}", flush=True)

    catalog = build_allele_catalog(df, loci, args.min_allele_freq, args.max_locus_missing)
    active_loci = [l for l in loci if catalog[l]]
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
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
pixi run pytest tests/test_microsat_input.py -v
```

Expected: all tests PASS (≥ 30 cases across the 7 task sections).

- [ ] **Step 5: Lint and commit**

```bash
pixi run ruff check scripts/microsat_to_locator.py tests/test_microsat_input.py
git add scripts/microsat_to_locator.py tests/test_microsat_input.py
git commit -m "$(cat <<'EOF'
microsat: wire the CLI — argparse, format detection, block composition

Replaces the stub main() with the full pipeline: parse args, read TSV,
detect input format (pair vs two-column), build allele catalog, run the
requested encoding blocks (dosage / geometry / repeat_norm) in fixed
order, concatenate, write output and optional encoding report.

Adds 6 end-to-end subprocess tests covering default mode, single-mode
subset, unknown-feature error path, two-column-format, combined-mode
column ordering, and encoding report.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Smoke run on the SLiM dataset

**Files:**
- No new files. Runs the script as-is and writes to `out/microsat_validation/smoke/`.

- [ ] **Step 1: Verify the SLiM input is reachable**

```bash
ls -L test_data/locator_microsats/notebooks/microsat_variants.txt
ls -L test_data/locator_microsats/notebooks/microsat_spatial_location.txt
wc -l test_data/locator_microsats/notebooks/microsat_variants.txt
head -1 test_data/locator_microsats/notebooks/microsat_variants.txt | awk -F"\t" '{print "ncols="NF}'
```

Expected: both files exist (resolve through symlink); 201 lines (header + 200 individuals); 201 columns (sampleID + 200 variant columns).

- [ ] **Step 2: Run the converter on the full dataset, all three modes**

```bash
mkdir -p out/microsat_validation/smoke
pixi run python scripts/microsat_to_locator.py \
    --microsat test_data/locator_microsats/notebooks/microsat_variants.txt \
    --out out/microsat_validation/smoke/features_all.tsv \
    --features dosage,geometry,repeat_norm \
    --report_encoding out/microsat_validation/smoke/encoding.tsv \
    2>&1 | tee out/microsat_validation/smoke/build_log.txt
```

Expected log lines: `Detected input format: two_column`; `Samples: 200, Loci: 100`; `Wrote out/microsat_validation/smoke/features_all.tsv: 200 samples × N features` where N is in the low thousands (dosage block dominates).

- [ ] **Step 3: Verify shape and basic value sanity**

```bash
wc -l out/microsat_validation/smoke/features_all.tsv  # should print 201 (header + 200)
head -1 out/microsat_validation/smoke/features_all.tsv | awk -F"\t" '{print "ncols="NF}'
pixi run python -c "
import pandas as pd
df = pd.read_csv('out/microsat_validation/smoke/features_all.tsv', sep='\t')
print('shape:', df.shape)
print('first 3 dosage cols:', [c for c in df.columns if c.startswith('dosage_')][:3])
print('first 3 geom cols:', [c for c in df.columns if c.startswith('geom_')][:3])
print('first 3 rnorm cols:', [c for c in df.columns if c.startswith('rnorm_')][:3])
print('rnorm column means (should be ~0):', df.filter(like='rnorm_').mean().abs().max())
print('rnorm column stds (should be ~1):', df.filter(like='rnorm_').std().describe()[['min','max']])
print('any NaN:', df.isna().any().any())
"
```

Expected: shape `(200, ~1300)`; column-prefix counts match what the encoders produce; `rnorm_` columns have mean ≈ 0 and std ≈ 1; no NaN values.

- [ ] **Step 4: Run a single-mode pass (each mode separately) for the validation runner to consume**

```bash
for mode in dosage geometry repeat_norm; do
    pixi run python scripts/microsat_to_locator.py \
        --microsat test_data/locator_microsats/notebooks/microsat_variants.txt \
        --out out/microsat_validation/smoke/features_${mode}.tsv \
        --features ${mode} \
        2>&1 | tail -3
done
ls -lh out/microsat_validation/smoke/features_*.tsv
```

Expected: three additional files written; the dosage file is the largest; the repeat_norm file is the smallest (200 individuals × 100 features ≈ small).

- [ ] **Step 5: Document the smoke run** (no commit needed; outputs are gitignored under `out/`)

The artifacts under `out/microsat_validation/smoke/` are local-only. Task 9's runner will regenerate them inside its per-fold directories.

---

## Task 9: Validation runner — random 80/20 + k-fold CV

**Files:**
- Create: `validation/run_microsat_modes.py`

This runner generates the feature matrix once per mode, performs random 80/20 (single seed) and 5-fold CV (3 seeds) splits, calls the existing `locator` CLI for each fold, and writes per-fold result JSONs.

- [ ] **Step 1: Inspect the existing LOSO runner to mirror its conventions**

```bash
sed -n '40,140p' validation/run_loso.py
```

Note the `_run_one_fold` pattern (subprocess to `locator`, parse predlocs, compute metric, write `fold_result.json`) and `validation.common.stream_run` for streaming subprocess output.

- [ ] **Step 2: Create the runner script**

Create `validation/run_microsat_modes.py`:

```python
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
import shutil
import subprocess
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
) -> tuple[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / "run"
    cmd = [
        "locator",
        "--matrix", str(features_tsv),
        "--sample_data", str(sample_data),
        "--out", str(out_prefix),
        "--gpu_number", str(gpu),
        "--seed", "42",
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
    p = pred.loc[holdout_ids, ["x", "y"]].to_numpy(dtype=np.float64)
    t = tru.to_numpy(dtype=np.float64)
    err = euclidean(p, t)
    return {
        "n_held_out": int(len(holdout_ids)),
        "median_error": float(np.median(err)),
        "mean_error": float(err.mean()),
        "max_error": float(err.max()),
    }


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
            fold_label=f"smoke_{mode}", gpu=args.gpu,
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
                    fold_label=f"kfold_{mode}_s{seed}_f{fold_idx}", gpu=args.gpu,
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
```

- [ ] **Step 3: Lint**

```bash
pixi run ruff check validation/run_microsat_modes.py
```

- [ ] **Step 4: Run a quick dry sanity check (one mode, 1 seed) to confirm it executes end-to-end before committing**

```bash
mkdir -p out/microsat_validation
pixi run python -m validation.run_microsat_modes \
    --microsat test_data/locator_microsats/notebooks/microsat_variants.txt \
    --spatial test_data/locator_microsats/notebooks/microsat_spatial_location.txt \
    --out_dir out/microsat_validation \
    --modes dosage \
    --kfold_seeds 1 \
    --gpu 0 \
    2>&1 | tail -40
```

Expected: smoke and 5 kfold fold_result.json files written under `out/microsat_validation/`. Run takes ~3 minutes per fold × 6 folds (1 smoke + 5 kfold) = ~18 minutes. If a fold fails, the JSON contains an `"error"` key — inspect `out/.../locator.log`.

- [ ] **Step 5: Commit**

```bash
git add validation/run_microsat_modes.py
git commit -m "$(cat <<'EOF'
validation: add microsat random-split + k-fold CV runner

Mirrors validation/run_loso.py's pattern but adapted for arena data:
random 80/20 single-seed smoke + k-fold CV (k=5, 3 seeds default).
Generates feature matrix once per mode, blanks held-out coords in
sample_data.txt, calls the locator CLI per fold, parses predicted vs
true coords, and writes fold_result.json. Euclidean distance metric
in arena units (no coastline / range mask — SLiM arena is 2D unconstrained).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Validation runner — polynomial experimental mode

**Files:**
- Create: `validation/run_microsat_polynomial.py`

This runner is functionally identical to `run_microsat_modes.py` for one mode (`polynomial`), except the feature matrix is built by applying `PolynomialFeatures(degree=2, interaction_only=True)` to the dosage matrix, then `PCA(n_components=200)`. Lives in a separate script to keep sklearn imports out of the merge-zone path.

- [ ] **Step 1: Create the runner**

Create `validation/run_microsat_polynomial.py`:

```python
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

from validation import common  # noqa: E402
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
    print(f"PCA: {Xp.shape[1]} → {Xr.shape[1]} components, "
          f"cumvar={pca.explained_variance_ratio_.sum():.4f}", flush=True)

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
        features_tsv=poly_tsv, sample_data=sd, out_dir=smoke_dir,
        fold_label="smoke_polynomial", gpu=args.gpu,
    )
    if rc == 0 and pred.exists():
        m = evaluate_fold(predlocs=pred, truth=truth, holdout_ids=held)
        (smoke_dir / "fold_result.json").write_text(json.dumps(
            {"mode": "polynomial", "split": "smoke80_20", "seed": args.smoke_seed, **m}
        ))
    else:
        (smoke_dir / "fold_result.json").write_text(json.dumps(
            {"mode": "polynomial", "split": "smoke80_20", "seed": args.smoke_seed,
             "error": f"rc={rc}, predlocs_exists={pred.exists()}"}
        ))

    # 4. K-fold CV.
    for seed in seeds:
        for fold_idx, (_, test_idx) in enumerate(kfold_indices(n, args.kfold_k, seed)):
            held = [sample_ids[i] for i in test_idx]
            kdir = args.out_dir / "kfold" / "polynomial" / f"seed{seed}_fold{fold_idx}"
            kdir.mkdir(parents=True, exist_ok=True)
            sd = kdir / "sample_data.txt"
            truth = write_sample_data(Path(args.spatial), held, sd)
            rc, pred = run_locator_fold(
                features_tsv=poly_tsv, sample_data=sd, out_dir=kdir,
                fold_label=f"kfold_poly_s{seed}_f{fold_idx}", gpu=args.gpu,
            )
            if rc == 0 and pred.exists():
                m = evaluate_fold(predlocs=pred, truth=truth, holdout_ids=held)
                (kdir / "fold_result.json").write_text(json.dumps(
                    {"mode": "polynomial", "split": "kfold", "seed": seed,
                     "fold": fold_idx, **m}
                ))
            else:
                (kdir / "fold_result.json").write_text(json.dumps(
                    {"mode": "polynomial", "split": "kfold", "seed": seed,
                     "fold": fold_idx,
                     "error": f"rc={rc}, predlocs_exists={pred.exists()}"}
                ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint**

```bash
pixi run ruff check validation/run_microsat_polynomial.py
```

- [ ] **Step 3: Smoke-execute (1 seed, the smoke fold only) to confirm it runs**

```bash
pixi run python -c "
from pathlib import Path
from validation.run_microsat_polynomial import expand_polynomial
expand_polynomial(
    Path('out/microsat_validation/features_dosage.tsv'),
    Path('out/microsat_validation/features_polynomial_smoke.tsv'),
    n_components=200,
)
"
```

Expected: prints expansion shape and PCA cumvar; writes the polynomial features TSV.

- [ ] **Step 4: Commit**

```bash
git add validation/run_microsat_polynomial.py
git commit -m "$(cat <<'EOF'
validation: add polynomial experimental runner (validation zone only)

Pipeline: scripts/microsat_to_locator.py --features dosage → dosage matrix
→ PolynomialFeatures(degree=2, interaction_only=True) → PCA(200 components)
→ standard ReLocator runs through the same harness as run_microsat_modes.
Re-uses the helper functions from run_microsat_modes (build_features,
evaluate_fold, kfold_indices, random_split_indices, run_locator_fold,
write_sample_data) to avoid duplication.

Lives in a separate script (not in the user-facing --features flag)
because the n=200 / 500k-pre-PCA-feature regime is expected to overfit;
this runner produces the receipts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Run the full validation arc

**Files:**
- No new files. Runs Tasks 9 + 10 to completion against the SLiM dataset.

- [ ] **Step 1: Run all four modes (3 user + 1 polynomial) full sweep**

```bash
pixi run python -m validation.run_microsat_modes \
    --microsat test_data/locator_microsats/notebooks/microsat_variants.txt \
    --spatial test_data/locator_microsats/notebooks/microsat_spatial_location.txt \
    --out_dir out/microsat_validation \
    --gpu 0 \
    2>&1 | tee out/microsat_validation/run_modes.log

pixi run python -m validation.run_microsat_polynomial \
    --microsat test_data/locator_microsats/notebooks/microsat_variants.txt \
    --spatial test_data/locator_microsats/notebooks/microsat_spatial_location.txt \
    --out_dir out/microsat_validation \
    --gpu 1 \
    2>&1 | tee out/microsat_validation/run_polynomial.log
```

Expected total runtime: 3 modes × (1 smoke + 5 folds × 3 seeds) + 1 polynomial mode × same = 64 ReLocator fits. At ~3 min each that is ~3.2 hours wall-clock. The two modes runners can run on separate GPUs in parallel — `--gpu 0` and `--gpu 1` to keep them isolated.

- [ ] **Step 2: Verify all per-fold JSONs were written**

```bash
find out/microsat_validation -name 'fold_result.json' | wc -l
find out/microsat_validation -name 'fold_result.json' -exec grep -l '"error"' {} \;
```

Expected: 64 JSON files; the second command should print nothing (no errors). If any folds errored, inspect `out/.../locator.log` and re-run those folds individually.

- [ ] **Step 3: Spot-check one JSON per mode for sanity**

```bash
for mode in dosage geometry repeat_norm polynomial; do
    echo "=== $mode (smoke) ==="
    cat out/microsat_validation/smoke/$mode/fold_result.json
    echo
done
```

Expected: each prints `{"mode": "...", "split": "smoke80_20", "seed": 42, "n_held_out": 40, "median_error": <float>, "mean_error": <float>, "max_error": <float>}`. Errors should be small relative to the arena diameter (~45 units); centroid baseline is ~13 (the std of arena coords).

- [ ] **Step 4: No commit yet** — outputs live under `out/` which is gitignored. Summaries are committed in Task 12.

---

## Task 12: Summarize — figures + summary.md

**Files:**
- Create: `validation/summarize_microsat.py`
- Create: `validation/summary/microsat_summary.md` (script-generated)
- Create: `validation/summary/microsat_kfold.tsv` (script-generated)
- Create: `validation/figures/microsat_modes.png` (script-generated)
- Create: `validation/figures/microsat_scatter.png` (script-generated)

- [ ] **Step 1: Create the summarizer**

Create `validation/summarize_microsat.py`:

```python
#!/usr/bin/env python3
"""Aggregate microsat validation per-fold JSONs → kfold.tsv, summary.md, figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ALL_MODES = ("dosage", "geometry", "repeat_norm", "polynomial")


def load_fold_jsons(out_dir: Path, mode: str) -> pd.DataFrame:
    rows = []
    for p in sorted((out_dir / "kfold" / mode).glob("seed*_fold*/fold_result.json")):
        rows.append(json.loads(p.read_text()))
    smoke_p = out_dir / "smoke" / mode / "fold_result.json"
    if smoke_p.exists():
        rows.append(json.loads(smoke_p.read_text()))
    return pd.DataFrame(rows)


def write_kfold_tsv(out_dir: Path, summary_dir: Path) -> pd.DataFrame:
    frames = []
    for mode in ALL_MODES:
        df = load_fold_jsons(out_dir, mode)
        if not df.empty:
            df["mode"] = mode
            frames.append(df)
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(summary_dir / "microsat_kfold.tsv", sep="\t", index=False)
    return full


def make_modes_figure(full: pd.DataFrame, fig_path: Path) -> None:
    if full.empty:
        return
    kfold = full[full["split"] == "kfold"].copy()
    agg = (
        kfold.groupby("mode")["median_error"]
        .agg(["mean", "std"])
        .reindex(ALL_MODES)
        .dropna()
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(agg))
    ax.bar(x, agg["mean"], yerr=agg["std"], capsize=5, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=15)
    ax.set_ylabel("Median Euclidean error (arena units)")
    ax.set_title("Microsat: median per-fold error by encoding mode\n(k=5, 3 seeds, mean ± std)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def make_scatter_figure(out_dir: Path, spatial_path: Path, fig_path: Path) -> None:
    spatial = pd.read_csv(spatial_path, sep="\t").set_index("sampleID")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, mode in zip(axes, ALL_MODES):
        smoke_dir = out_dir / "smoke" / mode
        pred_path = smoke_dir / "run_predlocs.txt"
        if not pred_path.exists():
            ax.set_title(f"{mode}: no predlocs")
            continue
        pred = pd.read_csv(pred_path).set_index("sampleID")
        held = sorted(set(pred.index) & set(spatial.index))
        ax.scatter(spatial.loc[held, "x"], pred.loc[held, "x"], s=14, alpha=0.7, label="x")
        ax.scatter(spatial.loc[held, "y"], pred.loc[held, "y"], s=14, alpha=0.7, label="y", marker="^")
        lo = min(spatial[["x", "y"]].min().min(), pred[["x", "y"]].min().min())
        hi = max(spatial[["x", "y"]].max().max(), pred[["x", "y"]].max().max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.set_title(f"{mode} (smoke 80/20)")
        ax.set_xlabel("true coord")
        ax.set_ylabel("predicted coord")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def render_summary_md(full: pd.DataFrame, summary_md: Path) -> None:
    if full.empty:
        summary_md.write_text("# Microsat validation summary\n\nNo fold results found.\n")
        return
    kfold = full[full["split"] == "kfold"]
    agg = (
        kfold.groupby("mode")
        .agg(median_mean=("median_error", "mean"),
             median_std=("median_error", "std"),
             mean_mean=("mean_error", "mean"),
             mean_std=("mean_error", "std"))
        .reindex(ALL_MODES)
        .dropna()
    )

    lines = [
        "# Microsat validation summary",
        "",
        "Validation arc for the `microsatellites` branch. SLiM-simulated dataset",
        "(200 individuals × 100 loci, arena coordinates). Metric: Euclidean",
        "distance in arena units. K-fold CV: k=5, 3 seeds, mean ± std reported.",
        "",
        "## Results",
        "",
        "| Mode | Median error (mean ± std) | Mean error (mean ± std) |",
        "|---|---|---|",
    ]
    for mode in agg.index:
        r = agg.loc[mode]
        lines.append(
            f"| `{mode}` | {r['median_mean']:.3f} ± {r['median_std']:.3f} | "
            f"{r['mean_mean']:.3f} ± {r['mean_std']:.3f} |"
        )

    lines += [
        "",
        "## Smoke (random 80/20, seed 42)",
        "",
        "| Mode | n_held_out | median_error | mean_error | max_error |",
        "|---|---|---|---|---|",
    ]
    smoke = full[full["split"] == "smoke80_20"]
    for _, r in smoke.iterrows():
        lines.append(
            f"| `{r['mode']}` | {int(r['n_held_out'])} | "
            f"{r['median_error']:.3f} | {r['mean_error']:.3f} | {r['max_error']:.3f} |"
        )

    lines += [
        "",
        "## Recommendation",
        "",
        "TODO: fill in after the run — pick the lowest-error mode for the recommendation.",
        "",
        "## Known limitations",
        "",
        "Validation here used the SLiM-simulated dataset bundled with the original",
        "locator paper. The following microsat-specific concerns are NOT covered by",
        "this arc and require a real-data follow-up:",
        "",
        "- Homoplasy (alleles of identical length but different ancestry).",
        "- Null alleles (PCR drop-out producing apparent homozygotes).",
        "- Sizing artefacts (off-by-one stutter calls).",
        "- Real-data ascertainment biases.",
        "",
        "See `CLAUDE.md` 'Microsats (active)' TODO list and",
        "`validation/docs/specs/2026-05-04-microsatellite-input-extension-design.md`",
        "for the full design rationale.",
        "",
        "## Figures",
        "",
        "- `validation/figures/microsat_modes.png` — bar chart of per-mode median",
        "  error (k-fold mean ± std).",
        "- `validation/figures/microsat_scatter.png` — predicted-vs-actual coords",
        "  for the smoke fold, one panel per mode.",
        "",
    ]
    summary_md.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--spatial", required=True)
    p.add_argument("--summary_dir", type=Path,
                   default=Path("validation/summary"))
    p.add_argument("--figures_dir", type=Path,
                   default=Path("validation/figures"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    full = write_kfold_tsv(args.out_dir, args.summary_dir)
    make_modes_figure(full, args.figures_dir / "microsat_modes.png")
    make_scatter_figure(args.out_dir, Path(args.spatial),
                        args.figures_dir / "microsat_scatter.png")
    render_summary_md(full, args.summary_dir / "microsat_summary.md")
    print(f"Wrote summary, kfold tsv, and figures under {args.summary_dir} and {args.figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint**

```bash
pixi run ruff check validation/summarize_microsat.py
```

- [ ] **Step 3: Run summarize**

```bash
pixi run python -m validation.summarize_microsat \
    --out_dir out/microsat_validation \
    --spatial test_data/locator_microsats/notebooks/microsat_spatial_location.txt
```

Expected: writes `validation/summary/microsat_kfold.tsv`, `validation/summary/microsat_summary.md`, `validation/figures/microsat_modes.png`, `validation/figures/microsat_scatter.png`.

- [ ] **Step 4: Manually edit the "Recommendation" section in `validation/summary/microsat_summary.md`**

Read the rendered table, pick the lowest-error mode, write 2–3 sentences explaining the recommendation. The `TODO` placeholder is intentional in the template — it gets replaced before commit. Example shape:

> The `geometry` mode produced the lowest median error across all k-fold seeds
> (X.XX ± Y.YY arena units), beating `dosage` (Z.ZZ ± ...) and `repeat_norm`
> (...). Polynomial degree-2 + PCA performed worst (...), confirming the
> design hypothesis that explicit pairwise expansion overfits at n=200.

- [ ] **Step 5: Commit**

```bash
pixi run ruff check validation/summarize_microsat.py
git add validation/summarize_microsat.py validation/summary/microsat_summary.md validation/summary/microsat_kfold.tsv validation/figures/microsat_modes.png validation/figures/microsat_scatter.png
git commit -m "$(cat <<'EOF'
validation: add summarize_microsat + run-output evidence

Aggregates per-fold result JSONs into:
- validation/summary/microsat_kfold.tsv (raw per-fold rows)
- validation/summary/microsat_summary.md (mean ± std table, smoke results,
  manually-edited recommendation, declared known limitations)
- validation/figures/microsat_modes.png (per-mode bar chart)
- validation/figures/microsat_scatter.png (predicted-vs-actual smoke scatter)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: User-facing docs (`docs/microsatellites.md`)

**Files:**
- Create: `docs/microsatellites.md`

- [ ] **Step 1: Write the doc**

Create `docs/microsatellites.md`:

```markdown
# Microsatellite Input

ReLocator accepts microsatellite genotype data through a preprocessing
converter (`scripts/microsat_to_locator.py`) that produces a continuous
float matrix consumed by the patched `--matrix` loader (see also
`docs/genotype_likelihoods.md` for the parallel GL pipeline).

## Input formats

Two formats are accepted, auto-detected from cell content:

### Pair format (preferred)

Tab-delimited, first column `sampleID`, remaining columns one per locus.
Each cell holds a diploid genotype as `allele1<sep>allele2` where `<sep>`
is comma, slash, pipe, or space. Missing genotypes: `NA`, `NaN`, `.`,
empty, `0,0`, `0/0`. Single-value cells (no separator) are interpreted
as homozygote shorthand.

```
sampleID    D3S1358    vWA      FGA
sample_001  15,16      17,18    22,22
sample_002  14,15      16,18    20,24
sample_003  NA         17,17    22,26
```

### Two-column format (legacy / simulation)

Tab-delimited, first column `sampleID`, then 2 × n_loci columns where
consecutive pairs (`variant_0`, `variant_1`) form the diploid genotype
at locus 0. Used by the SLiM simulation that ships with the original
locator paper. Detected automatically when no cell contains a pair
separator.

```
sampleID    variant_0  variant_1  variant_2  variant_3
sample_001  10         11         20         22
sample_002  10         10         20         20
```

## Encoding modes

Selected via `--features <subset>` (default = all three). Combined-mode
output concatenates blocks in fixed order: `dosage`, `geometry`,
`repeat_norm`.

| Mode | Output dim | What it captures |
|---|---|---|
| `dosage` | K = sum(k_l) | One-hot allele counts per (locus, allele) — categorical baseline. Equivalent to multi-allelic SNP encoding. |
| `geometry` | 3 × n_loci | Per-locus `mean_repeat`, `allele_diff`, `het` — encodes stepwise mutation: alleles of similar lengths are more closely related than alleles of dissimilar lengths. |
| `repeat_norm` | n_loci | Per-locus z-scored mean repeat — most compact continuous encoding. Under SMM at mutation-drift equilibrium, within-population repeat distributions are approximately Gaussian. |

`k_l` = number of distinct alleles at locus `l` after the
`--min_allele_freq` filter.

## CLI

```
python scripts/microsat_to_locator.py \
    --microsat <input.tsv> \
    --out <output.tsv> \
    --features dosage,geometry,repeat_norm \
    --min_allele_freq 0.01 \
    --max_locus_missing 1.0 \
    --report_encoding <encoding.tsv>
```

| Option | Required | Default | Description |
|---|---|---|---|
| `--microsat` | yes | — | Input TSV (pair or two-column format) |
| `--out` | yes | — | Output dosage TSV for ReLocator `--matrix` |
| `--features` | no | `dosage,geometry,repeat_norm` | Comma-separated subset of the three modes |
| `--min_allele_freq` | no | `0.01` | Drop alleles below this frequency (dosage mode only) |
| `--max_locus_missing` | no | `1.0` | Warn when locus missing rate exceeds this |
| `--report_encoding` | no | — | Optional column→source mapping TSV |

## Example

```bash
# 1. Convert microsat genotypes to a feature matrix
python scripts/microsat_to_locator.py \
    --microsat data/microsats.tsv \
    --out data/microsat_features.tsv \
    --features geometry,repeat_norm

# 2. Run ReLocator
locator \
    --matrix data/microsat_features.tsv \
    --sample_data data/sample_data.txt \
    --out out/microsat_run/run1
```

## Validation

The encoder was validated against the SLiM-simulated dataset that ships
with the original locator paper (200 individuals × 100 loci, arena
coordinates). See `validation/summary/microsat_summary.md` for results.

## Known limitations

The following microsat-specific concerns are NOT addressed by the
current encoder; they are tracked as TODOs and will be revisited in a
follow-up branch with a real-data validation arc.

- **Homoplasy** — alleles of identical length but different ancestry.
- **Null alleles** — PCR drop-out producing apparent homozygotes.
- **Sizing artefacts** — off-by-one stutter calls.
- **Native ReLocator CLI flag** (`--microsat <input>` directly).
- **Per-fold scaling stats persistence** — the validation harness runs
  the encoder once on the full dataset; per-fold μ/σ for `repeat_norm`
  is a follow-up if it proves consequential at larger N.
```

- [ ] **Step 2: Commit**

```bash
git add docs/microsatellites.md
git commit -m "$(cat <<'EOF'
docs: add docs/microsatellites.md user-facing guide

Mirrors docs/genotype_likelihoods.md's structure: input formats, encoding
modes with biological rationale, CLI table, example invocation, validation
pointer, known limitations.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: CLAUDE.md updates + final review + push

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Read the current CLAUDE.md microsat sections**

```bash
grep -n -B1 -A8 'Microsats' CLAUDE.md | head -60
```

Note the line numbers for the "Microsats (deferred)" section, the Scripts Summary table, and the Branch state section.

- [ ] **Step 2: Replace "Microsats (deferred)" with "Microsats (active)"**

In `CLAUDE.md`, find the section that begins with `## Microsats (deferred)` and replace it with:

```markdown
## Microsats (active)

The microsat input pipeline lives on the `microsatellites` branch (stacked
on `genotype_likelihoods`). One converter script —
`scripts/microsat_to_locator.py` — emits a continuous float matrix consumed
by ReLocator's patched `--matrix` loader. Three composable feature modes:

- `dosage` — one-hot allele counts (categorical baseline).
- `geometry` — per-locus `mean_repeat`, `allele_diff`, `het` (stepwise
  mutation geometry).
- `repeat_norm` — per-locus z-scored mean repeat (SMM-aware compact).

User-facing guide: `docs/microsatellites.md`. Validation evidence:
`validation/summary/microsat_summary.md` and `validation/figures/microsat_*.png`.

### Open TODOs (microsat)

- **TODO (Correctness):** Real-data validation against a published microsat
  study. The current arc validated only against SLiM-simulated arena data;
  homoplasy, null alleles, and sizing artefacts do not exist by construction.
- **TODO (Feature):** Native ReLocator CLI flag (`--microsat <input>` direct,
  skipping the explicit converter).
- **TODO (Feature):** Polynomial / tensor-decomposition modes as user-facing
  options. Currently restricted to validation-zone evidence (see
  `validation/run_microsat_polynomial.py`).
- **TODO (Correctness):** Per-fold scaling stats persistence for `repeat_norm`.
  v1 runs the encoder once on the full dataset; consequential only at larger N.
```

- [ ] **Step 3: Add a row to the Scripts Summary table**

Find the "Scripts Summary" table (lines roughly under `## Scripts Summary`) and add:

```markdown
| `microsat_to_locator.py` | microsat genotype TSV (pair or two-column) | continuous feature matrix TSV | `--features dosage,geometry,repeat_norm` selects modes |
```

- [ ] **Step 4: Update Branch state**

Find the `### Branch state` subsection. Update the active feature branch line:

> Active feature branch: `microsatellites` (stacked on `genotype_likelihoods` HEAD `2ee683d`) → push to `fork` remote (`git@github.com:stsmall/ReLocator.git`). PR target: `kr-colab/ReLocator:main`. The PR description must declare dependency on the GL PR; cannot merge until GL merges.

Add a new bullet under "Two zones on this branch":

> - **Microsat zone**: Merge zone adds `scripts/microsat_to_locator.py`, `tests/test_microsat_input.py`, `docs/microsatellites.md`, CLAUDE.md edits. Validation zone adds `validation/run_microsat_modes.py`, `validation/run_microsat_polynomial.py`, `validation/summarize_microsat.py`, `validation/summary/microsat_*`, `validation/figures/microsat_*.png`.

- [ ] **Step 5: Add operational notes**

Find the `### Test data` subsection. Append:

> - **Microsat test data**: `test_data/locator_microsats/notebooks/microsat_variants.txt` (200 individuals × 100 loci, two-column format with consecutive `variant_<i>` columns forming pairs) and `microsat_spatial_location.txt` (arena `x,y,sampleID`, no lat/lon — SLiM simulation). Source: original locator paper's notebook recipe.

- [ ] **Step 6: Verify the diff is clean**

```bash
git diff CLAUDE.md | head -120
```

Confirm: deferred section is replaced, Scripts Summary has the new row, Branch state is updated, microsat test data note is added.

- [ ] **Step 7: Final test sweep**

```bash
pixi run pytest tests/test_microsat_input.py -v
pixi run pytest tests/test_input_extensions.py -v   # GL tests still pass
pixi run ruff check scripts/microsat_to_locator.py tests/test_microsat_input.py validation/run_microsat_modes.py validation/run_microsat_polynomial.py validation/summarize_microsat.py
```

Expected: all microsat tests pass, all GL tests still pass, ruff is clean.

- [ ] **Step 8: Commit and push to fork**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: flip CLAUDE.md microsats to active; add operational notes

- Replace "Microsats (deferred)" with "Microsats (active)" describing the
  one-script converter, three feature modes, and validation summary pointers.
- Declare four open TODOs (real-data validation, native CLI flag, polynomial
  user-facing mode, per-fold scaling stats).
- Add microsat_to_locator.py row to Scripts Summary.
- Update Branch state to note the stacked-PR shape on microsatellites.
- Add SLiM microsat test data note under Test data.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

git log --oneline -8
git push fork microsatellites
```

Expected: 14 commits ahead of `genotype_likelihoods`; push succeeds. The PR can then be drafted via `gh` (HTTPS token expired — use the GitHub web UI to draft against `kr-colab/ReLocator:main` with description noting dependency on the GL PR), or via `gh` after refreshing the token.

---

## Self-review checklist

Run through this after the plan is committed; fix any gaps inline.

- [ ] **Spec coverage**: every section in the spec has a corresponding task or step.
  - Goals 1–4 → Tasks 1–7 (script), 8 (smoke), 9–11 (validation), 12 (summary), 13 (docs).
  - Non-goals (real-data, native flag, polynomial as user-facing, loader changes) → declared in CLAUDE.md (Task 14) and docs/microsatellites.md (Task 13).
  - Branch & PR shape → Task 14 push step.
  - Test data → Task 8 verification step.
  - CLI → Task 7.
  - Input formats → Tasks 6 (detection), 7 (CLI test for two-column).
  - Three encoding modes → Tasks 3, 4, 5.
  - Missing-data imputation → tested in each encoder task.
  - Known minor leakage → documented in spec; harness runs script once per mode (Task 9 build_features). Not engineered around in v1.
  - Why these three modes (rationale) → docs/microsatellites.md (Task 13), CLAUDE.md (Task 14).
  - Validation arc 6 steps → Tasks 8 (smoke), 9 (random + kfold), 11 (run sweep), 12 (summarize).
  - Polynomial validation-zone experiment → Task 10.
  - Tests (merge zone) → Tasks 1–7 each add tests; Task 14 final sweep.
  - Documentation → Task 13.
  - Out-of-scope TODOs → declared in Task 14.
- [ ] **Placeholder scan**: search "TBD", "TODO" in plan body. The only "TODO" tokens are in user-facing artifacts (CLAUDE.md "Open TODOs" section, summary recommendation placeholder Task 12 Step 4) — those are intentional content, not plan failures.
- [ ] **Type consistency**:
  - `parse_genotype` returns `tuple[int | None, int | None]` — used consistently across encoders.
  - Encoder return signatures: `(np.ndarray, list[str])` — same in all three.
  - `catalog: dict[str, list[int]]` — same in `build_allele_catalog` and `encode_dosage_block`.
  - Column-name prefix scheme (`dosage_`, `geom_`, `rnorm_`) used in encoders and tested in CLI integration tests.
  - `fold_result.json` schema — `mode`, `split`, `seed`, `fold` (kfold only), `n_held_out`, `median_error`, `mean_error`, `max_error`, optional `error` — same in `run_microsat_modes.py`, `run_microsat_polynomial.py`, and `summarize_microsat.py`.
