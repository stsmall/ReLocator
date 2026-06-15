# ReLocator Input Extensions: Genotype Likelihoods & Microsatellites

Design doc for extending `kr-colab/ReLocator` to support two new input data
types via preprocessing scripts that emit `--geno`-compatible tab-delimited
matrices. Native `DataLoader` integration is deferred to a follow-up
(see `CLAUDE.md` TODO list).

## Goals

1. Convert ANGSD `-doGlf 2` beagle output to a feature matrix usable as
   `--geno` input — both as expected dosage (default) and as the full GL
   probability triplet per site.
2. Convert tab-delimited microsatellite genotype calls to a `--geno`-compatible
   feature matrix, with optional stepwise-mutation geometry features and
   degree-2 polynomial expansion (with optional PCA reduction).
3. Ship as preprocessing CLI scripts that do not modify ReLocator's network or
   loader code.

## Non-goals (this spec)

- Native `--beagle` / `--microsat` loaders inside `locator.loaders` (TODO).
- Dual-stream network architecture for mixed inputs (TODO).
- Binary `-doGlf 1` parsing (TODO).
- Allele binning for capillary sizing artefacts (TODO).
- Non-flat-prior posterior dosage (TODO).

These are explicitly recorded as future work in `CLAUDE.md`.

## Architecture

Four standalone Python scripts in `scripts/`, each with a `main()` and an
`argparse` CLI:

```
scripts/gl_to_locator.py        # ANGSD beagle.gz → dosage TSV (or full-GL TSV)
scripts/microsat_to_locator.py  # microsat TSV → allele-dosage TSV
scripts/microsat_features.py    # microsat TSV → dosage + geometry features TSV
scripts/microsat_poly.py        # any feature TSV → degree-2 expansion (+PCA)
```

`microsat_features.py` imports parsing primitives from
`microsat_to_locator.py` so genotype-string parsing semantics
(separators, missing values, single-value homozygote shorthand) stay aligned.

Data flow:

```
ANGSD BAMs ─→ angsd -doGlf 2 ─→ *.beagle.gz
                                     │
                              gl_to_locator.py (--gl_mode {dosage,full_gl})
                                     │
                              feature TSV ──┐
                                            │
microsat TSV ─→ microsat_to_locator.py ────►├──► locator --geno <TSV>
        │                                   │
        └─→ microsat_features.py ──┐        │
                                   ▼        │
                         microsat_poly.py ──┘
```

## Component contracts

### `gl_to_locator.py`

**Inputs:** `--beagle <path.gz>`, `--bam_list <path>`, `--out <path>`.
**Optional:** `--gl_mode {dosage,full_gl}` (default `dosage`),
`--min_maf` (default 0.01), `--max_missing_frac` (default 0.10),
`--gl_missing_threshold` (default 0.40), `--sample_data` (cross-check IDs).

**Output (dosage mode):** TSV with `sampleID` first column and one
`site_{marker}` column per retained site. Values are
`E[dosage] = P(AB) + 2·P(BB)` under a flat prior, in `[0, 2]`.

**Output (full_gl mode):** TSV with `sampleID` first column and three columns
per retained site: `site_{marker}_AA`, `site_{marker}_AB`, `site_{marker}_BB`.
Values are the normalized GL probabilities directly, summing to ≈1 per triplet.

**Missingness:** a sample at a site is treated as missing when
`max(P_AA, P_AB, P_BB) < --gl_missing_threshold`. Imputation:
- dosage mode: site-mean dosage across non-missing samples.
- full_gl mode: site-mean GL triplet across non-missing samples.

**Site filters (applied in both modes):**
1. Drop sites where the missing fraction exceeds `--max_missing_frac`.
2. Drop sites where MAF (computed on imputed dosage) is below `--min_maf`.

The MAF filter uses the dosage representation in both modes intentionally — MAF
is only meaningful on dosage, and we want to drop monomorphic sites uniformly.

**Sample ordering:** rows are emitted in BAM-list order. `Path(bam).stem`
yields sample IDs; `--sample_data` enables a non-fatal cross-check that warns
on IDs present in one file but not the other.

### `microsat_to_locator.py`

**Inputs:** `--microsat <path>`, `--out <path>`. **Optional:**
`--min_allele_freq` (default 0.01), `--report_encoding <path>`,
`--max_locus_missing` (warn-only threshold, default 1.0), `--repeat_unit`
(stub for future allele binning).

**Output:** TSV with `sampleID` first column. Per locus with retained alleles
`[a₁, …, aₖ]`, k columns named `{locus}_{allele}` holding the count of that
allele in the diploid genotype (0/1/2 before imputation; float after).

**Missingness:** parsed missing strings are `NA`, `NaN`, `.`, empty, `0,0`,
`0/0`. A missing genotype at a locus is imputed with the column-group mean
(per-allele mean dosage at that locus across non-missing samples).

**Genotype parsing accepts:** `,`, `/`, ` `, `|` separators; a single value
without a separator is interpreted as a homozygote.

### `microsat_features.py`

**Inputs:** `--microsat <path>`, `--out <path>`. **Optional:**
`--features dosage,mean_repeat,allele_diff,het` (any non-empty subset, default
all four), `--min_allele_freq`, `--max_locus_missing`, `--report_encoding`.

**Output:** TSV with `sampleID` first column. Concatenated blocks in the order
the user requested via `--features`:
- `dosage` block — same columns as `microsat_to_locator.py`.
- `mean_repeat` block — `{locus}_mean_repeat` per active locus, value
  `(a₁ + a₂) / 2`.
- `allele_diff` block — `{locus}_allele_diff` per active locus, value
  `|a₁ − a₂|`.
- `het` block — `{locus}_het` per active locus, value `int(a₁ ≠ a₂)`.

**Missingness imputation:** locus mean for `mean_repeat` and `allele_diff`;
column mean for `het` (which equals the locus heterozygosity rate). The dosage
block reuses the imputation from `microsat_to_locator.py`.

### `microsat_poly.py`

**Inputs:** `--geno <feature_tsv>`, `--out <path>`. **Optional:**
`--degree` (default 2), `--standardize` / `--no_standardize` (default on),
`--interaction_only` (default off), `--pca N`, `--sample_data <path>`,
`--seed` (default 42).

**Output:** TSV with `sampleID` first column. Without `--pca`, columns are
sklearn's polynomial feature names (e.g. `f0`, `f1`, `f0 f1`, `f0^2`). With
`--pca N`, columns are `PC0…PC{N-1}`.

**Anti-leakage behavior:** when `--sample_data` is given, both `StandardScaler`
and `PCA` are fit on samples with non-NA `x` and `y` only, then applied to all
samples (including held-out / unknown-location). Without `--sample_data`,
both are fit on all samples.

**Component capping:** if requested `--pca N` exceeds either `n_features` or
the number of training samples, `n_components` is capped to the smaller of
the two and a warning is printed to stderr.

## Testing

`tests/test_input_extensions.py` exercises each script as a subprocess against
synthetic fixtures (small enough to keep the suite fast):

| Test | What it asserts |
|---|---|
| `test_gl_to_locator_dosage` | column count = n_sites, values in `[0, 2]`, sample order = BAM order |
| `test_gl_to_locator_full_gl` | column count = 3 × n_sites, suffixes `{AA,AB,BB}`, triplets sum to 1 |
| `test_gl_to_locator_dimension_mismatch_errors` | non-zero exit + clear error when bam_list and beagle disagree |
| `test_microsat_to_locator_dosage` | per-(locus, allele) counts correct (heterozygote = 1+1; homozygote = 2); encoding report is consistent |
| `test_microsat_features_geometry` | mean/diff/het correct on heterozygotes and homozygotes; missing genotype imputed with locus mean |
| `test_microsat_poly_expansion_and_pca` | degree-2 column count matches `n + n + C(n,2)`; PCA mode emits requested number of `PC*` columns |

Synthetic fixtures only — no large data committed to the repo. Real ANGSD
output is exercised by a one-shot smoke test (see Validation below) but does
not run in the test suite.

## Validation plan

1. **Unit tests pass** in the pixi environment via `pixi run pytest
   tests/test_input_extensions.py -v`.
2. **Smoke test on real ANGSD data**: run `gl_to_locator.py` against
   `test_data/angsd/snp_calling_global/combined.beagle.gz` (1.27M sites × 19
   samples) with a synthesized 19-line BAM list and standard filters
   (`--min_maf 0.05`, `--max_missing_frac 0.10`). Assert: non-zero exit code is
   absent, output TSV row count == n_samples + 1, column count is plausible
   given the filters. End-to-end ReLocator training is out of scope unless
   sample coordinates are provided.

## Known limitations & deferred work

Tracked in `CLAUDE.md` (TODO section). The most material:

- No native loader integration — preprocessing is required.
- Flat-prior dosage in GL mode (no `--doMaf`-driven posterior).
- No allele binning (`--repeat_unit` is a stub) for capillary sizing artefacts.
- Rare alleles below `--min_allele_freq` are silently zeroed, not pooled into a
  catch-all column or treated as missing.
- Mixed SNP+microsat inputs require column-concatenation outside of ReLocator
  (no dual-stream architecture yet).

## Dependencies

`numpy`, `pandas` for all four scripts. `scikit-learn`
(`PolynomialFeatures`, `StandardScaler`, `PCA`) only for `microsat_poly.py`.
All already present in ReLocator's `pyproject.toml`.
