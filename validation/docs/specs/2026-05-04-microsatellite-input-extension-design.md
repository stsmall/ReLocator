# Microsatellite Input Extension: Design

Design doc for the `microsatellites` branch. Adds microsatellite genotype
support to ReLocator as a preprocessing converter (`microsat_to_locator.py`)
that emits a continuous float matrix consumed by the existing `--matrix`
loader path patched on the `genotype_likelihoods` branch. Validation arc
mirrors the Path D pattern from the GL branch but adapted for the
SLiM-simulated dataset that ships with the original locator paper.

## Context

The `genotype_likelihoods` branch shipped a single-script preprocessor
(`gl_to_locator.py`) plus a loader patch (`is_dosage_matrix` predicate) that
lets ReLocator's `--matrix` accept continuous float input. An earlier
iteration of that branch also carried three microsat scripts
(`microsat_to_locator.py`, `microsat_features.py`, `microsat_poly.py`) but
they were dropped before handoff because they had only synthetic-fixture
unit tests and zero real-data validation. Microsats were deferred to a
separate branch with its own validation arc — this is that branch.

## Goals

1. Extend ReLocator to accept microsatellite genotype input via a
   preprocessing converter that emits a continuous dosage matrix
   compatible with the patched `--matrix` loader.
2. Encode microsat-specific biology (stepwise mutation, repeat-count
   ordinality, intra-individual diversity) through three composable
   feature modes.
3. Validate end-to-end on the SLiM-simulated dataset bundled with the
   original locator paper (200 individuals × 100 loci, arena coords),
   with mode-by-mode comparison evidence.
4. Produce a focused PR (single converter script + tests + docs +
   validation evidence) that maintainers can review without running
   anything.

## Non-goals

- Real-data microsat validation (homoplasy, null alleles, sizing
  artefacts). Declared as TODO; deferred to a follow-up branch with a
  real published microsat dataset.
- Native ReLocator CLI flag (`--microsat <input>` skipping the explicit
  converter step). UX improvement, not correctness.
- Polynomial / tensor-decomposition feature modes as user-facing
  options. A degree-2 polynomial run is included in the validation zone
  as evidence the design choice was deliberate, but does not ship in the
  user-facing `--features` flag.
- Loader changes. The microsat output is a continuous float matrix and
  flows through the existing `is_dosage_matrix` patch from the GL
  branch.

## Branch & PR shape

- **Branch:** `microsatellites`, off `genotype_likelihoods` HEAD
  `2ee683d`. Stacked PR.
- **Target:** `kr-colab/ReLocator:main`. PR description declares
  dependency on the GL PR; cannot merge until GL merges.
- **Two zones:**
  - **Merge zone** (would actually merge): `scripts/microsat_to_locator.py`,
    `tests/test_microsat_input.py`, `docs/microsatellites.md`, CLAUDE.md
    edits.
  - **Validation zone** (`validation/`, branch-only evidence):
    `validation/run_microsat_random_split.py`,
    `validation/run_microsat_modes.py`,
    `validation/summarize_microsat.py`,
    `validation/figures/microsat_*.png`,
    `validation/summary/microsat_summary.md`,
    `validation/notes/` for any out-of-scope ReLocator bugs found.

## Test data

`test_data/locator_microsats/notebooks/microsat_variants.txt` (symlinked
into shared storage outside the repo):

- 200 individuals (rows), 100 loci × 2 alleles each (200 columns of
  small-integer alleles).
- Two-column-pair format: `variant_0` and `variant_1` together form the
  diploid genotype at locus 0.
- Coords in `microsat_spatial_location.txt` (`x`, `y`, `sampleID` —
  arena coordinates in the simulation's unit square, not lat/lon).
- SLiM-simulated under the recipe at `recipes/locator.slim`. Ground
  truth is exact, so prediction error is well-defined as Euclidean
  distance in arena units.
- No coastline; no sites; no homoplasy or null alleles. Validation arc
  uses random splits and k-fold CV rather than LOSO.

## `microsat_to_locator.py`

Single script. Replaces what was previously three scripts on the dropped
`genotype_likelihoods` iteration.

### CLI

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
| `--microsat` | yes | — | Input TSV (see "Input formats" below) |
| `--out` | yes | — | Output dosage TSV for ReLocator `--matrix` |
| `--features` | no | `dosage,geometry,repeat_norm` | Comma-separated subset of the three modes |
| `--min_allele_freq` | no | `0.01` | Drop alleles below this frequency (dosage mode only) |
| `--max_locus_missing` | no | `1.0` | Drop loci with > X frac missing genotypes |
| `--report_encoding` | no | — | Optional column→source mapping TSV |

### Input formats

Auto-detected by header and cell content:

1. **Pair format** (preferred):
   `sampleID\tlocus1\tlocus2\t...` with cells like `15,16` (separator:
   comma, slash, space, or pipe). `NA`/`NaN`/`.`/empty for missing.

2. **Two-column format** (the SLiM dataset's shape):
   `sampleID\tvariant_0\tvariant_1\t...` with consecutive column pairs
   forming diploid genotypes. Detected when no pair-separator characters
   appear in any cell. Documented as a backward-compatibility shim, not
   the preferred format.

### Three encoding modes

Notation: `n_loci` = number of loci after `--max_locus_missing` filter;
`k_l` = number of distinct alleles at locus *l* after `--min_allele_freq`
filter; `K = sum(k_l)`.

| Mode | Output dim | Computation | Captures |
|---|---|---|---|
| `dosage` | K | one-hot allele counts (0/1/2) per `(locus, allele)` after MAF filter | which specific alleles (categorical) |
| `geometry` | 3 × n_loci | per-locus `mean_repeat = (a1+a2)/2`, `allele_diff = abs(a1-a2)`, `het = int(a1!=a2)` | per-individual heterozygosity / repeat magnitude |
| `repeat_norm` | n_loci | per-locus mean-repeat, z-scored across all input samples | SMM-aware compact continuous |

Output column names use mode prefixes: `dosage_<locus>_<allele>`,
`geom_<locus>_<feature>`, `rnorm_<locus>`. Order is deterministic:
`dosage` block, then `geometry` block, then `repeat_norm` block; loci
within a block sorted by input order.

### Missing-data imputation

- `dosage`: site-mean dosage (matches GL branch's pattern in
  `gl_to_locator.py`).
- `geometry`: per-locus mean for `mean_repeat` and `allele_diff`;
  per-locus heterozygosity rate for `het`.
- `repeat_norm`: per-locus mean (which becomes 0 after z-scoring — the
  "no information" prior).

### Known minor leakage (full-data feature statistics)

Three feature-level statistics are computed across the full input file
when the script runs:

- `dosage`: which alleles pass `--min_allele_freq` (frequencies computed
  across all samples).
- `geometry`: per-locus mean and heterozygosity rate used for
  missing-data imputation.
- `repeat_norm`: per-locus μ and σ used for z-scoring.

For the validation harness's k-fold CV, the script is run **once on the
full dataset**; the harness then splits the output matrix by row index
to form train and test folds. This means the test fold's encoding
implicitly used full-data statistics — a mild form of test-set leakage.
The magnitude is small because these statistics are population-level
(not individual-level labels), which is consistent with the GL branch's
practice in `gl_to_locator.py`. Documented as a known minor limitation;
not engineered around in v1. A real-data follow-up branch with
larger N would be the right place to add per-fold stats persistence
if the leakage proves consequential.

### Why these three modes (and not polynomial)

- **`dosage`** is the categorical baseline. It encodes "this individual
  has allele *a* at locus *l* with count 0/1/2" without using the fact
  that *a* is an integer repeat count. Equivalent to treating microsats
  as multi-allelic SNPs.
- **`geometry`** encodes the stepwise mutation model intuition that
  alleles of similar lengths are more closely related than alleles of
  dissimilar lengths. `mean_repeat` captures population-level drift in
  repeat lengths; `allele_diff` captures intra-individual diversity;
  `het` captures population-level heterozygosity.
- **`repeat_norm`** is the most compact continuous encoding. Under SMM
  at mutation-drift equilibrium, within-population repeat distributions
  are approximately Gaussian, so z-scoring per locus is the *correct*
  transformation, not just a generic ML hygiene step. One column per
  locus. (See "Known minor leakage" above for the trade-off in how the
  z-score stats are computed.)
- **Polynomial degree-2** is excluded from the user-facing flag because
  with n=200 and ~1000 dosage features, `interaction_only=True`
  expansion produces ~500k features. The NN backbone already learns
  pairwise interactions in its hidden layers; explicit polynomial
  features add severe overfit risk for marginal benefit. A degree-2 +
  PCA run is included in the validation zone as evidence — if the
  empirical result contradicts this analysis, the figure will say so.

## Validation arc

Mirrors Path D's structure (smoke → fold runner → comparison figure →
summary), adapted for SLiM-arena data.

| Step | What | Output |
|---|---|---|
| 1. Smoke | Run on full SLiM dataset, sanity-check shape/values | log only |
| 2. Random 80/20 split, single seed | All three user-facing modes + polynomial experimental mode | `validation/summary/microsat_smoke.txt` |
| 3. K-fold CV (k=5, 3 seeds each) | All four modes; mean ± std error | `validation/summary/microsat_kfold.tsv` |
| 4. Mode comparison figure | Bar chart: median Euclidean error per mode, with CV error bars | `validation/figures/microsat_modes.png` |
| 5. Predicted-vs-actual scatter | One panel per mode, true coords vs predicted, identity line | `validation/figures/microsat_scatter.png` |
| 6. Summary | Markdown report with results table, recommendation, known limitations | `validation/summary/microsat_summary.md` |

### Metric

Euclidean distance in arena units. Reported as median (robust) and
mean. No coastline, no range mask — the SLiM arena is unconstrained 2D
and ground truth is exact.

### Polynomial experimental mode (validation zone only)

`sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=True)`
applied to the dosage matrix, then `sklearn.decomposition.PCA(n_components=200)`
to keep the feature count tractable, then standard ReLocator run. Added
as the fourth column in the mode comparison figure. The expected
finding is that polynomial overfits at n=200; if surprising, the figure
documents it.

### What is NOT validated

Declared explicitly in `validation/summary/microsat_summary.md` "Known
limitations" and in CLAUDE.md "Microsats (active)" TODO list:

- Homoplasy (alleles of identical length but different ancestry).
- Null alleles (PCR drop-out producing apparent homozygotes).
- Sizing artefacts (off-by-one stutter calls).
- Real-data ascertainment biases.

These do not exist in the SLiM dataset by construction. Validating them
requires a published real-microsat study and a separate validation arc.

## Tests (merge zone)

`tests/test_microsat_input.py` — synthetic-fixture unit tests:

- Each of the three modes produces correct output shapes.
- `geometry` features computed correctly on a hand-built fixture.
- `repeat_norm` z-score is correct, scaling-stats round-trip works
  (write → read → re-apply produces identical output).
- Pair-format and two-column-format auto-detection both work on
  representative fixtures.
- Unparseable cells (non-integer alleles, malformed pair separators,
  odd column count in two-column format) raise clear errors.
- Missing-data imputation matches expectations for each mode.
- MAF filter drops the right alleles in `dosage` mode.
- `max_locus_missing` filter drops the right loci.
- Combined-mode (`dosage,geometry,repeat_norm`) output column ordering
  is deterministic and matches the documented prefix scheme.
- Errors on shape mismatch, unparseable cells, unknown feature names.

Run inside pixi env: `pixi run pytest tests/test_microsat_input.py -v`.

## Documentation

- **`docs/microsatellites.md`** (merge zone): user-facing guide. Input
  formats, feature mode descriptions with biological rationale, example
  CLI invocation, validation summary table copied from
  `validation/summary/microsat_summary.md`, list of known limitations
  and TODO items.
- **CLAUDE.md** updates:
  - Replace the "Microsats (deferred)" section with "Microsats
    (active)".
  - Add a row to the "Scripts Summary" table for
    `microsat_to_locator.py`.
  - Update "Branch state" to note the stacked PR shape.
  - Add operational notes (test data location, validation harness
    invocation).

## Out of scope (declared TODOs)

- Polynomial / tensor-decomposition modes as user-facing options.
- Homoplasy / null-allele / sizing-artefact handling.
- Real-data validation against a published microsat study.
- Native ReLocator CLI flag (`--microsat <input>` directly, skipping the
  explicit converter step).
- Combined SNP + microsat input runs (concatenate dosage matrices and
  feed through `--matrix` is documented as a manual workflow; no
  dedicated tooling).
- **Classifier output head** (`prediction_mode={classify, classify_then_avg}`)
  — the sculpin LOSO arc on the `microsat-sculpin` branch motivated this
  architectural follow-up. The regression head's regression-to-the-mean
  at geographic edges (snap-correctness rate = 8%, error vs FST r = +0.541)
  suggests a discrete-site classifier head would substantially improve
  fine-grained localization on data with discrete-sampling-design
  population structure. Designed in
  `validation/docs/specs/2026-05-05-classifier-head-design.md` and
  implemented on the `classifier-head` branch (off `microsat-sculpin`);
  not in scope for the microsat input PR proper.
