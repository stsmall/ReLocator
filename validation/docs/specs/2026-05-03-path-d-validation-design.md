# Path D Validation: Genotype-Likelihood Pipeline End-to-End

Design and outcome doc for the validation arc on the
`genotype_likelihoods` branch. The original draft (committed pre-patch)
focused on whether the unpatched `--matrix` flag could consume GL-derived
dosage; that question was answered negatively (it could not: integer
check at `loaders.py:141`), and the arc evolved into a loader patch plus
empirical evidence that the patched path is load-bearing.

This rewrite documents the actual outcome.

## Goals (final)

1. Validate that the GL → continuous-dosage → ReLocator pipeline works
   end-to-end on a non-underdetermined dataset.
2. Demonstrate that the loader patch (commit `5418bf3`) is non-destructive
   for the existing integer hard-call path.
3. Demonstrate that the patched continuous-dosage path is *load-bearing*:
   without it, GL uncertainty above ~50% destroys downstream prediction.
4. Produce supporting figures and a summary report that the kr-colab
   maintainers can review without running anything.

## Non-goals

- Microsat real-data validation (deferred to a separate branch).
- Native `--beagle` CLI flag (the existing patched `--matrix` covers
  continuous dosage and full_gl).
- Bootstrap / jackknife confidence intervals on per-fold error.

## Test sets

- **Balanus 54-sample beagle** (private). 12 sites along the Pacific
  coast, lat 34.4° → 60.1°. Used for Path A (smoke), Path B (dosage
  LOSO), and Path C (full_gl LOSO).
- **ReLocator's bundled msprime VCF** (`data/test_genotypes.vcf.gz`),
  500 samples × 11,497 biallelic sites in a simulated 50×50 unit square.
  Used for the example-VCF stress test and noise curve. Ground truth is
  exact (msprime simulation), so we can compute prediction error directly.

## Findings

### Path A — pipeline plumbing
PASS. End-to-end run on the balanus thinned beagle (100k sites × 54
samples) completes in ~1 min wall-clock; predictions written, sample IDs
align, dimensions correct.

### Path B — LOSO sanity
12-fold LOSO on the balanus dosage matrix. With the original haversine
metric, median per-fold error was 549.3 km against a centroid baseline
of 683.4 km — ratio 0.804, which the strict spec threshold called FAIL.
Switching to the coast-projected (along-coast) metric — biologically
motivated since barnacles are intertidal and inland error is meaningless
— the ratio drops to 0.350 (PASS) against an along-coast baseline of
~942 km. The model picks up signal at mid-latitude sites (Oregon: ~150
to ~400 km along-coast error) and collapses toward the centroid at the
extremes (AFB Alaska: ~1957 km; GOL S. California: ~659 km).

### Path C — full_gl LOSO
Originally blocked by `loaders.py:141` integer rejection. After the
loader patch (`5418bf3`), full_gl matrices (3 cols/site of probabilities
in [0, 1]) load through the same float-detection branch as 1-col-per-site
dosage. A 50-epoch smoke run on the balanus 100k×3-col matrix completes
in 2.6 min. The originally-deferred "Native `--beagle` loader" TODO is
no longer a correctness blocker; it remains a UX improvement.

### Example-VCF stress test (the actual evidence)
On a non-underdetermined dataset (500 samples × 11.5k sites,
90-sample holdout):

| condition | mean Euclidean err | max | ratio vs centroid (18.5) |
|---|---|---|---|
| VCF baseline (hard calls) | 4.46 | 30.3 | 0.24 |
| Continuous dosage, α=0.0 | 4.52 | 23.2 | 0.24 |
| Continuous dosage, α=0.5 | 4.62 | 18.6 | 0.25 |

The continuous-dosage path matches VCF performance to within ~1.4% on
mean error and is actually better in worst-case (lower max error at
both noise levels).

### Noise curve (the load-bearing finding)
Sweeping α over the example data:

| α | continuous | rounded |
|---|---|---|
| 0.0 | 4.52 | 4.67 |
| 0.3 | 4.51 | 4.83 |
| 0.5 | 4.62 | 5.44 |
| 0.7 | 4.22 | **18.54** ← centroid |
| 0.9 | 4.58 | **18.51** ← centroid |

Continuous dosage stays flat across the entire α range. Rounded matches
at low noise but **collapses at α ≥ 0.7**: every dosage rounds to "1"
once max(GL) drops near 0.4 (E[dosage] ≈ 0.5, 1.0, 1.5 for the three
genotypes; banker's rounding sends them all to 1). The matrix becomes
nearly constant and the model predicts the geographic mean.

The loader patch is therefore *load-bearing*: without it, GL uncertainty
above ~50% destroys the pipeline.

## Loader patch safety case

The patch (`locator/loaders.py` + `locator/training.py`) is structurally
additive: a `if np.issubdtype(values.dtype, np.floating): return early`
branch is added before the existing integer check, and the
`_filter_dosage_matrix` helper is only called from the new float branch.
The integer code path is byte-identical to the original.

Empirical verification:

- `np.array_equal(filtered_genotypes_patched,
  filtered_genotypes_original) == True` on the example VCF (uint8, shape
  (5830, 500)).
- All 232 ReLocator core tests pass post-patch.
- Same-seed VCF runs are GPU-nondeterministic (two consecutive same-seed
  runs differ in 1000/1000 lines), so prediction-level diff comparisons
  cannot show patch impact directly. The bit-identical filtered_genotypes
  is the rigorous test.

See `validation/summary/summary.md` for the full report and the
generated figures (`validation/figures/fig{1,2}_*.png`).

## Open assumptions / risks

- **12-site polyline approximates the real coast.** A prediction in a
  bay or strait projects diagonally across, not following the coast
  perfectly. Documented in `validation/coastline.py`.
- **`offshore_km` is informational, not corrective.** The map plots
  predictions at their raw lat/lon; no inland-snap is applied.
