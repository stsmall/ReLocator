# Validation evidence for the `genotype_likelihoods` branch

This directory is the validation zone of the feature-branch handoff. It
contains all the validation tooling, test outputs, figures, and design
notes that produced the branch's empirical evidence. None of these files
need to merge to `main`; they live alongside the code change so reviewers
can browse the evidence without running anything.

## What's here

- `coastline.py`, `common.py`, `build_inputs.py`, `run_smoke.py`,
  `run_loso.py`, `summarize.py`, `thin_beagle.py`, `synth_gls.py` —
  validation tooling.
- `tests/test_validation.py` — unit tests for the tooling.
- `figures/`
  - `fig1_test_data.png` — 4-panel: VCF / continuous-dosage α=0 /
    continuous-dosage α=0.5 truth→prediction scatter, plus a noise-curve
    panel showing how mean error scales with GL uncertainty for continuous
    vs rounded encodings.
  - `fig2_balanus_map.png` — cartopy map of the Pacific coast with the 12
    true site centroids and the 54 per-sample predictions (LOSO,
    continuous dosage), colored by along-coast error.
- `summary/`
  - `summary.md` — aggregated path D + stress-test report.
  - `summary.tsv` — flat per-fold table.
  - `noise_curve.tsv` — per-α prediction error for both encodings.
- `docs/specs/` — the design spec docs that drove this branch.
- `docs/plans/` — the implementation plans.

## Headline result

The continuous-dosage path through the patched `--matrix` loader is
within ~1% of VCF performance on the example test data (500 samples ×
~11.5k biallelic sites, 90-sample holdout). Under increasing GL
uncertainty (α ∈ {0, 0.3, 0.5, 0.7, 0.9}), prediction error stays flat
at ~4.2–4.6 mean Euclidean error, while the rounded path collapses to
the centroid baseline (~18.5) at α ≥ 0.7. **The loader patch is
load-bearing**: without it, GL uncertainty above ~50% destroys the
pipeline.

See `summary/summary.md` for the full report.

## Re-running anything

```bash
# Path B (LOSO over the balanus 54-sample test set, dosage mode):
pixi run python -m validation.run_loso \
    --inputs-dir out/path_d_validation/inputs \
    --samples-locations ../test_data/samples_locations.tsv \
    --sample-table ../test_data/sample_table.tsv \
    --beagle out/path_d_validation/inputs/combined.thin_byMiss100k.beagle.gz \
    --gl-mode dosage \
    --out-dir out/path_d_validation/loso_dosage \
    --max-epochs 500

# Regenerate the summary + figures:
pixi run python -m validation.summarize \
    --out-dir out/path_d_validation \
    --samples-locations ../test_data/samples_locations.tsv \
    --sample-table ../test_data/sample_table.tsv

# Run the validation unit tests:
pixi run pytest validation/tests/ -v
```

## Branch zones

The branch has two zones:

- **Merge zone** — `locator/`, `scripts/gl_to_locator.py`, `tests/`,
  `docs/genotype_likelihoods.md`. Small, focused, what would actually get
  merged to `main`.
- **Validation zone** — this directory. Branch-only; supports the merge
  but doesn't ship.
