# Feature-Branch Cleanup Arc — `genotype_likelihoods`

Design doc for the cleanup work that brings the `genotype_likelihoods` branch
to a state where it can be handed to the kr-colab authors as a self-contained
feature branch — they browse the work, look at the validation evidence, and
merge to main if happy. Lower stakes than a PR-to-main, lets the supporting
material live alongside the code change without polluting the merge target.

Picks up where the path D validation arc left off (commits
`fa04126…d8d366c`), with the loader patch (`5418bf3`) already landed and
verified non-destructive on the integer hard-call path.

## Goals

1. **Restructure the branch into two clear zones:**
   - **Merge-target zone** — `locator/`, `scripts/gl_to_locator.py`, `tests/`,
     `docs/genotype_likelihoods.md`. Small, focused, what would actually get
     merged to main.
   - **Validation evidence zone** — a new top-level `validation/` directory
     containing all the validation tooling, tests, figures, summary, and
     spec/plan docs. Branch-only; demonstrably supports the merge but doesn't
     clutter `main`.
2. **Drop microsats.** The 4 production scripts have synthetic-fixture unit
   tests but zero real-data validation. They become a separate spec/branch.
3. **Switch Path B's error metric to coast-projected.** Barnacles are
   intertidal — predictions inland of the coastline are biologically nonsense
   and the haversine metric overstates error.
4. **Add two figures** in `validation/figures/` (committed to the branch since
   they live in the validation zone, not the merge zone):
   - **Figure 1:** test-data noise curve — three condition scatters plus a
     fourth panel showing the noise-curve line plot.
   - **Figure 2:** balanus map — cartopy panel of the Pacific coast with true
     site centroids and per-sample predictions colored by along-coast error.
5. **Add a small user-facing doc** at `docs/genotype_likelihoods.md` (in the
   merge zone) — short, example-driven, shows the ANGSD beagle → matrix →
   locator workflow.
6. **Update existing spec docs** to reflect what shipped, not what was
   planned. Move all spec/plan docs into `validation/docs/`.
7. **Update parent-directory `CLAUDE.md`** TODOs to mark addressed items.
8. **Final test verification** — full ReLocator 232-test suite + path D
   extras green.

## Non-goals (this spec)

- Pushing the branch.
- Opening a PR or interacting with the upstream remote.
- Squashing or rewriting commit history.
- Microsat real-data validation (separate branch).
- Native `--beagle` CLI loader in ReLocator (separate spec; the existing
  patched `--matrix` already covers continuous dosage and full_gl matrices).
- Adding tests for figure generators (matplotlib output is awkward to assert
  on; integration testing is the visual inspection of PNGs).

## Branch structure (after the arc)

```
ReLocator/                              ← repo root

  ── MERGE ZONE (would be PR'd to main) ──

  locator/loaders.py                    (patched; commit 5418bf3)
  locator/training.py                   (patched; commit 5418bf3)
  scripts/
    gl_to_locator.py                    (production tool)
    setup_pre_commit.py                 (existing)
    vcf_to_zarr.py                      (existing)
  tests/
    test_data_loading.py                (+ 2 continuous-dosage tests)
    test_input_extensions.py            (3 GL tests; microsats removed)
    test_filters.py                     (existing)
    test_*.py                           (existing)
    conftest.py                         (revert sys.path hack — no longer needed
                                          here since the validation tests live
                                          under validation/tests/)
  docs/
    genotype_likelihoods.md             (NEW — small user doc)

  ── VALIDATION ZONE (branch-only evidence) ──

  validation/                           (NEW top-level directory)
    README.md                           (intro for authors)
    __init__.py
    common.py                           (was scripts/validation/common.py)
    coastline.py                        (NEW — 12-site polyline projection)
    build_inputs.py                     (was scripts/validation/build_inputs.py)
    run_smoke.py                        (was scripts/validation/run_smoke.py)
    run_loso.py                         (was scripts/validation/run_loso.py;
                                          updated to emit along-coast metric)
    summarize.py                        (was scripts/validation/summarize.py;
                                          updated with figures + new metric)
    thin_beagle.py                      (was scripts/validation/thin_beagle.py)
    synth_gls.py                        (was scripts/validation/synth_gls.py)
    tests/
      __init__.py
      conftest.py                       (sys.path adjustment so
                                          `from validation import …` works)
      test_validation.py                (was tests/test_validation.py)
    figures/
      fig1_test_data.png                (committed; PNG from summarize.py)
      fig2_balanus_map.png              (committed; PNG from summarize.py)
    summary/
      summary.md                        (committed; final report)
      summary.tsv                       (committed; flat per-fold table)
      noise_curve.tsv                   (committed; per-α prediction error)
    docs/
      specs/
        2026-04-28-relocator-input-extensions-design.md  (moved + trimmed)
        2026-05-03-path-d-validation-design.md           (moved + rewritten)
        2026-05-03-pr-cleanup-arc-design.md              (moved — this file)
      plans/
        2026-05-03-path-d-validation.md                  (moved)

  scripts/validation/                   (REMOVED — moved to validation/)
  scripts/microsat_*.py                 (REMOVED — dropped from scope)
  tests/test_validation.py              (REMOVED — moved to validation/tests/)
  docs/superpowers/                     (REMOVED — contents moved to validation/docs/)
```

The merge zone is intentionally tight. A reviewer skimming `locator/`,
`scripts/gl_to_locator.py`, the two new tests in `test_data_loading.py`,
`tests/test_input_extensions.py` (now 3 tests), and the small user doc
gets the full picture in <500 lines of diff. The `validation/` directory
is there for evidence-on-demand but isn't required reading.

## Components

### `validation/coastline.py` (NEW)

Pure-numpy module. Imports `haversine` from `validation.common`.

```python
BARNACLE_COAST: list[tuple[str, float, float]]  # 12 sites, N→S

def build_coast_index(sites=BARNACLE_COAST) -> tuple[np.ndarray, np.ndarray]:
    """Return (cum_km, points) where cum_km[i] = polyline distance from
    sites[0] to sites[i] in km; points = (n, 2) array of (lat, lon)."""

def project_to_coast(pred_lat, pred_lon, cum_km, points):
    """Returns (coast_pos_km, offshore_km).
    Planar projection per segment, picks segment with smallest perpendicular
    distance, returns along-polyline position + perpendicular distance."""

def along_coast_distance(pred_lat, pred_lon, true_site_idx, cum_km, points):
    """|project_to_coast(...).coast_pos_km - cum_km[true_site_idx]|."""
```

Tests in `validation/tests/test_validation.py`:
- `test_coast_index_zero_at_first_site`
- `test_project_at_known_site` (offshore_km < 1)
- `test_project_offshore` (~50 km from a site)
- `test_along_coast_distance_ordering`

### `validation/run_loso.py` modifications

Within `_run_one_fold`, after the existing haversine block, project each
held-out sample's prediction onto the coastline and add three fields to
`fold_result.json`:
- `mean_along_coast_err_km`
- `median_along_coast_err_km`
- `mean_offshore_km`

Existing haversine-based fields retained for backward compatibility.
Imports updated: `from validation.coastline import (...)` instead of
`from scripts.validation.coastline import (...)`.

### `validation/summarize.py` modifications

Three additions, all behind feature-detection so old `fold_result.json`
files (without the new fields) still summarize correctly:

1. Path B section gains `mean_along_coast_err_km` column; centroid baseline
   computed using along-coast distance to the polyline midpoint; PASS/FAIL
   ratio uses along-coast.
2. New `_render_test_data_figure(...)` — produces `fig1_test_data.png`
   (4-panel matplotlib).
3. New `_render_balanus_map(...)` — produces `fig2_balanus_map.png` (cartopy
   PlateCarree projection over Pacific coast bbox; coastlines + states; 12
   site centroids labeled and sized; 54 prediction points colored by
   along-coast error; colorbar).

PNG output paths default to `validation/figures/`. summary.md / summary.tsv
default to `validation/summary/`. CLI flags allow overriding for local
out-tree usage.

### `docs/genotype_likelihoods.md` (NEW)

Roughly 30–50 lines. Intent: a user lands here from the README or a search,
sees one paragraph of context, runs the example commands, has a working
GL → locator pipeline. Outline:

```
# Genotype likelihood input

ReLocator's --matrix flag now accepts continuous expected dosage
(floats in [0, 2]), not just hard-call dosage (integers 0/1/2).
This lets you feed ANGSD-derived genotype likelihoods directly,
without information-losing hard-call rounding.

## Workflow
  - bash example: ANGSD invocation
  - bash example: gl_to_locator.py
  - bash example: locator --matrix

## Notes
  - sample-order constraint
  - --gl_mode {dosage,full_gl}
  - reference to validation/ dir for evidence
```

The doc lives in the merge zone. It's the only PR-relevant documentation
addition.

### `validation/README.md` (NEW)

Intro for the kr-colab maintainers. Brief — what this directory contains,
why it's separate from the merge zone, where to find the figures and
summary, how to re-run anything they want to verify. Maybe 50 lines.

## Microsat removal

```bash
git rm scripts/microsat_to_locator.py
git rm scripts/microsat_features.py
git rm scripts/microsat_poly.py
```

Edit `tests/test_input_extensions.py`: remove `MICROSAT_SAMPLE` fixture and
`test_microsat_to_locator_dosage`, `test_microsat_features_geometry`,
`test_microsat_poly_expansion_and_pca`. Keep the 3 GL tests.

## Restructure mechanics

The move of `scripts/validation/` → `validation/` and `tests/test_validation.py` →
`validation/tests/test_validation.py` requires adjusting imports throughout.

- All files under `validation/` that previously did
  `from scripts.validation import common` → `from validation import common`.
- `validation/tests/conftest.py` adds `sys.path.insert(0, project_root)`
  (parallel to what `tests/conftest.py` did before).
- `tests/conftest.py` reverts to its pre-patch form (the sys.path hack we
  added is no longer needed since the new continuous-dosage tests in
  `tests/test_data_loading.py` don't import from `scripts.validation` —
  they only import from `locator`).
- `pyproject.toml`'s pytest config probably needs to add `validation/tests/`
  as a testpath (or we run it explicitly via `pixi run pytest validation/tests/`).
  Decide during implementation; minor.
- `git mv` preserves history; use it rather than rm + add for the moves.

## Spec rewrite

`docs/superpowers/specs/2026-05-03-path-d-validation-design.md` (now at
`validation/docs/specs/2026-05-03-path-d-validation-design.md` after the
move) is rewritten to:
- Lead with what shipped (loader patch, GL pipeline, validation tooling).
- Add a "Findings" section with the noise-curve story.
- Drop microsat references.
- Trim "Open assumptions" to current state.

`2026-04-28-relocator-input-extensions-design.md` is similarly trimmed of
microsat sections (or kept as historical with a header noting that microsats
were dropped from this branch).

## Parent CLAUDE.md updates

`../CLAUDE.md`:
- Mark TODO (Correctness) "ReLocator `--matrix` loader rejects continuous
  dosage" as resolved by `5418bf3`. Move to a "Resolved on
  `genotype_likelihoods` branch" subsection.
- Mark TODO (Feature) "Native `--beagle` loader" as **partially resolved** —
  continuous dosage and full_gl matrices both load through `--matrix`.
- Microsat sections trimmed to a one-line "deferred to follow-up branch"
  pointer.

## Sequencing / build order

1. **Restructure** — move `scripts/validation/` → `validation/`, move
   `tests/test_validation.py` → `validation/tests/test_validation.py`,
   move `docs/superpowers/{specs,plans}/` → `validation/docs/{specs,plans}/`.
   Update imports throughout. Add `validation/__init__.py`,
   `validation/tests/__init__.py`, `validation/tests/conftest.py`. Revert
   `tests/conftest.py` sys.path hack. Run `pixi run pytest tests/
   validation/tests/` and confirm everything still green.

2. **Build `validation/coastline.py` + 4 unit tests.** TDD-style. Single
   commit.

3. **Update `validation/run_loso.py`** to emit along-coast fields. Re-run
   Path B (~3 min wall-clock). Single commit (script change).

4. **Update `validation/summarize.py` for Path B's new metric.** Single
   commit.

5. **Add Figure 1 generator to `validation/summarize.py`.** Run summarize,
   verify the PNG renders. Commit script + the generated PNG to
   `validation/figures/fig1_test_data.png`.

6. **Add Figure 2 generator to `validation/summarize.py`.** Run, verify,
   commit script + `validation/figures/fig2_balanus_map.png`.

7. **Drop microsats** — `git rm` the 3 production scripts; remove the 3
   microsat tests from `tests/test_input_extensions.py`. Run remaining
   tests. Single commit.

8. **Add `docs/genotype_likelihoods.md`** — the small user doc. Single
   commit.

9. **Add `validation/README.md`** — intro for authors. Single commit.

10. **Rewrite spec docs** at `validation/docs/specs/`. Single commit.

11. **Update parent `CLAUDE.md`** (file is on disk but not in any git repo).
    No commit.

12. **Final verification** — `pixi run pytest tests/ validation/tests/
    --ignore=tests/test_bandwidth_optimization_integration.py -n 4` passes.
    Confirm `validation/figures/*.png` and `validation/summary/*.{md,tsv}`
    are committed and accessible.

## Testing

| Component | Test file | Tests | Coverage |
|---|---|---|---|
| `validation/coastline.py` | `validation/tests/test_validation.py` | 4 new | unit |
| `validation/run_loso.py` along-coast wiring | (no new test) | — | exercised by Path B re-run |
| `validation/summarize.py` figure generators | (no new test) | — | visual inspection |
| Microsat removal | `tests/test_input_extensions.py` | 3 (was 6) | confirms GL-only suite still passes |
| Loader patch | `tests/test_data_loading.py` | 2 (existing, unchanged) | already verified |

Total at end of arc: **19 path-D / input-extension tests** (3 GL in
`tests/test_input_extensions.py` + 12 existing validation in
`validation/tests/test_validation.py` + 4 new coastline). Plus the 232
ReLocator core tests unchanged (which include the 2 continuous-dosage
tests added to `tests/test_data_loading.py` by the loader patch).

## Error handling

- `validation/coastline.py`: hard-fails on degenerate input (zero-length
  segments, empty site list).
- `validation/run_loso.py`: if `coastline.py` raises during a fold, the
  fold-isolation logic catches it and records FAILED with the exception
  message. Sweep continues.
- `validation/summarize.py`:
  - If new along-coast fields missing from `fold_result.json`, fall back to
    haversine columns only. (Forward-compatible with old data.)
  - If `cartopy` import fails, Figure 2 is skipped with a stderr warning;
    rest of the report still emits.
  - If predlocs files for the example-VCF runs are missing, Figure 1 panels
    are skipped one-by-one rather than failing wholesale.

## Acceptance criteria

| Step | Criterion |
|---|---|
| Restructure | All tests green (232 ReLocator + 15 path-D pre-coastline); all imports updated; `git status` clean |
| coastline.py | 4 new unit tests pass; `cum_km[-1]` ∈ [2500, 3500] km |
| run_loso.py | Re-run Path B; all 12 fold_result.json files contain new along-coast fields |
| summarize.py | summary.md shows new column; along-coast ratio computed against polyline-midpoint baseline |
| Figure 1 | PNG exists at `validation/figures/fig1_test_data.png`; user visually confirms 4-panel layout |
| Figure 2 | PNG exists at `validation/figures/fig2_balanus_map.png`; cartopy renders coastlines + states |
| Microsat drop | 3 scripts deleted; `tests/test_input_extensions.py` shrinks to 3 tests; pytest passes |
| `docs/genotype_likelihoods.md` | exists in merge zone; <50 lines; example commands run cleanly |
| `validation/README.md` | exists; intro is clear |
| Spec rewrite | No microsat references; "Findings" section present; references commit `5418bf3` |
| Final test run | 232 ReLocator + 19 path-D tests pass |

## Open assumptions / load-bearing risks

- **12-site polyline approximates a real coast.** Documented in spec
  outcome.
- **`offshore_km` is informational, not corrective.** Predictions plotted at
  raw lat/lon on the map; no inland-snap.
- **Re-running Path B replaces the on-disk fold_result.json files.** The
  previous run's outputs in `out/path_d_validation/loso_dosage/` get
  overwritten. New fields are a strict superset of old.
- **Figure PNGs are committed to the branch.** ~1 MB total; acceptable for
  a feature branch carrying its own evidence. Since they live in the
  validation zone they don't show up in any future PR-to-main diff (the PR
  would only span the merge zone).
- **PR-to-main timing.** This spec produces a feature-branch handoff. If
  kr-colab later wants to merge, they can either merge the whole feature
  branch or cherry-pick just the merge-zone changes. The branch supports
  both workflows.
