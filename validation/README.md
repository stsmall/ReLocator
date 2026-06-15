# Validation tree (microsat + sculpin LOSO arc)

Branch-only validation evidence for the microsatellite-input PR (#46) and the sculpin location-masking demonstration that follows. The GL native loader (PR #47) shipped against main; its balanus path-D validation has been removed from this branch as no longer load-bearing.

## What's here

- `sculpin/` — *Cottus asper* LOSO + range-mask demonstration of the native `--microsat` loader. Headline reproduction recipe in [`sculpin/README.md`](sculpin/README.md).
- `elephants/` — KNP elephant microsat validation (14 loci × 50 km below resolution).
- `run_microsat_modes.py`, `run_microsat_polynomial.py`, `summarize_microsat.py` — microsat encoding-mode comparison on SLiM arenas (dosage vs polynomial expansion).
- `figures/`, `summary/` — figures and per-fold tables for the above.
- `docs/specs/`, `docs/plans/` — design specs and implementation plans for the input-extensions arc (GL + microsat).
- `notes/range_mask_bug.md` — context for the coord-space bug that `sculpin/run_loso_rangemask.py` works around.
- `common.py` — shared utilities (haversine, centroid baseline, predloc parser, GPU round-robin, subprocess streamer). Re-exported across the runners.
- `tests/test_validation.py` — unit tests for `common.py`.

## Reproduction

See `sculpin/README.md` for the full sculpin recipe end-to-end. The microsat-mode runners take similar arg surfaces (`--microsat`, `--sample_data`, `--out_dir`); each module exposes `--help`.

## What's not here anymore

The balanus path-D validation that produced `fig1`–`fig4` lived on this branch until PR #47 (the GL native loader) shipped. With PR #47 merged, the balanus runners, fig1–fig4, and the path-D specs/plans have been removed — their job was to prove the GL loader worked on real data, and main now carries the validated implementation. The history is preserved under tag `archive/microsat-sculpin-pre-rebuild` if you need to resurrect any of it.
