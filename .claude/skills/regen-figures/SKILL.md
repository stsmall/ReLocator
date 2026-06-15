---
name: regen-figures
description: Regenerate the three committed figures under validation/figures/ (fig1 test-data map, fig2 balanus map with snapped predictions, fig3 VCF-vs-GL scatter) from the existing fold_result.json files — without retraining anything. Use when validation/summarize.py rendering code changes (axes, labels, snap-to-coast logic, color choices), when CLAUDE.md or summary.md prose references a figure that needs visual refresh, or when reviewers ask for a tweak that doesn't touch any model output.
disable-model-invocation: true
---

# Regenerate validation figures

Re-renders `validation/figures/fig{1,2,3}.png` and rewrites
`validation/summary/{summary.md,summary.tsv,noise_curve.tsv}` from the
already-trained predictions in `out/path_d_validation/`. **No training, no
GPU, ~30 s.**

## When to use

- `validation/summarize.py` figure code changed (`_render_*_figure`,
  `_snap_to_coast`, axes, legend, color scheme).
- Coastline assets / cartopy version updated and the snap output looks
  different.
- A figure caption / summary.md prose referenced something that needs the
  figure refreshed to match.

**Do not use** if you've changed loader behavior, the LOSO sweep, or the
α-noise sweep — those need a full re-run via `run-validation`.

## Usage

```bash
.claude/skills/regen-figures/run.sh
```

Or run the underlying command directly:

```bash
pixi run python -m validation.summarize \
    --baseline-dir       out/path_d_validation/baseline \
    --loso-dosage-dir    out/path_d_validation/loso_dosage \
    --loso-full-gl-dir   out/path_d_validation/loso_full_gl \
    --samples-locations  data/balanus_locations.tsv \
    --out-dir            validation/summary \
    --figures-dir        validation/figures
```

## Sanity checks

After regenerating, eyeball:

1. `validation/figures/fig2.png` — every snapped prediction sits on the
   blue coastline polyline; nothing inland (the “samples in Utah” regression).
2. `validation/figures/fig3.png` — VCF-vs-GL scatter at α=0 hugs the 1:1 line.
3. `git diff validation/summary/summary.md` — prose changes look intentional;
   numeric tables match what summarize.py just wrote.
