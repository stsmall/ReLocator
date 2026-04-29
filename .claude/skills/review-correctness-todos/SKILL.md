---
name: review-correctness-todos
description: Audit a code change against the (Correctness)-tagged TODOs in CLAUDE.md to flag whether the change resolves any, regresses any, or sidesteps any without addressing them. Use after edits to scripts/gl_to_locator.py, scripts/microsat_*.py, or locator/loaders.py — or when the user asks "did this change touch any of our known correctness issues?"
---

# Review Correctness TODOs

The project's `CLAUDE.md` distinguishes `(Feature)` TODOs (nice-to-have new
capabilities) from `(Correctness)` TODOs (statistical / biological correctness
issues that materially affect results). Correctness items are the ones where a
silent regression matters most: sample-ID alignment, allele binning,
null-allele handling, homoplasy, prior-weighted dosage.

This skill walks through the `(Correctness)` items and reports the status of
each against the current diff.

## Procedure

1. **Read `CLAUDE.md`** and extract every TODO line tagged `(Correctness)`.
   These are typically formatted:
   `- [ ] **TODO (Correctness): <Title>**` followed by 1–3 lines of context.
2. **Determine the diff under review.** If the user did not name one, default
   to `git diff HEAD~1` (or the staged + unstaged diff if there are no
   commits since the last push).
3. **For each (Correctness) TODO, classify the change:**
   - **Resolves** — the diff implements the fix; mention the file(s) and
     where the TODO can be removed.
   - **Touches without fixing** — the diff modifies code in the area of the
     TODO but does not address it; flag the risk and quote the relevant
     hunk.
   - **Regresses** — the diff makes a change that contradicts the TODO's
     intent (e.g., assumes random missingness, hard-codes a flat prior).
     This is the most important category.
   - **Untouched** — the diff is unrelated; no action.
4. **Output a short numbered list**, one entry per Correctness TODO, with
   the classification and (for non-Untouched) the specific file:line
   reference. End with a 1-sentence summary.

## What this skill is NOT

- Not a general code review — for that, dispatch the
  `popgen-correctness-reviewer` subagent or use the standard review flow.
- Not a TODO-list editor — propose changes to `CLAUDE.md` only when the
  diff *resolves* a TODO; do not edit `CLAUDE.md` automatically.
- Not a Feature-TODO checker — explicitly limit the audit to
  `(Correctness)` items.

## Example output

```
1. (Correctness) Validate sample ID alignment — RESOLVES
   scripts/gl_to_locator.py:282-286 now adds an explicit cross-check via
   --sample_data; the silent-drop failure mode is closed for this entry point.
   Suggest: remove this item from CLAUDE.md.

2. (Correctness) Allele binning for sizing artefacts — UNTOUCHED.

3. (Correctness) Null allele / dropout handling — TOUCHES WITHOUT FIXING.
   scripts/microsat_to_locator.py:300-309 (impute_missing) still uses
   site-mean imputation, which is unbiased only under MAR. The diff did not
   add a per-locus missing-rate report.

Summary: 1 resolved, 1 touched-without-fixing, 4 untouched, 0 regressions.
```
