---
name: run-validation
description: Re-run the full Path D validation suite (smoke → LOSO dosage → LOSO full_gl → summarize) end-to-end against the balanus barnacle inputs. Use when the loader patch in locator/loaders.py, locator/training.py, or locator/prediction.py changes, or when validation/run_loso.py / validation/summarize.py are edited and we need fresh end-to-end evidence (figures + summary.md). Long-running — expect ≈ 60–90 min wall-clock on 3 GPUs.
disable-model-invocation: true
---

# Re-run Path D validation

Re-runs the complete Path D pipeline that produces the evidence committed under
`validation/`. This is what we cite in the PR. Invoke after edits to the loader
patch (`locator/loaders.py`, `locator/training.py`, `locator/prediction.py`,
`locator/data/filters.py`) or to the validation harness itself
(`validation/run_loso.py`, `validation/run_loso_rangemask.py`,
`validation/summarize.py`) to confirm metrics still match what the summary claims.

## Stages

| Stage | Script | Purpose | Wall-clock |
|---|---|---|---|
| 1. Smoke | `validation/run_smoke.py` | All-54 train, dosage mode | ~3 min |
| 2. LOSO dosage | `validation/run_loso.py --gl-mode dosage` | 12-fold LOSO, expected dosage features | ~25 min |
| 3. LOSO full_gl | `validation/run_loso.py --gl-mode full_gl` | 12-fold LOSO, 3-col-per-site features | ~30 min |
| 4. Summarize | `validation/summarize.py` | regenerate Fig 1/2/3 + summary.md/tsv | ~30 s |

Each LOSO sweep runs 3 folds in parallel via `ProcessPoolExecutor`, one per GPU,
through `--gpu_number` round-robin.

## Usage

```bash
# All four stages, default args:
.claude/skills/run-validation/run.sh

# Skip smoke (already passed):
.claude/skills/run-validation/run.sh --skip smoke

# Just the summarize stage (rebuild figures + summary from existing fold_result.json):
.claude/skills/run-validation/run.sh --only summarize
```

The script writes a stage-by-stage status line to stdout and aborts on the first
non-zero exit code. Check `out/path_d_validation/<stage>/*.log` for stage detail.

## Required inputs (paths assumed by the script)

- `out/path_d_validation/inputs/sample_data.txt` — locator-format sample table
- `out/path_d_validation/inputs/bam.filelist` — 54-sample BAM paths
- `out/path_d_validation/inputs/test_sample_data.txt` — example-VCF table (smoke)
- `data/test_data/...beagle.gz` — thinned beagle for LOSO sweeps
- `data/balanus_locations.tsv` — 12 site coords, used for the LOSO target list

## What to check after the run

1. **Fold success**: every `loso_*/<site>/fold_result.json` has `"status": "OK"`.
2. **Coast metric stability**: `validation/summary/summary.tsv` Path B
   `mean_along_coast_err_km` should be in the 200–400 km range; spikes >500 km
   suggest a broken loader path.
3. **VCF baseline parity**: `validation/figures/fig3.png` (VCF vs GL scatter at
   α=0) should sit on or near the 1:1 line — the loader patch must not change
   integer-VCF behavior.
4. **Summary regeneration**: `validation/summary/summary.md` should rebuild
   without manual edits and match committed content.
