#!/usr/bin/env bash
# Re-run Path D validation end-to-end: smoke -> LOSO dosage -> LOSO full_gl -> summarize.
# See SKILL.md for a description of each stage.
#
# Usage:
#   run.sh                       # all stages
#   run.sh --skip <stage>        # skip a stage (repeatable)
#   run.sh --only <stage>        # run only one stage (overrides --skip)
#
# Stages: smoke, loso_dosage, loso_full_gl, summarize

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

OUT_BASE="out/path_d_validation"
INPUTS="$OUT_BASE/inputs"
BEAGLE="data/test_data/balanus_thinned.beagle.gz"
SITES_TSV="data/balanus_locations.tsv"
SAMPLE_TABLE="data/sample_table.tsv"

SKIP=()
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip) SKIP+=("$2"); shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    -h|--help) sed -n '1,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

run_stage() {
  local name="$1"; shift
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then return 0; fi
  for s in "${SKIP[@]:-}"; do [[ "$s" == "$name" ]] && { echo "[skip] $name"; return 0; }; done
  echo "=== [$name] starting at $(date -Is) ==="
  "$@"
  echo "=== [$name] finished at $(date -Is) ==="
}

run_stage smoke pixi run python -m validation.run_smoke \
  --inputs-dir "$INPUTS" \
  --beagle "$BEAGLE" \
  --out-dir "$OUT_BASE/smoke"

run_stage loso_dosage pixi run python -m validation.run_loso \
  --inputs-dir "$INPUTS" \
  --samples-locations "$SITES_TSV" \
  --sample-table "$SAMPLE_TABLE" \
  --beagle "$BEAGLE" \
  --gl-mode dosage \
  --out-dir "$OUT_BASE/loso_dosage"

run_stage loso_full_gl pixi run python -m validation.run_loso \
  --inputs-dir "$INPUTS" \
  --samples-locations "$SITES_TSV" \
  --sample-table "$SAMPLE_TABLE" \
  --beagle "$BEAGLE" \
  --gl-mode full_gl \
  --out-dir "$OUT_BASE/loso_full_gl"

run_stage summarize pixi run python -m validation.summarize \
  --baseline-dir "$OUT_BASE/baseline" \
  --loso-dosage-dir "$OUT_BASE/loso_dosage" \
  --loso-full-gl-dir "$OUT_BASE/loso_full_gl" \
  --samples-locations "$SITES_TSV" \
  --out-dir validation/summary \
  --figures-dir validation/figures
