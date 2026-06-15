#!/usr/bin/env bash
# Regenerate figures + summary from existing predictions, no training.
# See SKILL.md.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

OUT="out/path_d_validation"

pixi run python -m validation.summarize \
  --baseline-dir       "$OUT/baseline" \
  --loso-dosage-dir    "$OUT/loso_dosage" \
  --loso-full-gl-dir   "$OUT/loso_full_gl" \
  --samples-locations  data/balanus_locations.tsv \
  --out-dir            validation/summary \
  --figures-dir        validation/figures
