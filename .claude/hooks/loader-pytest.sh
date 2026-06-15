#!/usr/bin/env bash
# PostToolUse hook: when the loader-patch core files are edited, run the
# focused dispatch tests so we catch regressions in the integer/float
# matrix dispatch immediately. The core files are the four touched in
# the genotype_likelihoods PR; an edit to any of them can change loader
# behavior in subtle ways.
#
# Always exits 0 — never blocks an edit. Test failures are reported to
# stderr but do not interrupt the workflow. We deliberately run only a
# subset (-k dispatch) so the post-edit feedback loop stays fast (<10s).

set -uo pipefail

input="$(cat)"
file_path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')"

case "$file_path" in
  */locator/loaders.py | \
  */locator/training.py | \
  */locator/prediction.py | \
  */locator/data/filters.py)
    ;;
  *) exit 0 ;;
esac

if ! command -v pixi >/dev/null 2>&1; then
  exit 0
fi

repo_root="$(git -C "$(dirname "$file_path")" rev-parse --show-toplevel 2>/dev/null)"
[ -n "$repo_root" ] || exit 0
cd "$repo_root" || exit 0

echo "[loader-pytest] running dispatch tests after edit to ${file_path#$repo_root/}" >&2
pixi run pytest tests/test_data_loading.py -k dispatch -x --tb=line -q 2>&1 \
  | tail -n 20 >&2 || true

exit 0
