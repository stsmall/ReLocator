#!/usr/bin/env bash
# PostToolUse hook: run ruff check --fix and ruff format on the just-edited
# Python file. Pre-commit catches lint at commit time, but running ruff on
# every Edit/Write keeps the working tree clean during exploratory changes
# and matches what .pre-commit-config.yaml would reject anyway.
#
# Always exits 0 — never blocks an edit. Ruff failures are reported to stderr
# but do not interrupt the workflow.

set -uo pipefail

input="$(cat)"
file_path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')"

case "$file_path" in
  *.py) ;;
  *) exit 0 ;;
esac

[ -f "$file_path" ] || exit 0

if ! command -v pixi >/dev/null 2>&1; then
  exit 0
fi

cd "$(dirname "$file_path")" 2>/dev/null || exit 0
pixi run --manifest-path "$(git rev-parse --show-toplevel 2>/dev/null)/pixi.toml" \
    ruff check --fix --quiet "$file_path" 2>/dev/null || true
pixi run --manifest-path "$(git rev-parse --show-toplevel 2>/dev/null)/pixi.toml" \
    ruff format --quiet "$file_path" 2>/dev/null || true

exit 0
