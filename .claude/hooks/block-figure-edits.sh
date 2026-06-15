#!/usr/bin/env bash
# PreToolUse hook: block direct edits to validation/figures/*.png.
# Those PNGs are reproducible artefacts written by validation/summarize.py.
# Editing them by hand decouples the figure from the data it claims to
# show. Use the regen-figures skill (or rerun summarize.py) instead.

set -uo pipefail

input="$(cat)"
file_path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')"

case "$file_path" in
  */validation/figures/*.png)
    cat >&2 <<'EOF'
[block-figure-edits] refusing to edit a tracked figure PNG directly.

  These figures are produced by validation/summarize.py and must stay in
  sync with the predictions in out/path_d_validation/. Edit the renderer
  (validation/summarize.py) or rerun:

      .claude/skills/regen-figures/run.sh

EOF
    exit 2
    ;;
esac

exit 0
