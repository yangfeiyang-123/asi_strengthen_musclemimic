#!/usr/bin/env bash
set -euo pipefail

# Run from the root of your repository after copying this package's prompts to ./codex_tasks.
# This uses the minimal non-interactive Codex form: codex exec "$(cat prompt.md)".

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found. Install it first, or run 'codex' interactively and paste the prompts." >&2
  exit 1
fi

for task in \
  01_audit_models.md \
  02_add_grip_sites_and_scene.md \
  03_implement_grip_ik_solver.md \
  04_implement_grip_training_env.md \
  05_validate_grip_pipeline.md
 do
  echo "=== Running Codex task: ${task} ==="
  codex exec "$(cat "codex_tasks/${task}")"
 done
