#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/experiment.env"
source "$script_dir/server_remote7.env"

if [[ "${MUSCLEMIMIC_CONFIRM_PRODUCTION:-0}" != 1 ]]; then
  echo "refusing production launch: set MUSCLEMIMIC_CONFIRM_PRODUCTION=1 only after preflight passes" >&2
  exit 2
fi

"$script_dir/preflight.sh" remote7 first-step
session=stage2_fc_stage1r_remote7_gpu1_v1
if tmux -S "$STAGE2_TMUX_SOCKET" has-session -t "$session" 2>/dev/null; then
  echo "refusing to reuse existing tmux session: $session" >&2
  exit 2
fi

tmux -S "$STAGE2_TMUX_SOCKET" new-session -d -s "$session" -n train \
  "bash '$script_dir/run_first_stage2_step.sh' remote7"
echo "started socket=$STAGE2_TMUX_SOCKET session=$session physical_gpu=$STAGE2_PHYSICAL_GPU"
