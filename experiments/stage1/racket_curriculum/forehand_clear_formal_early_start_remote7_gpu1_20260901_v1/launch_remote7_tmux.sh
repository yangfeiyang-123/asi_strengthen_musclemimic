#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/experiment.env"
source "$script_dir/server_remote7.env"
session=stage2_fc_formal_early_remote7_gpu1_v4

"$script_dir/preflight.sh" remote7
if tmux -S "$S2_TMUX_SOCKET" has-session -t "$session" 2>/dev/null; then
  echo "refusing to reuse existing tmux session: $session" >&2
  exit 2
fi
tmux -S "$S2_TMUX_SOCKET" new-session -d -s "$session" -n pipeline \
  "bash '$script_dir/run_pipeline.sh' remote7"
echo "started socket=$S2_TMUX_SOCKET session=$session physical_gpu=$S2_PHYSICAL_GPU"
