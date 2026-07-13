#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: musclemimic/badminton/scripts/eval_fullbody.sh <checkpoint-or-hf-path> <motion-path-without-npz> [extra eval args...]"
  exit 1
fi

CHECKPOINT_PATH="$1"
MOTION_PATH="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSCLEMIMIC_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${MUSCLEMIMIC_ROOT}/configs/env.sh"

cd "${MUSCLEMIMIC_ROOT}"

uv run fullbody/eval.py \
  --path "${CHECKPOINT_PATH}" \
  --motion_path "${MOTION_PATH}" \
  --use_mujoco \
  --stochastic \
  --eval_seed 0 \
  --n_steps 1000 \
  "$@"
