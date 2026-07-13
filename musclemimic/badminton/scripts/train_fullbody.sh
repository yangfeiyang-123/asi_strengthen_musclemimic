#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSCLEMIMIC_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${MUSCLEMIMIC_ROOT}/configs/env.sh"

cd "${MUSCLEMIMIC_ROOT}"

"${MUSCLEMIMIC_ROOT}/scripts/run_with_cuda_compat.sh" \
  uv run fullbody/experiment.py \
  --config-name=config_specific_task/stage1_body/conf_fullbody_forehand_clear_body_local \
  wandb.mode="${WANDB_MODE:-disabled}"
