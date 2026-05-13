#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BADMINTONMIMIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MUSCLEMIMIC_ROOT="$(cd "${BADMINTONMIMIC_ROOT}/.." && pwd)"

source "${BADMINTONMIMIC_ROOT}/configs/env.sh"
python "${BADMINTONMIMIC_ROOT}/scripts/build_config_from_manifests.py"
bash "${BADMINTONMIMIC_ROOT}/scripts/install_configs.sh"

cd "${MUSCLEMIMIC_ROOT}"

"${MUSCLEMIMIC_ROOT}/scripts/run_with_cuda_compat.sh" \
  uv run fullbody/experiment.py \
  --config-name=config_specific_task/conf_fullbody_badminton_gmr \
  wandb.mode="${WANDB_MODE:-disabled}"
