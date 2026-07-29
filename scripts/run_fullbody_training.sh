#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  export CUDA_VISIBLE_DEVICES=<physical_gpu_index>
  export MUSCLEMIMIC_JAX_CACHE_KEY=<task_cache_key>
  export MUSCLEMIMIC_TRAIN_LOG=<log_path>
  scripts/run_fullbody_training.sh --config-name=<hydra_config> [hydra_overrides...]

Example:
  export CUDA_VISIBLE_DEVICES=2
  export MUSCLEMIMIC_JAX_CACHE_KEY=chinajump_stage1
  export MUSCLEMIMIC_TRAIN_LOG=datasets/ChinaJump/training/logs/chinajump_root_control_v2_stage1_body_640m.log
  scripts/run_fullbody_training.sh \
    --config-name=config_specific_task/stage1_body/conf_fullbody_chinajump_root_control_v2 \
    wandb.mode=online

Set MUSCLEMIMIC_DRY_RUN=1 to resolve and print the environment without starting
the Python process.
EOF
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must name one physical GPU}"
: "${MUSCLEMIMIC_JAX_CACHE_KEY:?MUSCLEMIMIC_JAX_CACHE_KEY is required}"
: "${MUSCLEMIMIC_TRAIN_LOG:?MUSCLEMIMIC_TRAIN_LOG is required}"

if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "CUDA_VISIBLE_DEVICES must select exactly one physical GPU, got: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

has_config_name=false
has_wandb_mode=false
for arg in "$@"; do
  case "${arg}" in
    --config-name | --config-name=*) has_config_name=true ;;
    wandb.mode=*) has_wandb_mode=true ;;
  esac
done

if [[ "${has_config_name}" != true ]]; then
  echo "A Hydra --config-name argument is required." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# This is the authoritative source for datasets, local retarget caches, SMPL
# models, and the private CUDA compatibility library.
source "${REPO_ROOT}/configs/env.sh"

export CUDA_VISIBLE_DEVICES
export MM_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB="${MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB:-4}"
export MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB="${MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB:-4}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/data3/yangfeiyang/WorkSpace/ENV/jax-cache/${MUSCLEMIMIC_JAX_CACHE_KEY}}"

mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "$(dirname "${MUSCLEMIMIC_TRAIN_LOG}")"

command=(
  "${REPO_ROOT}/scripts/run_with_cuda_compat.sh"
  uv run --locked fullbody/experiment.py
  "$@"
)
if [[ "${has_wandb_mode}" != true ]]; then
  command+=("wandb.mode=${WANDB_MODE}")
fi

{
  echo "[launch] repo=${REPO_ROOT}"
  echo "[launch] physical_gpu=${CUDA_VISIBLE_DEVICES}"
  echo "[launch] config_cache=${JAX_COMPILATION_CACHE_DIR}"
  echo "[launch] gmr_cache=${MUSCLEMIMIC_GMR_CACHE_PATH}"
  echo "[launch] log=${MUSCLEMIMIC_TRAIN_LOG}"
  printf '[launch] command='
  printf ' %q' "${command[@]}"
  printf '\n'
} | tee -a "${MUSCLEMIMIC_TRAIN_LOG}"

if [[ "${MUSCLEMIMIC_DRY_RUN:-0}" == "1" ]]; then
  echo "[launch] dry-run complete; training was not started" | tee -a "${MUSCLEMIMIC_TRAIN_LOG}"
  exit 0
fi

"${command[@]}" 2>&1 | tee -a "${MUSCLEMIMIC_TRAIN_LOG}"
