#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <local9|remote7>" >&2
  exit 2
fi

server="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/experiment.env"
case "$server" in
  local9) source "$script_dir/server_local9.env" ;;
  remote7) source "$script_dir/server_remote7.env" ;;
  *) echo "unknown server profile: $server" >&2; exit 2 ;;
esac

"$script_dir/preflight.sh" "$server" first-step

export PATH="$STAGE2_TOOLS_BIN:$PATH"
export UV_CACHE_DIR="$STAGE2_UV_CACHE_DIR"
if [[ -n "$STAGE2_UV_PYTHON_INSTALL_DIR" ]]; then
  export UV_PYTHON_INSTALL_DIR="$STAGE2_UV_PYTHON_INSTALL_DIR"
fi
if [[ -n "$STAGE2_UV_PROJECT_ENVIRONMENT" ]]; then
  export UV_PROJECT_ENVIRONMENT="$STAGE2_UV_PROJECT_ENVIRONMENT"
  export UV_NO_SYNC=1
fi
export MUSCLEMIMIC_JAX_CACHE_ROOT="$STAGE2_JAX_CACHE_ROOT"
export MUSCLEMIMIC_DATASETS_ROOT="$STAGE2_REPO_ROOT/datasets"
export MUSCLEMIMIC_GMR_CACHE_PATH="$STAGE2_REPO_ROOT/datasets"
export MUSCLEMIMIC_SMPL_MODEL_PATH="$STAGE2_REPO_ROOT/smpl_models/smplh"
export TMPDIR="$STAGE2_TMPDIR"
export MPLCONFIGDIR="$STAGE2_MPLCONFIGDIR"
export CUDA_VISIBLE_DEVICES="$STAGE2_PHYSICAL_GPU"
export MUSCLEMIMIC_JAX_CACHE_KEY="${STAGE2_CACHE_PREFIX}_stage1r_train"
export MUSCLEMIMIC_TRAIN_LOG="$STAGE2_ASSET_ROOT/$STAGE2_OUTPUT_ROOT/logs/stage1r_train.log"
export MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB=4
export MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$STAGE2_REPO_ROOT"
exec uv run --locked python -m fullbody.run_forehand_clear_pipeline \
  --profile "$STAGE2_PROFILE" \
  --action "$STAGE2_ACTION" \
  --output_dir "$STAGE2_ASSET_ROOT/$STAGE2_OUTPUT_ROOT" \
  --stage1_checkpoint "$STAGE2_ASSET_ROOT/$STAGE2_T3_CHECKPOINT" \
  --stage1_peasd_promotion_manifest "$STAGE2_ASSET_ROOT/$STAGE2_STAGE1_PROMOTION" \
  --emg_reference_manifest "$STAGE2_ASSET_ROOT/$STAGE2_TUBE" \
  --stage1_peasd_latent_arm disabled \
  --execute_step stage1r_train
