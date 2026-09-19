#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <local9|remote7> [--dry-run]" >&2
  exit 2
fi
server="$1"
mode="${2:-run}"
if [[ "$mode" != run && "$mode" != --dry-run ]]; then
  echo "unknown mode: $mode" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/experiment.env"
case "$server" in
  local9) source "$script_dir/server_local9.env" ;;
  remote7) source "$script_dir/server_remote7.env" ;;
  *) echo "unknown server profile: $server" >&2; exit 2 ;;
esac

"$script_dir/preflight.sh" "$server"
[[ "$S2_RACKET_TRACKING_EVAL" == 1 ]] || {
  echo "racket tracking evaluation contract is not enabled" >&2
  exit 2
}

export PATH="$S2_TOOLS_BIN:$PATH"
export UV_CACHE_DIR="$S2_UV_CACHE_DIR"
if [[ -n "$S2_UV_PYTHON_INSTALL_DIR" ]]; then
  export UV_PYTHON_INSTALL_DIR="$S2_UV_PYTHON_INSTALL_DIR"
fi
if [[ -n "$S2_UV_PROJECT_ENVIRONMENT" ]]; then
  export UV_PROJECT_ENVIRONMENT="$S2_UV_PROJECT_ENVIRONMENT"
  export UV_NO_SYNC=1
fi
export MUSCLEMIMIC_JAX_CACHE_ROOT="$S2_JAX_CACHE_ROOT"
export MUSCLEMIMIC_DATASETS_ROOT="$S2_DATASETS_ROOT"
export MUSCLEMIMIC_GMR_CACHE_PATH="$S2_DATASETS_ROOT"
export MUSCLEMIMIC_SMPL_MODEL_PATH="$S2_SMPL_MODEL_PATH"
export MM_CUDA_COMPAT_ROOT="$S2_CUDA_COMPAT_ROOT"
export TMPDIR="$S2_TMPDIR"
export MPLCONFIGDIR="$S2_MPLCONFIGDIR"
export CUDA_VISIBLE_DEVICES="$S2_PHYSICAL_GPU"
export MM_CUDA_VISIBLE_DEVICES="$S2_PHYSICAL_GPU"
export MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB=4
export MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$S2_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_root="$S2_ASSET_ROOT/$S2_OUTPUT_ROOT"
teacher="$S2_ASSET_ROOT/$S2_TEACHER_CHECKPOINT"
dataset="$run_root/dataset_seed0"
bc_ckpt="$run_root/bc_seed0/checkpoints/checkpoint_$S2_BC_STEPS"
dagger_ckpt="$run_root/dagger_seed0/iter_000/checkpoints/checkpoint_$S2_DAGGER_TRAIN_STEPS"
compare_root="$run_root/compare_seed0"
logs="$run_root/logs"
events="$run_root/experiment_events.jsonl"

[[ -d "$bc_ckpt" ]] || { echo "missing completed BC checkpoint: $bc_ckpt" >&2; exit 1; }
[[ -d "$dagger_ckpt" ]] || { echo "missing completed DAgger checkpoint: $dagger_ckpt" >&2; exit 1; }
[[ -f "$dataset/dataset_manifest.json" ]] || { echo "missing dataset manifest" >&2; exit 1; }
[[ "$(jq -r '[.collections[] | .num_samples] | add' "$dataset/dataset_manifest.json")" == 256 ]] || {
  echo "dataset manifest does not contain the expected 128+64+64 samples" >&2
  exit 1
}
if [[ "$mode" == run && -e "$compare_root/comparison_metrics.json" ]]; then
  echo "refusing to overwrite completed comparison: $compare_root/comparison_metrics.json" >&2
  exit 2
fi
mkdir -p "$logs"

record_event() {
  printf '{"timestamp":"%s","event":"%s","server":"%s","run_uid":"%s","experiment_class":"%s","evaluation_code_sha":"%s"}\n' \
    "$(date -Is)" "$1" "$S2_SERVER_ID" "$S2_RUN_UID" "$S2_EXPERIMENT_CLASS" "$S2_CODE_SHA" >> "$events"
}

export MUSCLEMIMIC_JAX_CACHE_KEY="${S2_CACHE_PREFIX}_compare_racket_eval_v2"
export MUSCLEMIMIC_TRAIN_LOG="$logs/compare_racket_eval_v2.log"
if [[ "$mode" == --dry-run ]]; then
  export MUSCLEMIMIC_DRY_RUN=1
  record_event comparison_racket_eval_v2_dry_run_started
else
  record_event comparison_racket_eval_v2_started
fi

set +e
"$S2_REPO_ROOT/scripts/run_fullbody_training.sh" --distill-compare \
  --teacher_ckpt "$teacher" --student_ckpt "$bc_ckpt" \
  --student_dagger_ckpt "$dagger_ckpt" --output_dir "$compare_root" \
  --motion_path "$S2_HELDOUT_MOTION_1" "$S2_HELDOUT_MOTION_2" \
    "$S2_HELDOUT_MOTION_3" "$S2_HELDOUT_MOTION_4" \
  --metrics_envs "$S2_COMPARE_ENVS" --metrics_steps "$S2_COMPARE_STEPS" \
  --eval_seed "$S2_SEED" --deterministic --racket_tracking_eval
rc=$?
set -e

if [[ "$mode" == --dry-run ]]; then
  [[ "$rc" == 0 ]] || exit "$rc"
  record_event comparison_racket_eval_v2_dry_run_completed
  echo "COMPARE DRY RUN PASSED: evaluation was not started"
  exit 0
fi
if [[ "$rc" != 0 ]]; then
  record_event "comparison_racket_eval_v2_failed_rc_${rc}"
  exit "$rc"
fi
record_event comparison_racket_eval_v2_completed
record_event pipeline_completed_after_comparison_retry
echo "COMPARISON COMPLETED: $compare_root"
