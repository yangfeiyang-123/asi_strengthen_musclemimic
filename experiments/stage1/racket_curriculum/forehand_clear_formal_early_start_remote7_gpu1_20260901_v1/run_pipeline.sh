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
# The remote server intentionally reuses a dependency-complete virtualenv from
# the Stage-1 checkout.  That environment contains an editable install pointing
# at the old worktree, so pin imports to this experiment's immutable checkout.
export PYTHONPATH="$S2_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_root="$S2_ASSET_ROOT/$S2_OUTPUT_ROOT"
teacher="$S2_ASSET_ROOT/$S2_TEACHER_CHECKPOINT"
dataset="$run_root/dataset_seed0"
bc_root="$run_root/bc_seed0"
bc_ckpt="$bc_root/checkpoints/checkpoint_$S2_BC_STEPS"
dagger_root="$run_root/dagger_seed0"
dagger_ckpt="$dagger_root/iter_000/checkpoints/checkpoint_$S2_DAGGER_TRAIN_STEPS"
compare_root="$run_root/compare_seed0"
logs="$run_root/logs"
events="$run_root/experiment_events.jsonl"
mkdir -p "$logs"

record_event() {
  printf '{"timestamp":"%s","event":"%s","server":"%s","run_uid":"%s","experiment_class":"%s"}\n' \
    "$(date -Is)" "$1" "$S2_SERVER_ID" "$S2_RUN_UID" "$S2_EXPERIMENT_CLASS" >> "$events"
}
on_error() {
  local rc=$?
  record_event "pipeline_failed_rc_${rc}"
  exit "$rc"
}
trap on_error ERR

cd "$S2_REPO_ROOT"
collect_cmd=(
  "$S2_REPO_ROOT/scripts/run_with_cuda_compat.sh"
  uv run --locked python -m fullbody.distill_collect
  --teacher_ckpt "$teacher"
  --output_dir "$dataset"
  --num_envs "$S2_NUM_ENVS"
  --num_transitions "$S2_TEACHER_TRANSITIONS"
  --shard_size "$S2_SHARD_SIZE"
  --seed "$S2_SEED"
  --split train
  --run_uid "$S2_RUN_UID"
  --deterministic_teacher
  --teacher_action_target mean
  --save_reference_features
  --test_only_allow_unpromoted_teacher
)
if [[ -f "$dataset/dataset_manifest.json" ]]; then
  collect_cmd+=(--resume_dataset)
fi
val_collect_cmd=(
  "$S2_REPO_ROOT/scripts/run_with_cuda_compat.sh"
  uv run --locked python -m fullbody.distill_collect
  --teacher_ckpt "$teacher"
  --output_dir "$dataset"
  --num_envs "$S2_NUM_ENVS"
  --num_transitions "$S2_VALIDATION_TRANSITIONS"
  --shard_size "$S2_SHARD_SIZE"
  --seed "$S2_SEED"
  --split val
  --motion_path "$S2_HELDOUT_MOTION_1"
  --run_uid "$S2_RUN_UID"
  --deterministic_teacher
  --teacher_action_target mean
  --save_reference_features
  --test_only_allow_unpromoted_teacher
  --resume_dataset
)

if [[ "$mode" == --dry-run ]]; then
  record_event dry_run_started
  printf '[dry-run] collection command=' | tee -a "$logs/dry_run.log"
  printf ' %q' "${collect_cmd[@]}" | tee -a "$logs/dry_run.log"
  printf '\n' | tee -a "$logs/dry_run.log"
  printf '[dry-run] validation collection command=' | tee -a "$logs/dry_run.log"
  printf ' %q' "${val_collect_cmd[@]}" | tee -a "$logs/dry_run.log"
  printf '\n' | tee -a "$logs/dry_run.log"
  export MUSCLEMIMIC_DRY_RUN=1

  export MUSCLEMIMIC_JAX_CACHE_KEY="${S2_CACHE_PREFIX}_bc"
  export MUSCLEMIMIC_TRAIN_LOG="$logs/bc.log"
  "$S2_REPO_ROOT/scripts/run_fullbody_training.sh" --distill-bc \
    --dataset_dir "$dataset" \
    --student_config "$S2_REPO_ROOT/$S2_STUDENT_CONFIG" \
    --output_dir "$bc_root" \
    --batch_size "$S2_BATCH_SIZE" --num_steps "$S2_BC_STEPS" \
    --lr "$S2_LR" --seed "$S2_SEED" --log_interval 1 \
    --convergence_eval_interval 5 --require_dataset_manifest

  export MUSCLEMIMIC_JAX_CACHE_KEY="${S2_CACHE_PREFIX}_dagger"
  export MUSCLEMIMIC_TRAIN_LOG="$logs/dagger.log"
  "$S2_REPO_ROOT/scripts/run_fullbody_training.sh" --distill-dagger \
    --teacher_ckpt "$teacher" --test_only_allow_unpromoted_teacher \
    --initial_student_ckpt "$bc_ckpt" \
    --student_config "$S2_REPO_ROOT/$S2_STUDENT_CONFIG" \
    --dataset_dir "$dataset" --output_dir "$dagger_root" \
    --num_iters "$S2_DAGGER_ITERS" --num_envs "$S2_NUM_ENVS" \
    --num_transitions "$S2_DAGGER_TRANSITIONS" --shard_size "$S2_SHARD_SIZE" \
    --train_steps "$S2_DAGGER_TRAIN_STEPS" --batch_size "$S2_BATCH_SIZE" \
    --lr "$S2_LR" --seed "$S2_SEED" --resume_dataset \
    --run_uid "$S2_RUN_UID" --save_reference_features \
    --physical_gpu "$S2_PHYSICAL_GPU" \
    --jax_cache_key_prefix "${S2_CACHE_PREFIX}_dagger" \
    --train_log_dir "$logs/dagger_iterations"

  export MUSCLEMIMIC_JAX_CACHE_KEY="${S2_CACHE_PREFIX}_compare"
  export MUSCLEMIMIC_TRAIN_LOG="$logs/compare.log"
  "$S2_REPO_ROOT/scripts/run_fullbody_training.sh" --distill-compare \
    --teacher_ckpt "$teacher" --student_ckpt "$bc_ckpt" \
    --student_dagger_ckpt "$dagger_ckpt" --output_dir "$compare_root" \
    --motion_path "$S2_HELDOUT_MOTION_1" "$S2_HELDOUT_MOTION_2" \
      "$S2_HELDOUT_MOTION_3" "$S2_HELDOUT_MOTION_4" \
    --metrics_envs "$S2_COMPARE_ENVS" --metrics_steps "$S2_COMPARE_STEPS" \
    --eval_seed "$S2_SEED" --deterministic --racket_tracking_eval
  record_event dry_run_completed
  echo "DRY RUN PASSED: no collection or training process was started"
  exit 0
fi

record_event pipeline_started
export MUSCLEMIMIC_JAX_CACHE_KEY="${S2_CACHE_PREFIX}_collect"
teacher_existing=0
validation_existing=0
if [[ -f "$dataset/dataset_manifest.json" ]]; then
  teacher_existing="$(jq -r '[.collections[]? | select(.collection_id == "teacher_train") | .num_samples] | add // 0' "$dataset/dataset_manifest.json")"
  validation_existing="$(jq -r '[.collections[]? | select(.collection_id == "teacher_val") | .num_samples] | add // 0' "$dataset/dataset_manifest.json")"
fi
if [[ "$teacher_existing" == "$S2_TEACHER_TRANSITIONS" ]]; then
  record_event teacher_collection_reused_exact
else
  [[ "$teacher_existing" == 0 ]] || {
    echo "teacher dataset has $teacher_existing samples; expected either 0 or exactly $S2_TEACHER_TRANSITIONS" >&2
    false
  }
  record_event teacher_collection_started
  "${collect_cmd[@]}" 2>&1 | tee -a "$logs/teacher_collection.log"
  record_event teacher_collection_completed
fi

if [[ "$validation_existing" == "$S2_VALIDATION_TRANSITIONS" ]]; then
  record_event validation_collection_reused_exact
else
  [[ "$validation_existing" == 0 ]] || {
    echo "validation dataset has $validation_existing samples; expected either 0 or exactly $S2_VALIDATION_TRANSITIONS" >&2
    false
  }
  record_event validation_collection_started
  "${val_collect_cmd[@]}" 2>&1 | tee -a "$logs/validation_collection.log"
  record_event validation_collection_completed
fi

"$S2_REPO_ROOT/scripts/run_with_cuda_compat.sh" \
  uv run --locked musclemimic-distill-inspect-dataset \
  --dataset_dir "$dataset" --output_json "$run_root/dataset_inspect_after_teacher.json" \
  2>&1 | tee -a "$logs/dataset_inspect.log"

export MUSCLEMIMIC_JAX_CACHE_KEY="${S2_CACHE_PREFIX}_bc"
export MUSCLEMIMIC_TRAIN_LOG="$logs/bc.log"
record_event bc_training_started
"$S2_REPO_ROOT/scripts/run_fullbody_training.sh" --distill-bc \
  --dataset_dir "$dataset" \
  --student_config "$S2_REPO_ROOT/$S2_STUDENT_CONFIG" \
  --output_dir "$bc_root" \
  --batch_size "$S2_BATCH_SIZE" --num_steps "$S2_BC_STEPS" \
  --lr "$S2_LR" --seed "$S2_SEED" --log_interval 1 \
  --convergence_eval_interval 5 --require_dataset_manifest
record_event bc_training_completed

export MUSCLEMIMIC_JAX_CACHE_KEY="${S2_CACHE_PREFIX}_dagger"
export MUSCLEMIMIC_TRAIN_LOG="$logs/dagger.log"
record_event dagger_started
"$S2_REPO_ROOT/scripts/run_fullbody_training.sh" --distill-dagger \
  --teacher_ckpt "$teacher" --test_only_allow_unpromoted_teacher \
  --initial_student_ckpt "$bc_ckpt" \
  --student_config "$S2_REPO_ROOT/$S2_STUDENT_CONFIG" \
  --dataset_dir "$dataset" --output_dir "$dagger_root" \
  --num_iters "$S2_DAGGER_ITERS" --num_envs "$S2_NUM_ENVS" \
  --num_transitions "$S2_DAGGER_TRANSITIONS" --shard_size "$S2_SHARD_SIZE" \
  --train_steps "$S2_DAGGER_TRAIN_STEPS" --batch_size "$S2_BATCH_SIZE" \
  --lr "$S2_LR" --seed "$S2_SEED" --resume_dataset \
  --run_uid "$S2_RUN_UID" --save_reference_features \
  --physical_gpu "$S2_PHYSICAL_GPU" \
  --jax_cache_key_prefix "${S2_CACHE_PREFIX}_dagger" \
  --train_log_dir "$logs/dagger_iterations"
record_event dagger_completed

export MUSCLEMIMIC_JAX_CACHE_KEY="${S2_CACHE_PREFIX}_compare"
export MUSCLEMIMIC_TRAIN_LOG="$logs/compare.log"
record_event comparison_started
"$S2_REPO_ROOT/scripts/run_fullbody_training.sh" --distill-compare \
  --teacher_ckpt "$teacher" --student_ckpt "$bc_ckpt" \
  --student_dagger_ckpt "$dagger_ckpt" --output_dir "$compare_root" \
  --motion_path "$S2_HELDOUT_MOTION_1" "$S2_HELDOUT_MOTION_2" \
    "$S2_HELDOUT_MOTION_3" "$S2_HELDOUT_MOTION_4" \
  --metrics_envs "$S2_COMPARE_ENVS" --metrics_steps "$S2_COMPARE_STEPS" \
  --eval_seed "$S2_SEED" --deterministic --racket_tracking_eval
record_event comparison_completed
record_event pipeline_completed
echo "PIPELINE COMPLETED: $run_root"
