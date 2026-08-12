#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  export CUDA_VISIBLE_DEVICES=<physical_gpu_index>
  export MUSCLEMIMIC_JAX_CACHE_KEY=<task_cache_key>
  export MUSCLEMIMIC_TRAIN_LOG=<log_path>
  # Continuity reward runs additionally require:
  export MUSCLEMIMIC_CONTINUITY_SMOKE_ARTIFACT=<passing_smoke.json>
  scripts/run_fullbody_training.sh --config-name=<hydra_config> [hydra_overrides...]

Stage-3 incoming-shuttle PPO:
  scripts/run_fullbody_training.sh --incoming-hit \
    --spec experiments/posttrain/<spec>.yaml --stage train-gpu [runner_args...]

Stage-3 single-feed MJX-CEM teacher search:
  scripts/run_fullbody_training.sh --incoming-hit-cem \
    --spec experiments/posttrain/<spec>.yaml --checkpoint <policy.npz> \
    --out-dir <run_dir> [search_args...]

Latent / direct-distillation training:
  scripts/run_fullbody_training.sh --latent --config <latent.yaml> [trainer_args...]
  scripts/run_fullbody_training.sh --distill-bc --dataset_dir <dataset> [trainer_args...]
  scripts/run_fullbody_training.sh --distill-dagger \
    --teacher_ckpt <teacher> --initial_student_ckpt <bc> [runner_args...]

Deterministic direct-distillation evaluation (read-only):
  scripts/run_fullbody_training.sh --distill-compare \
    --teacher_ckpt <teacher> --student_ckpt <bc> [evaluator_args...]

Stage-1 PEASD sealed endpoint evaluation (read-only; never trains):
  scripts/run_fullbody_training.sh --stage1-peasd-eval \
    --checkpoint <exact_checkpoint_leaf> [--reference-cache <verified_tube>]

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

launch_mode="fullbody"
if [[ "${1}" == "--incoming-hit" ]]; then
  launch_mode="incoming-hit"
  shift
  if [[ $# -eq 0 ]]; then
    echo "--incoming-hit requires runner arguments" >&2
    exit 2
  fi
fi
if [[ "${1}" == "--incoming-hit-cem" ]]; then
  launch_mode="incoming-hit-cem"
  shift
  if [[ $# -eq 0 ]]; then
    echo "--incoming-hit-cem requires optimizer arguments" >&2
    exit 2
  fi
fi
if [[ "${1}" == "--latent" ]]; then
  launch_mode="latent"
  shift
  if [[ $# -eq 0 ]]; then
    echo "--latent requires latent trainer arguments" >&2
    exit 2
  fi
fi
if [[ "${1}" == "--distill-bc" ]]; then
  launch_mode="distill-bc"
  shift
  if [[ $# -eq 0 ]]; then
    echo "--distill-bc requires distillation trainer arguments" >&2
    exit 2
  fi
fi
if [[ "${1}" == "--distill-dagger" ]]; then
  launch_mode="distill-dagger"
  shift
  if [[ $# -eq 0 ]]; then
    echo "--distill-dagger requires DAgger runner arguments" >&2
    exit 2
  fi
fi
if [[ "${1}" == "--distill-compare" ]]; then
  launch_mode="distill-compare"
  shift
  if [[ $# -eq 0 ]]; then
    echo "--distill-compare requires evaluator arguments" >&2
    exit 2
  fi
fi
if [[ "${1}" == "--stage1-peasd-eval" ]]; then
  launch_mode="stage1-peasd-eval"
  shift
  if [[ $# -eq 0 ]]; then
    echo "--stage1-peasd-eval requires evaluator arguments" >&2
    exit 2
  fi
fi

: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must name one physical GPU}"
: "${MUSCLEMIMIC_JAX_CACHE_KEY:?MUSCLEMIMIC_JAX_CACHE_KEY is required}"
: "${MUSCLEMIMIC_TRAIN_LOG:?MUSCLEMIMIC_TRAIN_LOG is required}"

if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must be one non-negative physical GPU index, got: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

has_config_name=false
has_wandb_mode=false
has_spec=false
has_train_gpu_stage=false
has_checkpoint=false
requires_continuity_smoke=false
for arg in "$@"; do
  case "${arg}" in
    --config-name | --config-name=*) has_config_name=true ;;
    wandb.mode=*) has_wandb_mode=true ;;
    --spec | --spec=*) has_spec=true ;;
    train-gpu | --stage=train-gpu) has_train_gpu_stage=true ;;
    --checkpoint | --checkpoint=*) has_checkpoint=true ;;
  esac
  case "${arg}" in
    *continuity_[abcg]1_s* | *continuity_reward*) requires_continuity_smoke=true ;;
  esac
done

if [[ "${launch_mode}" == "fullbody" && "${has_config_name}" != true ]]; then
    echo "A Hydra --config-name argument is required." >&2
    exit 2
fi
if [[ "${launch_mode}" == "incoming-hit" ]]; then
  if [[ "${has_spec}" != true || "${has_train_gpu_stage}" != true ]]; then
    echo "--incoming-hit requires --spec and --stage train-gpu." >&2
    exit 2
  fi
fi
if [[ "${launch_mode}" == "incoming-hit-cem" && "${has_spec}" != true ]]; then
  echo "--incoming-hit-cem requires --spec." >&2
  exit 2
fi
if [[ "${launch_mode}" == "stage1-peasd-eval" && "${has_checkpoint}" != true ]]; then
  echo "--stage1-peasd-eval requires --checkpoint." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${requires_continuity_smoke}" == true ]]; then
  if [[ -z "${MUSCLEMIMIC_CONTINUITY_SMOKE_ARTIFACT:-}" ]]; then
    if [[ "${MUSCLEMIMIC_DRY_RUN:-0}" == "1" ]]; then
      echo "[launch] continuity smoke gate=pending (dry-run only)" >&2
    else
      echo "MUSCLEMIMIC_CONTINUITY_SMOKE_ARTIFACT is required for continuity reward training." >&2
      exit 2
    fi
  elif [[ ! -f "${MUSCLEMIMIC_CONTINUITY_SMOKE_ARTIFACT}" ]]; then
    echo "Continuity smoke artifact does not exist: ${MUSCLEMIMIC_CONTINUITY_SMOKE_ARTIFACT}" >&2
    exit 2
  fi
fi

# This is the authoritative source for datasets, local retarget caches, SMPL
# models, and the private CUDA compatibility library.
source "${REPO_ROOT}/configs/env.sh"

export CUDA_VISIBLE_DEVICES
export MM_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB="${MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB:-4}"
export MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB="${MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB:-4}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${MUSCLEMIMIC_JAX_CACHE_ROOT}/${MUSCLEMIMIC_JAX_CACHE_KEY}}"

mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "$(dirname "${MUSCLEMIMIC_TRAIN_LOG}")"

launch_args=("$@")
if [[ "${launch_mode}" == "fullbody" && "${has_wandb_mode}" != true ]]; then
  launch_args+=("wandb.mode=${WANDB_MODE}")
fi

if [[ "${launch_mode}" == "incoming-hit" ]]; then
  command=(
    "${REPO_ROOT}/scripts/run_with_cuda_compat.sh"
    uv run --locked python -m musclemimic.badminton.scripts.run_incoming_shuttle_hit
    "${launch_args[@]}"
  )
elif [[ "${launch_mode}" == "incoming-hit-cem" ]]; then
  command=(
    "${REPO_ROOT}/scripts/run_with_cuda_compat.sh"
    uv run --locked python scripts/optimize_single_feed_hit_mjx.py
    "${launch_args[@]}"
  )
elif [[ "${launch_mode}" == "latent" ]]; then
  command=(
    "${REPO_ROOT}/scripts/run_with_cuda_compat.sh"
    uv run --locked python -m fullbody.latent_train
    "${launch_args[@]}"
  )
elif [[ "${launch_mode}" == "distill-bc" ]]; then
  command=(
    "${REPO_ROOT}/scripts/run_with_cuda_compat.sh"
    uv run --locked python -m fullbody.distill_train_bc
    "${launch_args[@]}"
  )
elif [[ "${launch_mode}" == "distill-dagger" ]]; then
  command=(
    "${REPO_ROOT}/scripts/run_with_cuda_compat.sh"
    uv run --locked python -m fullbody.distill_run_dagger
    "${launch_args[@]}"
  )
elif [[ "${launch_mode}" == "distill-compare" ]]; then
  command=(
    "${REPO_ROOT}/scripts/run_with_cuda_compat.sh"
    uv run --locked python -m fullbody.distill_compare
    "${launch_args[@]}"
  )
elif [[ "${launch_mode}" == "stage1-peasd-eval" ]]; then
  command=(
    "${REPO_ROOT}/scripts/run_with_cuda_compat.sh"
    uv run --locked python scripts/evaluate_stage1_peasd.py
    "${launch_args[@]}"
  )
else
  command=(
    "${REPO_ROOT}/scripts/run_with_cuda_compat.sh"
    uv run --locked fullbody/experiment.py
    "${launch_args[@]}"
  )
fi

{
  echo "[launch] repo=${REPO_ROOT}"
  echo "[launch] mode=${launch_mode}"
  if [[ "${launch_mode}" == "stage1-peasd-eval" || "${launch_mode}" == "distill-compare" ]]; then
    echo "[launch] workload=read-only-evaluation (training is disabled)"
  fi
  echo "[launch] physical_gpu=${CUDA_VISIBLE_DEVICES}"
  echo "[launch] config_cache=${JAX_COMPILATION_CACHE_DIR}"
  echo "[launch] gmr_cache=${MUSCLEMIMIC_GMR_CACHE_PATH}"
  echo "[launch] continuity_smoke=${MUSCLEMIMIC_CONTINUITY_SMOKE_ARTIFACT:-not-required}"
  echo "[launch] log=${MUSCLEMIMIC_TRAIN_LOG}"
  printf '[launch] command='
  printf ' %q' "${command[@]}"
  printf '\n'
} | tee -a "${MUSCLEMIMIC_TRAIN_LOG}"

if [[ "${MUSCLEMIMIC_DRY_RUN:-0}" == "1" ]]; then
  if [[ "${launch_mode}" == "stage1-peasd-eval" ]]; then
    echo "[launch] dry-run complete; evaluation was not started and training is disabled" | tee -a "${MUSCLEMIMIC_TRAIN_LOG}"
    exit 0
  fi
  if [[ "${launch_mode}" != "fullbody" ]]; then
    echo "[launch] dry-run complete; training was not started" | tee -a "${MUSCLEMIMIC_TRAIN_LOG}"
    exit 0
  fi
  dry_run_command=(
    "${REPO_ROOT}/scripts/run_with_cuda_compat.sh"
    uv run --locked python scripts/resolve_fullbody_training.py
    "${launch_args[@]}"
  )
  "${dry_run_command[@]}" 2>&1 | tee -a "${MUSCLEMIMIC_TRAIN_LOG}"
  echo "[launch] dry-run complete; training was not started" | tee -a "${MUSCLEMIMIC_TRAIN_LOG}"
  exit 0
fi

if [[ "${launch_mode}" == "fullbody" ]]; then
  preflight_command=(
    "${REPO_ROOT}/scripts/run_with_cuda_compat.sh"
    uv run --locked python scripts/resolve_fullbody_training.py
    --validate-continuity-smoke
    "${launch_args[@]}"
  )
  "${preflight_command[@]}" 2>&1 | tee -a "${MUSCLEMIMIC_TRAIN_LOG}"
fi

"${command[@]}" 2>&1 | tee -a "${MUSCLEMIMIC_TRAIN_LOG}"
