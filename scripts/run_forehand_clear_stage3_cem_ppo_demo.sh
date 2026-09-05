#!/usr/bin/env bash
set -Eeuo pipefail

# Historical-baseline Stage-3 demo pipeline:
#   audited stance/feed -> low-dimensional CEM -> independent CPU gate
#   -> zero-delta BC anchor -> feedback PPO.
# This intentionally does not claim the formal PEASD Stage-3 release.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-check}"
PHYSICAL_GPU="${2:-0}"
SPEC="experiments/posttrain/incoming_shuttle_hit_forehand_clear_cem_ppo_demo_v6.yaml"
OUTPUT_ROOT="outputs/posttrain_forehand_clear_cem_ppo_demo_v6"
CHECK_ROOT="${OUTPUT_ROOT}/preflight_seed0"
CEM_BOOTSTRAP_ROOT="${OUTPUT_ROOT}/cem_bootstrap_seed0"
CEM_STRICT_ROOT="${OUTPUT_ROOT}/cem_cpu_verified_seed0"
PPO_ROOT="${OUTPUT_ROOT}/ppo_seed0"
LOG="${OUTPUT_ROOT}/logs/forehand_clear_cem_ppo_demo_v6_seed0.log"

SOURCE_CHECKPOINT="${MUSCLEMIMIC_STAGE3_SOURCE_CHECKPOINT:-/raid/yangfeiyang/musclemimic_runs/forehand_clear_stage3_right_arm_mean_consolidation_v24c/train_gpu_256/checkpoints/checkpoint_000000491520/policy.npz}"
BASE_POLICY="${MUSCLEMIMIC_STAGE3_BASE_POLICY:-/raid/yangfeiyang/musclemimic_runs/forehand_clear_stage3_direct_residual_overhead_v2/frozen_base_ckpt156}"
CHECK_PYTHON="${REPO_ROOT}/.venv/bin/python"
JAX_CACHE_ROOT="${MUSCLEMIMIC_STAGE3_JAX_CACHE_ROOT:-/data3/yangfeiyang/WorkSpace/ENV/jax-cache}"
MPL_CACHE_ROOT="${MUSCLEMIMIC_STAGE3_MPL_CACHE_ROOT:-/data3/yangfeiyang/WorkSpace/ENV/tmp/matplotlib-stage3-cem-v6}"
WANDB_PROJECT="${MUSCLEMIMIC_STAGE3_WANDB_PROJECT:-musclemimic-stage3-demo}"

CEM_SYNERGIES=(
  shoulder_elevation
  shoulder_retraction
  shoulder_internal_rotation
  elbow_extension
  forearm_pronation
  wrist_extension
  wrist_radial_deviation
)
CEM_KNOTS=(1 2 3 4 5)

usage() {
  cat <<'EOF'
Usage:
  scripts/run_forehand_clear_stage3_cem_ppo_demo.sh check
  scripts/run_forehand_clear_stage3_cem_ppo_demo.sh dry-run <physical_gpu>
  scripts/run_forehand_clear_stage3_cem_ppo_demo.sh search <physical_gpu>
  scripts/run_forehand_clear_stage3_cem_ppo_demo.sh train <physical_gpu>
  scripts/run_forehand_clear_stage3_cem_ppo_demo.sh pipeline <physical_gpu>
  scripts/run_forehand_clear_stage3_cem_ppo_demo.sh status <physical_gpu>

Modes:
  check     CPU-only asset, stance, feed and frozen-policy checks.
  dry-run   Print both canonical GPU commands without starting them.
  search    Run bootstrap CEM followed by the strict CPU-gated CEM.
  train     Start PPO only from an already passed strict CEM artifact.
  pipeline  Run checks/search as needed, then automatically start PPO.
  status    Show CEM/PPO reports, log tail and processes on the physical GPU.

The default mode is check.  CEM searches 7 anatomical synergies at temporal
knots 1..5: 35 active variables, while the saved teacher keeps the complete
6-knot artifact.  PPO never starts unless the strict CEM report passes.
EOF
}

require_gpu() {
  if [[ ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
    echo "physical_gpu must be one non-negative integer, got: ${PHYSICAL_GPU}" >&2
    exit 2
  fi
}

require_assets() {
  if [[ ! -x "${CHECK_PYTHON}" ]]; then
    echo "Repository check interpreter is missing: ${CHECK_PYTHON}" >&2
    exit 2
  fi
  if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "Stage-3 source checkpoint is missing: ${SOURCE_CHECKPOINT}" >&2
    exit 2
  fi
  if [[ ! -f "${BASE_POLICY}/manifest.json" ]]; then
    echo "Frozen base-policy manifest is missing: ${BASE_POLICY}/manifest.json" >&2
    exit 2
  fi
}

run_cpu_stage() {
  local stage="$1"
  mkdir -p "${MPL_CACHE_ROOT}"
  CUDA_VISIBLE_DEVICES=-1 JAX_PLATFORMS=cpu MPLCONFIGDIR="${MPL_CACHE_ROOT}" \
    "${REPO_ROOT}/scripts/run_with_cuda_compat.sh" "${CHECK_PYTHON}" -m \
    musclemimic.badminton.scripts.run_incoming_shuttle_hit \
    --spec "${SPEC}" \
    --stage "${stage}" \
    --out-dir "${CHECK_ROOT}" \
    --base-policy-artifact "${BASE_POLICY}" \
    --seed 0 \
    >/dev/null
}

run_checks() {
  require_assets
  source "${REPO_ROOT}/configs/env.sh"
  mkdir -p "${CHECK_ROOT}"
  run_cpu_stage preflight
  run_cpu_stage feed-check
  run_cpu_stage base-only-check
  run_cpu_stage contact-seed-check
  CUDA_VISIBLE_DEVICES=-1 JAX_PLATFORMS=cpu \
    "${REPO_ROOT}/scripts/run_with_cuda_compat.sh" "${CHECK_PYTHON}" -c '
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2])
for name in (
    "preflight_report.json",
    "feed_check_report.json",
    "base_only_report.json",
    "contact_seed_check_report.json",
):
    payload = json.loads((root / name).read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise SystemExit(f"{name} did not pass")
preflight = json.loads((root / "preflight_report.json").read_text(encoding="utf-8"))
stance = preflight["reference_ready_pose"]
left_lead = float(stance["left_foot_forward_lead_m"])
stance_width = float(stance["lateral_stance_width_m"])
print(f"[stage3-v6] stance_left_lead_m={left_lead:.6f}")
print(f"[stage3-v6] stance_width_m={stance_width:.6f}")
print(f"[stage3-v6] source_checkpoint_sha256={hashlib.sha256(source.read_bytes()).hexdigest()}")
print("[stage3-v6] all CPU prerequisite reports passed")
' "${CHECK_ROOT}" "${SOURCE_CHECKPOINT}"
}

export_launch_contract() {
  local cache_key="$1"
  export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
  export MUSCLEMIMIC_JAX_CACHE_KEY="${cache_key}"
  export JAX_COMPILATION_CACHE_DIR="${JAX_CACHE_ROOT}/${cache_key}"
  export MUSCLEMIMIC_TRAIN_LOG="${LOG}"
  export MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB="${MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB:-4}"
  export MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB="${MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB:-4}"
  export WANDB_MODE="online"
}

cem_common_args() {
  printf '%s\n' \
    --spec "${SPEC}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --base-policy-artifact "${BASE_POLICY}" \
    --parameterization anatomical_synergies \
    --time-knots 6 \
    --trainable-synergies "${CEM_SYNERGIES[@]}" \
    --trainable-knot-indices "${CEM_KNOTS[@]}" \
    --replicas 3 \
    --min-replica-fraction 0.6666666666666666 \
    --verification-repeats 2 \
    --max-episode-steps 420 \
    --min-outgoing-z-m-s 0.5 \
    --min-forward-m-s 2.0 \
    --min-racket-face-forward-alignment 0.5 \
    --max-stringbed-height-deficit-m 0.10 \
    --max-hand-height-deficit-m 0.10 \
    --seed 0 \
    --wandb-project "${WANDB_PROJECT}"
}

run_cem_bootstrap() {
  mkdir -p "${CEM_BOOTSTRAP_ROOT}"
  export_launch_contract forehand_clear_stage3_cem_v6_bootstrap_seed0
  mapfile -t common < <(cem_common_args)
  if "${REPO_ROOT}/scripts/run_fullbody_training.sh" \
    --incoming-hit-cem \
    "${common[@]}" \
    --out-dir "${CEM_BOOTSTRAP_ROOT}" \
    --population 512 \
    --iterations 24 \
    --elite-fraction 0.08 \
    --initial-std 0.45 \
    --min-std 0.04 \
    --coordinate-probe-radius 0.35 \
    --search-frontier-copies 3 \
    --wandb-name forehand-clear-stage3-cem-v6-bootstrap-seed0; then
    return 0
  fi
  if ! find "${CEM_BOOTSTRAP_ROOT}" -maxdepth 1 -type f \
    -name 'search_seed_frontier_*_unqualified.json' -print -quit \
    | grep -q .; then
    echo "[stage3-v6] bootstrap CEM failed before producing a reusable candidate" >&2
    return 2
  fi
  echo "[stage3-v6] bootstrap did not yet pass; continuing from its sealed best candidate" >&2
}

run_cem_strict() {
  local initial_candidate
  initial_candidate="$(
    find "${CEM_BOOTSTRAP_ROOT}" -maxdepth 1 -type f \
      -name 'search_seed_frontier_*_unqualified.json' \
      | sort \
      | tail -n 1
  )"
  if [[ -z "${initial_candidate}" || ! -f "${initial_candidate}" ]]; then
    echo "Strict CEM requires an explicitly unqualified bootstrap search seed" >&2
    return 2
  fi
  mkdir -p "${CEM_STRICT_ROOT}"
  export_launch_contract forehand_clear_stage3_cem_v6_cpu_verified_seed0
  mapfile -t common < <(cem_common_args)
  "${REPO_ROOT}/scripts/run_fullbody_training.sh" \
    --incoming-hit-cem \
    "${common[@]}" \
    --out-dir "${CEM_STRICT_ROOT}" \
    --population 512 \
    --iterations 16 \
    --elite-fraction 0.08 \
    --initial-std 0.18 \
    --min-std 0.02 \
    --coordinate-probe-radius 0.15 \
    --search-frontier-copies 3 \
    --initial-candidate "${initial_candidate}" \
    --require-cpu-quality-for-best \
    --cpu-guide-unqualified-mean \
    --cpu-promotion-audit-limit 8 \
    --wandb-name forehand-clear-stage3-cem-v6-cpu-verified-seed0
}

validate_strict_teacher() {
  local report="${CEM_STRICT_ROOT}/cem_report.json"
  local teacher="${CEM_STRICT_ROOT}/teacher_trajectory_mjx.npz"
  if [[ ! -f "${report}" || ! -f "${teacher}" ]]; then
    echo "Strict CEM teacher artifacts are incomplete" >&2
    return 2
  fi
  "${CHECK_PYTHON}" -c '
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("passed") is not True:
    raise SystemExit("strict CEM report did not pass")
metrics = payload["verified_metrics"]
if float(metrics["hit_racket_face_forward_alignment"]) < 0.5:
    raise SystemExit("strict CEM teacher violates the forward racket-face gate")
cpu = payload.get("cpu_gated_best_audit") or {}
if cpu.get("cpu_quality_passed") is not True:
    raise SystemExit("strict CEM teacher lacks the independent CPU quality gate")
print("[stage3-v6] strict CEM teacher passed MJX replicas and CPU replay")
' "${report}"
}

run_search() {
  require_gpu
  require_assets
  if [[ -f "${CEM_STRICT_ROOT}/cem_report.json" ]]; then
    if validate_strict_teacher; then
      echo "[stage3-v6] reusing the existing immutable strict CEM teacher"
      return 0
    fi
    echo "[stage3-v6] existing strict CEM output is invalid; refusing to overwrite it" >&2
    return 2
  fi
  run_cem_bootstrap
  run_cem_strict
  validate_strict_teacher
}

run_ppo() {
  require_gpu
  require_assets
  validate_strict_teacher
  export_launch_contract forehand_clear_stage3_cem_ppo_v6_seed0
  export MUSCLEMIMIC_STAGE3_WANDB_PROJECT="${WANDB_PROJECT}"
  export MUSCLEMIMIC_STAGE3_WANDB_RUN_ID="forehand-clear-stage3-cem-ppo-v6-seed0"
  export MUSCLEMIMIC_STAGE3_WANDB_NAME="${MUSCLEMIMIC_STAGE3_WANDB_RUN_ID}"
  export MUSCLEMIMIC_STAGE3_WANDB_MODE="online"
  "${REPO_ROOT}/scripts/run_fullbody_training.sh" \
    --incoming-hit \
    --spec "${SPEC}" \
    --stage train-gpu \
    --out-dir "${PPO_ROOT}" \
    --initialize-policy-from "${SOURCE_CHECKPOINT}" \
    --teacher-dataset "${CEM_STRICT_ROOT}/teacher_trajectory_mjx.npz" \
    --base-policy-artifact "${BASE_POLICY}" \
    --num-envs 256 \
    --rollout-steps 128 \
    --total-env-steps 12000000 \
    --seed 0
}

dry_run() {
  require_gpu
  require_assets
  export MUSCLEMIMIC_DRY_RUN=1
  export_launch_contract forehand_clear_stage3_cem_v6_dry_run
  mapfile -t common < <(cem_common_args)
  "${REPO_ROOT}/scripts/run_fullbody_training.sh" \
    --incoming-hit-cem \
    "${common[@]}" \
    --out-dir "${CEM_BOOTSTRAP_ROOT}" \
    --population 512 \
    --iterations 24
  "${REPO_ROOT}/scripts/run_fullbody_training.sh" \
    --incoming-hit \
    --spec "${SPEC}" \
    --stage train-gpu \
    --out-dir "${PPO_ROOT}" \
    --initialize-policy-from "${SOURCE_CHECKPOINT}" \
    --teacher-dataset "${CEM_STRICT_ROOT}/teacher_trajectory_mjx.npz" \
    --base-policy-artifact "${BASE_POLICY}" \
    --num-envs 256 \
    --rollout-steps 128 \
    --total-env-steps 12000000 \
    --seed 0
}

status() {
  require_gpu
  echo "[stage3-v6] spec=${SPEC}"
  echo "[stage3-v6] log=${LOG}"
  for artifact in \
    "${CEM_BOOTSTRAP_ROOT}/cem_report.json" \
    "${CEM_STRICT_ROOT}/cem_report.json" \
    "${PPO_ROOT}/policy_latest.json" \
    "${PPO_ROOT}/wandb_run.json" \
    "${PPO_ROOT}/train_report.json"; do
    if [[ -f "${artifact}" ]]; then
      echo "[stage3-v6] artifact=${artifact}"
    fi
  done
  if [[ -f "${LOG}" ]]; then
    tail -n 60 "${LOG}"
  else
    echo "[stage3-v6] log has not been created"
  fi
  nvidia-smi --id="${PHYSICAL_GPU}" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader
}

case "${MODE}" in
  check)
    run_checks
    ;;
  dry-run)
    dry_run
    ;;
  search)
    run_checks
    run_search
    ;;
  train)
    run_ppo
    ;;
  pipeline)
    run_checks
    run_search
    run_ppo
    ;;
  status)
    status
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac
