#!/usr/bin/env bash
set -Eeuo pipefail

# Re-render the known successful old-baseline CEM correction on independent
# CPU MuJoCo.  This is an engineering demo, not a promoted PEASD/Stage-3
# teacher and not a replacement for the formal paper evaluation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_RUN="/raid/yangfeiyang/musclemimic_runs/stage3_hit_v46h_prepostgate_distal_direction_k6to9_p96r8_i24_s761_v92"
SOURCE_CANDIDATE="${SOURCE_RUN}/cpu_candidate_audits/candidate_paramshaf78964732af9fe72.json"
OUTPUT_DIR="${1:-artifacts/stage3_legacy_demo/forehand_clear_v46h_f789}"

if [[ ! -f "${SOURCE_RUN}/cem_contract.json" || ! -f "${SOURCE_CANDIDATE}" ]]; then
  echo "The immutable legacy CEM source assets are missing." >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}/replay_trace.npz" || -e "${OUTPUT_DIR}/forehand_clear_hit_demo.mp4" ]]; then
  echo "Refusing to overwrite an existing demo in ${OUTPUT_DIR}" >&2
  exit 2
fi

source "${REPO_ROOT}/configs/env.sh"
export JAX_PLATFORMS=cpu
export MUJOCO_GL=osmesa

"${REPO_ROOT}/scripts/run_with_cuda_compat.sh" uv run --locked python \
  scripts/audit_cem_candidate_cpu.py \
  --run-dir "${SOURCE_RUN}" \
  --candidate "${SOURCE_CANDIDATE}" \
  --engineering-demo-only \
  --output "${OUTPUT_DIR}/replay_trace.npz" \
  --record-video "${OUTPUT_DIR}/forehand_clear_hit_demo.mp4"

echo "Legacy engineering demo: ${OUTPUT_DIR}/forehand_clear_hit_demo.mp4"
echo "Replay audit: ${OUTPUT_DIR}/replay_trace.json"
