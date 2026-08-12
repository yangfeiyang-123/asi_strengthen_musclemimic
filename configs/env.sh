#!/usr/bin/env bash
set -euo pipefail

# Source this file from the repository root:
#   source configs/env.sh

# This file lives at <repo>/configs/env.sh, so the repo root is its parent's parent.
MUSCLEMIMIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Backward-compat alias for scripts/docs that still reference BADMINTONMIMIC_ROOT.
BADMINTONMIMIC_ROOT="${MUSCLEMIMIC_ROOT}"

export BADMINTONMIMIC_ROOT
export MUSCLEMIMIC_ROOT

export MUSCLEMIMIC_DATASETS_ROOT="${MUSCLEMIMIC_DATASETS_ROOT:-${MUSCLEMIMIC_ROOT}/datasets}"

export MUSCLEMIMIC_AMASS_PATH="${MUSCLEMIMIC_AMASS_PATH:-${MUSCLEMIMIC_DATASETS_ROOT}/_global/amass_npz}"
export AMASS_PATH="${MUSCLEMIMIC_AMASS_PATH}"

export MUSCLEMIMIC_CONVERTED_AMASS_PATH="${MUSCLEMIMIC_CONVERTED_AMASS_PATH:-${MUSCLEMIMIC_DATASETS_ROOT}/_global/muscle_trajectory/gmr_cache}"
export CONVERTED_AMASS_PATH="${MUSCLEMIMIC_CONVERTED_AMASS_PATH}"

# Preprocessed muscle-trajectory root. All motions are already retargeted to the
# MyoFullBody skeleton under datasets/<action>/muscle_trajectory/, so no GMR/SMPL
# retargeting is needed at train time. With this set, ImitationFactory resolves a
# rel_dataset_path entry ``X`` directly to ``<root>/X.npz`` and loads it if present
# (loco_mujoco/smpl/retargeting.py:get_gmr_cache_dataset_path). Override by exporting
# MUSCLEMIMIC_GMR_CACHE_PATH before sourcing to point at a different cache root
# (e.g. the legacy gmr_cache) or unset it to restore the per-env gmr_cache default.
export MUSCLEMIMIC_GMR_CACHE_PATH="${MUSCLEMIMIC_GMR_CACHE_PATH:-${MUSCLEMIMIC_DATASETS_ROOT}}"

export MUSCLEMIMIC_SMPL_MODEL_PATH="${MUSCLEMIMIC_SMPL_MODEL_PATH:-${MUSCLEMIMIC_ROOT}/smpl_models/smplh}"
export SMPL_MODEL_PATH="${MUSCLEMIMIC_SMPL_MODEL_PATH}"

# Keep the current workstation contract as the default while allowing another
# server to select its own large, writable compilation-cache volume without
# patching source files.
export MUSCLEMIMIC_JAX_CACHE_ROOT="${MUSCLEMIMIC_JAX_CACHE_ROOT:-/data3/yangfeiyang/WorkSpace/ENV/jax-cache}"

# Drop system CUDA toolkit paths (e.g. /usr/local/cuda-12.1/lib64) inherited
# from the shell profile: they shadow the venv's pip-provided CUDA libraries and
# break GPU jaxlib (outdated cuSPARSE). The system CUDA install is untouched.
_mm_sanitized_ld_path=""
IFS=':' read -ra _mm_ld_entries <<< "${LD_LIBRARY_PATH:-}"
for _mm_entry in "${_mm_ld_entries[@]}"; do
  [[ -z "${_mm_entry}" ]] && continue
  [[ "${_mm_entry}" == /usr/local/cuda* ]] && continue
  _mm_sanitized_ld_path="${_mm_sanitized_ld_path:+${_mm_sanitized_ld_path}:}${_mm_entry}"
done
export LD_LIBRARY_PATH="${_mm_sanitized_ld_path}"
unset _mm_sanitized_ld_path _mm_ld_entries _mm_entry

export MM_CUDA_COMPAT_ROOT="${MM_CUDA_COMPAT_ROOT:-${MUSCLEMIMIC_ROOT}/.local/cuda-compat-12.4}"
export MM_CUDA_COMPAT_DIR="${MM_CUDA_COMPAT_ROOT}/compat"
if [[ -d "${MM_CUDA_COMPAT_DIR}" ]]; then
  export LD_LIBRARY_PATH="${MM_CUDA_COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
fi

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# Use MuJoCo's headless EGL backend by default so validation videos work from
# tmux/SSH sessions without an X server. Callers can still override MUJOCO_GL.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "${MUSCLEMIMIC_AMASS_PATH}" "${MUSCLEMIMIC_CONVERTED_AMASS_PATH}"

echo "BADMINTONMIMIC_ROOT=${BADMINTONMIMIC_ROOT}"
echo "MUSCLEMIMIC_ROOT=${MUSCLEMIMIC_ROOT}"
echo "AMASS_PATH=${AMASS_PATH}"
echo "CONVERTED_AMASS_PATH=${CONVERTED_AMASS_PATH}"
echo "GMR_CACHE_PATH=${MUSCLEMIMIC_GMR_CACHE_PATH}"
echo "SMPL_MODEL_PATH=${SMPL_MODEL_PATH}"
echo "JAX_CACHE_ROOT=${MUSCLEMIMIC_JAX_CACHE_ROOT}"
echo "CUDA_COMPAT_DIR=${MM_CUDA_COMPAT_DIR}"
