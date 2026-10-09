#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
source configs/env.sh

export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/musclemimic-racket-editor-matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

exec .venv/bin/python -m musclemimic.badminton.scripts.racket_pose_editor "$@"
