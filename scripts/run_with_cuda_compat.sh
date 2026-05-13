#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CUDA_COMPAT_ROOT="${MM_CUDA_COMPAT_ROOT:-${REPO_ROOT}/.local/cuda-compat-12.4}"
CUDA_COMPAT_RPM="${CUDA_COMPAT_ROOT}/cuda-compat-12-4-550.163.01-1.el9.x86_64.rpm"
CUDA_COMPAT_URL="${MM_CUDA_COMPAT_URL:-https://developer.download.nvidia.com/compute/cuda/preview/repos/rhel9/x86_64/cuda-compat-12-4-550.163.01-1.el9.x86_64.rpm}"
CUDA_COMPAT_DIR="${CUDA_COMPAT_ROOT}/compat"
CUDA_TMP_DIR="${CUDA_COMPAT_ROOT}/tmp"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_with_cuda_compat.sh <command> [args...]

Examples:
  scripts/run_with_cuda_compat.sh uv run fullbody/experiment.py --config-name=conf_fullbody_demo wandb.mode=disabled
  MM_CUDA_VISIBLE_DEVICES=1 scripts/run_with_cuda_compat.sh uv run bimanual/eval.py --path outputs/.../checkpoint_123

Environment overrides:
  MM_CUDA_COMPAT_ROOT          Install/cache root for the private compat package
  MM_CUDA_COMPAT_URL           Download URL for the compat RPM
  MM_CUDA_VISIBLE_DEVICES      CUDA_VISIBLE_DEVICES value passed to the command
EOF
}

ensure_cuda_compat() {
  mkdir -p "$CUDA_COMPAT_DIR" "$CUDA_TMP_DIR"

  if [[ ! -f "$CUDA_COMPAT_RPM" ]]; then
    wget -O "$CUDA_COMPAT_RPM" "$CUDA_COMPAT_URL"
  fi

  if [[ ! -f "$CUDA_COMPAT_DIR/libcuda.so.1" ]]; then
    rm -rf "${CUDA_TMP_DIR:?}"/*
    bsdtar -xf "$CUDA_COMPAT_RPM" -C "$CUDA_TMP_DIR"
    cp -af "$CUDA_TMP_DIR/usr/local/cuda-12.4/compat/"* "$CUDA_COMPAT_DIR"/
  fi
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

ensure_cuda_compat

export MM_CUDA_COMPAT_ROOT="${CUDA_COMPAT_ROOT}"
export MM_CUDA_COMPAT_DIR="${CUDA_COMPAT_DIR}"
export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${MM_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-1}}"
export MUSCLEMIMIC_CONVERTED_AMASS_PATH="${MM_CONVERTED_AMASS_PATH:-${REPO_ROOT}/caches/AMASS}"
export CONVERTED_AMASS_PATH="${MUSCLEMIMIC_CONVERTED_AMASS_PATH}"

cd "$REPO_ROOT"
exec "$@"
