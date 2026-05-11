#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BADMINTONMIMIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MUSCLEMIMIC_ROOT="$(cd "${BADMINTONMIMIC_ROOT}/.." && pwd)"

SRC="${BADMINTONMIMIC_ROOT}/experiments/fullbody/config_specific_task/conf_fullbody_badminton_gmr.yaml"
DST_DIR="${MUSCLEMIMIC_ROOT}/fullbody/config_specific_task"
DST="${DST_DIR}/conf_fullbody_badminton_gmr.yaml"

mkdir -p "${DST_DIR}"
cp "${SRC}" "${DST}"

echo "Installed ${SRC}"
echo "       -> ${DST}"
echo "Run with: uv run fullbody/experiment.py --config-name=config_specific_task/conf_fullbody_badminton_gmr"
