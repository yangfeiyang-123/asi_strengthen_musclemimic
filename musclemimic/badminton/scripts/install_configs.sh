#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSCLEMIMIC_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SRC="${MUSCLEMIMIC_ROOT}/experiments/fullbody/config_specific_task/conf_fullbody_badminton_gmr.yaml"
DST_DIR="${MUSCLEMIMIC_ROOT}/fullbody/config_specific_task"
DST="${DST_DIR}/generated/conf_fullbody_badminton_manifest_gmr.yaml"

if [[ ! -f "${SRC}" ]]; then
  echo "Missing generated manifest config: ${SRC}" >&2
  echo "Generate it first with:" >&2
  echo "  uv run python musclemimic/badminton/scripts/build_config_from_manifests.py" >&2
  exit 2
fi

mkdir -p "${DST_DIR}/generated"
cp "${SRC}" "${DST}"

echo "Installed ${SRC}"
echo "       -> ${DST}"
echo "Run with: uv run fullbody/experiment.py --config-name=config_specific_task/generated/conf_fullbody_badminton_manifest_gmr"
echo "This generic manifest config is not the canonical ForehandClear Stage-1 config."
echo "The canonical base file is never overwritten by this installer."
