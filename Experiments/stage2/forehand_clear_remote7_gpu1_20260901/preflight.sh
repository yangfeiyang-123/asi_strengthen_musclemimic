#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "usage: $0 <local9|remote7> [first-step|full]" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

server="$1"
scope="${2:-full}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/experiment.env"
case "$server" in
  local9) source "$script_dir/server_local9.env" ;;
  remote7) source "$script_dir/server_remote7.env" ;;
  *) usage; exit 2 ;;
esac
if [[ "$scope" != first-step && "$scope" != full ]]; then
  usage
  exit 2
fi

export PATH="$STAGE2_TOOLS_BIN:$PATH"
export UV_CACHE_DIR="$STAGE2_UV_CACHE_DIR"
if [[ -n "$STAGE2_UV_PYTHON_INSTALL_DIR" ]]; then
  export UV_PYTHON_INSTALL_DIR="$STAGE2_UV_PYTHON_INSTALL_DIR"
fi
if [[ -n "$STAGE2_UV_PROJECT_ENVIRONMENT" ]]; then
  export UV_PROJECT_ENVIRONMENT="$STAGE2_UV_PROJECT_ENVIRONMENT"
  export UV_NO_SYNC=1
fi
export MUSCLEMIMIC_JAX_CACHE_ROOT="$STAGE2_JAX_CACHE_ROOT"
export TMPDIR="$STAGE2_TMPDIR"
export MPLCONFIGDIR="$STAGE2_MPLCONFIGDIR"

failures=0
fail() {
  echo "BLOCKED: $*" >&2
  failures=$((failures + 1))
}
check_file() {
  local label="$1" relative="$2"
  if [[ -z "$relative" ]]; then
    fail "$label path is not configured"
  elif [[ ! -f "$STAGE2_ASSET_ROOT/$relative" ]]; then
    fail "$label is missing: $relative"
  else
    echo "PASS: $label=$relative"
  fi
}
check_dir() {
  local label="$1" relative="$2"
  if [[ -z "$relative" ]]; then
    fail "$label path is not configured"
  elif [[ ! -d "$STAGE2_ASSET_ROOT/$relative" ]]; then
    fail "$label is missing: $relative"
  else
    echo "PASS: $label=$relative"
  fi
}
check_value() {
  local label="$1" value="$2"
  if [[ -z "$value" ]]; then
    fail "$label is not configured"
  else
    echo "PASS: $label configured"
  fi
}

if [[ ! -d "$STAGE2_REPO_ROOT/.git" && ! -f "$STAGE2_REPO_ROOT/.git" ]]; then
  fail "repository/worktree is missing: $STAGE2_REPO_ROOT"
else
  actual_sha="$(git -C "$STAGE2_REPO_ROOT" rev-parse HEAD)"
  [[ "$actual_sha" == "$STAGE2_CODE_SHA" ]] || fail "Git SHA $actual_sha != $STAGE2_CODE_SHA"
  dirty="$(git -C "$STAGE2_REPO_ROOT" status --short --untracked-files=no -- fullbody musclemimic scripts configs)"
  [[ -z "$dirty" ]] || fail "scoped source/config is dirty"
  if ! (cd "$STAGE2_REPO_ROOT" && sha256sum -c "$script_dir/source_files.sha256"); then
    fail "source/config hashes differ from the shared contract"
  fi
fi

check_dir "T3 seed-0 checkpoint" "$STAGE2_T3_CHECKPOINT"
check_file "Stage-1 PEASD teacher promotion" "$STAGE2_STAGE1_PROMOTION"
check_file "verified EMG tube" "$STAGE2_TUBE"
check_file "opaque blind review" "$STAGE2_BLIND_REVIEW"
if [[ -f "$STAGE2_ASSET_ROOT/$STAGE2_BLIND_REVIEW" ]] && \
   rg -q '"passed": null|"reviewer_id": null' "$STAGE2_ASSET_ROOT/$STAGE2_BLIND_REVIEW"; then
  fail "opaque blind review is still unsigned/incomplete"
fi

if [[ "$scope" == full ]]; then
  check_file "train event manifest list" "$STAGE2_TRAIN_EVENT_MANIFEST_LIST"
  check_file "validation event manifest list" "$STAGE2_VAL_EVENT_MANIFEST_LIST"
  check_file "train event bank" "$STAGE2_TRAIN_EVENT_BANK"
  check_file "validation event bank" "$STAGE2_VAL_EVENT_BANK"
  check_file "frozen Forehand Clear body decoder" "$STAGE2_FROZEN_BODY_DECODER"
  check_value "frozen decoder fingerprint" "$STAGE2_FROZEN_BODY_DECODER_FINGERPRINT"
  check_value "body synergy contract fingerprint" "$STAGE2_BODY_SYNERGY_CONTRACT_FINGERPRINT"
  check_value "body synergy portable-core fingerprint" "$STAGE2_BODY_SYNERGY_PORTABLE_CORE_FINGERPRINT"
fi

if [[ "$STAGE2_ALLOW_UNPROMOTED_TEACHER" != 0 ]]; then
  fail "production contract must not allow an unpromoted teacher"
fi
if [[ "$STAGE2_RESUME_OLD_FAMILY" != 0 ]]; then
  fail "production contract must not resume an old Stage-2 family"
fi

echo "GPU inventory (informational; physical assignment is GPU $STAGE2_PHYSICAL_GPU):"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader || fail "nvidia-smi failed"

if (( failures > 0 )); then
  echo "RESULT: BLOCKED ($failures gate failures; no training started)" >&2
  exit 1
fi
echo "RESULT: PASSED ($scope)"
