#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <local9|remote7>" >&2
  exit 2
fi

server="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/experiment.env"
case "$server" in
  local9) source "$script_dir/server_local9.env" ;;
  remote7) source "$script_dir/server_remote7.env" ;;
  *) echo "unknown server profile: $server" >&2; exit 2 ;;
esac

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
export TMPDIR="$S2_TMPDIR"
export MPLCONFIGDIR="$S2_MPLCONFIGDIR"
export PYTHONPATH="$S2_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

failures=0
fail() {
  echo "BLOCKED: $*" >&2
  failures=$((failures + 1))
}

actual_sha="$(git -C "$S2_REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
[[ "$actual_sha" == "$S2_CODE_SHA" ]] || fail "Git SHA $actual_sha != $S2_CODE_SHA"
dirty="$(git -C "$S2_REPO_ROOT" status --short --untracked-files=no -- fullbody musclemimic scripts configs || true)"
[[ -z "$dirty" ]] || fail "scoped source/config is dirty: $dirty"
if ! (cd "$S2_REPO_ROOT" && sha256sum -c "$script_dir/source_files.sha256"); then
  fail "source/config hashes differ from the shared contract"
fi
actual_source_fingerprint="$(
  cd "$S2_REPO_ROOT" && \
    "$S2_PYTHON_BIN" -c 'from musclemimic.runner.checkpointing import stage1_source_tree_snapshot; print(stage1_source_tree_snapshot()["source_tree_fingerprint"])'
)" || fail "could not compute source-tree fingerprint"
[[ "$actual_source_fingerprint" == "$S2_SOURCE_FINGERPRINT" ]] || \
  fail "source-tree fingerprint $actual_source_fingerprint != $S2_SOURCE_FINGERPRINT"
runtime_imports="$(
  cd "$S2_REPO_ROOT" && \
    "$S2_PYTHON_BIN" -c 'import fullbody.distill_train_bc as entry; import musclemimic.distill.train_bc as impl; print(entry.__file__); print(impl.__file__)'
)" || fail "could not resolve Stage-2 runtime module paths"
while IFS= read -r runtime_module; do
  [[ "$runtime_module" == "$S2_REPO_ROOT"/* ]] || \
    fail "runtime module escaped the fixed checkout: $runtime_module"
done <<< "$runtime_imports"
echo "Runtime modules:"
printf '%s\n' "$runtime_imports"

teacher="$S2_ASSET_ROOT/$S2_TEACHER_CHECKPOINT"
teacher_manifest="$S2_ASSET_ROOT/$S2_TEACHER_RUN_MANIFEST"
[[ -d "$teacher" ]] || fail "teacher checkpoint is missing: $teacher"
[[ -f "$teacher_manifest" ]] || fail "teacher run manifest is missing: $teacher_manifest"
if [[ -d "$teacher" ]] && ! (cd "$teacher" && sha256sum -c "$script_dir/teacher_checkpoint_files.sha256"); then
  fail "teacher checkpoint inventory/hash mismatch"
fi
if [[ -f "$teacher_manifest" ]]; then
  actual_manifest_sha="$(sha256sum "$teacher_manifest" | awk '{print $1}')"
  [[ "$actual_manifest_sha" == "$S2_TEACHER_RUN_MANIFEST_SHA256" ]] || \
    fail "teacher manifest SHA-256 mismatch: $actual_manifest_sha"
fi

if [[ -f "$teacher_manifest" ]]; then
  train_count="$(jq '.experiment_config.task_factory.params.amass_dataset_conf.rel_dataset_path | length' "$teacher_manifest")"
  val_count="$(jq '.experiment_config.validation.amass_dataset_conf.rel_dataset_path | length' "$teacher_manifest")"
  [[ "$train_count" == 80 ]] || fail "teacher train split count is $train_count, expected 80"
  [[ "$val_count" == 20 ]] || fail "teacher validation split count is $val_count, expected 20"
  missing_motions=0
  while IFS= read -r motion; do
    if [[ ! -f "$S2_DATASETS_ROOT/$motion.npz" ]]; then
      echo "MISSING MOTION: $motion" >&2
      missing_motions=$((missing_motions + 1))
    fi
  done < <(jq -r '.experiment_config.task_factory.params.amass_dataset_conf.rel_dataset_path[], .experiment_config.validation.amass_dataset_conf.rel_dataset_path[]' "$teacher_manifest")
  [[ "$missing_motions" == 0 ]] || fail "$missing_motions teacher motions are not local"
fi

if [[ "$S2_EXPERIMENT_CLASS" != FORMAL_EARLY_START_PENDING_UPSTREAM_ACCEPTANCE ]]; then
  fail "unexpected experiment class: $S2_EXPERIMENT_CLASS"
fi
[[ "$S2_ALLOW_PENDING_UPSTREAM_TEACHER" == 1 ]] || fail "pending-upstream contract was not explicitly enabled"
[[ "$S2_RESUME_OLD_STAGE2_FAMILY" == 0 ]] || fail "old Stage-2 family resume is forbidden"

gpu_line="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits -i "$S2_PHYSICAL_GPU" 2>/dev/null || true)"
if [[ -z "$gpu_line" ]]; then
  fail "nvidia-smi could not inspect physical GPU $S2_PHYSICAL_GPU"
else
  echo "GPU: $gpu_line"
  gpu_free="$(awk -F',' '{gsub(/ /,"",$5); print $5}' <<< "$gpu_line")"
  if [[ ! "$gpu_free" =~ ^[0-9]+$ ]] || (( gpu_free < 20000 )); then
    fail "physical GPU $S2_PHYSICAL_GPU has only ${gpu_free:-unknown} MiB free; require >= 20000 MiB"
  fi
  echo "GPU processes (informational):"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader -i "$S2_PHYSICAL_GPU" || true
fi

if (( failures > 0 )); then
  echo "RESULT: BLOCKED ($failures failures; training not started)" >&2
  exit 1
fi
echo "RESULT: PASSED server=$S2_SERVER_ID class=$S2_EXPERIMENT_CLASS train=80 val=20"
