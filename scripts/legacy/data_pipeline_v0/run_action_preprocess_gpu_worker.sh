#!/usr/bin/env bash
set -u

if [ "$#" -lt 1 ]; then
  echo "usage: $0 ACTION" >&2
  exit 2
fi

ACTION="$1"
WORKDIR="${WORKDIR:-/data3/yangfeiyang/WorkSpace/musclemimic}"
DATASETS_ROOT="${DATASETS_ROOT:-$WORKDIR/datasets}"
OPT_WHAM="${OPT_WHAM:-/data3/yangfeiyang/WorkSpace/optimized_wham}"
WHAM_PY="${WHAM_PY:-/data3/yangfeiyang/conda_envs/wham/bin/python}"
MM_PY="${MM_PY:-$WORKDIR/.venv/bin/python}"
LOG_ROOT="${LOG_ROOT:-$WORKDIR/logs/preprocess_gpu_parallel}"
LOG_DIR="$LOG_ROOT/$ACTION"

export ACTION
export DATASETS_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WHAM_CUDA_LOCK_PATH="${WHAM_CUDA_LOCK_PATH:-/tmp/wham_cuda_runtime.lock}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/ultralytics}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export PYTHONUNBUFFERED=1

cd "$WORKDIR" || exit 1
mkdir -p "$LOG_DIR"
: > "$LOG_DIR/failed_videos.txt"
: > "$LOG_DIR/processed_videos.txt"

TODO="$LOG_DIR/videos_to_process.txt"
"$MM_PY" - <<'PY' > "$TODO"
from pathlib import Path
import os
import sys

sys.path.insert(0, "/data3/yangfeiyang/WorkSpace/optimized_wham/tools")
import musclemimic_dataset_bridge as bridge

root = Path(os.environ["DATASETS_ROOT"])
action_dir = root / os.environ["ACTION"]
for video in bridge.iter_raw_videos(action_dir):
    seq = bridge.sequence_name_for_video(action_dir, video)
    expected = [
        action_dir / "wham" / "raw_wham" / seq / "wham_output.pkl",
        action_dir / "wham" / "optimized_wham" / seq / "lower_body_corrected" / "corrected_smpl.pkl",
        action_dir / "muscle_trajectory" / "raw" / f"{seq}.npz",
        action_dir / "muscle_trajectory" / "optimized" / f"{seq}.npz",
    ]
    if any(not path.exists() for path in expected):
        print(video)
PY

TOTAL="$(wc -l < "$TODO" | tr -d ' ')"
echo "action=$ACTION videos_to_process=$TOTAL log_dir=$LOG_DIR"

INDEX=0
while IFS= read -r VIDEO; do
  [ -n "$VIDEO" ] || continue
  INDEX=$((INDEX + 1))
  BASE="$(basename "$VIDEO")"
  SAFE_NAME="${BASE//[^A-Za-z0-9_.-]/_}"
  LOG_FILE="$LOG_DIR/$(printf '%03d_of_%03d_%s.log' "$INDEX" "$TOTAL" "$SAFE_NAME")"

  echo "=== [$ACTION][$INDEX/$TOTAL] $VIDEO ===" | tee -a "$LOG_DIR/processed_videos.txt"
  "$MM_PY" "$OPT_WHAM/tools/musclemimic_dataset_bridge.py" pipeline \
    --datasets-root "$DATASETS_ROOT" \
    --action "$ACTION" \
    --video "$VIDEO" \
    --optimized-wham-root "$OPT_WHAM" \
    --musclemimic-root "$WORKDIR" \
    --wham-python-exe "$WHAM_PY" \
    --musclemimic-python-exe "$MM_PY" \
    --device cuda \
    --pose-backend rtmpose \
    --optimize-lower-body \
    --run > "$LOG_FILE" 2>&1
  RC=$?
  cat "$LOG_FILE"
  if [ "$RC" -eq 0 ]; then
    echo "[OK] $VIDEO" | tee -a "$LOG_DIR/processed_videos.txt"
  else
    echo "[FAILED] $VIDEO" | tee -a "$LOG_DIR/processed_videos.txt"
    echo "$VIDEO" >> "$LOG_DIR/failed_videos.txt"
  fi
done < "$TODO"

echo "action=$ACTION complete"
echo "failed=$(wc -l < "$LOG_DIR/failed_videos.txt" | tr -d ' ')"
