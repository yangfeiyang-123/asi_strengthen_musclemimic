#!/bin/bash
# Regenerate optimized muscle trajectories for ALL collapsed actions after the
# foot-frame reconciliation fix. Resumes from each sequence's canonical pkl and
# auto-detects fps per sequence from that pkl. Chain per sequence:
#   world_grounded -> lower_body(--enable-pose-pass) -> reference_bundle
#   -> convert_wham_to_amass -> run_retarget --stance-bundle
# Backs up the old (broken) optimized npz, then sanity-checks root height.
set -u
MM=/data3/yangfeiyang/WorkSpace/musclemimic
OW=/data3/yangfeiyang/WorkSpace/optimized_wham
WHAM_PY=/data3/yangfeiyang/conda_envs/wham/bin/python
MM_PY=$MM/.venv/bin/python
BACKUP=$MM/outputs/broken_traj_backup
mkdir -p "$BACKUP"
cd "$MM"

ACTIONS=(ChinaJump JumpAndSmash backhand_clear backhand_light backhandlift \
         forehandClear_insufficient_arm forehandLift overhead_clear smash)

total_ok=0; total_fail=0
for ACTION in "${ACTIONS[@]}"; do
  ACT="$MM/datasets/$ACTION"
  echo "########## ACTION: $ACTION ##########"
  seqs=()
  for d in "$ACT/wham/optimized_wham"/*/; do
    [[ -f "${d}canonical_wham_output.pkl" ]] && seqs+=("$(basename "$d")")
  done
  echo "  sequences: ${#seqs[@]}"

  for SEQ in "${seqs[@]}"; do
    SEQ_DIR="$ACT/wham/optimized_wham/$SEQ"
    CANON="$SEQ_DIR/canonical_wham_output.pkl"
    # per-sequence fps straight from the canonical pkl
    FPS=$("$MM_PY" - "$CANON" <<'PY'
import sys, joblib, numpy as np
d = joblib.load(sys.argv[1]); r = d[list(d.keys())[0]] if isinstance(d, dict) else d
for k in ("mocap_framerate","fps","mocap_frame_rate"):
    if k in r:
        print(f"{float(np.asarray(r[k]).reshape(-1)[0]):g}"); break
else:
    print("")
PY
)
    if [[ -z "$FPS" ]]; then echo "  [FAIL] $SEQ: no fps in canonical"; total_fail=$((total_fail+1)); continue; fi

    (cd "$OW" && "$WHAM_PY" scripts/world_grounded_smpl_optimizer.py \
        --input-pkl "$CANON" --out-dir "$SEQ_DIR/world_grounded" \
        --fps "$FPS" --track-id merge --device cpu) >/dev/null 2>&1 \
      || { echo "  [FAIL] $SEQ world_grounded"; total_fail=$((total_fail+1)); continue; }
    (cd "$OW" && "$WHAM_PY" scripts/optimize_smpl_lower_body.py \
        --input-pkl "$SEQ_DIR/world_grounded/optimized_canonical_wham_output.pkl" \
        --out-dir "$SEQ_DIR/lower_body_corrected" \
        --fps "$FPS" --track-id merge --device cpu --enable-pose-pass) >/dev/null 2>&1 \
      || { echo "  [FAIL] $SEQ lower_body"; total_fail=$((total_fail+1)); continue; }
    (cd "$OW" && "$WHAM_PY" scripts/export_contact_preserving_reference.py \
        --input-pkl "$SEQ_DIR/lower_body_corrected/corrected_smpl.pkl" \
        --out-dir "$SEQ_DIR/reference_bundle" --sequence "$SEQ" \
        --fps "$FPS" --track-id merge \
        --quality-report "$SEQ_DIR/lower_body_corrected/validation_summary.json" \
        --source-json "$SEQ_DIR/reference_bundle/source.json") >/dev/null 2>&1 \
      || { echo "  [FAIL] $SEQ reference_bundle"; total_fail=$((total_fail+1)); continue; }
    "$MM_PY" musclemimic/badminton/scripts/convert_wham_to_amass.py \
        --input "$SEQ_DIR/lower_body_corrected/corrected_smpl.pkl" \
        --output "$ACT/wham/optimized_wham/$SEQ.npz" \
        --fps "$FPS" --force-fps --gender neutral --merge-tracks >/dev/null 2>&1 \
      || { echo "  [FAIL] $SEQ convert"; total_fail=$((total_fail+1)); continue; }

    # back up + remove old broken cache before retarget
    mkdir -p "$BACKUP/$ACTION"
    cp "$ACT/muscle_trajectory/optimized/$SEQ.npz" "$BACKUP/$ACTION/" 2>/dev/null
    rm -f "$ACT/muscle_trajectory/optimized/$SEQ.npz" "$ACT/muscle_trajectory/optimized/${SEQ}_analysis.npz"
    printf '%s\n' "$SEQ" > "$MM/outputs/.regen_one.txt"
    "$MM_PY" musclemimic/badminton/scripts/run_retarget.py \
        --manifest "$MM/outputs/.regen_one.txt" \
        --amass-root "$ACT/wham/optimized_wham" \
        --gmr-cache-root "$ACT/muscle_trajectory/optimized" \
        --fps "$FPS" \
        --stance-bundle "$SEQ_DIR/reference_bundle/manifest.json" >/dev/null 2>&1 \
      || { echo "  [FAIL] $SEQ retarget"; total_fail=$((total_fail+1)); continue; }

    "$MM_PY" - "$ACT/muscle_trajectory/optimized/$SEQ.npz" "$SEQ" <<'PY' || { echo "  [BAD-HEIGHT] $SEQ"; total_fail=$((total_fail+1)); continue; }
import sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
z = np.asarray(d["qpos"])[:, 2]
zmed = float(np.median(z))
assert 0.55 < zmed < 1.4, f"median root z {zmed:.3f} not standing"
print(f"  [OK] {sys.argv[2]}  root_z_med={zmed:.3f}")
PY
    total_ok=$((total_ok+1))
  done
done
echo "############################################"
echo "DONE  ok=$total_ok  fail=$total_fail"
