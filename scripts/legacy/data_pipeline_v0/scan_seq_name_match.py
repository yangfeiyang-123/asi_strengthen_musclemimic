"""Verify per-action that optimized npz basenames match the WHAM sequence dir
names (the retarget manifest keys off these). Reports any mismatch so the batch
regen doesn't silently skip or misfile a sequence."""
import glob
import os

DATASETS = "/data3/yangfeiyang/WorkSpace/musclemimic/datasets"
ACTIONS = [
    "ChinaJump", "JumpAndSmash", "backhand_clear", "backhand_light",
    "backhandlift", "forehandClear_insufficient_arm", "forehandLift",
    "overhead_clear", "smash",
]

for action in ACTIONS:
    base = f"{DATASETS}/{action}"
    seq_dirs = {os.path.basename(d) for d in glob.glob(f"{base}/wham/optimized_wham/*")
                if os.path.isdir(d)}
    opt_npz = {os.path.basename(p)[:-4] for p in glob.glob(f"{base}/muscle_trajectory/optimized/*.npz")
               if not p.endswith("_analysis.npz")}
    only_seq = sorted(seq_dirs - opt_npz)
    only_opt = sorted(opt_npz - seq_dirs)
    matched = len(seq_dirs & opt_npz)
    flag = "" if not only_opt else "  <-- opt npz with NO seq dir!"
    print(f"{action:32s} matched={matched:3d}  seq_only={len(only_seq)}  opt_only={len(only_opt)}{flag}")
    if only_opt:
        print("   opt_only:", only_opt[:5])
    if only_seq:
        print("   seq_only:", only_seq[:5])
