"""Confirm raw/initial branches are healthy and check what intermediate WHAM
artifacts survive per action (decides whether re-processing must restart from
video or can resume from the canonical pkl)."""
import glob
import os

import numpy as np

DATASETS = "/data3/yangfeiyang/WorkSpace/musclemimic/datasets"
ACTIONS = [
    "ChinaJump", "JumpAndSmash", "backhand_clear", "backhand_light",
    "backhandlift", "forehandClear_insufficient_arm", "forehandLift",
    "overhead_clear", "smash",
]


def root_zmed(path):
    d = np.load(path, allow_pickle=True)
    return float(np.median(np.asarray(d["qpos"])[:, 2]))


for action in ACTIONS:
    base = f"{DATASETS}/{action}"
    # raw branch health (one sample)
    raw_clips = sorted(p for p in glob.glob(f"{base}/muscle_trajectory/raw/*.npz")
                       if not p.endswith("_analysis.npz"))
    raw_z = root_zmed(raw_clips[0]) if raw_clips else None
    # canonical pkl availability per sequence
    wham_root = f"{base}/wham/optimized_wham"
    seq_dirs = [d for d in glob.glob(f"{wham_root}/*") if os.path.isdir(d)]
    n_canon = sum(os.path.isfile(f"{d}/canonical_wham_output.pkl") for d in seq_dirs)
    n_opt = len([p for p in glob.glob(f"{base}/muscle_trajectory/optimized/*.npz")
                 if not p.endswith("_analysis.npz")])
    print(f"{action:32s} raw_clips={len(raw_clips):3d} raw_root_z={raw_z if raw_z is None else round(raw_z,2)}"
          f"  opt_clips={n_opt:3d}  seq_dirs={len(seq_dirs):3d}  canonical_pkl={n_canon}")
