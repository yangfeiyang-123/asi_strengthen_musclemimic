"""Probe fps availability in each action's canonical WHAM pkls (auto-detect source)."""
import glob
import os

import joblib
import numpy as np

DATASETS = "/data3/yangfeiyang/WorkSpace/musclemimic/datasets"
ACTIONS = [
    "ChinaJump", "JumpAndSmash", "backhand_clear", "backhand_light",
    "backhandlift", "forehandClear_insufficient_arm", "forehandLift",
    "overhead_clear", "smash",
]
FPS_KEYS = ("mocap_framerate", "fps", "mocap_frame_rate")


def fps_of(pkl):
    d = joblib.load(pkl)
    rec = d[list(d.keys())[0]] if isinstance(d, dict) else d
    for k in FPS_KEYS:
        if k in rec:
            try:
                return float(np.asarray(rec[k]).reshape(-1)[0])
            except Exception:
                pass
    return None


for action in ACTIONS:
    seq_dirs = sorted(d for d in glob.glob(f"{DATASETS}/{action}/wham/optimized_wham/*") if os.path.isdir(d))
    fpss = {}
    missing = 0
    for d in seq_dirs:
        pkl = f"{d}/canonical_wham_output.pkl"
        if not os.path.isfile(pkl):
            missing += 1
            continue
        f = fps_of(pkl)
        if f is None:
            missing += 1
        else:
            fpss[round(f, 3)] = fpss.get(round(f, 3), 0) + 1
    print(f"{action:32s} seqs={len(seq_dirs):3d} fps_hist={fpss} missing_fps={missing}")
