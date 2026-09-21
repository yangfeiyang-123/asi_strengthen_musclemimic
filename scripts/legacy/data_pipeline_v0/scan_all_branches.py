"""Full-library health scan of all three retargeted muscle-trajectory branches
(raw / initial / optimized) across every action. Reports per-branch clip count
and root-height range; flags any clip whose median root z < 0.55 (the collapse
signature). This is the ground truth that matters for training/rendering and it
directly reflects the health of the SMPL branch that produced each retarget.
"""
import glob
import os

import numpy as np

DATASETS = "/data3/yangfeiyang/WorkSpace/musclemimic/datasets"
BRANCHES = ("raw", "initial", "optimized")
STANDING_MIN = 0.55


def scan_branch(action, branch):
    d = f"{DATASETS}/{action}/muscle_trajectory/{branch}"
    if not os.path.isdir(d):
        return None
    clips = sorted(p for p in glob.glob(f"{d}/*.npz") if not p.endswith("_analysis.npz"))
    zmeds, bad = [], []
    for p in clips:
        try:
            z = float(np.median(np.asarray(np.load(p, allow_pickle=True)["qpos"])[:, 2]))
        except Exception as e:
            bad.append(f"{os.path.basename(p)}(err)")
            continue
        zmeds.append(z)
        if z < STANDING_MIN:
            bad.append(f"{os.path.basename(p)}(z={z:.2f})")
    return {"n": len(clips), "zrange": [round(min(zmeds), 2), round(max(zmeds), 2)] if zmeds else None, "bad": bad}


actions = sorted(os.path.basename(os.path.dirname(p))
                 for p in glob.glob(f"{DATASETS}/*/muscle_trajectory"))
print(f"{'action':32s} | {'raw':22s} | {'initial':22s} | {'optimized':22s}")
print("-" * 108)
any_bad = False
for action in actions:
    cells = []
    for br in BRANCHES:
        r = scan_branch(action, br)
        if r is None:
            cells.append("(none)".ljust(22))
        else:
            tag = "OK" if not r["bad"] else f"BAD:{len(r['bad'])}"
            if r["bad"]:
                any_bad = True
            cells.append(f"n={r['n']:2d} z{r['zrange']} {tag}".ljust(22))
    print(f"{action:32s} | {cells[0]} | {cells[1]} | {cells[2]}")
print("-" * 108)
print("ALL HEALTHY" if not any_bad else "SOME BRANCHES HAVE COLLAPSED CLIPS (see BAD counts)")
