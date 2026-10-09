"""Scan every optimized muscle-trajectory npz and flag the collapse bug.

The foot-frame-mismatch ground-alignment bug sinks the whole body ~1.7 m below
the floor, so the retargeted root height collapses from ~0.9 m (standing) to
~0.1-0.25 m (kneeling). We read the root qpos height directly from the npz
(no env build needed) and classify each clip.
"""
import glob
import json
import os

import numpy as np

DATASETS = "/data3/yangfeiyang/WorkSpace/musclemimic/datasets"
STANDING_MIN = 0.55  # median root z above this = upright; well below = collapsed

rows = []
for opt_dir in sorted(glob.glob(f"{DATASETS}/*/muscle_trajectory/optimized")):
    action = opt_dir.split("/")[-3]
    clips = sorted(p for p in glob.glob(f"{opt_dir}/*.npz") if not p.endswith("_analysis.npz"))
    n_ok = n_bad = 0
    bad_names = []
    zmeds = []
    for p in clips:
        try:
            d = np.load(p, allow_pickle=True)
            q = np.asarray(d["qpos"])
            zmed = float(np.median(q[:, 2]))  # root free-joint z is qpos[2]
        except Exception as e:
            bad_names.append(os.path.basename(p) + f"(load-err:{e})")
            n_bad += 1
            continue
        zmeds.append(zmed)
        if zmed >= STANDING_MIN:
            n_ok += 1
        else:
            n_bad += 1
            bad_names.append(f"{os.path.basename(p)}(z={zmed:.2f})")
    rows.append({
        "action": action,
        "clips": len(clips),
        "upright": n_ok,
        "collapsed": n_bad,
        "root_z_med_range": [round(min(zmeds), 2), round(max(zmeds), 2)] if zmeds else None,
        "bad": bad_names[:6],
    })

print(json.dumps(rows, ensure_ascii=False, indent=1))
print("\n=== SUMMARY ===")
for r in rows:
    status = "OK" if r["collapsed"] == 0 else f"NEEDS-REDO ({r['collapsed']}/{r['clips']} collapsed)"
    print(f"{r['action']:32s} clips={r['clips']:3d}  z_med_range={r['root_z_med_range']}  {status}")
