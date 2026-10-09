"""SMPL-level check of the optimized branch (the only SMPL branch that passes
through the buggy world_grounded ground-alignment). For every sequence, read the
regenerated reference_bundle motion.npz z-up pelvis height (trans[:,2]) and the
corrected_smpl.pkl trans_world, and confirm the reconciliation metadata is
present + the pelvis sits at a standing height above the floor.
raw / init SMPL never touch world_grounded, and their retargeted muscle
trajectories already scan healthy, so they need no SMPL re-check here.
"""
import glob
import os

import joblib
import numpy as np

DATASETS = "/data3/yangfeiyang/WorkSpace/musclemimic/datasets"

print(f"{'action':32s} seqs  bundle_z_med_range     recon_meta  corrected_trans_z_range")
print("-" * 100)
for action_dir in sorted(glob.glob(f"{DATASETS}/*/wham/optimized_wham")):
    action = action_dir.split("/")[-3]
    seq_dirs = sorted(d for d in glob.glob(f"{action_dir}/*") if os.path.isdir(d))
    bz, cz, recon_ok, missing = [], [], 0, 0
    for d in seq_dirs:
        mnpz = f"{d}/reference_bundle/motion.npz"
        pkl = f"{d}/lower_body_corrected/corrected_smpl.pkl"
        if os.path.isfile(mnpz):
            bz.append(float(np.median(np.load(mnpz, allow_pickle=True)["trans"][:, 2])))
        else:
            missing += 1
        if os.path.isfile(pkl):
            rec = joblib.load(pkl)["merged"]
            cz.append(float(np.median(np.asarray(rec["trans_world"])[:, 1])))
            if "foot_frame_reconciliation" in rec:
                recon_ok += 1
    bz_r = [round(min(bz), 2), round(max(bz), 2)] if bz else None
    cz_r = [round(min(cz), 2), round(max(cz), 2)] if cz else None
    print(f"{action:32s} {len(seq_dirs):3d}   {str(bz_r):22s} {recon_ok}/{len(seq_dirs):<3d}     {cz_r}   miss={missing}")
