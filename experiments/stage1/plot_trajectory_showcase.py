#!/usr/bin/env python3
"""Per-trajectory showcase figure: synergy coefficients (human tube vs arms) + 15-channel timing strip + hand speed.
Usage: .venv/bin/python experiments/stage1/plot_trajectory_showcase.py --trajs 4 13 --seed 0
"""
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
REPO=Path(__file__).resolve().parents[2]; OUT=REPO/"outputs/stage1_endpoint_compare"; FIG=OUT/"figures"
TUBE=REPO/"artifacts/emg_human_review_v2/verified_tubes/forehand_high_clear"
ARMS=["T0","T1","T2","T3","T4"]; COL={"T0":"#7f7f7f","T1":"#ff7f0e","T2":"#2ca02c","T3":"#d62728","T4":"#9467bd"}
LABEL={"T0":"T0 no EMG","T1":"T1 anchor only","T2":"T2 synergy only","T3":"T3 PEASD-Lite","T4":"T4 phase-shifted"}
NBIN=20
def bin_mean(c,p):
    b=np.clip((p*NBIN).astype(int),0,NBIN-1); out=np.full((NBIN,c.shape[1]),np.nan)
    for k in range(NBIN):
        m=b==k
        if m.any(): out[k]=c[m].mean(0)
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--trajs",nargs="+",type=int,default=[4,13]); ap.add_argument("--seed",type=int,default=0); a=ap.parse_args()
    tube=np.load(TUBE/"emg_reference_tube.npz"); hs_mean=tube["synergy_mean"][0]; hs_scale=tube["synergy_scale"][0]; x=(np.arange(NBIN)+0.5)/NBIN
    summ={arm: {r["traj"]:r for r in json.load(open(OUT/f"{arm}_s{a.seed}.json"))["per_traj"]} for arm in ARMS}
    for ti in a.trajs:
        fig,axes=plt.subplots(1,4,figsize=(20,4.2))
        for j in range(3):
            ax=axes[j]; lo=np.clip(hs_mean[:,j]-hs_scale[:,j],0,None); hi=hs_mean[:,j]+hs_scale[:,j]
            ax.fill_between(x,lo,hi,color="k",alpha=0.12,label="human tube"); ax.plot(x,hs_mean[:,j],"k-",lw=2.2,label="human median")
            for arm in ARMS:
                z=np.load(OUT/f"{arm}_s{a.seed}.npz"); c=z[f"synergy_coeff_traj{ti}"]; p=z[f"phase_traj{ti}"]; M=bin_mean(c,p)
                r=summ[arm][ti]; ax.plot(x,M[:,j],color=COL[arm],lw=1.8,label=f"{LABEL[arm]}  loss {r['synergy_loss']:.2f}, cos {r['synergy_shape_cosine']:.2f}, cov {r['coverage']:.2f}")
            ax.set_title(f"synergy S{j+1} coefficient"); ax.set_xlabel("normalised motion progress")
        axes[0].set_ylabel("coefficient"); axes[0].legend(fontsize=7)
        # hand speed from kin
        ax=axes[3]; kin0=np.load(OUT/f"kin/T0_s{a.seed}.npz"); site=[str(s) for s in kin0["site_names"]]; hand=site.index("right_hand_mimic"); dt=float(kin0["dt"])
        RS=kin0[f"ref_site_traj{ti}"]; hr=np.linalg.norm(np.diff(RS[:,hand],axis=0),axis=1)/dt; ax.plot(np.arange(len(hr))/max(len(hr)-1,1),hr,"k-",lw=2.2,label="human reference")
        for arm in ARMS:
            S=np.load(OUT/f"kin/{arm}_s{a.seed}.npz")[f"site_traj{ti}"]; hs=np.linalg.norm(np.diff(S[:,hand],axis=0),axis=1)/dt
            ax.plot(np.arange(len(hs))/max(len(RS)-2,1),hs,color=COL[arm],lw=1.6,label=LABEL[arm])
        ax.set_title("right-hand speed (m/s)"); ax.set_xlabel("normalised motion progress"); ax.legend(fontsize=7)
        fig.suptitle(f"held-out trajectory {ti} ({RS.shape[0]} frames), seed {a.seed}: human reference vs five arms"); fig.tight_layout()
        out=FIG/f"showcase_traj{ti}_s{a.seed}.png"; fig.savefig(out,dpi=140); plt.close(fig); print("wrote",out)
if __name__=="__main__": main()
