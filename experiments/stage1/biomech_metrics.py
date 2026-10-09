#!/usr/bin/env python3
"""Biomechanical / neuromuscular metrics for the Stage-1 endpoints (beyond tube & synergy losses).

Inputs : outputs/stage1_endpoint_compare/kin/<arm>_s<seed>.npz (capture_endpoint_kinematics.py), verified tube.
Outputs: outputs/stage1_endpoint_compare/biomech/{metrics.json, summary.md, figures/*.png}

Metric families
  1 muscle sequencing   peak-phase of the 15 measured channels; Spearman rank agreement with the human order;
                        group ladder legs -> trunk -> shoulder -> elbow -> forearm (proximal-to-distal)
  2 kinetic chain       peak angular-speed timing of trunk rotation, shoulder rotation, elbow, forearm, wrist and hand speed,
                        simulated vs the retargeted human reference
  3 co-contraction      Rudolph CCI = mean 2*min(A,B)/(A+B) for antagonist groups; human values where both sides are measured
  4 recruitment         participation ratio (effective # active muscles), 7-region energy share, dead / saturated actuators
  5 dimensionality      # PCA components for 90 % variance of the 354-D activation
  6 smoothness          RMS second difference of activation; hand-speed profile correlation with reference
  7 tracking            per-body-region joint RMSE, right-hand position error, peak hand speed ratio
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
KIN = REPO / "outputs/stage1_endpoint_compare/kin"
OUT = REPO / "outputs/stage1_endpoint_compare/biomech"
FIG = OUT / "figures"
TUBE = REPO / "artifacts/emg_human_review_v2/verified_tubes/forehand_high_clear"
ARMS = ["T0", "T1", "T2", "T3", "T4"]
COL = {"T0": "#7f7f7f", "T1": "#ff7f0e", "T2": "#2ca02c", "T3": "#d62728", "T4": "#9467bd", "human": "k"}
NBIN = 20
CHANNEL_GROUP = {  # 15 comparable channels -> chain group
    "S2 right:anterior_deltoid": "shoulder", "S3 right:posterior_deltoid": "shoulder", "S4 right:pectoralis_major_clavicular": "shoulder",
    "S5 right:latissimus_dorsi": "shoulder", "S6 right:triceps_lateral": "elbow", "S7 right:pronator_teres": "forearm",
    "S8 right:extensor_carpi_radialis": "forearm", "S9 right:external_oblique": "trunk", "S10 left:external_oblique": "trunk",
    "S11 right:vastus_lateralis": "legs", "S12 left:vastus_lateralis": "legs", "S13 right:biceps_femoris_long_head": "legs",
    "S14 left:biceps_femoris_long_head": "legs", "S15 right:gastrocnemius_medialis": "legs", "S16 left:gastrocnemius_medialis": "legs",
}
GROUP_ORDER = ["legs", "trunk", "shoulder", "elbow", "forearm"]
CHAIN_JOINTS = [("trunk rotation", "axial_rotation"), ("shoulder rotation", "shoulder_rot_r"), ("elbow flexion", "elbow_flex_r"),
                ("forearm pro/sup", "pro_sup_r"), ("wrist flexion", "flexion_r")]
REGIONS = {"pelvis_trunk": range(0, 210), "R_shoulder_upperarm": range(210, 225), "R_elbow_forearm": list(range(225, 234)) + [240, 241],
           "R_wrist": range(234, 240), "L_shoulder_arm": range(242, 274), "R_leg": range(274, 314), "L_leg": range(314, 354)}
JOINT_GROUPS = {"trunk": range(1, 19), "right_arm": range(19, 37), "left_arm": range(37, 55), "right_leg": range(55, 69), "left_leg": range(69, 83)}
CCI_PAIRS = {  # name: (agonist actuator names, antagonist actuator names, human channel pair or None)
    "R shoulder ant/post deltoid": (["DELT1"], ["DELT3"], ("S2 right:anterior_deltoid", "S3 right:posterior_deltoid")),
    "R elbow biceps/triceps": (["BIClong", "BICshort", "BRA"], ["TRIlong", "TRIlat", "TRImed"], None),
    "R wrist extensors/flexors": (["ECRL", "ECRB", "ECU"], ["FCR", "FCU", "PL"], None),
    "R knee quad/hamstring": (["vaslat_r", "vasmed_r", "vasint_r", "recfem_r"], ["bflh_r", "bfsh_r", "semimem_r", "semiten_r"], ("S11 right:vastus_lateralis", "S13 right:biceps_femoris_long_head")),
    "L knee quad/hamstring": (["vaslat_l", "vasmed_l", "vasint_l", "recfem_l"], ["bflh_l", "bfsh_l", "semimem_l", "semiten_l"], ("S12 left:vastus_lateralis", "S14 left:biceps_femoris_long_head")),
    "R ankle tib ant/triceps surae": (["tibant_r"], ["gasmed_r", "gaslat_r", "soleus_r"], None),
}


def bin_curve(x, nbin=NBIN):
    """(T,D) -> (nbin,D) mean per normalised-progress bin over the episode's own length."""
    T = x.shape[0]; b = np.minimum((np.arange(T) / max(T - 1, 1) * nbin).astype(int), nbin - 1)
    out = np.full((nbin, x.shape[1]), np.nan)
    for k in range(nbin):
        m = b == k
        if m.any():
            out[k] = x[m].mean(0)
    return out


TRANSIENT_BINS = 1      # skip the activation spin-up at the very start
MIN_RANGE = 0.02        # a channel that never moves more than this has no meaningful peak


def peak_phase(curve):
    """curve (nbin,) -> progress of the maximum after the transient; NaN for flat / dead channels."""
    c = np.array(curve, dtype=float); c[:TRANSIENT_BINS] = np.nan
    if np.all(np.isnan(c)) or (np.nanmax(c) - np.nanmin(c)) < MIN_RANGE:
        return np.nan
    c = np.where(np.isnan(c), -np.inf, c)
    return (np.argmax(c) + 0.5) / len(curve)


def peak_progress_signal(w, skip_frac=0.05, smooth=5):
    """peak location of a 1-D speed signal after light smoothing, ignoring the first skip_frac."""
    w = np.asarray(w, dtype=float)
    if len(w) >= smooth:
        w = np.convolve(w, np.ones(smooth) / smooth, mode="same")
    s = int(len(w) * skip_frac)
    if len(w) - s < 3 or np.nanmax(w[s:]) <= 1e-6:
        return np.nan
    return (s + int(np.argmax(w[s:]))) / max(len(w) - 1, 1)


def cci(a, b, eps=1e-6):
    return float(np.mean(2 * np.minimum(a, b) / (a + b + eps)))


def load_projection(actuator_names):
    from musclemimic.physiology.emg_anchor import build_emg_observation_projection, load_json_mapping
    mapping = load_json_mapping(TUBE / "emg_observation_mapping.json")
    P, chans = build_emg_observation_projection(mapping, actuator_names)
    return np.asarray(P, dtype=np.float64), list(chans)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", default=["kin"], help="capture dirs under outputs/stage1_endpoint_compare (kin = held-out, kin_train = train)")
    ap.add_argument("--stable-json", default=None, help="restrict to motions listed in t3_stable_motions.json (split, traj)")
    ap.add_argument("--out", default=None, help="output dir name under outputs/stage1_endpoint_compare (default biomech)")
    args = ap.parse_args()
    global OUT, FIG
    if args.out:
        OUT = REPO / "outputs/stage1_endpoint_compare" / args.out; FIG = OUT / "figures"
    allow = None
    if args.stable_json:
        allow = {(r["split"], r["traj"]) for r in json.load(open(args.stable_json))["stable"]}
    FIG.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(KIN / "T*_s*.npz")))
    first = np.load(files[0])
    actuator_names = [str(n) for n in first["actuator_names"]]
    joint_names = [str(j) for j in first["joint_names"]]
    site_names = [str(s) for s in first["site_names"]]
    dt = float(first["dt"])
    P, channels = load_projection(actuator_names)
    tube = np.load(TUBE / "emg_reference_tube.npz")
    human_anchor = tube["anchor_mean"][0]  # (20,15)
    aidx = {n: i for i, n in enumerate(actuator_names)}
    hand = site_names.index("right_hand_mimic")

    def jq(name):  # qpos / qvel index of a 1-dof joint
        k = joint_names.index(name); return 7 + (k - 1), 6 + (k - 1)

    # ---- human references ----
    human_peak = {c: peak_phase(human_anchor[:, i]) for i, c in enumerate(channels)}
    human_group_peak = {g: float(np.nanmean([human_peak[c] for c in channels if CHANNEL_GROUP[c] == g])) for g in GROUP_ORDER}
    human_cci = {}
    for name, (_, _, hp) in CCI_PAIRS.items():
        if hp:
            i, j = channels.index(hp[0]), channels.index(hp[1]); human_cci[name] = cci(human_anchor[:, i], human_anchor[:, j])

    per_run = []
    ref_chain_cache = {}
    file_list = []
    for d in args.dirs:
        split = "val" if d == "kin" else "train"
        file_list += [(f, split) for f in sorted(glob.glob(str(REPO / "outputs/stage1_endpoint_compare" / d / "T*_s*.npz")))]
    per_run_map = {}
    for f, split in file_list:
        z = np.load(f); arm, seed = Path(f).stem.split("_s"); seed = int(seed)
        idx = sorted(int(k[8:]) for k in z.files if k.startswith("act_traj"))
        if allow is not None:
            idx = [i for i in idx if (split, i) in allow]
        if not idx:
            continue
        rec = {"arm": arm, "seed": seed, "n_traj": len(idx), "split": split}
        acc = {k: [] for k in ("peak_spearman", "group_peaks", "chain_sim", "chain_ref", "hand_peak_ratio", "hand_corr", "cci", "pr", "region_share",
                               "dead", "saturated", "pca90", "act_jerk", "joint_rmse", "hand_err", "seq_err")}
        for i in idx:
            A = z[f"act_traj{i}"]; Q = z[f"qpos_traj{i}"]; V = z[f"qvel_traj{i}"]; S = z[f"site_traj{i}"]
            RQ = z[f"ref_qpos_traj{i}"]; RV = z[f"ref_qvel_traj{i}"]; RS = z[f"ref_site_traj{i}"]
            T = A.shape[0]; L = min(T, RQ.shape[0])
            # 1 muscle sequencing
            Y = A @ P.T; Yb = bin_curve(Y)
            sim_peak = np.array([peak_phase(Yb[:, c]) for c in range(len(channels))]); hum_peak = np.array([human_peak[c] for c in channels])
            ok = ~np.isnan(sim_peak) & ~np.isnan(hum_peak)
            acc["peak_spearman"].append(spearmanr(sim_peak[ok], hum_peak[ok]).correlation if ok.sum() >= 4 else np.nan)
            acc["seq_err"].append(float(np.nanmean(np.abs(sim_peak - hum_peak))))
            acc["group_peaks"].append([float(np.nanmean([sim_peak[k] for k, c in enumerate(channels) if CHANNEL_GROUP[c] == g])) for g in GROUP_ORDER])
            # 2 kinetic chain (peak |angular speed| progress) sim vs ref, + hand speed
            def chain(Vm, Sm):
                out = []
                for _, jn in CHAIN_JOINTS:
                    _, vi = jq(jn); out.append(peak_progress_signal(np.abs(Vm[:, vi])))
                hs = np.linalg.norm(np.diff(Sm[:, hand], axis=0), axis=1) / dt
                out.append(peak_progress_signal(hs)); return out, hs
            cs, hs_sim = chain(V, S); cr, hs_ref = chain(RV, RS)
            acc["chain_sim"].append(cs); acc["chain_ref"].append(cr)
            acc["hand_peak_ratio"].append(float(hs_sim.max() / max(hs_ref.max(), 1e-6)))
            n = min(len(hs_sim), len(hs_ref)); acc["hand_corr"].append(float(np.corrcoef(hs_sim[:n], hs_ref[:n])[0, 1]) if n > 5 else np.nan)
            # 3 CCI
            row = {}
            for name, (ag, an, _) in CCI_PAIRS.items():
                a = A[:, [aidx[x] for x in ag if x in aidx]].mean(1); b = A[:, [aidx[x] for x in an if x in aidx]].mean(1); row[name] = cci(a, b)
            acc["cci"].append(row)
            # 4 recruitment
            s1 = A.sum(1); s2 = np.square(A).sum(1); acc["pr"].append(float(np.mean(np.square(s1) / np.maximum(s2, 1e-9))))
            energy = np.square(A).sum(0); tot = energy.sum()
            acc["region_share"].append([float(energy[list(r)].sum() / tot) for r in REGIONS.values()])
            acc["dead"].append(int((A.max(0) < 0.01).sum())); acc["saturated"].append(int((A.mean(0) > 0.9).sum()))
            # 5 dimensionality
            Ac = A - A.mean(0); sv = np.linalg.svd(Ac, compute_uv=False); ev = np.cumsum(sv**2) / max((sv**2).sum(), 1e-12)
            acc["pca90"].append(int(np.searchsorted(ev, 0.90) + 1))
            # 6 smoothness
            acc["act_jerk"].append(float(np.sqrt(np.mean(np.square(np.diff(A, 2, axis=0))))) if T > 3 else np.nan)
            # 7 tracking
            d = Q[:L, 7:] - RQ[:L, 7:]
            acc["joint_rmse"].append({g: float(np.sqrt(np.mean(np.square(d[:, [k - 1 for k in r]])))) for g, r in JOINT_GROUPS.items()})
            acc["hand_err"].append(float(np.mean(np.linalg.norm(S[:L, hand] - RS[:L, hand], axis=1))))
        # aggregate
        rec.update({
            "seq_spearman": float(np.nanmean(acc["peak_spearman"])), "seq_abs_err": float(np.nanmean(acc["seq_err"])),
            "group_peaks": dict(zip(GROUP_ORDER, np.nanmean(np.array(acc["group_peaks"], dtype=float), 0).tolist())),
            "chain_sim": dict(zip([n for n, _ in CHAIN_JOINTS] + ["hand speed"], np.nanmean(np.array(acc["chain_sim"], dtype=float), 0).tolist())),
            "chain_ref": dict(zip([n for n, _ in CHAIN_JOINTS] + ["hand speed"], np.nanmean(np.array(acc["chain_ref"], dtype=float), 0).tolist())),
            "hand_peak_speed_ratio": float(np.nanmean(acc["hand_peak_ratio"])), "hand_speed_corr": float(np.nanmean(acc["hand_corr"])),
            "cci": {k: float(np.mean([r[k] for r in acc["cci"]])) for k in CCI_PAIRS},
            "participation_ratio": float(np.mean(acc["pr"])), "region_share": dict(zip(REGIONS.keys(), np.mean(acc["region_share"], 0).tolist())),
            "dead_actuators": float(np.mean(acc["dead"])), "saturated_actuators": float(np.mean(acc["saturated"])),
            "pca90_components": float(np.mean(acc["pca90"])), "activation_jerk_rms": float(np.nanmean(acc["act_jerk"])),
            "joint_rmse": {g: float(np.mean([r[g] for r in acc["joint_rmse"]])) for g in JOINT_GROUPS},
            "hand_pos_err_m": float(np.mean(acc["hand_err"])),
        })
        per_run.append(rec); print(f"{arm} s{seed}: seq_rho={rec['seq_spearman']:.2f} PR={rec['participation_ratio']:.0f} pca90={rec['pca90_components']:.0f} dead={rec['dead_actuators']:.0f} hand_err={rec['hand_pos_err_m']:.3f}", flush=True)

    def arm_mean(key, sub=None):
        out = {}
        for a in ARMS:
            rs = [r for r in per_run if r["arm"] == a]
            if not rs:
                continue
            out[a] = float(np.nanmean([(r[key][sub] if sub else r[key]) for r in rs]))
        return out

    summary = {"human": {"group_peaks": human_group_peak, "cci": human_cci, "channel_peaks": human_peak}, "per_run": per_run, "arm_mean": {}}
    keys = ["seq_spearman", "seq_abs_err", "hand_peak_speed_ratio", "hand_speed_corr", "participation_ratio", "dead_actuators", "saturated_actuators", "pca90_components", "activation_jerk_rms", "hand_pos_err_m"]
    for k in keys:
        summary["arm_mean"][k] = arm_mean(k)
    for g in GROUP_ORDER:
        summary["arm_mean"][f"group_peak.{g}"] = arm_mean("group_peaks", g)
    for name in CCI_PAIRS:
        summary["arm_mean"][f"cci.{name}"] = arm_mean("cci", name)
    for r in REGIONS:
        summary["arm_mean"][f"region.{r}"] = arm_mean("region_share", r)
    for g in JOINT_GROUPS:
        summary["arm_mean"][f"rmse.{g}"] = arm_mean("joint_rmse", g)
    for n in [x for x, _ in CHAIN_JOINTS] + ["hand speed"]:
        summary["arm_mean"][f"chain_sim.{n}"] = arm_mean("chain_sim", n); summary["arm_mean"][f"chain_ref.{n}"] = arm_mean("chain_ref", n)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT / "metrics.json", "w"), indent=1, ensure_ascii=False)

    # ---------------- figures ----------------
    am = summary["arm_mean"]
    # A: sequencing ladder (muscle groups) + kinetic chain
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]; x = np.arange(len(GROUP_ORDER))
    ax.plot(x, [human_group_peak[g] for g in GROUP_ORDER], "k-o", lw=2.5, label="human sEMG")
    for a in ARMS:
        if a in am["seq_spearman"]:
            ax.plot(x, [am[f"group_peak.{g}"][a] for g in GROUP_ORDER], "-o", color=COL[a], label=f"{a} (rank r={am['seq_spearman'][a]:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(GROUP_ORDER); ax.set_ylabel("peak activation progress"); ax.set_title("Proximal-to-distal muscle sequencing"); ax.legend(fontsize=8)
    ax = axes[1]; names = [n for n, _ in CHAIN_JOINTS] + ["hand speed"]; x = np.arange(len(names))
    ax.plot(x, [am[f"chain_ref.{n}"]["T0"] for n in names], "k-o", lw=2.5, label="human reference kinematics")
    for a in ("T0", "T2", "T3", "T4"):
        if a in am["seq_spearman"]:
            ax.plot(x, [am[f"chain_sim.{n}"][a] for n in names], "-o", color=COL[a], label=a)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20); ax.set_ylabel("peak |angular speed| progress"); ax.set_title("Kinetic chain: peak timing, sim vs reference"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "figA_sequencing.png", dpi=150); plt.close(fig)
    # B: CCI
    fig, ax = plt.subplots(figsize=(13, 4.5)); names = list(CCI_PAIRS); x = np.arange(len(names)); w = 0.14
    for i, a in enumerate(ARMS):
        if a in am["seq_spearman"]:
            ax.bar(x + (i - 2) * w, [am[f"cci.{n}"][a] for n in names], w, color=COL[a], label=a)
    hx = [k for k, n in enumerate(names) if n in human_cci]
    ax.scatter(hx, [human_cci[names[k]] for k in hx], marker="_", s=600, color="k", linewidths=3, label="human (measured pairs)", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, fontsize=8); ax.set_ylabel("co-contraction index"); ax.set_title("Antagonist co-contraction (Rudolph CCI, lower = less wasted co-activation)"); ax.legend(fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(FIG / "figB_cocontraction.png", dpi=150); plt.close(fig)
    # C: region share + recruitment
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]; bottom = np.zeros(len(ARMS)); arms_present = [a for a in ARMS if a in am["seq_spearman"]]
    for r in REGIONS:
        vals = [am[f"region.{r}"][a] for a in arms_present]; ax.bar(arms_present, vals, bottom=bottom[:len(arms_present)], label=r); bottom[:len(arms_present)] += vals
    ax.set_ylabel("share of activation energy"); ax.set_title("Where the effort goes (7 regions)"); ax.legend(fontsize=7)
    ax = axes[1]; ax.bar(arms_present, [am["participation_ratio"][a] for a in arms_present], color=[COL[a] for a in arms_present]); ax.set_title("Effective # of active muscles (participation ratio)")
    ax2 = ax.twinx(); ax2.plot(arms_present, [am["dead_actuators"][a] for a in arms_present], "k^--", label="dead (max<0.01)"); ax2.plot(arms_present, [am["saturated_actuators"][a] for a in arms_present], "rv--", label="saturated (mean>0.9)"); ax2.legend(fontsize=8); ax2.set_ylabel("# actuators")
    ax = axes[2]; ax.bar(arms_present, [am["pca90_components"][a] for a in arms_present], color=[COL[a] for a in arms_present]); ax.set_title("# PCA components for 90% variance (354-D activation)")
    fig.tight_layout(); fig.savefig(FIG / "figC_recruitment.png", dpi=150); plt.close(fig)
    # D: hand speed profiles for trajectory 0 (ref vs arms, seed 0)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    files = [f for f, s in file_list if s == "val"] or files
    for f in files:
        arm, seed = Path(f).stem.split("_s")
        if int(seed) != 0 or arm not in ("T0", "T2", "T3", "T4"):
            continue
        z = np.load(f); S = z["site_traj0"]; RS = z["ref_site_traj0"]
        hs = np.linalg.norm(np.diff(S[:, hand], axis=0), axis=1) / dt; ax.plot(np.arange(len(hs)) / max(len(RS) - 2, 1), hs, color=COL[arm], label=arm)
        if arm == "T0":
            hr = np.linalg.norm(np.diff(RS[:, hand], axis=0), axis=1) / dt; ax.plot(np.arange(len(hr)) / max(len(hr) - 1, 1), hr, "k-", lw=2.5, label="human reference")
    ax.set_xlabel("normalised progress"); ax.set_ylabel("right-hand speed (m/s)"); ax.set_title("Racket-hand speed profile, held-out trajectory 0 (seed 0)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "figD_hand_speed.png", dpi=150); plt.close(fig)

    # ---------------- markdown ----------------
    def row(label, key, fmt="{:.3f}", human=None):
        cells = [fmt.format(am[key][a]) if a in am[key] else "—" for a in ARMS]
        return f"| {label} | {human if human is not None else '—'} | " + " | ".join(cells) + " |"
    lines = ["# Stage-1 endpoint 生物力学 / 神经肌肉指标（80/20 held-out，按 arm 平均）", "",
             "| 指标 | 人体 | T0 | T1 | T2 | T3 | T4 |", "|---|---:|---:|---:|---:|---:|---:|",
             row("15 通道峰值时序 Spearman ρ（vs 人体）↑", "seq_spearman", human="1.00"),
             row("15 通道峰值相位平均绝对误差 ↓", "seq_abs_err"),
             *[row(f"峰值相位 · {g}", f"group_peak.{g}", "{:.2f}", human=f"{human_group_peak[g]:.2f}") for g in GROUP_ORDER],
             row("右手峰值速度 / 参考 ↑→1", "hand_peak_speed_ratio", "{:.2f}", human="1.00"),
             row("右手速度曲线相关 ↑", "hand_speed_corr", "{:.2f}", human="1.00"),
             *[row(f"CCI · {n}", f"cci.{n}", "{:.2f}", human=(f"{human_cci[n]:.2f}" if n in human_cci else None)) for n in CCI_PAIRS],
             row("有效活跃肌肉数（participation ratio）", "participation_ratio", "{:.0f}"),
             row("死区肌肉数（max<0.01）↓", "dead_actuators", "{:.0f}"), row("饱和肌肉数（mean>0.9）↓", "saturated_actuators", "{:.0f}"),
             row("PCA 90% 方差所需分量数", "pca90_components", "{:.1f}"), row("激活二阶差分 RMS ↓", "activation_jerk_rms", "{:.4f}"),
             *[row(f"能量占比 · {r}", f"region.{r}", "{:.3f}") for r in REGIONS],
             *[row(f"关节 RMSE · {g}", f"rmse.{g}", "{:.3f}") for g in JOINT_GROUPS], row("右手位置误差 (m) ↓", "hand_pos_err_m"),
             "", "动力链峰值时刻（|角速度| 峰值的归一化进度；参考 = 重定向的人体动作）", "",
             "| 环节 | 参考 | T0 | T1 | T2 | T3 | T4 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for n in [x for x, _ in CHAIN_JOINTS] + ["hand speed"]:
        lines.append(f"| {n} | {am[f'chain_ref.{n}']['T0']:.2f} | " + " | ".join(f"{am[f'chain_sim.{n}'][a]:.2f}" if a in am[f'chain_sim.{n}'] else "—" for a in ARMS) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines)); print("figures ->", FIG)


if __name__ == "__main__":
    main()
