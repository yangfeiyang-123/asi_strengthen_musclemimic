#!/usr/bin/env python3
"""Visualise the Stage-1 endpoint comparison (outputs of compare_endpoints_80_20.py).

Figures (outputs/stage1_endpoint_compare/figures/):
  fig1_channel_phase_heatmaps.png  human tube median vs sim (T0/T2/T3/T4), 15 channels x 20 phase bins, per-channel z-score
  fig2_synergy_coefficients.png    3 synergy coefficient curves over phase: human tube band vs sim arms
  fig3_channel_timing_correlation.png  per-channel Pearson r between sim and human phase curves (scale-invariant), per arm
  fig4_channel_anchor_loss.png     per-channel anchor (amplitude tube) loss, per arm
  fig5_envelope_psd.png            frequency domain: normalised PSD of human envelope vs sim projected activation
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, welch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/stage1_endpoint_compare"
FIG = OUT / "figures"
TUBE = REPO / "artifacts/emg_human_review_v2/verified_tubes/forehand_high_clear"
HUMAN_TRIALS = REPO / "jidian_measurement/data/P002/S20260721_A/trials/forehand_high_clear"
ARMS = ["T0", "T1", "T2", "T3", "T4"]
COLORS = {"T0": "#7f7f7f", "T1": "#ff7f0e", "T2": "#2ca02c", "T3": "#d62728", "T4": "#9467bd"}
LABEL = {"T0": "T0 no EMG", "T1": "T1 anchor only", "T2": "T2 synergy only", "T3": "T3 PEASD-Lite", "T4": "T4 phase-shifted"}
NBIN = 20


ALLOW = None  # set from --stable-json: {(split, traj)}


def load_sim():
    """held-out from compare npz (OUT/), train from kin_train npz; optionally restricted to ALLOW."""
    runs = {}
    sources = [("val", str(OUT / "T*_s*.npz"))]
    if ALLOW is not None:
        sources.append(("train", str(OUT / "kin_train" / "T*_s*.npz")))
    for split, pattern in sources:
        for f in sorted(glob.glob(pattern)):
            arm, seed = Path(f).stem.split("_s"); seed = int(seed)
            z = np.load(f)
            idx = sorted(int(k[len("act_traj"):]) for k in z.files if k.startswith("act_traj"))
            if ALLOW is not None:
                idx = [i for i in idx if (split, i) in ALLOW]
            if not idx:
                continue
            r = runs.setdefault((arm, seed), {"proj": [], "coef": [], "phase": [], "summary": None})
            r["proj"] += [z[f"projected_traj{i}"] for i in idx]; r["coef"] += [z[f"synergy_coeff_traj{i}"] for i in idx]; r["phase"] += [z[f"phase_traj{i}"] for i in idx]
            jf = Path(f).with_suffix(".json")
            if jf.exists():
                s = json.load(open(jf)); per = [x for x in s["per_traj"] if x["traj"] in idx]
                base = r["summary"] or {"anchor_channel_loss_traj_mean": None, "_n": 0}
                if "anchor_channel_loss_traj_mean" in s:  # compare json carries per-channel anchor loss
                    prev = base.get("anchor_channel_loss_traj_mean"); n0 = base.get("_n", 0)
                    cur = np.mean([x["anchor_channel_loss"] for x in per], axis=0) if per and "anchor_channel_loss" in per[0] else None
                    if cur is not None:
                        base["anchor_channel_loss_traj_mean"] = (cur if prev is None else (np.asarray(prev) * n0 + cur * len(per)) / (n0 + len(per))).tolist(); base["_n"] = n0 + len(per)
                r["summary"] = base
    channels = [str(c) for c in np.load(str(OUT / "T3_s0.npz"))["channel_names"]]
    return runs, channels


def bin_mean(curves, phases, nbin=NBIN):
    """Average (T,D) curves into nbin phase bins across trajectories -> (nbin, D)."""
    acc = np.zeros((nbin, curves[0].shape[1])); cnt = np.zeros(nbin)
    for c, p in zip(curves, phases):
        b = np.clip((p * nbin).astype(int), 0, nbin - 1)
        for k in range(nbin):
            m = b == k
            if m.any():
                acc[k] += c[m].mean(0); cnt[k] += 1
    return acc / np.maximum(cnt, 1)[:, None]


def arm_bin_mean(runs, key, arm):
    mats = [bin_mean(r[key], r["phase"]) for (a, _), r in runs.items() if a == arm]
    return np.mean(mats, axis=0) if mats else None


def zrow(M):
    mu = M.mean(0, keepdims=True); sd = M.std(0, keepdims=True) + 1e-8
    return (M - mu) / sd


def short(ch):
    return ch.split(" ", 1)[1].replace("right:", "R ").replace("left:", "L ").replace("_", " ")


def main():
    import argparse
    global ALLOW, FIG
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stable-json", default=None); ap.add_argument("--fig-dir", default=None)
    args = ap.parse_args()
    if args.stable_json:
        ALLOW = {(r["split"], r["traj"]) for r in json.load(open(args.stable_json))["stable"]}
    if args.fig_dir:
        FIG = OUT / args.fig_dir
    FIG.mkdir(parents=True, exist_ok=True)
    runs, channels = load_sim()
    tube = np.load(TUBE / "emg_reference_tube.npz")
    human_anchor = tube["anchor_mean"][0]  # (20,15)
    human_syn = tube["synergy_mean"][0]    # (20,3)
    human_syn_scale = tube["synergy_scale"][0]
    x = (np.arange(NBIN) + 0.5) / NBIN

    # ---- fig1: heatmaps ----
    panels = [("Human tube (median)", human_anchor)] + [(LABEL[a], arm_bin_mean(runs, "proj", a)) for a in ("T0", "T2", "T3", "T4")]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 6), sharey=True)
    for ax, (title, M) in zip(axes, panels):
        im = ax.imshow(zrow(M).T, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, extent=[0, 1, len(channels) - 0.5, -0.5])
        ax.set_title(title, fontsize=10); ax.set_xlabel("normalised motion progress")
    axes[0].set_yticks(range(len(channels))); axes[0].set_yticklabels([short(c) for c in channels], fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.015, label="per-channel z-score over phase")
    fig.suptitle("15 measured channels x 20 phase bins: activation timing pattern (row-wise z-score, amplitude removed)")
    fig.savefig(FIG / "fig1_channel_phase_heatmaps.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- fig2: synergy coefficient curves ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for j, ax in enumerate(axes):
        lo = np.clip(human_syn[:, j] - human_syn_scale[:, j], 0, None); hi = human_syn[:, j] + human_syn_scale[:, j]
        ax.fill_between(x, lo, hi, color="k", alpha=0.12, label="human tube (median ± MAD scale)")
        ax.plot(x, human_syn[:, j], "k-", lw=2, label="human median")
        for a in ("T0", "T2", "T3", "T4"):
            M = arm_bin_mean(runs, "coef", a)
            ax.plot(x, M[:, j], color=COLORS[a], lw=1.8, label=LABEL[a])
        ax.set_title(f"synergy S{j + 1} coefficient"); ax.set_xlabel("normalised motion progress")
    axes[0].set_ylabel("coefficient (relu(Q y))"); axes[0].legend(fontsize=8)
    fig.suptitle("Synergy coefficients over the stroke: human reference vs simulated arms (seed-averaged)")
    fig.savefig(FIG / "fig2_synergy_coefficients.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- fig3: per-channel timing correlation (scale invariant) ----
    corr = {}
    for a in ARMS:
        M = arm_bin_mean(runs, "proj", a)
        corr[a] = np.array([np.corrcoef(M[:, c], human_anchor[:, c])[0, 1] for c in range(len(channels))])
    fig, ax = plt.subplots(figsize=(14, 4.5))
    w = 0.16
    for i, a in enumerate(ARMS):
        ax.bar(np.arange(len(channels)) + (i - 2) * w, corr[a], w, color=COLORS[a], label=f"{LABEL[a]} (mean r={np.nanmean(corr[a]):.2f})")
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(len(channels))); ax.set_xticklabels([short(c) for c in channels], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r (sim vs human, over 20 phase bins)"); ax.legend(fontsize=8, ncol=3)
    ax.set_title("Per-channel timing correlation with human EMG (amplitude-free)")
    fig.savefig(FIG / "fig3_channel_timing_correlation.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- fig4: per-channel anchor loss ----
    fig, ax = plt.subplots(figsize=(14, 4.5))
    for i, a in enumerate(ARMS):
        vals = [np.array(r["summary"]["anchor_channel_loss_traj_mean"]) for (aa, _), r in runs.items() if aa == a and r["summary"] and r["summary"].get("anchor_channel_loss_traj_mean") is not None]
        if not vals:
            continue
        v = np.mean(vals, axis=0)
        ax.bar(np.arange(len(channels)) + (i - 2) * w, v, w, color=COLORS[a], label=f"{LABEL[a]} (mean {v.mean():.2f})")
    ax.set_yscale("log"); ax.set_xticks(range(len(channels))); ax.set_xticklabels([short(c) for c in channels], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("anchor (amplitude tube) loss, log"); ax.legend(fontsize=8, ncol=3)
    ax.set_title("Per-channel amplitude-tube loss (lower = simulated amplitude closer to human range)")
    fig.savefig(FIG / "fig4_channel_anchor_loss.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- fig5: envelope PSD (frequency domain) ----
    human_psd = []; freqs_h = None
    for tdir in sorted(HUMAN_TRIALS.glob("trial_*")):
        try:
            z = np.load(tdir / "mvc_normalized_emg.npz")
            names = [str(n) for n in z["channel_names"]]
            fs = float(z["fs_hz"]); env = z["normalized_envelope"]
            cue = 0
            with open(tdir / "events.csv") as f:
                for row in csv.DictReader(f):
                    if row["event_name"] == "movement_cue" and row["sample_index"]:
                        cue = int(row["sample_index"])
            seg = env[cue:, [names.index(c) for c in channels]]
            seg = seg - seg.mean(0, keepdims=True)
            f_h, p = welch(seg, fs=fs, nperseg=min(len(seg), 4096), axis=0)
            keep = f_h <= 10.0
            pm = p[keep].mean(1)
            freqs_h = f_h[keep]; human_psd.append(pm / np.trapezoid(pm, f_h[keep]))
        except Exception as exc:  # noqa: BLE001
            print("skip", tdir.name, exc)
    human_psd = np.mean(human_psd, axis=0)  # (F,)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(freqs_h, human_psd, "k-", lw=2, label=f"human envelope (n={len(list(HUMAN_TRIALS.glob('trial_*')))} trials)")
    for a in ("T0", "T2", "T3", "T4"):
        ps = []
        for (aa, _), r in runs.items():
            if aa != a:
                continue
            for c in r["proj"]:
                # same 4 Hz zero-phase Butterworth envelope filter as the human preprocessing
                b, a_ = butter(4, 4.0 / (100.0 / 2), btype="low")
                c = filtfilt(b, a_, c, axis=0) if len(c) > 30 else c
                c = c - c.mean(0, keepdims=True)
                f_s, p = welch(c, fs=100.0, nperseg=min(len(c), 64), axis=0)
                keep = f_s <= 10.0
                pm = p[keep].mean(1)
                area = np.trapezoid(pm, f_s[keep])
                if not np.isfinite(area) or area <= 0:
                    continue
                ps.append(np.interp(freqs_h, f_s[keep], pm / area))
        ax.plot(freqs_h, np.mean(ps, axis=0), color=COLORS[a], lw=1.8, label=LABEL[a])
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("normalised PSD of activation envelope (area = 1)"); ax.set_yscale("log")
    ax.legend(fontsize=8); ax.axvline(4.0, color="k", ls=":", lw=0.8); ax.text(4.1, ax.get_ylim()[0] if False else 1e-4, "4 Hz envelope low-pass (both)", fontsize=8)
    ax.set_title("Frequency domain: envelope spectrum shape after identical 4 Hz low-pass, human vs simulated")
    fig.savefig(FIG / "fig5_envelope_psd.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    json.dump({a: corr[a].tolist() for a in ARMS} | {"channels": channels}, open(OUT / "timing_correlation.json", "w"), indent=1)
    print("timing r mean per arm:", {a: round(float(np.nanmean(corr[a])), 3) for a in ARMS})
    print("figures in", FIG)


if __name__ == "__main__":
    main()
