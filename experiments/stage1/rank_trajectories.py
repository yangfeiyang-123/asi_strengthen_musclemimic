#!/usr/bin/env python3
"""Per-trajectory view over all 100 aug100 motions (80 train + 20 held-out) for the Stage-1 endpoints.

Inputs : outputs/stage1_endpoint_compare/<arm>_s<seed>.json   (held-out, from compare_endpoints_80_20.py)
         outputs/stage1_endpoint_compare/kin_train/<arm>_s<seed>.json (train, from capture_endpoint_kinematics.py --split train)
Outputs: outputs/stage1_endpoint_compare/trajectories/{trajectory_table_100.md, trajectory_table_100.json, fig_*.png}
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/stage1_endpoint_compare"
DST = OUT / "trajectories"
ARMS = ["T0", "T1", "T2", "T3", "T4"]
COL = {"T0": "#7f7f7f", "T1": "#ff7f0e", "T2": "#2ca02c", "T3": "#d62728", "T4": "#9467bd"}


def val_motion_names():
    cfg = json.load(open(REPO / "checkpoints/stage1/seed0/T3/checkpoint_39063/config/metadata"))
    return [Path(p).name for p in cfg["experiment"]["validation"]["amass_dataset_conf"]["rel_dataset_path"]]


def load():
    rows = {}  # (split, traj) -> {"motion":..., "len":..., arms: {arm: {seed: metrics}}}
    vnames = val_motion_names()
    for f in sorted(glob.glob(str(OUT / "T*_s*.json"))):
        d = json.load(open(f)); arm, seed = d["arm"], int(d["seed"])
        for r in d["per_traj"]:
            key = ("val", r["traj"]); e = rows.setdefault(key, {"motion": vnames[r["traj"]] if r["traj"] < len(vnames) else "", "len": r["traj_len"], "arms": {}})
            e["arms"].setdefault(arm, {})[seed] = r
    for f in sorted(glob.glob(str(OUT / "kin_train/T*_s*.json"))):
        d = json.load(open(f)); arm, seed = d["arm"], int(d["seed"])
        for r in d["per_traj"]:
            key = ("train", r["traj"]); e = rows.setdefault(key, {"motion": r.get("motion", ""), "len": r["traj_len"], "arms": {}})
            e["arms"].setdefault(arm, {})[seed] = r
    return rows


def mean_over_seeds(d, key):
    vals = [v[key] for v in d.values() if v.get(key) is not None]
    return float(np.mean(vals)) if vals else np.nan


def main():
    DST.mkdir(parents=True, exist_ok=True)
    rows = load()
    table = []
    for (split, ti), e in sorted(rows.items(), key=lambda kv: (kv[0][0] != "val", kv[0][1])):
        rec = {"split": split, "traj": ti, "motion": e["motion"], "len": e["len"]}
        for a in ARMS:
            d = e["arms"].get(a, {})
            rec[f"{a}.syn"] = mean_over_seeds(d, "synergy_loss"); rec[f"{a}.cos"] = mean_over_seeds(d, "synergy_shape_cosine")
            rec[f"{a}.anchor"] = mean_over_seeds(d, "anchor_loss"); rec[f"{a}.cov"] = mean_over_seeds(d, "coverage"); rec[f"{a}.n"] = len(d)
        rec["T3-T2"] = rec["T3.syn"] - rec["T2.syn"]; rec["T4-T3"] = rec["T4.syn"] - rec["T3.syn"]; rec["T0-T3"] = rec["T0.syn"] - rec["T3.syn"]
        table.append(rec)
    json.dump(table, open(DST / "trajectory_table_100.json", "w"), indent=1, ensure_ascii=False)

    # ---- summaries ----
    def wins(split=None):
        sub = [r for r in table if split is None or r["split"] == split]
        return {
            "n": len(sub),
            "T3<T4": int(sum(r["T4-T3"] > 0 for r in sub)), "T3<T0": int(sum(r["T0-T3"] > 0 for r in sub)), "T3<T2": int(sum(r["T3-T2"] < 0 for r in sub)),
            "T3 best of 5": int(sum(min(ARMS, key=lambda a: r[f"{a}.syn"] if not np.isnan(r[f"{a}.syn"]) else 9) == "T3" for r in sub)),
            "T2 best of 5": int(sum(min(ARMS, key=lambda a: r[f"{a}.syn"] if not np.isnan(r[f"{a}.syn"]) else 9) == "T2" for r in sub)),
        }
    def wins_anchor(split=None):
        sub = [r for r in table if split is None or r["split"] == split]
        best = lambda r, key: min(ARMS, key=lambda a: r[f"{a}.{key}"] if not np.isnan(r[f"{a}.{key}"]) else 9)
        def combined_best(r):
            # rank-sum of synergy loss and anchor loss across the five arms (lower is better)
            ranks = {a: 0.0 for a in ARMS}
            for key in ("syn", "anchor"):
                order = sorted(ARMS, key=lambda a: r[f"{a}.{key}"] if not np.isnan(r[f"{a}.{key}"]) else 9)
                for k, a in enumerate(order): ranks[a] += k
            return min(ARMS, key=lambda a: ranks[a])
        return {"n": len(sub), "T3 anchor best": int(sum(best(r, "anchor") == "T3" for r in sub)), "T2 anchor best": int(sum(best(r, "anchor") == "T2" for r in sub)),
                "T3 anchor<T2": int(sum(r["T3.anchor"] < r["T2.anchor"] for r in sub)),
                "T3 combined best": int(sum(combined_best(r) == "T3" for r in sub)), "T2 combined best": int(sum(combined_best(r) == "T2" for r in sub))}
    W = {"all": wins(), "val": wins("val"), "train": wins("train")}
    WA = {"all": wins_anchor(), "val": wins_anchor("val"), "train": wins_anchor("train")}
    med = {s: {a: float(np.nanmedian([r[f"{a}.syn"] for r in table if s == "all" or r["split"] == s])) for a in ARMS} for s in ("all", "val", "train")}
    cov = {s: {a: float(np.nanmean([r[f"{a}.cov"] for r in table if s == "all" or r["split"] == s])) for a in ARMS} for s in ("all", "val", "train")}

    # ---- figures ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]
    for i, a in enumerate(ARMS):
        for split, mk in (("train", "o"), ("val", "s")):
            ys = [r[f"{a}.syn"] for r in table if r["split"] == split]; xs = i + (np.random.RandomState(0).rand(len(ys)) - 0.5) * 0.5
            ax.scatter(xs, ys, s=14, color=COL[a], marker=mk, alpha=0.6 if split == "train" else 0.95, edgecolor="k" if split == "val" else "none", linewidths=0.4)
        ax.hlines(med["all"][a], i - 0.3, i + 0.3, color="k", lw=2)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS); ax.set_ylabel("synergy loss (lower = closer to human)"); ax.set_yscale("log")
    ax.set_title("All 100 motions (o train, ■ held-out); black = median")
    ax = axes[1]
    for split, mk in (("train", "o"), ("val", "s")):
        sub = [r for r in table if r["split"] == split]
        ax.scatter([r["T2.syn"] for r in sub], [r["T3.syn"] for r in sub], s=18, marker=mk, color=COL["T3"], alpha=0.6 if split == "train" else 0.95, edgecolor="k" if split == "val" else "none", linewidths=0.4, label=split)
    lim = [0.02, 1.0]; ax.plot(lim, lim, "k--", lw=1); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("T2 synergy loss"); ax.set_ylabel("T3 synergy loss"); ax.set_title("Per motion: T3 vs T2 (below the line = T3 better)"); ax.legend()
    ax = axes[2]
    for split, mk in (("train", "o"), ("val", "s")):
        sub = [r for r in table if r["split"] == split]
        ax.scatter([r["T4.syn"] for r in sub], [r["T3.syn"] for r in sub], s=18, marker=mk, color=COL["T4"], alpha=0.6 if split == "train" else 0.95, edgecolor="k" if split == "val" else "none", linewidths=0.4, label=split)
    ax.plot(lim, lim, "k--", lw=1); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("T4 (phase-shifted) synergy loss"); ax.set_ylabel("T3 synergy loss"); ax.set_title("Per motion: T3 vs T4 (below the line = T3 better)"); ax.legend()
    fig.tight_layout(); fig.savefig(DST / "fig_trajectories_100.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(17, 4.2))
    order = sorted(table, key=lambda r: (r["split"] != "val", r["len"], r["traj"]))
    x = np.arange(len(order))
    for a in ("T0", "T2", "T3", "T4"):
        ax.plot(x, [r[f"{a}.syn"] for r in order], "-", color=COL[a], lw=1.2, label=a)
    for k, r in enumerate(order):
        if r["split"] == "val":
            ax.axvspan(k - 0.5, k + 0.5, color="#1F6F8B", alpha=0.08)
    ax.set_yscale("log"); ax.set_ylabel("synergy loss"); ax.set_xlabel("motion (held-out shaded, then train; sorted by length)"); ax.legend(ncol=4, fontsize=8); ax.set_title("Synergy loss per motion, all 100 (seed-averaged)")
    fig.tight_layout(); fig.savefig(DST / "fig_trajectories_100_lines.png", dpi=150); plt.close(fig)

    # ---- markdown ----
    L = ["# 全部 100 条动作（80 train + 20 held-out）逐条对比", "",
         "协同 loss / 余弦 / 覆盖率按 seed 平均（T0 1 seed，T1 2，T2/T3/T4 各 3）。训练集上的结果衡量拟合，不是泛化；held-out 才是泛化。", "",
         "## 胜负计数（协同 loss，越低越好）", "", "| 范围 | n | T3 优于 T4 | T3 优于 T0 | T3 优于 T2 | T3 五臂最优 | T2 五臂最优 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s in ("all", "val", "train"):
        w = W[s]; L.append(f"| {s} | {w['n']} | {w['T3<T4']} | {w['T3<T0']} | {w['T3<T2']} | {w['T3 best of 5']} | {w['T2 best of 5']} |")
    L += ["", "## 胜负计数（anchor loss 与 协同+anchor 综合排名）", "", "| 范围 | n | T3 anchor 最优 | T2 anchor 最优 | T3 anchor 优于 T2 | T3 综合最优 | T2 综合最优 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s in ("all", "val", "train"):
        w = WA[s]; L.append(f"| {s} | {w['n']} | {w['T3 anchor best']} | {w['T2 anchor best']} | {w['T3 anchor<T2']} | {w['T3 combined best']} | {w['T2 combined best']} |")
    L += ["", "## 中位数与覆盖率", "", "| 范围 | " + " | ".join(f"{a} syn 中位" for a in ARMS) + " | " + " | ".join(f"{a} 覆盖率" for a in ARMS) + " |", "|---|" + "---:|" * 10]
    for s in ("all", "val", "train"):
        L.append(f"| {s} | " + " | ".join(f"{med[s][a]:.3f}" for a in ARMS) + " | " + " | ".join(f"{cov[s][a]:.2f}" for a in ARMS) + " |")
    top = sorted([r for r in table if not np.isnan(r["T3.syn"])], key=lambda r: -(r["T4-T3"] + r["T0-T3"] - 2 * max(r["T3-T2"], 0)))[:12]
    L += ["", "## T3 优势最大的 12 条（按 T3 对 T4、T0 的领先减去对 T2 的落后排序）", "", "| split | traj | 动作 | 帧 | T0 | T1 | T2 | T3 | T4 | T3 覆盖 | T3−T2 |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in top:
        L.append(f"| {r['split']} | {r['traj']} | {r['motion']} | {r['len']} | " + " | ".join(f"{r[f'{a}.syn']:.2f}" for a in ARMS) + f" | {r['T3.cov']:.2f} | {r['T3-T2']:+.2f} |")
    L += ["", "## 全表", "", "| split | traj | 动作 | 帧 | T0 syn | T1 syn | T2 syn | T3 syn | T4 syn | T3 cos | T3 cov | T0 cov | T2 cov | T4 cov |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        L.append(f"| {r['split']} | {r['traj']} | {r['motion']} | {r['len']} | " + " | ".join(f"{r[f'{a}.syn']:.2f}" for a in ARMS) + f" | {r['T3.cos']:.2f} | {r['T3.cov']:.2f} | {r['T0.cov']:.2f} | {r['T2.cov']:.2f} | {r['T4.cov']:.2f} |")
    (DST / "trajectory_table_100.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40])); print("... full table in", DST / "trajectory_table_100.md")


if __name__ == "__main__":
    main()
