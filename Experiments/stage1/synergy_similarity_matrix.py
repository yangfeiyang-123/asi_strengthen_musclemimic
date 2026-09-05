#!/usr/bin/env python3
"""3 实验臂 × 3 发力链的"学成 vs 实测"相似度矩阵 + T3 相对 T2/T4 的改善百分比。

相似度定义（主指标）：每条协同的策略系数曲线（逐轨迹，归一化进度网格）与实测 tube
synergy_mean 参考曲线的**余弦相似度**（对有效进度区间），取 4 条共同近全程轨迹
（0/3/4/5）的均值±std。辅助指标：Pearson 相关（中心化形状）、强度比（曲线积分比值）。

T3 改善口径（对 Tx ∈ {T2, T4}）：
  * Δpp      = sim(T3) - sim(Tx)（百分点，最稳健）；
  * 相对提升 = (sim(T3) - sim(Tx)) / |sim(Tx)|（相似度相对值，sim(Tx)≈0 时无意义，标 n/a）；
  * 误差缩减 = (sim(T3) - sim(Tx)) / (1 - sim(Tx))（"剩余差距"被消掉的比例）。

输入：{t2,t3,t4}_320m_activation_rollout.npz + verified tube。
输出：synergy_similarity_matrix.json / .md、figures/synergy6_similarity_matrix.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TUBE_DIR = REPO / "artifacts/emg_human_review_v2/verified_tubes/forehand_high_clear"
FIG_DIR = HERE / "figures"

ARMS = ["T2", "T3", "T4"]
ARM_COLORS = {"T2": "#9467bd", "T3": "#2ca02c", "T4": "#d62728"}
ARM_DESC = {"T2": "synergy-only", "T3": "real anchor+synergy", "T4": "相移负对照"}
SYNERGY_LABELS = ["S1 蹬地转体链", "S2 支撑稳定链", "S3 上肢引拍链"]
SYNERGY_SHORT = ["S1", "S2", "S3"]
COMMON_TRAJS = (0, 3, 4, 5)  # 三个 arm 共同的近全程轨迹集（coverage≥99%）
GRID_N = 200
STAGE_BOUNDS = (0.0, 0.45, 0.65, 1.0)

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def resample_progress(progress, values, grid):
    out = np.full((grid.size, values.shape[1]), np.nan)
    if progress.size < 2:
        return out
    lo, hi = float(progress[0]), float(progress[-1])
    inside = (grid >= lo) & (grid <= hi)
    for d in range(values.shape[1]):
        out[inside, d] = np.interp(grid[inside], progress, values[:, d])
    return out


def cosine_masked(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10:
        return np.nan
    x, y = a[m], b[m]
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 1e-12 else np.nan


def pearson_masked(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10:
        return np.nan
    x, y = a[m] - np.mean(a[m]), b[m] - np.mean(b[m])
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 1e-12 else np.nan


def main() -> int:
    from musclemimic.physiology.emg_anchor import build_emg_observation_projection
    from musclemimic.physiology.emg_reference import synergy_projection_matrix

    with open(TUBE_DIR / "emg_reference_manifest.json") as f:
        manifest = json.load(f)
    with open(TUBE_DIR / "emg_observation_mapping.json") as f:
        mapping = json.load(f)
    tube = np.load(TUBE_DIR / "emg_reference_tube.npz", allow_pickle=True)

    ref_names = None
    per_arm_curves = {}
    grid = np.linspace(0.0, 1.0, GRID_N)
    for arm in ARMS:
        roll = np.load(HERE / f"{arm.lower()}_320m_activation_rollout.npz", allow_pickle=True)
        names = [str(n) for n in roll["actuator_names"]]
        if ref_names is None:
            ref_names = names
            P, channel_names = build_emg_observation_projection(mapping, ref_names)
            if list(channel_names) != list(manifest["channel_names"]):
                raise ValueError("channel order mismatch")
            P = P.astype(np.float64)
            ridge = float(manifest["synergy_binding"]["projection_ridge"])
            Q = synergy_projection_matrix(
                np.asarray(tube["synergy_basis"], dtype=np.float64), ridge=ridge
            )
        elif names != ref_names:
            raise ValueError(f"{arm} actuator order mismatch")
        traj_indices = roll["trajectory_indices"].tolist()
        traj_lens = roll["traj_lengths"]
        curves = {}
        for k, ti in enumerate(traj_indices):
            act = np.asarray(roll[f"act_traj{ti}"], dtype=np.float64)
            progress = np.arange(act.shape[0]) / max(int(traj_lens[k]) - 1, 1)
            c = np.maximum((act @ P.T) @ Q.T, 0.0)
            curves[int(ti)] = resample_progress(progress, c, grid)
        per_arm_curves[arm] = curves

    ref_syn = np.asarray(tube["synergy_mean"][0], dtype=np.float64)  # (20,3)
    bin_centers = (np.arange(20) + 0.5) / 20.0
    ref_g = np.stack([np.interp(grid, bin_centers, ref_syn[:, j]) for j in range(3)], axis=1)

    # ---- 相似度矩阵 ----
    result = {"grid_n": GRID_N, "common_trajectories": list(COMMON_TRAJS),
              "synergies": SYNERGY_LABELS, "arms": {}}
    for arm in ARMS:
        per_syn = []
        for j in range(3):
            cos_list, pear_list, inten_list = [], [], []
            for ti in COMMON_TRAJS:
                c = per_arm_curves[arm][ti][:, j]
                r = ref_g[:, j]
                cos_list.append(cosine_masked(c, r))
                pear_list.append(pearson_masked(c, r))
                m = ~(np.isnan(c) | np.isnan(r))
                inten_list.append(float(np.trapezoid(c[m], grid[m]) / np.trapezoid(r[m], grid[m])))
            per_syn.append({
                "cosine_mean": float(np.mean(cos_list)),
                "cosine_std": float(np.std(cos_list)),
                "cosine_per_traj": {str(t): v for t, v in zip(COMMON_TRAJS, cos_list)},
                "pearson_mean": float(np.mean(pear_list)),
                "pearson_std": float(np.std(pear_list)),
                "intensity_ratio_mean": float(np.mean(inten_list)),
                "intensity_ratio_std": float(np.std(inten_list)),
            })
        result["arms"][arm] = {"per_synergy": per_syn}

    # ---- T3 相对 T2/T4 改善 ----
    comparisons = {}
    for other in ("T2", "T4"):
        rows = []
        for j in range(3):
            s3 = result["arms"]["T3"]["per_synergy"][j]["cosine_mean"]
            sx = result["arms"][other]["per_synergy"][j]["cosine_mean"]
            dpp = s3 - sx
            rel = (s3 - sx) / abs(sx) if abs(sx) > 1e-3 else None
            err_red = (s3 - sx) / (1.0 - sx) if (1.0 - sx) > 1e-3 else None
            rows.append({"synergy": SYNERGY_LABELS[j], "sim_t3": s3, "sim_other": sx,
                         "delta_pp": dpp, "relative_improvement": rel, "error_reduction": err_red})
        comparisons[f"T3_vs_{other}"] = rows
    result["comparisons"] = comparisons

    with open(HERE / "synergy_similarity_matrix.json", "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)

    # ---- 图 ----
    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=(1, 1.6), wspace=0.25)
    axm = fig.add_subplot(gs[0])
    sim_mat = np.array([[result["arms"][a]["per_synergy"][j]["cosine_mean"] for j in range(3)]
                        for a in ARMS])
    im = axm.imshow(sim_mat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    axm.set_xticks(range(3), SYNERGY_SHORT, fontsize=10)
    axm.set_yticks(range(3), ARMS, fontsize=10)
    for i in range(3):
        for j in range(3):
            axm.text(j, i, f"{sim_mat[i, j]:.3f}", ha="center", va="center", fontsize=11,
                     fontweight="bold")
    axm.set_title("相似度矩阵（cosine，1=与实测完全一致）", fontsize=11)
    fig.colorbar(im, ax=axm, fraction=0.04, pad=0.03)

    axb = fig.add_subplot(gs[1])
    x = np.arange(3)
    width = 0.26
    for i, arm in enumerate(ARMS):
        vals = [result["arms"][arm]["per_synergy"][j]["cosine_mean"] for j in range(3)]
        stds = [result["arms"][arm]["per_synergy"][j]["cosine_std"] for j in range(3)]
        axb.bar(x + (i - 1) * width, vals, width, yerr=stds, capsize=3,
                color=ARM_COLORS[arm], label=f"{arm}（{ARM_DESC[arm]}）", alpha=0.9)
        for xx, v in zip(x + (i - 1) * width, vals):
            axb.text(xx, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axb.axhline(1.0, color="gray", ls="--", lw=0.8)
    axb.set_xticks(x, SYNERGY_LABELS, fontsize=9)
    axb.set_ylim(0, 1.12)
    axb.set_ylabel("与实测的余弦相似度")
    axb.legend(fontsize=8, loc="upper right")
    axb.grid(axis="y", alpha=0.3)
    axb.set_title("各协同的学成-实测相似度（均值±std，4 条共同轨迹）", fontsize=11)
    fig.suptitle("T2/T3/T4 @ 320M：三条发力链的学成-实测相似度", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synergy6_similarity_matrix.png", dpi=150)
    plt.close(fig)

    # ---- markdown ----
    lines = ["# 三实验臂 × 三发力链：学成-实测相似度矩阵（@320M）", "",
             "相似度 = 策略协同系数曲线与实测 tube 参考曲线的余弦相似度（逐轨迹计算后取均值；",
             "三个臂均使用共同的 4 条近全程轨迹 0/3/4/5，进度覆盖 0–0.994）。", "",
             "## 相似度矩阵（cosine，均值±std）", "",
             "| 臂 | S1 蹬地转体链 | S2 支撑稳定链 | S3 上肢引拍链 |",
             "|---|---|---|---|"]
    for arm in ARMS:
        cells = [f"{result['arms'][arm]['per_synergy'][j]['cosine_mean']:.3f}"
                 f"±{result['arms'][arm]['per_synergy'][j]['cosine_std']:.3f}" for j in range(3)]
        lines.append(f"| {arm}（{ARM_DESC[arm]}） | " + " | ".join(cells) + " |")
    lines += ["", "辅助口径：Pearson 相关（中心化形状）与强度比（曲线积分 策略/实测）见 "
              "`synergy_similarity_matrix.json`。", "",
              "## T3 相对另外两个实验臂的改善", "",
              "| 协同 | 对比 | 相似度 T3 vs 对方 | 绝对差 | 相对提升 | 误差缩减（1-cos 口径） |",
              "|---|---|---|---|---|---|"]
    for other in ("T2", "T4"):
        for row in comparisons[f"T3_vs_{other}"]:
            rel = "n/a" if row["relative_improvement"] is None else f"{row['relative_improvement'] * 100:+.1f}%"
            err = "n/a" if row["error_reduction"] is None else f"{row['error_reduction'] * 100:+.1f}%"
            lines.append(
                f"| {row['synergy']} | T3 vs {other} | {row['sim_t3']:.3f} vs {row['sim_other']:.3f} "
                f"| {row['delta_pp'] * 100:+.1f}pp | {rel} | {err} |")
    lines += ["", "![matrix](figures/synergy6_similarity_matrix.png)", "",
              "口径备注：相对提升 = (simT3−simX)/|simX|；误差缩减 = (simT3−simX)/(1−simX)，"
              "即“距离完全一致还剩多少差距被消掉”的比例；对方相似度≈0 时相对提升无意义（标 n/a）。"]
    with open(HERE / "synergy_similarity_matrix.md", "w") as f:
        f.write("\n".join(lines))

    print(json.dumps({a: [round(result["arms"][a]["per_synergy"][j]["cosine_mean"], 4) for j in range(3)]
                      for a in ARMS}, indent=1))
    for other in ("T2", "T4"):
        print(f"T3 vs {other}:")
        for row in comparisons[f"T3_vs_{other}"]:
            rel = "n/a" if row["relative_improvement"] is None else f"{row['relative_improvement'] * 100:+.1f}%"
            err = "n/a" if row["error_reduction"] is None else f"{row['error_reduction'] * 100:+.1f}%"
            print(f"  {row['synergy']}: {row['sim_t3']:.3f} vs {row['sim_other']:.3f} "
                  f"Δ={row['delta_pp'] * 100:+.1f}pp rel={rel} err_red={err}")
    print("wrote synergy_similarity_matrix.{json,md} and figures/synergy6_similarity_matrix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
