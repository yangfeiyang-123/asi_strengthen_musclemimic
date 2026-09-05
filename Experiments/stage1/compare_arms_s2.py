#!/usr/bin/env python3
"""跨 arm 协同曲线对比：T2/T3/T4 @ 320M 的 3 条协同系数 vs 实测 tube 参考。

回答“S2 缺失是 T3 特有还是三个 arm 共有”：若三个 arm 阶段一都缺早期 S2，
则原因是任务/奖励/可观测性层面的，而非某个 treatment 的失败。

输入：Experiments/stage1/{t2,t3,t4}_320m_activation_rollout.npz
输出：figures/synergy5_arms_comparison.png、arms_synergy_stage_means.json
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
OUT_JSON = HERE / "arms_synergy_stage_means.json"

ARMS = ["T2", "T3", "T4"]
ARM_COLORS = {"T2": "#9467bd", "T3": "#2ca02c", "T4": "#d62728"}
ARM_DESC = {
    "T2": "synergy-only",
    "T3": "real anchor + real synergy",
    "T4": "anchor + 相移 synergy（负对照）",
}
SYNERGY_LABELS = ["S1 蹬地转体链", "S2 支撑稳定链", "S3 上肢引拍链"]
STAGE_BOUNDS = (0.0, 0.45, 0.65, 1.0)
GRID_N = 200

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


def arm_synergy_curves(arm, P, Q, grid):
    roll = np.load(HERE / f"{arm.lower()}_320m_activation_rollout.npz", allow_pickle=True)
    actuator_names = [str(n) for n in roll["actuator_names"]]
    if actuator_names != EXPECTED_NAMES:
        raise ValueError(f"{arm} actuator order mismatch")
    traj_indices = roll["trajectory_indices"].tolist()
    traj_lens = roll["traj_lengths"]
    ep_lens = roll["episode_lengths"]
    curves = []
    for k, ti in enumerate(traj_indices):
        act = np.asarray(roll[f"act_traj{ti}"], dtype=np.float64)
        progress = np.arange(act.shape[0]) / max(int(traj_lens[k]) - 1, 1)
        c = np.maximum((act @ P.T) @ Q.T, 0.0)
        curves.append(resample_progress(progress, c, grid))
    curves = np.stack(curves)  # (10, G, 3)
    covered = np.asarray(
        [int(ep_lens[k]) / int(traj_lens[k]) >= 0.99 for k in range(len(traj_indices))], dtype=bool
    )
    return curves, covered


def main() -> int:
    from musclemimic.physiology.emg_anchor import build_emg_observation_projection
    from musclemimic.physiology.emg_reference import synergy_projection_matrix

    with open(TUBE_DIR / "emg_reference_manifest.json") as f:
        manifest = json.load(f)
    with open(TUBE_DIR / "emg_observation_mapping.json") as f:
        mapping = json.load(f)
    tube = np.load(TUBE_DIR / "emg_reference_tube.npz", allow_pickle=True)

    roll3 = np.load(HERE / "t3_320m_activation_rollout.npz", allow_pickle=True)
    global EXPECTED_NAMES
    EXPECTED_NAMES = [str(n) for n in roll3["actuator_names"]]
    P, channel_names = build_emg_observation_projection(mapping, EXPECTED_NAMES)
    if list(channel_names) != list(manifest["channel_names"]):
        raise ValueError("channel order mismatch")
    P = P.astype(np.float64)
    ridge = float(manifest["synergy_binding"]["projection_ridge"])
    Q = synergy_projection_matrix(np.asarray(tube["synergy_basis"], dtype=np.float64), ridge=ridge)

    grid = np.linspace(0.0, 1.0, GRID_N)
    bin_centers = (np.arange(20) + 0.5) / 20.0
    ref_syn = np.asarray(tube["synergy_mean"][0], dtype=np.float64)
    ref_g = np.stack([np.interp(grid, bin_centers, ref_syn[:, j]) for j in range(3)], axis=1)

    per_arm = {}
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True)
    for arm in ARMS:
        curves, covered = arm_synergy_curves(arm, P, Q, grid)
        mean_curve = np.nanmean(curves[covered], axis=0)  # (G,3)
        stage_means = np.stack(
            [np.nanmean(mean_curve[(grid >= a) & (grid < b)], axis=0)
             for a, b in zip(STAGE_BOUNDS[:-1], STAGE_BOUNDS[1:])]
        )  # (3 stages, 3 synergies)
        per_arm[arm] = {
            "n_near_complete": int(covered.sum()),
            "stage_means": stage_means.tolist(),
        }
        for j, ax in enumerate(axes):
            ax.plot(grid, mean_curve[:, j], color=ARM_COLORS[arm], lw=1.8,
                    label=f"{arm}（{ARM_DESC[arm]}）")
    for j, ax in enumerate(axes):
        ax.plot(grid, ref_g[:, j], color="k", lw=2.2, ls="--", label="实测 tube 参考")
        for b in STAGE_BOUNDS[1:-1]:
            ax.axvline(b, color="gray", ls=":", lw=1)
        ax.axvspan(STAGE_BOUNDS[0], STAGE_BOUNDS[1], color="#f5f5f5", zorder=0)
        ax.set_ylabel("协同系数")
        ax.set_title(SYNERGY_LABELS[j], fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("归一化运动进度")
    fig.suptitle("三个实验臂的协同系数全程对比 @320M（各臂近全程轨迹均值；黑虚线=实测参考）", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synergy5_arms_comparison.png", dpi=150)
    plt.close(fig)

    ref_stage = np.stack(
        [np.nanmean(ref_g[(grid >= a) & (grid < b)], axis=0)
         for a, b in zip(STAGE_BOUNDS[:-1], STAGE_BOUNDS[1:])]
    )
    out = {
        "stage_bounds": STAGE_BOUNDS,
        "synergies": SYNERGY_LABELS,
        "measured_stage_means": ref_stage.tolist(),
        "arms": per_arm,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote figures/synergy5_arms_comparison.png and {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
