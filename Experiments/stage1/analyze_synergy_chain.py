#!/usr/bin/env python3
"""T3@320M 策略的肌肉协同链条分析（基于用户已有 k=3 协同提取）。

数据链路：
  rollout npz（354 维激活，capture_t3_320m_activation_rollout.py 产出）
    -> y = P @ a          （15 个实测可比通道，mapping 与训练 reward 相同）
    -> c = relu(Q @ y)    （3 条协同系数，Q = (WᵀW + λI)⁻¹Wᵀ，λ=0.001，与训练 reward 相同）
  参考侧：verified tube 的 synergy_mean / anchor_mean（20 个归一化相位 bin，来自用户实测 EMG）

三阶段划分（由实测协同结构驱动，见报告）：阶段一 0.00–0.45、阶段二 0.45–0.65、阶段三 0.65–1.00。

输出（均在 Experiments/stage1/）：
  t3_320m_synergy_chain_analysis.npz
  figures/synergy1_basis.png synergy2_coefficients.png synergy3_channel_heatmap.png synergy4_chain_order.png
  t3_320m_synergy_chain_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # Experiments/stage1 -> 仓库根
TUBE_DIR = REPO / "artifacts/emg_human_review_v2/verified_tubes/forehand_high_clear"
K3_DIR = REPO / "jidian_measurement/data/analysis/P002_S20260721_A_study/fits/forehand_high_clear_k3"
ROLLOUT_NPZ = HERE / "t3_320m_activation_rollout.npz"
OUT_NPZ = HERE / "t3_320m_synergy_chain_analysis.npz"
OUT_JSON = HERE / "t3_320m_synergy_chain_summary.json"
FIG_DIR = HERE / "figures"

STAGE_BOUNDS = (0.0, 0.45, 0.65, 1.0)
STAGE_NAMES = ["阶段一·准备引拍", "阶段二·挥拍击球", "阶段三·随挥恢复"]
GRID_N = 200
ONSET_FRAC = 0.30  # 通道 onset = 曲线首次达到自身峰值 30% 的归一化进度
TRANSIENT_PROGRESS = 0.05  # 肌肉激活动力学 spin-up 区（act 从 0 爬升），链条排序用 post-transient 值

SYNERGY_LABELS = ["S1 蹬地转体链", "S2 支撑稳定链", "S3 上肢引拍链"]
SYNERGY_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_inputs():
    roll = np.load(ROLLOUT_NPZ, allow_pickle=True)
    tube = np.load(TUBE_DIR / "emg_reference_tube.npz", allow_pickle=True)
    k3 = np.load(K3_DIR / "synergy_basis.npz", allow_pickle=True)
    with open(TUBE_DIR / "emg_reference_manifest.json") as f:
        manifest = json.load(f)
    with open(TUBE_DIR / "emg_observation_mapping.json") as f:
        mapping = json.load(f)
    with open(K3_DIR / "synergy_metadata.json") as f:
        k3_meta = json.load(f)
    return roll, tube, k3, manifest, mapping, k3_meta


def build_projection(mapping, actuator_names, manifest):
    from musclemimic.physiology.emg_anchor import build_emg_observation_projection

    P, channel_names = build_emg_observation_projection(mapping, actuator_names)
    expected = manifest["channel_names"]
    if list(channel_names) != list(expected):
        raise ValueError(f"projection channel order diverges from tube: {channel_names} vs {expected}")
    return P.astype(np.float64), list(channel_names)


def zh_name_maps(k3_meta):
    out = {}
    for c in k3_meta["channel_profile_snapshot"]["channels"]:
        out[(c["side"], c["muscle_slug"])] = (c["name_zh"], c["abbreviation"])
    return out


def resample_progress(progress, values, grid):
    """values: (T, D) -> (G, D) linear interp on progress in [0,1]; NaN outside coverage."""
    out = np.full((grid.size, values.shape[1]), np.nan)
    if progress.size < 2:
        return out
    lo, hi = float(progress[0]), float(progress[-1])
    inside = (grid >= lo) & (grid <= hi)
    for d in range(values.shape[1]):
        out[inside, d] = np.interp(grid[inside], progress, values[:, d])
    return out


def smooth(x, w=5):
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")[: x.shape[0]]


def onset_peak(curve, grid, frac=ONSET_FRAC, min_progress=None):
    """curve: (G,) with possible NaN tail. Return (onset_progress, peak_progress) or (nan, nan).

    min_progress: 若给定，只在 progress >= min_progress 的区段上计算（用于跳过
    肌肉激活从 0 爬升的初始化瞬态）。
    """
    valid = ~np.isnan(curve)
    if min_progress is not None:
        valid &= grid >= min_progress
    if valid.sum() < 5:
        return np.nan, np.nan
    c = np.where(valid, curve, 0.0)
    peak = float(np.max(c))
    if peak <= 1e-8:
        return np.nan, np.nan
    peak_idx = int(np.argmax(c))
    thr = frac * peak
    above = np.flatnonzero(c[: peak_idx + 1] >= thr)
    if above.size == 0:
        return np.nan, float(grid[peak_idx])
    return float(grid[int(above[0])]), float(grid[peak_idx])


def main() -> int:
    roll, tube, k3, manifest, mapping, k3_meta = load_inputs()
    actuator_names = [str(n) for n in roll["actuator_names"]]
    P, channel_names = build_projection(mapping, actuator_names, manifest)
    ridge = float(manifest["synergy_binding"]["projection_ridge"])
    W_tube = np.asarray(tube["synergy_basis"], dtype=np.float64)  # (15,3)
    from musclemimic.physiology.emg_reference import synergy_projection_matrix

    Q = synergy_projection_matrix(W_tube, ridge=ridge)  # (3,15)

    names_map = zh_name_maps(k3_meta)
    ch_labels = []
    for ch in channel_names:  # 'right:anterior_deltoid'
        side, slug = ch.split(":", 1)
        zh, ab = names_map.get((side, slug), (ch, ""))
        ch_labels.append(f"{zh} {ab}")

    traj_indices = roll["trajectory_indices"].tolist()
    traj_lens = roll["traj_lengths"]
    ep_lens = roll["episode_lengths"]
    early = roll["early_terminated"]
    grid = np.linspace(0.0, 1.0, GRID_N)

    y_all, c_all, covered = [], [], []
    for k, ti in enumerate(traj_indices):
        act = np.asarray(roll[f"act_traj{ti}"], dtype=np.float64)  # (T,354)
        T = act.shape[0]
        progress = np.arange(T) / max(int(traj_lens[k]) - 1, 1)
        y = act @ P.T  # (T,15)
        c = np.maximum(y @ Q.T, 0.0)  # relu(Q y), 与 reward 相同
        y_all.append(resample_progress(progress, y, grid))
        c_all.append(resample_progress(progress, c, grid))
        # “完整覆盖动作全程”的定义：覆盖 ≥99% 归一化进度（对 run 间边界帧抖动稳健，
        # 不依赖 early-termination 的阈值语义）。
        covered.append(T / int(traj_lens[k]) >= 0.99)
    y_all = np.stack(y_all)  # (10, G, 15)
    c_all = np.stack(c_all)  # (10, G, 3)
    covered = np.asarray(covered, dtype=bool)

    # 参考（实测）曲线：tube 20 bins -> grid
    bin_centers = (np.arange(20) + 0.5) / 20.0
    ref_syn = np.asarray(tube["synergy_mean"][0], dtype=np.float64)  # (20,3)
    ref_syn_valid = np.asarray(tube["synergy_valid"][0], dtype=bool)
    ref_anchor = np.asarray(tube["anchor_mean"][0], dtype=np.float64)  # (20,15)
    ref_anchor_valid = np.asarray(tube["anchor_valid"][0], dtype=bool)
    ref_syn_g = np.stack([np.interp(grid, bin_centers, ref_syn[:, j]) for j in range(3)], axis=1)
    ref_anchor_g = np.stack(
        [np.interp(grid, bin_centers, ref_anchor[:, m]) for m in range(15)], axis=1
    )

    pol_c = np.nanmean(c_all[covered], axis=0)  # (G,3) 完成轨迹均值
    pol_c_std = np.nanstd(c_all[covered], axis=0)
    pol_y = np.nanmean(y_all[covered], axis=0)  # (G,15)

    # ---------- 链条顺序指标 ----------
    summary = {"stage_bounds": STAGE_BOUNDS, "stage_names": STAGE_NAMES,
               "n_trajectories": len(traj_indices), "n_completed": int(covered.sum()),
               "completed_traj_indices": [int(t) for t, ok in zip(traj_indices, covered) if ok]}

    # 协同级：每条协同的 onset / peak / 各阶段均值（策略 vs 实测）
    syn_info = []
    for j in range(3):
        p_curve = smooth(pol_c[:, j])
        r_curve = smooth(ref_syn_g[:, j])
        p_on, p_pk = onset_peak(p_curve, grid)
        p_on_post, p_pk_post = onset_peak(p_curve, grid, min_progress=TRANSIENT_PROGRESS)
        r_on, r_pk = onset_peak(r_curve, grid)
        stage_means_p = [float(np.nanmean(pol_c[(grid >= a) & (grid < b), j]))
                         for a, b in zip(STAGE_BOUNDS[:-1], STAGE_BOUNDS[1:])]
        stage_means_r = [float(np.nanmean(ref_syn_g[(grid >= a) & (grid < b), j]))
                         for a, b in zip(STAGE_BOUNDS[:-1], STAGE_BOUNDS[1:])]
        syn_info.append({
            "synergy": SYNERGY_LABELS[j],
            "policy_onset": p_on, "policy_peak": p_pk,
            "policy_onset_post_transient": p_on_post, "policy_peak_post_transient": p_pk_post,
            "measured_onset": r_on, "measured_peak": r_pk,
            "policy_stage_mean": stage_means_p, "measured_stage_mean": stage_means_r,
        })
    summary["synergy_timing"] = syn_info

    # 阶段主导协同（按阶段均值）
    for side, key in (("policy", "policy_stage_mean"), ("measured", "measured_stage_mean")):
        dom = []
        for si in range(3):
            vals = [syn_info[j][key][si] for j in range(3)]
            dom.append(int(np.argmax(vals)))
        summary[f"{side}_stage_dominant_synergy"] = dom

    # 分阶段的协同形状/强度一致性（策略 vs 实测，与 reward 的 shape cosine 同定义）
    stage_cos, stage_intensity_ratio = [], []
    for a, b in zip(STAGE_BOUNDS[:-1], STAGE_BOUNDS[1:]):
        m = (grid >= a) & (grid < b)
        p = np.nanmean(pol_c[m], axis=0)
        r = np.nanmean(ref_syn_g[m], axis=0)
        ps, rs = p.sum(), r.sum()
        p_sh = p / (ps + 1e-8)
        r_sh = r / (rs + 1e-8)
        denom = np.linalg.norm(p_sh) * np.linalg.norm(r_sh)
        stage_cos.append(float(p_sh @ r_sh / (denom + 1e-8)))
        stage_intensity_ratio.append(float(ps / (rs + 1e-8)))
    summary["stage_shape_cosine_policy_vs_measured"] = stage_cos
    summary["stage_intensity_ratio_policy_over_measured"] = stage_intensity_ratio

    # 通道级：onset / peak（策略 vs 实测 anchor_mean）；策略侧另计 post-transient 版本
    ch_info = []
    for m in range(15):
        p_curve = smooth(pol_y[:, m])
        r_curve = smooth(ref_anchor_g[:, m])
        p_on, p_pk = onset_peak(p_curve, grid)
        p_on_post, p_pk_post = onset_peak(p_curve, grid, min_progress=TRANSIENT_PROGRESS)
        r_on, r_pk = onset_peak(r_curve, grid)
        # 主导协同归属 = W_tube 该行最大列
        dom_syn = int(np.argmax(W_tube[m]))
        ch_info.append({
            "channel": channel_names[m], "label": ch_labels[m],
            "dominant_synergy": dom_syn,
            "policy_onset": p_on, "policy_peak": p_pk,
            "policy_onset_post_transient": p_on_post, "policy_peak_post_transient": p_pk_post,
            "measured_onset": r_on, "measured_peak": r_pk,
            "policy_stage_mean": [float(np.nanmean(pol_y[(grid >= a) & (grid < b), m]))
                                  for a, b in zip(STAGE_BOUNDS[:-1], STAGE_BOUNDS[1:])],
            "measured_stage_mean": [float(np.nanmean(ref_anchor_g[(grid >= a) & (grid < b), m]))
                                    for a, b in zip(STAGE_BOUNDS[:-1], STAGE_BOUNDS[1:])],
        })
    summary["channels"] = ch_info
    summary["policy_onset_order"] = sorted(
        range(15), key=lambda m: (np.nan_to_num(ch_info[m]["policy_onset_post_transient"], nan=9.9), m))
    summary["policy_peak_order"] = sorted(
        range(15), key=lambda m: (np.nan_to_num(ch_info[m]["policy_peak_post_transient"], nan=9.9), m))
    summary["measured_onset_order"] = sorted(
        range(15), key=lambda m: (np.nan_to_num(ch_info[m]["measured_onset"], nan=9.9), m))
    summary["measured_peak_order"] = sorted(
        range(15), key=lambda m: (np.nan_to_num(ch_info[m]["measured_peak"], nan=9.9), m))

    # ---------- 图 1：3 条协同的肌肉空间模式（用户 k3，16 通道） ----------
    W16 = np.asarray(k3["W"], dtype=np.float64)  # (16,3)
    k3_slugs = [str(s) for s in k3["muscle_slugs"]]
    k3_sides = [str(s) for s in k3["sides"]]
    k3_labels = []
    for side, slug in zip(k3_sides, k3_slugs):
        zh, ab = names_map.get((side, slug), (slug, ""))
        k3_labels.append(f"{zh} {ab}")
    excluded_slug = "upper_trapezius"  # S1 通道无模型同系肌肉，不进映射

    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    for j, ax in enumerate(axes):
        order = np.argsort(-W16[:, j])
        vals = W16[:, j][order]
        labels = [k3_labels[i] for i in order]
        colors = ["#999999" if k3_slugs[i] == excluded_slug else SYNERGY_COLORS[j] for i in order]
        ax.barh(range(16), vals, color=colors)
        ax.set_yticks(range(16), labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_title(f"{SYNERGY_LABELS[j]}\n(k3 协同基 W 列 {j + 1})", fontsize=11)
        ax.set_xlabel("synergy weight")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("用户实测 EMG 提取的 3 条肌肉协同（P002/S20260721_A，10 trials，global VAF=0.75；灰色=未进模型映射的 S1 斜方肌上束）", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synergy1_basis.png", dpi=150)
    plt.close(fig)

    # ---------- 图 2：协同系数全程曲线（策略 vs 实测） ----------
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for j, ax in enumerate(axes):
        ax.axvspan(STAGE_BOUNDS[0], STAGE_BOUNDS[1], color="#f0f0f0", zorder=0)
        ax.axvspan(STAGE_BOUNDS[2], STAGE_BOUNDS[3], color="#f0f0f0", zorder=0)
        for k in range(len(traj_indices)):
            if covered[k]:
                ax.plot(grid, c_all[k][:, j], color=SYNERGY_COLORS[j], alpha=0.15, lw=0.8)
        mean = pol_c[:, j]
        std = pol_c_std[:, j]
        ax.plot(grid, mean, color=SYNERGY_COLORS[j], lw=2.2, label="策略（T3@320M，近全程轨迹均值）")
        ax.fill_between(grid, mean - std, mean + std, color=SYNERGY_COLORS[j], alpha=0.25)
        ax.plot(grid, ref_syn_g[:, j], color="k", lw=1.8, ls="--", label="实测 tube 参考（synergy_mean）")
        for b in STAGE_BOUNDS[1:-1]:
            ax.axvline(b, color="gray", ls=":", lw=1)
        ax.set_ylabel("协同系数")
        ax.set_title(f"{SYNERGY_LABELS[j]}：策略 vs 实测", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
    axes[-1].set_xlabel("归一化运动进度 (0=起始帧, 1=结束帧)")
    fig.suptitle("T3@320M 策略的 3 条肌肉协同系数全程曲线（浅线=各完成轨迹）", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synergy2_coefficients.png", dpi=150)
    plt.close(fig)

    # ---------- 图 3：15 通道激活热图（策略 vs 实测） ----------
    chan_order = sorted(range(15), key=lambda m: (W_tube[m].argmax(), -W_tube[m].max()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    for ax, mat, title in (
        (axes[0], pol_y, "策略 T3@320M（近全程轨迹均值）"),
        (axes[1], ref_anchor_g, "实测 tube（anchor_mean）"),
    ):
        sub = mat[:, chan_order].T
        sub = sub / np.maximum(np.nanmax(sub, axis=1, keepdims=True), 1e-8)  # 每通道归一（忽略尾部 NaN）
        im = ax.imshow(sub, aspect="auto", origin="lower", extent=[0, 1, -0.5, 14.5],
                       cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(15), [ch_labels[m] for m in chan_order], fontsize=8)
        for b in STAGE_BOUNDS[1:-1]:
            ax.axvline(b, color="w", ls="--", lw=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("归一化运动进度")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="激活（行内归一）")
    fig.suptitle("15 个实测可比通道的激活热图（通道按主导协同分组排序；白虚线=阶段边界 0.45 / 0.65）", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synergy3_channel_heatmap.png", dpi=150)
    plt.close(fig)

    # ---------- 图 4：发力链条顺序（onset○→peak● 时间线，按峰值时刻排序） ----------
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5), sharex=True)
    panels = (
        (axes[0], "policy", "策略 T3@320M 的通道募集链条（onset/peak 均跳过 0–5% 的肌肉激活 spin-up 瞬态）"),
        (axes[1], "measured", "实测 EMG 的通道募集链条（anchor_mean）"),
    )
    for ax, side, title in panels:
        if side == "policy":
            order = summary["policy_peak_order"]
            on_key, pk_key = "policy_onset_post_transient", "policy_peak_post_transient"
        else:
            order = summary["measured_peak_order"]
            on_key, pk_key = "measured_onset", "measured_peak"
        rank = 0
        for m in order:
            info = ch_info[m]
            on, pk = info[on_key], info[pk_key]
            if np.isnan(on) or np.isnan(pk):
                ax.text(0.005, rank, f"{info['label']}（全程近零）", fontsize=8, va="center", color="#888888")
                rank += 1
                continue
            col = SYNERGY_COLORS[info["dominant_synergy"]]
            ax.plot([on, pk], [rank, rank], color=col, lw=2, alpha=0.55, zorder=2)
            ax.scatter(on, rank, s=70, facecolor="white", edgecolor=col, lw=1.8, zorder=3)
            ax.scatter(pk, rank, s=110, color=col, zorder=4)
            if pk > 0.62:  # 晚期通道：标签放线段左侧，防止越界
                ax.text(min(on, pk) - 0.008, rank, info["label"], fontsize=8, va="center", ha="right")
            else:
                ax.text(max(pk, on) + 0.008, rank, info["label"], fontsize=8, va="center")
            rank += 1
        for b in STAGE_BOUNDS[1:-1]:
            ax.axvline(b, color="gray", ls=":", lw=1)
        for si, (a, b) in enumerate(zip(STAGE_BOUNDS[:-1], STAGE_BOUNDS[1:])):
            ax.text((a + b) / 2, -1.8, STAGE_NAMES[si], ha="center", fontsize=9, color="gray")
        ax.set_ylim(-2.4, 15.4)
        ax.set_yticks([])
        ax.set_xlim(-0.28, 1.1)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.3)
    axes[-1].set_xlabel("归一化运动进度")
    handles = [plt.Line2D([], [], marker="o", ls="", color=SYNERGY_COLORS[j], label=SYNERGY_LABELS[j]) for j in range(3)]
    handles += [plt.Line2D([], [], marker="o", mfc="white", mec="k", ls="", label="onset（达峰值30%）"),
                plt.Line2D([], [], marker="o", ls="", color="k", label="peak（峰值时刻）")]
    axes[0].legend(handles=handles, fontsize=8, loc="lower right")
    fig.suptitle("肌肉发力链条顺序：○=onset ●=peak，点/线颜色=该通道的主导协同", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synergy4_chain_order.png", dpi=150)
    plt.close(fig)

    # ---------- 保存 ----------
    np.savez_compressed(
        OUT_NPZ,
        grid=grid,
        policy_channel_mean=pol_y,
        policy_synergy_mean=pol_c,
        policy_synergy_std=pol_c_std,
        policy_channel_all=y_all,
        policy_synergy_all=c_all,
        measured_synergy_curve=ref_syn_g,
        measured_anchor_curve=ref_anchor_g,
        synergy_basis_tube=W_tube,
        synergy_basis_k3_16ch=W16,
        covered=covered,
        channel_names=np.asarray(channel_names),
    )
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f"wrote {OUT_NPZ.name}, {OUT_JSON.name} and 4 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
