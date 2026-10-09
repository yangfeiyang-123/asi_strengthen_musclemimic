#!/usr/bin/env python3
"""Stage-1 320M 全指标仪表盘：每个指标 → 图 + 具体改善百分比。

数据源（均已生成，只读）：
  stage1_t2t3t4_320m_snapshot.json      五维验证指标（extract_stage1_320m_metrics.py）
  t3_320m_synergy_chain_summary.json    链条时序指标（analyze_synergy_chain.py）
  arms_synergy_stage_means.json         跨臂协同阶段均值（compare_arms_s2.py）

约定：
  * 所有百分比统一为"正 = 更优"（对 lower-better 指标已翻转符号）；
  * 全程变化 = 各自 arm 首次验证（9.99M，update 488）→ 320M（update 15625）；
  * 臂间对比 = 320M 处相对改善率（对 T4/T2 等基线臂）；
  * gate 距离用百分点或相对差表示。

输出：
  figures/dash_dim1..5_*.png, dash_chain.png
  stage1_320m_metrics_dashboard.json
  stage1_320m_metrics_dashboard.md
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ARMS = ["T2", "T3", "T4"]
ARM_COLORS = {"T2": "#9467bd", "T3": "#2ca02c", "T4": "#d62728"}

# (key, 中文名, 单位, 方向, gate(<=/>=,阈值) 或 None)
DIMENSIONS = {
    "dim1": {
        "title": "动作激活度（activation / action）",
        "curve_fig": "fig1_action_activation.png",
        "metrics": [
            ("val_activation_energy", "activation energy", "arb.", "lower", ("<=", 0.35)),
            ("val_activation_saturation_fraction", "activation saturation", "frac", "lower", None),
            ("val_activation_rate_mean_square", "activation rate", "ms", "lower", None),
            ("val_action_saturation_fraction", "action saturation", "frac", "lower", ("<=", 0.05)),
            ("val_action_rate_mean_square", "action rate", "ms", "lower", None),
        ],
    },
    "dim2": {
        "title": "动作追踪精度（tracking error）",
        "curve_fig": "fig2_tracking_accuracy.png",
        "metrics": [
            ("val_err_rpos", "RPos error", "m", "lower", ("<=", 0.09)),
            ("val_err_joint_pos", "joint pos error", "rad", "lower", None),
            ("val_err_joint_vel", "joint vel error", "rad/s", "lower", None),
            ("val_err_root_xyz", "root xyz error", "m", "lower", None),
            ("val_err_root_yaw", "root yaw error", "rad", "lower", None),
        ],
    },
    "dim3": {
        "title": "动作成功度（success）",
        "curve_fig": "fig3_motion_success.png",
        "metrics": [
            ("val_frame_coverage", "frame coverage", "frac", "higher", (">=", 0.95)),
            ("val_early_termination_rate", "early termination", "frac", "lower", ("<=", 0.05)),
        ],
    },
    "dim4": {
        "title": "肌肉协同误差（vs 实测 synergy 参考）",
        "curve_fig": "fig4_synergy_error.png",
        "metrics": [
            ("val_emg_synergy_real_reference_loss", "synergy loss", "loss", "lower", None),
            ("val_emg_synergy_real_reference_shape_loss", "shape loss", "1-cos", "lower", None),
            ("val_emg_synergy_real_reference_shape_cosine", "shape cosine", "cos", "higher", None),
            ("val_emg_synergy_real_reference_intensity_loss", "intensity loss", "loss", "lower", None),
        ],
    },
    "dim5": {
        "title": "肌肉误差（M-channel anchor vs 实测 sEMG）",
        "curve_fig": "fig5_muscle_anchor_error.png",
        "metrics": [
            ("val_emg_anchor_loss", "anchor loss", "loss", "lower", None),
            ("val_emg_anchor_correlation", "anchor correlation", "corr", "higher", None),
            ("val_emg_anchor_mean_abs_deviation", "mean |deviation|", "act", "lower", None),
            ("val_emg_anchor_max_abs_deviation", "max |deviation|", "act", "lower", None),
            ("val_emg_anchor_violation_fraction", "violation fraction", "frac", "lower", None),
        ],
    },
}


def rel_improve(a, b, direction):
    """a 相对 b 的改善率（正=a 更优）。b 为基线。"""
    if b == 0:
        return None
    if direction == "lower":
        return (b - a) / abs(b)
    return (a - b) / abs(b)


def fmt_course(row, arm):
    """全程变化的显示：基线过小时相对百分比失真，改显绝对变化。"""
    first = row["first_values"][arm]
    final = row["values"][arm]
    delta = final - first
    if row["direction"] == "lower":
        delta = -delta  # 正=改善
    tiny_frac = row["unit"] == "frac" and abs(first) < 0.10
    tiny_abs = row["unit"] != "frac" and abs(first) < 0.05
    if tiny_frac:
        return f"{delta * 100:+.1f}pp"
    if tiny_abs:
        return f"{delta:+.1e} {row['unit']}"
    return fmt_pct(row["course_change"][arm])


def fmt_pct(x, pp=False):
    if x is None:
        return "—"
    if pp:
        return f"{x * 100:+.1f}pp"
    return f"{x * 100:+.1f}%"


def main() -> int:
    snap = json.load(open(HERE / "stage1_t2t3t4_320m_snapshot.json"))
    chain = json.load(open(HERE / "t3_320m_synergy_chain_summary.json"))
    arms_syn = json.load(open(HERE / "arms_synergy_stage_means.json"))

    dash = {"dimensions": {}, "chain_metrics": {}, "notes": []}

    for dim_id, dim in DIMENSIONS.items():
        rows = []
        for key, zh, unit, direction, gate in dim["metrics"]:
            vals = {}
            firsts = {}
            for a in ARMS:
                for dname, keys in snap["dimensions"].items():
                    if key in keys:
                        vals[a] = snap["per_arm"][a]["endpoint_metrics"][dname][key]
                        break
                firsts[a] = snap["per_arm"][a]["improvement_first_to_320m"][key]["first_value"]
            row = {
                "key": key, "name": zh, "unit": unit, "direction": direction,
                "values": {a: vals[a] for a in ARMS},
                "first_values": {a: firsts[a] for a in ARMS},
                "course_change": {a: rel_improve(vals[a], firsts[a], direction) for a in ARMS},
                "t3_vs_t4": rel_improve(vals["T3"], vals["T4"], direction),
                "t3_vs_t2": rel_improve(vals["T3"], vals["T2"], direction),
                "t2_vs_t4": rel_improve(vals["T2"], vals["T4"], direction),
                "gate": None,
            }
            if gate is not None:
                op, thr = gate
                row["gate"] = {
                    "op": op, "threshold": thr,
                    "pass": {a: (vals[a] <= thr if op == "<=" else vals[a] >= thr) for a in ARMS},
                    "distance": {a: (vals[a] - thr) if op == "<=" else (thr - vals[a]) for a in ARMS},
                }
            rows.append(row)
        dash["dimensions"][dim_id] = {"title": dim["title"], "rows": rows}

        # ---- 该维度的 bar 图 ----
        n = len(rows)
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 1.4 + 0.62 * n), sharey=True)
        yticks = np.arange(n)
        labels = [r["name"] for r in rows]
        # 左：全程变化（每 arm 一条）
        width = 0.26
        for i, a in enumerate(ARMS):
            vals_a = [
                (r["course_change"][a]
                 if (abs(r["first_values"][a]) >= (0.10 if r["unit"] == "frac" else 0.05)) else None)
                for r in rows
            ]
            axes[0].barh(yticks + (i - 1) * width, [v if v is not None else 0 for v in vals_a],
                         height=width, color=ARM_COLORS[a], label=a)
        axes[0].axvline(0, color="k", lw=0.8)
        axes[0].set_title("训练全程变化（9.99M→320M，正=改善；基线≈0 的指标不画 bar、见表）", fontsize=10)
        axes[0].grid(axis="x", alpha=0.3)
        # 右：320M 臂间对比
        for i, (pair, key2, col) in enumerate((("T3 vs T4", "t3_vs_t4", "#1a9850"), ("T3 vs T2", "t3_vs_t2", "#4575b4"))):
            vals_p = [r[key2] for r in rows]
            axes[1].barh(yticks + (i - 0.5) * 0.34, [v if v is not None else 0 for v in vals_p],
                         height=0.34, color=col, label=pair)
        axes[1].axvline(0, color="k", lw=0.8)
        axes[1].axvline(0.05, color="gray", ls="--", lw=0.8)
        axes[1].set_title("320M 臂间对比（正=T3 更优；灰虚线=+5% 预注册阈值）", fontsize=10)
        axes[1].grid(axis="x", alpha=0.3)
        # 数值标注 + xlim 留白
        all_vals = [r["course_change"][a] for r in rows for a in ARMS
                    if abs(r["first_values"][a]) >= (0.10 if r["unit"] == "frac" else 0.05)]
        pair_vals = [r["t3_vs_t4"] for r in rows] + [r["t3_vs_t2"] for r in rows]
        all_vals = [v for v in all_vals if v is not None]
        pair_vals = [v for v in pair_vals if v is not None]
        vmax = max(abs(v) for v in all_vals) if all_vals else 1.0
        vmax_r = max(abs(v) for v in pair_vals) if pair_vals else 1.0
        pad = vmax * 1.28
        pad_r = max(vmax_r * 1.35, 0.08)
        for i, a in enumerate(ARMS):
            for y, r in zip(yticks + (i - 1) * width, rows):
                v = r["course_change"][a]
                if abs(r["first_values"][a]) < (0.10 if r["unit"] == "frac" else 0.05):
                    if i == 1:
                        axes[0].text(0, y - width, "基线≈0→见表", fontsize=6.5, color="gray",
                                     va="center", ha="left")
                    continue
                if v is not None:
                    axes[0].text(v + (0.015 * vmax if v >= 0 else -0.015 * vmax), y, fmt_pct(v),
                                 va="center", ha="left" if v >= 0 else "right", fontsize=7)
        for i, key2 in enumerate(("t3_vs_t4", "t3_vs_t2")):
            for y, r in zip(yticks + (i - 0.5) * 0.34, rows):
                v = r[key2]
                if v is not None:
                    axes[1].text(v + (0.015 * vmax_r if v >= 0 else -0.015 * vmax_r), y, fmt_pct(v),
                                 va="center", ha="left" if v >= 0 else "right", fontsize=7)
        axes[0].set_xlim(-pad, pad)
        axes[1].set_xlim(-pad_r * 0.5, pad_r)
        axes[0].set_yticks(yticks, labels, fontsize=9)
        axes[0].invert_yaxis()
        fig.legend(loc="upper center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(f"{dim['title']} — 改善幅度总览", fontsize=12, y=1.04)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = FIG / f"dash_{dim_id}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)

    # ---- 链条量化指标 ----
    syn = chain["synergy_timing"]
    chain_rows = []
    for j, s in enumerate(syn):
        chain_rows.append({
            "metric": f"{s['synergy']} 峰值时刻差（策略-实测）",
            "value": s["policy_peak_post_transient"] - s["measured_peak"],
            "fmt": "progress",
        })
    cos = chain["stage_shape_cosine_policy_vs_measured"]
    inten = chain["stage_intensity_ratio_policy_over_measured"]
    for si in range(3):
        chain_rows.append({"metric": f"阶段{'一二三'[si]} 协同形状余弦", "value": cos[si], "fmt": "cos"})
        chain_rows.append({"metric": f"阶段{'一二三'[si]} 强度比（策略/实测）", "value": inten[si], "fmt": "ratio"})
    s2_stage1_policy = arms_syn["arms"]["T3"]["stage_means"][0][1]
    s2_stage1_meas = arms_syn["measured_stage_means"][0][1]
    chain_rows.append({"metric": "阶段一 S2 支撑链缺口（策略 vs 实测）",
                       "value": (s2_stage1_policy - s2_stage1_meas) / s2_stage1_meas, "fmt": "pct_signed"})
    chain_rows.append({"metric": "S2 不可表达能量占比（静默+近零肌肉）", "value": 0.547, "fmt": "pct"})
    chain_rows.append({"metric": "全程静默映射通道占比（3/15）", "value": 0.20, "fmt": "pct"})
    dash["chain_metrics"] = chain_rows

    # 链条图：两个 panel —— 峰值时差 + 阶段余弦/强度
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8))
    names3 = ["S1 蹬地转体链", "S2 支撑稳定链", "S3 上肢引拍链"]
    peak_diffs = [syn[j]["policy_peak_post_transient"] - syn[j]["measured_peak"] for j in range(3)]
    axes[0].bar(names3, peak_diffs, color=["#d62728", "#2ca02c", "#1f77b4"])
    axes[0].axhline(0, color="k", lw=0.8)
    for i, v in enumerate(peak_diffs):
        axes[0].text(i, v + (0.008 if v >= 0 else -0.008), f"{v:+.3f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=9)
    axes[0].set_title("协同峰值时刻差（策略 - 实测，进度单位）", fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", labelsize=8)
    x = np.arange(3)
    axes[1].bar(x - 0.2, cos, width=0.4, color="#1a9850", label="形状余弦（越高越像）")
    axes[1].bar(x + 0.2, inten, width=0.4, color="#fdae61", label="强度比 策略/实测（1=等同人体）")
    for i, (c, r) in enumerate(zip(cos, inten)):
        axes[1].text(i - 0.2, c + 0.01, f"{c:.3f}", ha="center", fontsize=9)
        axes[1].text(i + 0.2, r + 0.01, f"{r:.3f}", ha="center", fontsize=9)
    axes[1].set_xticks(x, [f"阶段{s}" for s in "一二三"])
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(1.0, color="gray", ls="--", lw=0.8)
    axes[1].legend(fontsize=8, loc="lower left")
    axes[1].set_title("T3@320M 各阶段协同形状/强度一致性", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("肌肉发力链条的量化指标", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "dash_chain.png", dpi=150)
    plt.close(fig)

    with open(HERE / "stage1_320m_metrics_dashboard.json", "w") as f:
        json.dump(dash, f, indent=1, ensure_ascii=False)

    # ---- markdown 仪表盘 ----
    lines = ["# Stage-1 320M 全指标仪表盘", "",
             "口径：三条 seed-0 run（T2/T3/T4，同 snapshot `c1ccd93`、同 40/10 split）在合同终点 320M 的固定验证；",
             "10 条 held-out 轨迹、deterministic policy、eval seed 0。", "",
             "**百分比约定：正数一律表示“更优”**（lower-better 指标已翻转符号）。", "",
             "- 全程变化：各 arm 首次验证（9.99M）→ 320M 的自身改善幅度；",
             "- 臂间对比：320M 处 T3 相对 T4/T2 的改善率；",
             "- gate 距离：与数值 promotion 门槛的差距（pp=百分点）。", ""]
    for dim_id, dim in DIMENSIONS.items():
        d = dash["dimensions"][dim_id]
        lines += [f"## {dim['title']}", "",
                  f"训练曲线：![{dim_id}]({dim['curve_fig']})", "",
                  f"改善幅度：![dash_{dim_id}](dash_{dim_id}.png)", "",
                  "| 指标 | T2 | T3 | T4 | 全程变化 T2 | 全程变化 T3 | 全程变化 T4 | T3 vs T4 | T3 vs T2 | gate |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in d["rows"]:
            v = r["values"]
            gate_str = "—"
            if r["gate"] is not None:
                g = r["gate"]
                sym = "≤" if g["op"] == "<=" else "≥"
                dist = g["distance"]["T3"]
                if r["unit"] == "frac":
                    gate_str = f"门槛 {sym}{g['threshold']:.0%}；T3 差 {abs(dist):.1%}pp"
                else:
                    gate_str = f"门槛 {sym}{g['threshold']}；T3 差 {abs(dist):.3f} {r['unit']}"
                gate_str += ("（**达标**）" if g["pass"]["T3"] else "（未达标）")
            lines.append(
                f"| {r['name']}（{r['unit']}） | {v['T2']:.4f} | {v['T3']:.4f} | {v['T4']:.4f} "
                f"| {fmt_course(r, 'T2')} | {fmt_course(r, 'T3')} "
                f"| {fmt_course(r, 'T4')} | **{fmt_pct(r['t3_vs_t4'])}** "
                f"| {fmt_pct(r['t3_vs_t2'])} | {gate_str} |")
        lines.append("")

    lines += ["## 发力链条量化指标（T3@320M vs 实测 EMG）", "",
              "![dash_chain](dash_chain.png)", "",
              "| 指标 | 数值 |",
              "|---|---:|"]
    for r in chain_rows:
        if r["fmt"] == "progress":
            sval = f"{r['value']:+.3f} 进度"
        elif r["fmt"] == "cos":
            sval = f"{r['value']:.3f}"
        elif r["fmt"] == "ratio":
            sval = f"{r['value']:.3f}（欠激活 {(1 - r['value']) * 100:.1f}%）"
        else:
            sval = f"{r['value'] * 100:+.1f}%" if r["fmt"] == "pct_signed" else f"{r['value'] * 100:.1f}%"
        lines.append(f"| {r['metric']} | {sval} |")
    lines += ["", "详表与机制分析见 `stage1_t3_320m_synergy_chain_report.md`；五维明细见 "
              "`stage1_t2t3t4_320m_report.md`；**三臂 × 三协同的学成-实测相似度矩阵与 T3 相对 "
              "T2/T4 的改善百分比**见 `synergy_similarity_matrix.md`"
              "（`figures/synergy6_similarity_matrix.png`）。", ""]
    with open(HERE / "stage1_320m_metrics_dashboard.md", "w") as f:
        f.write("\n".join(lines))

    print("wrote dashboard md/json and dash_dim1..5 + dash_chain figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
