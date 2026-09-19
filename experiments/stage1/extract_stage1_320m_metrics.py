#!/usr/bin/env python3
"""Stage-1 T2/T3/T4 @ 320M 五维验证指标提取与对比分析。

数据源（只读）：
  datasets/forehandClear_standard/training_aug100_40train10val/checkpoints/
    260813T045112-pid1445039-50f5cf  -> T2 seed 0 (synergy-only)
    260813T045116-pid1445175-9e68dd  -> T3 seed 0 (real anchor + real synergy)
    260813T045113-pid1445040-6fff9b  -> T4 seed 0 (real anchor + phase-shifted synergy)

五个验证维度：
  1. 动作激活度   activation energy / activation & action saturation / rate
  2. 动作追踪精度 rpos / joint pos / joint vel / root xyz / root yaw error
  3. 动作成功度   frame coverage / early termination / episode return
  4. 肌肉协同误差 real-reference synergy loss / shape cosine / intensity loss
  5. 肌肉误差     M-channel anchor loss / correlation / abs deviation / violation

输出（全部写在本目录，不修改仓库其他任何文件）：
  stage1_t2t3t4_320m_snapshot.json   机器可读快照与对比
  figures/fig1..fig5.png             五个维度的全程训练曲线
"""

from __future__ import annotations

import json
from pathlib import Path

CKPT_BASE = Path(
    "/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/datasets/"
    "forehandClear_standard/training_aug100_40train10val/checkpoints"
)
RUN_DIRS = {
    "T2": "260813T045112-pid1445039-50f5cf",
    "T3": "260813T045116-pid1445175-9e68dd",
    "T4": "260813T045113-pid1445040-6fff9b",
}
ARM_DESC = {
    "T2": "synergy-only (anchor 0.00 + synergy 0.05)",
    "T3": "real anchor 0.02 + real synergy 0.05 (PEASD-Lite)",
    "T4": "real anchor 0.02 + phase-shifted synergy 0.05 (negative control)",
}
TARGET_STEP = 320_000_000
FULL_WEIGHT_UPDATE = 5000  # curriculum: ramp 1000->5000, full weight from 5000
STEPS_PER_UPDATE = 20480

DIMENSIONS = {
    "1_action_activation": [
        "val_activation_energy",
        "val_activation_saturation_fraction",
        "val_activation_rate_mean_square",
        "val_action_saturation_fraction",
        "val_action_rate_mean_square",
    ],
    "2_tracking_accuracy": [
        "val_err_rpos",
        "val_err_joint_pos",
        "val_err_joint_vel",
        "val_err_root_xyz",
        "val_err_root_yaw",
    ],
    "3_motion_success": [
        "val_frame_coverage",
        "val_early_termination_rate",
        "val_mean_episode_return",
    ],
    "4_synergy_error": [
        "val_emg_synergy_real_reference_loss",
        "val_emg_synergy_real_reference_shape_loss",
        "val_emg_synergy_real_reference_shape_cosine",
        "val_emg_synergy_real_reference_intensity_loss",
        "val_emg_synergy_real_reference_intensity",
    ],
    "5_muscle_anchor_error": [
        "val_emg_anchor_loss",
        "val_emg_anchor_correlation",
        "val_emg_anchor_mean_abs_deviation",
        "val_emg_anchor_max_abs_deviation",
        "val_emg_anchor_violation_fraction",
        "val_emg_anchor_valid_channel_fraction",
    ],
}
CONTEXT_KEYS = [
    "val_emg_anchor_weight",
    "val_emg_synergy_weight",
    "val_emg_curriculum_factor_anchor",
    "val_emg_curriculum_factor_synergy",
]

PROMOTION_GATES = {
    "val_early_termination_rate": ("<=", 0.05),
    "val_frame_coverage": (">=", 0.95),
    "val_err_rpos": ("<=", 0.09),
    "val_action_saturation_fraction": ("<=", 0.05),
    "val_activation_energy": ("<=", 0.35),
}


def load_histories() -> dict:
    histories = {}
    for arm, d in RUN_DIRS.items():
        path = CKPT_BASE / d / "stage1_peasd_validation_history.json"
        with open(path) as f:
            h = json.load(f)
        assert h["arm"] == arm, f"{path}: arm mismatch {h['arm']} != {arm}"
        assert h["seed"] == 0
        assert h["action_id"] == "forehandClear_standard"
        histories[arm] = h
    return histories


def get_entry_at(hist: dict, step: int) -> dict:
    for e in hist["entries"]:
        if e["checkpoint_identity"]["global_timestep"] == step:
            return e
    raise KeyError(f"{hist['arm']}: no entry at step {step}")


def metric_series(hist: dict, key: str) -> list:
    """[(global_step, update, value), ...] sorted by step."""
    out = []
    for e in hist["entries"]:
        ci = e["checkpoint_identity"]
        out.append((ci["global_timestep"], ci["update_number"], e["metrics"][key]))
    out.sort(key=lambda r: r[0])
    return out


def build_snapshot(histories: dict) -> dict:
    snap = {"per_arm": {}, "pairwise_at_320m": {}, "gate_check": {}, "series_meta": {}}
    for arm, h in histories.items():
        e320 = get_entry_at(h, TARGET_STEP)
        ci = e320["checkpoint_identity"]
        m = e320["metrics"]
        first = h["entries"][0]
        arm_block = {
            "run_id": ci["run_id"],
            "arm_description": ARM_DESC[arm],
            "config_hash": ci["config_hash"],
            "checkpoint_path": ci["checkpoint_path"],
            "checkpoint_content_sha256": ci["checkpoint_content_sha256"],
            "global_timestep": ci["global_timestep"],
            "update_number": ci["update_number"],
            "validation_protocol": h["validation_provenance"],
            "context_weights_at_320m": {k: m[k] for k in CONTEXT_KEYS},
            "endpoint_metrics": {},
            "improvement_first_to_320m": {},
        }
        for dim, keys in DIMENSIONS.items():
            arm_block["endpoint_metrics"][dim] = {k: m[k] for k in keys}
        # first -> 320m change (first entry shares the same metric schema)
        fm = first["metrics"]
        for dim, keys in DIMENSIONS.items():
            for k in keys:
                v0, v1 = fm[k], m[k]
                rel = None if v0 == 0 else (v1 - v0) / abs(v0)
                arm_block["improvement_first_to_320m"][k] = {
                    "first_step": first["checkpoint_identity"]["global_timestep"],
                    "first_value": v0,
                    "final_value": v1,
                    "abs_change": v1 - v0,
                    "rel_change": rel,
                }
        # last-3 consecutive validations (for gate stability)
        last3 = h["entries"][-3:]
        arm_block["last3_validations"] = [
            {
                "step": e["checkpoint_identity"]["global_timestep"],
                "update": e["checkpoint_identity"]["update_number"],
                **{k: e["metrics"][k] for k in PROMOTION_GATES},
            }
            for e in last3
        ]
        # gate check at 320m and over last-3
        gate = {}
        for k, (op, thr) in PROMOTION_GATES.items():
            v = m[k]
            ok = v <= thr if op == "<=" else v >= thr
            ok3 = all(
                (e["metrics"][k] <= thr if op == "<=" else e["metrics"][k] >= thr)
                for e in last3
            )
            gate[k] = {"op": op, "threshold": thr, "value_at_320m": v,
                       "pass_at_320m": ok, "pass_last3_consecutive": ok3}
        arm_block["gate_check"] = gate
        snap["per_arm"][arm] = arm_block

    # pairwise contrasts at 320m (relative improvement of row over column: (B-A)/B for
    # "lower is better" metrics, i.e. positive means row arm is better)
    lower_better = [
        "val_err_rpos", "val_err_joint_pos", "val_err_joint_vel", "val_err_root_xyz",
        "val_err_root_yaw", "val_early_termination_rate", "val_activation_energy",
        "val_activation_saturation_fraction", "val_action_saturation_fraction",
        "val_emg_synergy_real_reference_loss",
        "val_emg_synergy_real_reference_shape_loss",
        "val_emg_synergy_real_reference_intensity_loss",
        "val_emg_anchor_loss", "val_emg_anchor_mean_abs_deviation",
        "val_emg_anchor_max_abs_deviation", "val_emg_anchor_violation_fraction",
    ]
    higher_better = [
        "val_frame_coverage", "val_mean_episode_return",
        "val_emg_synergy_real_reference_shape_cosine",
        "val_emg_synergy_real_reference_intensity",
        "val_emg_anchor_correlation",
    ]
    vals = {a: get_entry_at(h, TARGET_STEP)["metrics"] for a, h in histories.items()}
    for a, b in [("T3", "T4"), ("T3", "T2"), ("T2", "T4")]:
        pair = {}
        for k in lower_better:
            va, vb = vals[a][k], vals[b][k]
            pair[k] = {"A": a, "A_value": va, "B": b, "B_value": vb,
                       "A_rel_improvement_over_B": None if vb == 0 else (vb - va) / abs(vb),
                       "direction": "lower_better"}
        for k in higher_better:
            va, vb = vals[a][k], vals[b][k]
            pair[k] = {"A": a, "A_value": va, "B": b, "B_value": vb,
                       "A_rel_improvement_over_B": None if vb == 0 else (va - vb) / abs(vb),
                       "direction": "higher_better"}
        snap["pairwise_at_320m"][f"{a}_vs_{b}"] = pair

    # T3-vs-T4 primary contrast at every common full-weight checkpoint
    t3_syn = metric_series(histories["T3"], "val_emg_synergy_real_reference_loss")
    t4_syn = metric_series(histories["T4"], "val_emg_synergy_real_reference_loss")
    t4_by_step = {s: v for s, _, v in t4_syn}
    contrast = []
    for s, u, v3 in t3_syn:
        if u >= FULL_WEIGHT_UPDATE and s in t4_by_step:
            v4 = t4_by_step[s]
            contrast.append({
                "step": s, "update": u, "t3": v3, "t4": v4,
                "t3_rel_improvement": (v4 - v3) / v4 if v4 else None,
            })
    snap["t3_vs_t4_synergy_full_weight_history"] = contrast

    # series metadata sanity
    for arm, h in histories.items():
        steps = [e["checkpoint_identity"]["global_timestep"] for e in h["entries"]]
        snap["series_meta"][arm] = {
            "n_entries": len(steps), "first_step": min(steps), "last_step": max(steps),
        }
    return snap


def make_figures(histories: dict, out_dir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"T2": "#1f77b4", "T3": "#2ca02c", "T4": "#d62728"}
    fw_step = FULL_WEIGHT_UPDATE * STEPS_PER_UPDATE / 1e6  # 102.4 M

    def series(arm, key):
        rows = metric_series(histories[arm], key)
        return [r[0] / 1e6 for r in rows], [r[2] for r in rows]

    def panel(fig_path, subplots, suptitle):
        n = len(subplots)
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.6), squeeze=False)
        for ax, (key, title, ylabel) in zip(axes[0], subplots):
            for arm in ("T2", "T3", "T4"):
                x, y = series(arm, key)
                ax.plot(x, y, marker="o", ms=3, lw=1.2, color=colors[arm], label=arm)
            ax.axvline(320.0, color="k", ls="--", lw=0.8, alpha=0.6)
            if key.startswith("val_emg"):
                ax.axvline(fw_step, color="gray", ls=":", lw=0.8, alpha=0.7)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("global step (M)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        return fig_path.name

    written = []
    written.append(panel(
        out_dir / "fig1_action_activation.png",
        [("val_activation_energy", "activation energy", "energy"),
         ("val_activation_saturation_fraction", "activation saturation (|a|>=0.98)", "fraction"),
         ("val_action_saturation_fraction", "action saturation", "fraction"),
         ("val_activation_rate_mean_square", "activation rate (mean square)", "ms")],
        "Dim 1: action/muscle activation level"))
    written.append(panel(
        out_dir / "fig2_tracking_accuracy.png",
        [("val_err_rpos", "RPos error", "m"),
         ("val_err_joint_pos", "joint pos error", "rad"),
         ("val_err_root_xyz", "root xyz error", "m"),
         ("val_err_root_yaw", "root yaw error", "rad")],
        "Dim 2: motion tracking accuracy"))
    written.append(panel(
        out_dir / "fig3_motion_success.png",
        [("val_frame_coverage", "frame coverage", "fraction"),
         ("val_early_termination_rate", "early termination rate", "fraction"),
         ("val_mean_episode_return", "mean episode return", "return")],
        "Dim 3: motion success"))
    written.append(panel(
        out_dir / "fig4_synergy_error.png",
        [("val_emg_synergy_real_reference_loss", "real-reference synergy loss", "loss"),
         ("val_emg_synergy_real_reference_shape_cosine", "synergy shape cosine", "cosine"),
         ("val_emg_synergy_real_reference_intensity_loss", "synergy intensity loss", "loss")],
        "Dim 4: muscle synergy error (vs real EMG reference; dotted line = full weight)"))
    written.append(panel(
        out_dir / "fig5_muscle_anchor_error.png",
        [("val_emg_anchor_loss", "M-channel anchor loss", "loss"),
         ("val_emg_anchor_correlation", "anchor correlation", "corr"),
         ("val_emg_anchor_mean_abs_deviation", "anchor mean |deviation|", "activation"),
         ("val_emg_anchor_violation_fraction", "anchor violation fraction", "fraction")],
        "Dim 5: muscle anchor error (M-channel vs measured EMG; dotted line = full weight)"))
    return written


def main() -> None:
    here = Path(__file__).resolve().parent
    histories = load_histories()
    snap = build_snapshot(histories)
    snap["target_step"] = TARGET_STEP
    snap["full_weight_update"] = FULL_WEIGHT_UPDATE
    snap["arms"] = ARM_DESC
    snap["dimensions"] = DIMENSIONS
    out_json = here / "stage1_t2t3t4_320m_snapshot.json"
    with open(out_json, "w") as f:
        json.dump(snap, f, indent=1, ensure_ascii=False)
    figs = make_figures(histories, here / "figures")
    print(f"wrote {out_json.name}")
    for name in figs:
        print(f"wrote figures/{name}")


if __name__ == "__main__":
    main()
