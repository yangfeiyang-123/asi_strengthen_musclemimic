#!/usr/bin/env python3
"""Stage-1 T2/T3/T4 @ 640M 五维验证指标提取与对比分析。

数据源（只读，seed 0；run 由 320M 合同 extend 到 640M 并训完）：
  ≤320M：datasets/.../checkpoints/<run_dir>/stage1_peasd_validation_history.json
  320M→640M：datasets/.../wandb/wandb/<extension_session>/run-*.wandb 本地 datastore
              的 `Validation Measures/*`（extension 阶段验证只写 wandb，
              未回写本地 history JSON）。

  T2 = 260813T045112-pid1445039-50f5cf / wandb jgnczvt2  (synergy-only)
  T3 = 260813T045116-pid1445175-9e68dd / wandb r8ue7l8s  (real anchor + real synergy)
  T4 = 260813T045113-pid1445040-6fff9b / wandb pxjbjk0q  (real anchor + phase-shifted synergy)

口径映射已验证：T3 原始 run 在 update 15616/15625 同时写了本地 JSON 的 val_* 与
wandb 的 Validation Measures/*，28/28 项逐一相等（见 snapshot 的 mapping_verification）。

640M 终点口径：extension 最后一次验证在 update 31232（639,631,360 步），
最终 checkpoint 为 update 31250（精确 640,000,000 步），二者相差 18 个 update
（368,640 步）。本分析以 update 31232 为"640M 终点验证"。

输出（全部写在本目录，不修改仓库其他任何文件）：
  stage1_t2t3t4_640m_snapshot.json     机器可读快照与对比
  figures_640m/fig1..fig5.png          五维全程训练曲线（9.99M→639.6M）
"""

from __future__ import annotations

import json
from pathlib import Path

from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2

ROOT = Path("/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic")
CKPT_BASE = ROOT / "datasets/forehandClear_standard/training_aug100_40train10val/checkpoints"
WANDB_BASE = ROOT / "datasets/forehandClear_standard/training_aug100_40train10val/wandb/wandb"

RUNS = {
    "T2": {
        "ckpt_dir": "260813T045112-pid1445039-50f5cf",
        "wandb_file": "run-20260815_162714-jgnczvt2/run-jgnczvt2.wandb",
        "wandb_id": "jgnczvt2",
        "desc": "synergy-only (anchor 0.00 + synergy 0.05)",
    },
    "T3": {
        "ckpt_dir": "260813T045116-pid1445175-9e68dd",
        "wandb_file": "run-20260815_155815-r8ue7l8s/run-r8ue7l8s.wandb",
        "wandb_id": "r8ue7l8s",
        "desc": "real anchor 0.02 + real synergy 0.05 (PEASD-Lite)",
    },
    "T4": {
        "ckpt_dir": "260813T045113-pid1445040-6fff9b",
        "wandb_file": "run-20260815_162714-pxjbjk0q/run-pxjbjk0q.wandb",
        "wandb_id": "pxjbjk0q",
        "desc": "real anchor 0.02 + phase-shifted synergy 0.05 (negative control)",
    },
}

TARGET_UPDATE = 31232            # last extension validation
TARGET_STEP = 639_631_360        # = 31232 * 20480
FINAL_STEP = 640_000_000         # update 31250 checkpoint, no validation
MID_STEP = 320_000_000           # previous contract endpoint (update 15625)
MID_UPDATE = 15625
FULL_WEIGHT_UPDATE = 5000
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
TREATMENT_FLAGS = ["val_emg_synergy_phase_shuffled"]

WANDB_KEY_MAP = {
    "val_activation_energy": "Validation Measures/activation_energy",
    "val_activation_saturation_fraction": "Validation Measures/activation_saturation_fraction",
    "val_activation_rate_mean_square": "Validation Measures/activation_rate_mean_square",
    "val_action_saturation_fraction": "Validation Measures/action_saturation_fraction",
    "val_action_rate_mean_square": "Validation Measures/action_rate_mean_square",
    "val_err_rpos": "Validation Measures/err_rpos",
    "val_err_joint_pos": "Validation Measures/err_joint_pos",
    "val_err_joint_vel": "Validation Measures/err_joint_vel",
    "val_err_root_xyz": "Validation Measures/err_root_xyz",
    "val_err_root_yaw": "Validation Measures/err_root_yaw",
    "val_frame_coverage": "Validation Measures/frame_coverage",
    "val_early_termination_rate": "Validation Measures/early_termination_rate",
    "val_mean_episode_return": "Validation/Mean Episode Return",
    "val_emg_synergy_real_reference_loss": "Validation Measures/emg_synergy_real_reference_loss",
    "val_emg_synergy_real_reference_shape_loss": "Validation Measures/emg_synergy_real_reference_shape_loss",
    "val_emg_synergy_real_reference_shape_cosine": "Validation Measures/emg_synergy_real_reference_shape_cosine",
    "val_emg_synergy_real_reference_intensity_loss": "Validation Measures/emg_synergy_real_reference_intensity_loss",
    "val_emg_synergy_real_reference_intensity": "Validation Measures/emg_synergy_real_reference_intensity",
    "val_emg_anchor_loss": "Validation Measures/emg_anchor_loss",
    "val_emg_anchor_correlation": "Validation Measures/emg_anchor_correlation",
    "val_emg_anchor_mean_abs_deviation": "Validation Measures/emg_anchor_mean_abs_deviation",
    "val_emg_anchor_max_abs_deviation": "Validation Measures/emg_anchor_max_abs_deviation",
    "val_emg_anchor_violation_fraction": "Validation Measures/emg_anchor_violation_fraction",
    "val_emg_anchor_valid_channel_fraction": "Validation Measures/emg_anchor_valid_channel_fraction",
    "val_emg_anchor_weight": "Validation Measures/emg_anchor_weight",
    "val_emg_synergy_weight": "Validation Measures/emg_synergy_weight",
    "val_emg_curriculum_factor_anchor": "Validation Measures/emg_curriculum_factor_anchor",
    "val_emg_curriculum_factor_synergy": "Validation Measures/emg_curriculum_factor_synergy",
    "val_emg_synergy_phase_shuffled": "Validation Measures/emg_synergy_phase_shuffled",
}

PROMOTION_GATES = {
    "val_early_termination_rate": ("<=", 0.05),
    "val_frame_coverage": (">=", 0.95),
    "val_err_rpos": ("<=", 0.09),
    "val_action_saturation_fraction": ("<=", 0.05),
    "val_activation_energy": ("<=", 0.35),
}

LOWER_BETTER = [
    "val_err_rpos", "val_err_joint_pos", "val_err_joint_vel", "val_err_root_xyz",
    "val_err_root_yaw", "val_early_termination_rate", "val_activation_energy",
    "val_activation_saturation_fraction", "val_action_saturation_fraction",
    "val_emg_synergy_real_reference_loss",
    "val_emg_synergy_real_reference_shape_loss",
    "val_emg_synergy_real_reference_intensity_loss",
    "val_emg_anchor_loss", "val_emg_anchor_mean_abs_deviation",
    "val_emg_anchor_max_abs_deviation", "val_emg_anchor_violation_fraction",
]
HIGHER_BETTER = [
    "val_frame_coverage", "val_mean_episode_return",
    "val_emg_synergy_real_reference_shape_cosine",
    "val_emg_synergy_real_reference_intensity",
    "val_emg_anchor_correlation",
]


def read_wandb_validations(path: Path) -> dict[int, dict]:
    """update_number -> {val_*: value} for records carrying Validation Measures."""
    rev = {v: k for k, v in WANDB_KEY_MAP.items()}
    ds = DataStore()
    ds.open_for_scan(str(path))
    out = {}
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        upd = rec.history.step.num // STEPS_PER_UPDATE
        for it in rec.history.item:
            nm = "/".join(it.nested_key) if it.nested_key else it.key
            if nm in rev:
                try:
                    out.setdefault(upd, {})[rev[nm]] = json.loads(it.value_json)
                except (json.JSONDecodeError, TypeError):
                    pass
    ds.close()
    # keep only updates that actually ran validation (err_rpos present)
    return {u: m for u, m in sorted(out.items()) if "val_err_rpos" in m}


def build_series(arm: str) -> list[dict]:
    info = RUNS[arm]
    series = []
    with open(CKPT_BASE / info["ckpt_dir"] / "stage1_peasd_validation_history.json") as f:
        h = json.load(f)
    assert h["arm"] == arm
    for e in h["entries"]:
        series.append({
            "update": e["checkpoint_identity"]["update_number"],
            "global_timestep": e["checkpoint_identity"]["global_timestep"],
            "source": "local_history_json",
            "metrics": e["metrics"],
        })
    for upd, m in read_wandb_validations(WANDB_BASE / info["wandb_file"]).items():
        series.append({
            "update": upd,
            "global_timestep": upd * STEPS_PER_UPDATE,
            "source": "wandb_extension",
            "metrics": m,
        })
    seen, dedup = set(), []
    for e in sorted(series, key=lambda e: e["update"]):
        if e["update"] in seen:
            continue
        seen.add(e["update"])
        dedup.append(e)
    return dedup


def at_update(series: list[dict], upd: int) -> dict:
    for e in series:
        if e["update"] == upd:
            return e
    raise KeyError(f"no entry at update {upd}")


def rel_improvement(a, b, key):
    """Positive = arm A better than arm B on key."""
    va, vb = a[key], b[key]
    if key in LOWER_BETTER:
        return None if vb == 0 else (vb - va) / abs(vb)
    return None if vb == 0 else (va - vb) / abs(vb)


def make_figures(series: dict, out_dir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"T2": "#1f77b4", "T3": "#2ca02c", "T4": "#d62728"}
    fw_m = FULL_WEIGHT_UPDATE * STEPS_PER_UPDATE / 1e6      # 102.4
    mid_m = MID_STEP / 1e6                                   # 320.0
    end_m = TARGET_STEP / 1e6                                # 639.63

    def ser(arm, key):
        pts = [(e["global_timestep"] / 1e6, e["metrics"][key])
               for e in series[arm] if e["metrics"].get(key) is not None]
        return [p[0] for p in pts], [p[1] for p in pts]

    def panel(fig_path, subplots, suptitle):
        n = len(subplots)
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.6), squeeze=False)
        for ax, (key, title, ylabel) in zip(axes[0], subplots):
            for arm in ("T2", "T3", "T4"):
                x, y = ser(arm, key)
                ax.plot(x, y, marker="o", ms=2.5, lw=1.1, color=colors[arm], label=arm)
            ax.axvline(end_m, color="k", ls="--", lw=0.9, alpha=0.7)
            ax.axvline(mid_m, color="gray", ls="--", lw=0.8, alpha=0.6)
            if key.startswith("val_emg"):
                ax.axvline(fw_m, color="gray", ls=":", lw=0.8, alpha=0.7)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("global step (M)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(suptitle + "  [dashed: 320M old / 639.6M new endpoint]", fontsize=11)
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
        "Dim 4: muscle synergy error (vs real EMG reference; dotted = full weight)"))
    written.append(panel(
        out_dir / "fig5_muscle_anchor_error.png",
        [("val_emg_anchor_loss", "M-channel anchor loss", "loss"),
         ("val_emg_anchor_correlation", "anchor correlation", "corr"),
         ("val_emg_anchor_mean_abs_deviation", "anchor mean |deviation|", "activation"),
         ("val_emg_anchor_violation_fraction", "anchor violation fraction", "fraction")],
        "Dim 5: muscle anchor error (M-channel vs measured EMG; dotted = full weight)"))
    return written


def main() -> None:
    here = Path(__file__).resolve().parent
    series = {arm: build_series(arm) for arm in RUNS}
    for arm, s in series.items():
        steps = [e["global_timestep"] for e in s]
        print(f"{arm}: {len(s)} validations, {min(steps):,} -> {max(steps):,} "
              f"(endpoint update {s[-1]['update']}, source {s[-1]['source']})")

    snap = {
        "target": {"endpoint_update": TARGET_UPDATE, "endpoint_step": TARGET_STEP,
                   "final_checkpoint_step": FINAL_STEP, "mid_step": MID_STEP},
        "arms": {a: {**{"desc": RUNS[a]["desc"]}, "wandb_id": RUNS[a]["wandb_id"],
                     "ckpt_dir": RUNS[a]["ckpt_dir"]} for a in RUNS},
        "mapping_verification": (
            "T3 original run wrote both local val_* and wandb Validation Measures/*; "
            "28/28 metrics bit-identical at updates 15616 and 15625."
        ),
        "dimensions": DIMENSIONS,
        "per_arm": {},
        "pairwise_at_endpoint": {},
        "gate_check": {},
    }

    for arm, s in series.items():
        e_end = at_update(s, TARGET_UPDATE)
        e_mid = at_update(s, MID_UPDATE)
        e_first = s[0]
        m_end, m_mid, m_first = e_end["metrics"], e_mid["metrics"], e_first["metrics"]
        block = {
            "endpoint": {"update": TARGET_UPDATE, "global_timestep": TARGET_STEP},
            "context_at_endpoint": {k: m_end.get(k) for k in CONTEXT_KEYS},
            "treatment_flags_at_endpoint": {k: m_end.get(k) for k in TREATMENT_FLAGS},
            "endpoint_metrics": {d: {k: m_end.get(k) for k in ks} for d, ks in DIMENSIONS.items()},
            "midpoint_320m_metrics": {d: {k: m_mid.get(k) for k in ks} for d, ks in DIMENSIONS.items()},
            "change_320m_to_endpoint": {},
            "change_first_to_endpoint": {},
        }
        for ks in DIMENSIONS.values():
            for k in ks:
                v0, v1 = m_first.get(k), m_end.get(k)
                if v0 is not None and v1 is not None:
                    block["change_first_to_endpoint"][k] = {
                        "first_step": e_first["global_timestep"], "first_value": v0,
                        "final_value": v1, "abs_change": v1 - v0,
                        "rel_change": None if v0 == 0 else (v1 - v0) / abs(v0)}
                vm, ve = m_mid.get(k), m_end.get(k)
                if vm is not None and ve is not None:
                    block["change_320m_to_endpoint"][k] = {
                        "mid_value": vm, "final_value": ve, "abs_change": ve - vm,
                        "rel_change": None if vm == 0 else (ve - vm) / abs(vm)}
        # gate check at endpoint + last-3 consecutive validations
        last3 = s[-3:]
        gate = {}
        for k, (op, thr) in PROMOTION_GATES.items():
            v = m_end.get(k)
            ok = (v <= thr if op == "<=" else v >= thr) if v is not None else None
            ok3 = all((e["metrics"][k] <= thr if op == "<=" else e["metrics"][k] >= thr)
                      for e in last3)
            gate[k] = {"op": op, "threshold": thr, "value_at_endpoint": v,
                       "pass_at_endpoint": ok, "pass_last3_consecutive": ok3}
        block["gate_check"] = gate
        block["last3_validations"] = [
            {"update": e["update"], "step": e["global_timestep"],
             **{k: e["metrics"].get(k) for k in PROMOTION_GATES}} for e in last3]
        snap["per_arm"][arm] = block

    # pairwise at endpoint
    vals = {a: at_update(s, TARGET_UPDATE)["metrics"] for a, s in series.items()}
    for a, b in [("T3", "T4"), ("T3", "T2"), ("T2", "T4")]:
        pair = {}
        for k in LOWER_BETTER + HIGHER_BETTER:
            pair[k] = {"A": a, "A_value": vals[a].get(k), "B": b, "B_value": vals[b].get(k),
                       "A_rel_improvement_over_B": rel_improvement(vals[a], vals[b], k),
                       "direction": "lower_better" if k in LOWER_BETTER else "higher_better"}
        snap["pairwise_at_endpoint"][f"{a}_vs_{b}"] = pair

    # T3-vs-T4 primary contrast at every common full-weight validation
    t3 = {e["update"]: e["metrics"].get("val_emg_synergy_real_reference_loss")
          for e in series["T3"]}
    t4 = {e["update"]: e["metrics"].get("val_emg_synergy_real_reference_loss")
          for e in series["T4"]}
    common = sorted(u for u in t3 if u in t4 and u >= FULL_WEIGHT_UPDATE
                    and t3[u] is not None and t4[u] is not None)
    contrast = [{"update": u, "step": u * STEPS_PER_UPDATE, "t3": t3[u], "t4": t4[u],
                 "t3_rel_improvement": (t4[u] - t3[u]) / t4[u] if t4[u] else None}
                for u in common]
    pos = [c for c in contrast if c["t3_rel_improvement"] and c["t3_rel_improvement"] > 0]
    snap["t3_vs_t4_synergy_full_weight_history"] = contrast
    snap["t3_vs_t4_synergy_summary"] = {
        "n_common_full_weight_points": len(contrast),
        "n_positive": len(pos),
        "mean_improvement_all": sum(c["t3_rel_improvement"] for c in contrast) / len(contrast),
        "mean_improvement_last17": sum(c["t3_rel_improvement"] for c in contrast[-17:]) / 17,
        "improvement_at_320m": next(c["t3_rel_improvement"] for c in contrast
                                    if c["update"] == MID_UPDATE),
        "improvement_at_endpoint": contrast[-1]["t3_rel_improvement"],
        "extension_points": [c for c in contrast if c["update"] > MID_UPDATE],
    }

    # raw series (for reproducibility)
    snap["series_index"] = {
        a: [{"update": e["update"], "step": e["global_timestep"], "source": e["source"]}
            for e in s] for a, s in series.items()}

    out = here / "stage1_t2t3t4_640m_snapshot.json"
    with open(out, "w") as f:
        json.dump(snap, f, indent=1, ensure_ascii=False)
    print(f"wrote {out.name}")

    # ---- console summary tables ----
    def row(k, label):
        r = f"{label:<34}"
        for a in ("T2", "T3", "T4"):
            v = snap["per_arm"][a]["endpoint_metrics"]
            v = [v[d][k] for d in v if k in v[d]][0]
            r += f" {v:>12.4g}"
        for pair in ("T3_vs_T4", "T3_vs_T2"):
            ent = snap["pairwise_at_endpoint"][pair].get(k)
            imp = ent["A_rel_improvement_over_B"] if ent else None
            r += f" {imp*100:>+9.1f}%" if imp is not None else f" {'—':>9}"
        return r

    print("\n=== endpoint @639.63M (update 31232) ===")
    print(f"{'metric':<34} {'T2':>12} {'T3':>12} {'T4':>12} {'T3vsT4':>9} {'T3vsT2':>9}")
    for d, ks in DIMENSIONS.items():
        print(f"-- {d}")
        for k in ks:
            print(row(k, k.replace("val_", "").replace("emg_", "")))

    print("\n=== gates @endpoint ===")
    for k, (op, thr) in PROMOTION_GATES.items():
        line = f"{k:<40} {op}{thr:<7}"
        for a in ("T2", "T3", "T4"):
            g = snap["per_arm"][a]["gate_check"][k]
            line += f"  {a}={g['value_at_endpoint']:.4g}({'P' if g['pass_at_endpoint'] else 'F'}/l3:{'P' if g['pass_last3_consecutive'] else 'F'})"
        print(line)

    print("\n=== context/treatment at endpoint ===")
    for a in ("T2", "T3", "T4"):
        pa = snap["per_arm"][a]
        print(a, {k: round(v, 5) if isinstance(v, float) else v
                  for k, v in {**pa["context_at_endpoint"],
                               **pa["treatment_flags_at_endpoint"]}.items()})

    s = snap["t3_vs_t4_synergy_summary"]
    print("\n=== T3 vs T4 synergy primary contrast ===")
    print(f"common full-weight points: {s['n_common_full_weight_points']}, "
          f"positive: {s['n_positive']}")
    print(f"improvement @320M: {s['improvement_at_320m']*100:+.2f}%  "
          f"@endpoint: {s['improvement_at_endpoint']*100:+.2f}%")
    print(f"mean all: {s['mean_improvement_all']*100:+.2f}%  "
          f"mean last-17: {s['mean_improvement_last17']*100:+.2f}%")
    ext = s["extension_points"]
    print(f"extension points: {len(ext)}, positive: "
          f"{sum(1 for c in ext if c['t3_rel_improvement'] > 0)}, "
          f"mean: {sum(c['t3_rel_improvement'] for c in ext)/len(ext)*100:+.2f}%")

    # key 320M->640M deltas
    print("\n=== key deltas 320M -> endpoint (rel change) ===")
    for k in ("val_emg_synergy_real_reference_loss", "val_err_rpos", "val_frame_coverage",
              "val_early_termination_rate", "val_activation_energy",
              "val_action_saturation_fraction", "val_emg_anchor_loss",
              "val_emg_anchor_correlation"):
        line = f"{k:<46}"
        for a in ("T2", "T3", "T4"):
            c = snap["per_arm"][a]["change_320m_to_endpoint"].get(k)
            line += f" {a}:{c['rel_change']*100:+.1f}%" if c and c["rel_change"] is not None else f" {a}:—"
        print(line)

    figs = make_figures(series, here / "figures_640m")
    for name in figs:
        print(f"wrote figures_640m/{name}")


if __name__ == "__main__":
    main()
