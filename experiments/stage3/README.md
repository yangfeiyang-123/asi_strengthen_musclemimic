# 阶段三：自适应击球

runner：`python -m musclemimic.badminton.scripts.run_incoming_shuttle_hit --spec <yaml> --stage {preflight,base-only-check,feed-check,train-gpu,evaluate}`。
registry 指针在 `musclemimic/badminton/action_registry.py`（`stage3_spec` / `stage3_v2_spec` / `stage3_direct_spec`）。

## `lab/` 主线：冻结 prior/decoder + `z = μ + λσ·tanh(u)`

| spec | 说明 |
|---|---|
| `incoming_shuttle_hit_v1.yaml` | LAB，latent_dim 16，registry `stage3_spec` |
| `incoming_shuttle_hit_impact_recovery_v2.yaml` | LAB，含 impact/recovery 奖励 profile，registry `stage3_v2_spec`；H1/H2/H3 从这里派生 |

H1/H2/H3 只差 `--latent-checkpoint`（S2-B / S2-C）与 H3 的 bounded right-arm residual 字段（alpha ≤ 0.10）；见 `docs/runbooks/peasd_implementation_guide.md` §5。训练入口 `fullbody/latent_run_lab_ppo.py`。正式训练前必须依次通过 preflight、base-only-check、feed-check 和单 feed 物理可达性门。

## `direct_residual/` 已归档：直接残差探索线（`stage3_lab.enabled: false`）

`incoming_shuttle_hit_full354_v1.yaml`（registry `stage3_direct_spec`）以及 v3–v31：冻结 direct Stage-2 策略 + 354 维残差（`frozen_base_residual`、`selected_physical_correction` 等）。复盘见 `docs/archive/stage3_return_training_findings_20260805.md`。保留下来的结论：单 feed 物理可达性门、aero 出球目标校准（22 m/s @ up 0.84，`outputs/stage3_calibration/`）。

离线服务器上的 v32–v45 尚未回传（`outputs/transfer_checklist_20260905.md`）；`incoming_shuttle_hit_selected_correction_v31.yaml` 是本机未入库的草案，封 artifact 前需改名 ≥ v46 并与离线线对账。

## `early_tasks/`

`forehand_clear_grip_hold_v1.yaml`、`forehand_clear_static_hit_v1.yaml`、`forehand_net_lift_v1.yaml`：早期握拍 / 静态击球 / 挑球实验，非主线，仍被 `run_forehand_clear_racket_curriculum.py` 等脚本作为默认路径引用。

## 双人对打

`environment/double_play/` 不是阶段三的训练目标，而是之后的评估场；其 README 在该目录。
