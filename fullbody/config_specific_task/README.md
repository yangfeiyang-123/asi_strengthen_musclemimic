# 训练配置导航

Hydra `--config-name` 相对于 `fullbody/`，不带 `.yaml`。主线只用下面这些目录；`archive/` 是已归档支线，代码与测试保留、默认不排期（见 `docs/archive/README.md`）。

## 主线

| 目录 / 文件 | 用途 |
|---|---|
| `base/conf_fullbody_badminton_{gmr,body_only_gmr,racket_gmr}.yaml` | body / racket 共享基座 |
| `stage1_body/conf_fullbody_forehand_clear_body_local.yaml` | legacy 22/5 body 基座（aug100 配置的父级，本身不再排期） |
| `stage1_body/conf_fullbody_forehand_clear_aug100_body_local.yaml` | S1 数据合同：aug100 80/20 名单、training_source、验证 lane |
| `stage1_body/peasd_lite_v1/conf_fullbody_forehand_clear_aug100_peasd_t{0..4}.yaml` + `presets/stage1_peasd_lite_{common,t0..t4}_v1.yaml` | S1 T0–T4 matched family（五个文件同形，只换 preset）。注意 `action_registry.py` / pipeline planner 的 forehand_clear 数据合同仍是 22/5，其 `stage1_peasd_configs` 指向 `archive/legacy_22_5/`；aug100 family 目前通过 `scripts/run_fullbody_training.sh --config-name=` 直接启动 |
| `stage1_body/peasd_lite_v1/conf_fullbody_forehand_clear_aug100_peasd_t{1,3,4}_anchor_v2.yaml` + `presets/stage1_peasd_lite_t{1,3,4}_anchor_v2_v1.yaml` | anchor v2（待跑，顺序见 `docs/narrative/03_当前状态与待办.md`） |
| `stage1_body/conf_fullbody_forehand_clear_body_finger_isolated{,_005}.yaml`、`*_fingerperturb.yaml`、`*_repair_v2.yaml` | Stage 1R 手指隔离 rung 与修复配置 |
| `stage2_racket_v2/conf_fullbody_forehand_clear_aug100_racket_derived_rigid.yaml` | **持拍挥拍训练入口**：aug100 + `MjxMyoFullBodyRacket` + `RacketMimicReward`（`derived_rigid`，不需要 event bank），从 Stage-1 endpoint `resume_from`；启动步骤见 `docs/runbooks/racket_grip_hit_assets_20260920.md` |
| `stage2_racket_v2/conf_fullbody_forehand_clear_racket_{event_bank,event_cache_single,mass_025..100}.yaml` | event_reference_v2 版球拍课程；需要 train/val event bank 清单（目前没有） |
| `stage2_racket/conf_fullbody_badminton_racket_*.yaml` | Stage 2 v1 球拍配置；`stage2_racket_v2` 的 defaults 仍继承 `conf_fullbody_badminton_racket_local`，所以留在原位 |
| `distill/conf_fullbody_forehandclear_*_student_*.yaml`、`conf_fullbody_badminton_student_*.yaml` | S2-A direct student（BC / DAgger / PPO） |
| `distill/latent_forehandclear_lab.yaml` | S2-B..E latent 与 S3 LAB 基座 |

## 归档（`archive/`）

| 子目录 | 内容 | 文件数 |
|---|---|---:|
| `chinajump/` | ChinaJump 支线：Stage-1 配置、peasd_lite_v1 变体、`primitive_catalog/`、coverage schema、latent 配置 | 80 |
| `continuity_graph_nmf/` | 肌束连续性 / Graph-NMF 支线：`continuity_ablation_v1/`、continuity reward/diag 配置与 preset | 32 |
| `forehand_lift/` | 正手挑球支线：Stage-1/1R、球拍、peasd_lite_v1、distill | 15 |
| `fixed_synergy/` | fixed-synergy W/R 动作空间：early_unified_synergy_v4 及其球拍课程、latent synergy_v3 | 8 |
| `legacy_22_5/` | 22/5 数据上的 PEASD T0–T4（正式 family 只认 aug100 80/20） | 5 |
| `subset_40_10/` | aug100 的 40/10 分组子集（body_local + T2/T3/T4），探索用 | 4 |
| `strokes/`、`skill/` | 上游/早期单动作实验（turnleft、contact tracking、net lift、expert skill） | 5 |

归档配置内部的 `defaults:` 与 `musclemimic/badminton/action_registry.py`、相关测试中的路径已随迁移同步改写；`--config-name=config_specific_task/archive/<family>/...` 仍可加载。
