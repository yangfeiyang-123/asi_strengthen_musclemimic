# 训练配置导航

Hydra `--config-name` 相对于 `fullbody/`，不带 `.yaml`。主线只用下表"主线"列标为 ✓ 的目录与文件；其余为已归档支线，代码保留、默认不排期（见 `docs/archive/README.md`）。

| 目录 / 文件 | 主线 | 用途 |
|---|:-:|---|
| `base/conf_fullbody_badminton_{gmr,body_only_gmr,racket_gmr}.yaml` | ✓ | body / racket 共享基座 |
| `stage1_body/conf_fullbody_forehand_clear_aug100_body_local.yaml` | ✓ | S1 无 treatment 参照（80/20 aug100） |
| `stage1_body/peasd_lite_v1/conf_fullbody_forehand_clear_aug100_peasd_t{0..4}.yaml` | ✓ | S1 T0–T4 matched family（t0/t1 已有，t2–t4 待照 t1 派生） |
| `presets/stage1_peasd_lite_common_v1.yaml`、`stage1_peasd_lite_t{0..4}_v1.yaml` | ✓ | 五个 arm 的 EMG treatment preset（唯一差异） |
| `presets/stage1_peasd_lite_t{1,3,4}_anchor_v2_v1.yaml`、`peasd_lite_v1/conf_fullbody_forehand_clear_aug100_peasd_t{1,3,4}_anchor_v2.yaml` | **待跑** | anchor v2（带内拉力、burst 加权、跨通道形状项、噪声底加宽、逐通道封顶），针对 12 endpoint 诊断出的"被测肌肉被关掉"问题；预算 800,010,240 与已完成 endpoint 一致。重跑顺序见 `docs/narrative/03_当前状态与待办.md` |
| `stage1_body/conf_fullbody_forehand_clear_body_finger_isolated*.yaml` | ✓ | Stage 1R 手指隔离 rung（003/005） |
| `stage2_racket_v2/conf_fullbody_forehand_clear_racket_{event_bank,event_cache_single,mass_025..100}.yaml` | ✓ | 25→50→75→100% 球拍质量课程 |
| `distill/conf_fullbody_forehandclear_*_student_*.yaml`、`conf_fullbody_badminton_student_*.yaml` | ✓ | S2-A direct student（BC / DAgger / PPO） |
| `distill/latent_forehandclear_lab.yaml` | ✓ | S2-B..E latent 与 S3 LAB 基座 |
| `stage1_body/peasd_lite_v1/conf_fullbody_forehand_clear_peasd_t*.yaml`、`*_40train10val_*` | 探索 | legacy 22/5 与 40/10 子集，不进 formal family |
| `stage1_body/conf_fullbody_chinajump_*.yaml`、`peasd_lite_v1/conf_fullbody_chinajump_*`、`chinajump_coverage_phase_schema_v1.json`、`primitive_catalog/` | 归档 | ChinaJump 支线 |
| `stage1_body/conf_fullbody_forehand_lift_*.yaml`、`peasd_lite_v1/conf_fullbody_forehand_lift_*`、`distill/*forehandlift*` | 归档 | 正手挑球支线 |
| `stage1_body/continuity_ablation_v1/`、`*_continuity_*.yaml`、`presets/*fascicle*`、`presets/forehand_graph_nmf_*`、`presets/forehand_raw_unit_standard_nmf_*` | 归档 | 肌束连续性 / Graph-NMF 支线 |
| `stage1_body/*early_unified_synergy*`、`presets/forehand_early_unified_*`、`stage2_racket_v2/*_early_unified_synergy_v4.yaml`、`distill/latent_*_synergy_v3.yaml` | 归档 | fixed-synergy W/R 动作空间支线 |
| `stage2_racket/` | 归档 | Stage 2 v1（grip 与旧 racket 配置），已被 `stage2_racket_v2/` 取代 |
| `strokes/`、`skill/` | 归档 | 上游/早期单动作实验（turnleft、contact tracking、net lift、expert skill） |

这些支线配置没有物理搬移，因为 `musclemimic/badminton/action_registry.py` 与多份测试按路径引用它们；如需彻底清理，应连同 registry 与测试一起改。
