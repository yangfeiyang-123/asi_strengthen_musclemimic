# Forehand Clear 实验记录表

> 阶段编号按 `进展/工作汇报0816.pdf`：阶段一 = 轨迹跟踪（身体 + 球拍课程），阶段二 = 技能蒸馏，阶段三 = 自适应击球。括号内为仓库代码中的 Stage 编号。

维护规则：append-only，一行一个 run；结果必须同时记 commit、config hash（resolved config 的 `run_manifest.json`）、seed、checkpoint 路径与 W&B run；失败 run 不删行，状态改"失败"并在备注写原因。状态取值：`未开始 / 训练中 / 完成 / 失败 / 已晋级`。

---

## 阶段一 a — 身体轨迹模仿（仓库 Stage 1；PEASD 对照，3 seed × 5 组 = 15 runs）

数据：**唯一数据组 = 80/20，即 aug100 release 的 80 train / 20 held-out validation**（`train80_val20`，`grouped_source_split`：按 metadata.augmentation.source_cache 沿用审阅过的 22/5 source 划分，held-out 为 yaw 分组轨迹）。
本表只覆盖此数据组；**40/10 子集线（`train40_val10` 全部配置）与 legacy 22/5 不在本轮范围，不得混入对照。**
五组对照 = PEASD 消融矩阵（`fullbody/config_specific_task/presets/stage1_peasd_lite_t{0..4}_v1.yaml`）：

| 组 | control_kind | anchor_w | synergy_w | 含义 |
|---|---|---:|---:|---|
| T0 | no_emg_reward | 0 | 0 | 无 EMG 一致性奖励基线（纯轨迹模仿） |
| T1 | activation_anchor_only | 0.02 | 0 | 只加 activation anchor 项 |
| T2 | synergy_anchor_only | 0 | 0.05 | 只加协同项 |
| T3 | real_peasd_lite | 0.02 | 0.05 | 完整 PEASD lite |
| T4 | deterministic_half_cycle_circular_phase_shift | — | — | 协同相位半周期打乱的负对照（安慰剂） |

命名规范（沿用 80/20 线现有约定，见 t1 配置）：`forehand_clear_aug100_stage1_peasd_lite_v1_t{X}_s{Y}`，seed Y ∈ {0,1,2}。

**80/20 配置现状与缺口（开跑前须补）**：
- 已有：baseline `conf_fullbody_forehand_clear_aug100_body_local.yaml`（train80_val20）、T1 `peasd_lite_v1/conf_fullbody_forehand_clear_aug100_peasd_t1.yaml`；
- **缺 t0/t2/t3/t4 的 80/20 版**——照 t1 派生（把 preset 换成 `stage1_peasd_lite_t{0,2,3,4}_v1`，改 run name/tag），每个 seed 独立 run_id + seed 覆盖；
- 注意：baseline `body_local` ≠ T0——公平对照要求五个 arm 共用 matched 配置形状（T0 是显式 `emg_consistency.enabled: false` 的 preset 版），body_local 只作 sanity 参照，不进对照统计。

**Endpoint checkpoint 位置（2026-09-18）**：12 个已完成 run 的 800M endpoint 已迁到本机 `checkpoints/stage1/seed{0,1,2}/T*/checkpoint_39063`（gitignored），run 身份、逐文件哈希、W&B 与 endpoint 指标见 `checkpoints/stage1/COMPLETED_20260918.json`；`datasets/` 下对应的中间快照（六个 checkpoint，逐文件哈希核对一致）已删除。完成 ≠ 晋级：pairwise gate、post-hoc physiology 与 T3 seed0 blind review 尚未运行。

**2026-09-18 anchor v2 之后需要重跑**：T1 / T3 / T4 的奖励定义已变（`presets/stage1_peasd_lite_t{1,3,4}_anchor_v2_v1.yaml`），上表这三个 arm 的 12 行结果保留为 anchor v1；T0、T2 不变、不重跑。重跑顺序与配置名见 `docs/narrative/03_当前状态与待办.md`「需要重跑的实验」。新 run 请在下表追加行（run_id 后缀 `t3av2_s0` 等），不要覆盖 v1 行。

| # | 组 | seed | run_id | config（80/20 版） | 状态 | 服务器/GPU | 步数 | val tracking err | no-fall | PEASD 诊断 | checkpoint | W&B | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | T0 | s0 | `forehand_clear_aug100_80train20val_peasd_c1ccd93_local9_t0t4_direct800_v1_t0_s0` | 离线 c1ccd93 快照配置（hash 见 COMPLETED json） | 完成（800M endpoint，未晋级） | 远端（离线快照 20260903） | 800,010,240 | joint_pos 0.237；coverage 0.869 | early_term 0.40 | — | `checkpoints/stage1/seed0/T0/checkpoint_39063` | [l9t0s0v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t0s0v1) | validation 65 次；W&B finished；哈希见 `checkpoints/stage1/SHA256SUMS_20260918` |
| 2 | T0 | s1 | forehand_clear_aug100_stage1_peasd_lite_v1_t0_s1_msclab_20260908 | peasd_lite_v1/conf_fullbody_forehand_clear_aug100_peasd_t0.yaml | **中断**：2026-09-11 停在 299,827,200 / 800,010,240（进程消失，无 traceback，机器未重启） | msclab / GPU 6 | 299,827,200 | joint_pos 0.291（第 24 次验证） | early_term 0.65 | — | `datasets/forehandClear_standard/training_aug100_t0_s1_msclab_20260908/checkpoints/260908T183904-pid1035777-60c6ff/checkpoint_14640` | [np2vb5cd](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/np2vb5cd) | 2026-09-08 启动，PID 1035777；待决定 auto_resume 或 fresh 重跑 |
| 3 | T0 | s2 | …_t0_s2 | （待创建，照 t1 派生） | 未开始 | | | | | | | | |
| 4 | T1 | s0 | `forehand_clear_aug100_80train20val_peasd_c1ccd93_full5_direct800_v3_t1_s0` | 离线 c1ccd93 快照配置（hash 见 COMPLETED json） | 完成（800M endpoint，未晋级） | 远端（离线快照 20260903） | 800,010,240 | joint_pos 0.264；coverage 0.773 | early_term 0.75 | real-synergy loss 0.987 | `checkpoints/stage1/seed0/T1/checkpoint_39063` | [c8t1s0v3](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t1s0v3) | validation 65 次；W&B finished；哈希见 `checkpoints/stage1/SHA256SUMS_20260918` |
| 5 | T1 | s1 | `forehand_clear_aug100_80train20val_peasd_c1ccd93_local9_multiseed_direct800_v1_t1_s1` | 离线 c1ccd93 快照配置（hash 见 COMPLETED json） | 完成（800M endpoint，未晋级） | 远端（离线快照 20260903） | 800,010,240 | joint_pos 0.205；coverage 0.803 | early_term 0.40 | real-synergy loss 1.099 | `checkpoints/stage1/seed1/T1/checkpoint_39063` | [l9t1s1v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t1s1v1) | validation 65 次；W&B finished；哈希见 `checkpoints/stage1/SHA256SUMS_20260918` |
| 6 | T1 | s2 | …_t1_s2 | （待创建，照 t1 派生） | 未开始 | | | | | | | | |
| 7 | T2 | s0 | `forehand_clear_aug100_80train20val_peasd_c1ccd93_full5_direct800_v3_t2_s0` | 离线 c1ccd93 快照配置（hash 见 COMPLETED json） | 完成（800M endpoint，未晋级） | 远端（离线快照 20260903） | 800,010,240 | joint_pos 0.253；coverage 0.917 | early_term 0.30 | real-synergy loss 0.173 | `checkpoints/stage1/seed0/T2/checkpoint_39063` | [c8t2s0v3](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t2s0v3) | validation 65 次；W&B finished；哈希见 `checkpoints/stage1/SHA256SUMS_20260918` |
| 8 | T2 | s1 | `forehand_clear_aug100_80train20val_peasd_c1ccd93_local9_multiseed_direct800_v1_t2_s1` | 离线 c1ccd93 快照配置（hash 见 COMPLETED json） | 完成（800M endpoint，未晋级） | 远端（离线快照 20260903） | 800,010,240 | joint_pos 0.197；coverage 0.923 | early_term 0.30 | real-synergy loss 0.137 | `checkpoints/stage1/seed1/T2/checkpoint_39063` | [l9t2s1v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t2s1v1) | validation 65 次；W&B finished；哈希见 `checkpoints/stage1/SHA256SUMS_20260918` |
| 9 | T2 | s2 | `forehand_clear_aug100_80train20val_peasd_c1ccd93_local9_multiseed_direct800_v1_t2_s2` | 离线 c1ccd93 快照配置（hash 见 COMPLETED json） | 完成（800M endpoint，未晋级） | 远端（离线快照 20260903） | 800,010,240 | joint_pos 0.235；coverage 0.890 | early_term 0.30 | real-synergy loss 0.126 | `checkpoints/stage1/seed2/T2/checkpoint_39063` | [l9t2s2v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t2s2v1) | validation 65 次；W&B finished；哈希见 `checkpoints/stage1/SHA256SUMS_20260918` |
| 10 | T3 | s0 | `forehand_clear_aug100_80train20val_peasd_c1ccd93_full5_direct800_v3_t3_s0` | 离线 c1ccd93 快照配置（hash 见 COMPLETED json） | 完成（800M endpoint，未晋级） | 远端（离线快照 20260903） | 800,010,240 | joint_pos 0.210；coverage 0.899 | early_term 0.40 | real-synergy loss 0.196 | `checkpoints/stage1/seed0/T3/checkpoint_39063` | [c8t3s0v3](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t3s0v3) | validation 65 次；W&B finished；哈希见 `checkpoints/stage1/SHA256SUMS_20260918` |
| 11 | T3 | s1 | `forehand_clear_aug100_80train20val_peasd_c1ccd93_local9_multiseed_direct800_v1_t3_s1` | 离线 c1ccd93 快照配置（hash 见 COMPLETED json） | 完成（800M endpoint，未晋级） | 远端（离线快照 20260903） | 800,010,240 | joint_pos 0.232；coverage 0.877 | early_term 0.20 | real-synergy loss 0.195 | `checkpoints/stage1/seed1/T3/checkpoint_39063` | [l9t3s1v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t3s1v1) | validation 65 次；W&B finished；哈希见 `checkpoints/stage1/SHA256SUMS_20260918` |
| 12 | T3 | s2 | `forehand_clear_aug100_80train20val_peasd_c1ccd93_local9_multiseed_direct800_v1_t3_s2` | 离线 c1ccd93 快照配置（hash 见 COMPLETED json） | 完成（800M endpoint，未晋级） | 远端（离线快照 20260903） | 800,010,240 | joint_pos 0.261；coverage 0.855 | early_term 0.30 | real-synergy loss 0.186 | `checkpoints/stage1/seed2/T3/checkpoint_39063` | [l9t3s2v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t3s2v1) | validation 65 次；W&B finished；哈希见 `checkpoints/stage1/SHA256SUMS_20260918` |
| 13 | T4 | s0 | `forehand_clear_aug100_80train20val_peasd_c1ccd93_local9_t0t4_direct800_v1_t4_s0` | 离线 c1ccd93 快照配置（hash 见 COMPLETED json） | 完成（800M endpoint，未晋级） | 远端（离线快照 20260903） | 800,010,240 | joint_pos 0.235；coverage 0.856 | early_term 0.30 | real-synergy loss 0.485 | `checkpoints/stage1/seed0/T4/checkpoint_39063` | [l9t4s0v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t4s0v1) | validation 65 次；W&B finished；哈希见 `checkpoints/stage1/SHA256SUMS_20260918` |
| 14 | T4 | s1 | `forehand_clear_aug100_80train20val_peasd_c1ccd93_local9_multiseed_direct800_v1_t4_s1` | 离线 c1ccd93 快照配置（hash 见 COMPLETED json） | 完成（800M endpoint，未晋级） | 远端（离线快照 20260903） | 800,010,240 | joint_pos 0.225；coverage 0.873 | early_term 0.40 | real-synergy loss 0.556 | `checkpoints/stage1/seed1/T4/checkpoint_39063` | [l9t4s1v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t4s1v1) | validation 65 次；W&B finished；哈希见 `checkpoints/stage1/SHA256SUMS_20260918` |
| 15 | T4 | s2 | `forehand_clear_aug100_80train20val_peasd_c1ccd93_local9_multiseed_direct800_v1_t4_s2` | 离线 c1ccd93 快照配置（hash 见 COMPLETED json） | 完成（800M endpoint，未晋级） | 远端（离线快照 20260903） | 800,010,240 | joint_pos 0.227；coverage 0.922 | early_term 0.20 | real-synergy loss 0.539 | `checkpoints/stage1/seed2/T4/checkpoint_39063` | [l9t4s2v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t4s2v1) | validation 65 次；W&B finished；哈希见 `checkpoints/stage1/SHA256SUMS_20260918` |

结论规则（预注册，避免事后挑指标）：
- 主指标：held-out 20 条 validation 的 tracking 指标 + no-fall + promotion gate 通过与否（80/20 配置要求 20 条视觉验收 clip）；
- PEASD 是否"有效"：T3 相对 T0 的改善必须显著大于 T4 相对 T0 的改善（负对照校准），且跨 3 seed 方向一致；
- T1/T2 用于归因（anchor vs synergy 各自贡献）；
- 每组报 3-seed 均值±区间，不以单 seed 下结论。

---

## 阶段一 b — 球拍质量课程（仓库 Stage 2；对照消融）

目标：从 Stage-1 promoted checkpoint 起，25%→50%→75%→100% 逐档适应球拍惯量，保持挥拍质量；每档新 run_id + fresh optimizer + parent lineage。

消融轴（候选，来自仓库既有设计，**开跑前圈定实际组合，全因子 = 2×2 太贵则选主对角）**：

| 轴 | 取值 A | 取值 B | 依据 |
|---|---|---|---|
| 起点 | formal（promoted Stage-1） | early_start（未满训 Stage-1 早期 ckpt） | `experiments/stage1/racket_curriculum/forehand_clear_formal_early_start_*` 既有对比线 |
| Stage-1 父 arm | 胜出 PEASD arm（如 T3） | T0 基线 arm | 检验 PEASD 增益是否传递到带拍阶段 |

配置：`stage2_racket_v2/conf_fullbody_forehand_clear_racket_mass_{025,050,075,100}.yaml`（synergy_v4 变体暂不进本轮）。
注意：`experiments/stage1/racket_curriculum/forehand_clear_formal_early_start_remote7_gpu1_20260901_v1/` 的脚本引用 `experiment.env`、`server_local9.env`、`server_remote7.env`，**这三个 env 文件未随快照回传**——已加入迁移清单意识范围，回传后这套 pipeline 才能直接跑。

| # | 消融组 | 父 Stage-1 run | 档位 | run_id | 状态 | 步数 | 挥拍质量指标 | promotion | checkpoint | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|
| S2-1 | formal × 胜出arm | （Stage-1 定后填） | 25% | | 未开始 | | | | | |
| S2-2 | 〃 | 〃 | 50% | | 未开始 | | | | | |
| S2-3 | 〃 | 〃 | 75% | | 未开始 | | | | | |
| S2-4 | 〃 | 〃 | 100% | | 未开始 | | | | | |
| S2-5 | early_start × 胜出arm | | 25→100% | | 未开始 | | | | | 按档位续行 |
| S2-6 | formal × T0 | | 25→100% | | 未开始 | | | | | 按档位续行 |

结论规则：各消融组用**相同**的 25→50→75→100 档位序列、相同晋级判据；比较点是 100% 档的挥拍质量与所需总步数；单 seed 先行，胜出组合再补 seed。

---

## 阶段二 — 技能蒸馏（未开始）

S2-A direct lifecycle 与 S2-B/C/D/E latent family 尚无 run；启动顺序见 `docs/runbooks/peasd_implementation_guide.md` §4，记录格式沿用上表。

---

## 阶段三 — 来球击打（未正式开始）

现状（2026-09-05）：离线线已迭代到 v45（spec 待回传对账）；本机已完成 aero 出球目标校准（12→22 m/s @ up 0.84，见 `outputs/stage3_calibration/`）和 v46 统一 spec 草案（现名 `experiments/stage3/direct_residual/incoming_shuttle_hit_selected_correction_v31.yaml`，待改名对账）。

开始前置（见 `outputs/transfer_checklist_20260905.md` B–D 组）：v24c ckpt、冻结 Stage-2 base、feed bank、CEM 可达性产物、v31–v45 spec 与 eval 报告。

| # | 实验 | spec | 状态 | 备注 |
|---|---|---|---|---|
| S3-0 | v31–v45 对账 + v46 定稿 | selected_correction（→≥v46） | 等 spec 回传 | |
| S3-1 | 单 feed CEM 可达性（22 m/s 目标） | v46 | 未开始 | 通过标准：真实击中 + outgoing_z>0，最好过网 |
| S3-2 | v46 短训 + 64-feed held-out | v46 | 未开始 | 163,840 步一评，两连零正 z 即停 |

## 2026-09-07 本机 T0 seed1 启动准备

状态：未开始，等待本机 W&B 登录。选择物理 GPU 3、T0 seed1，80/20 数据与 800,010,240 步预算。
数据哈希、配置、runtime preflight 与 GPU 检查已通过；训练尚未启动。
详见 [本次实验记录](stage1/T0_SEED1_GPU3_20260907.md)。

## 2026-09-08：T0 seed1 已在物理 GPU 6 启动

W&B 在线 running，已上报 143,360 步；100 条轨迹均从本地加载，manifest 和 GPU/PID 已验收。
详见 [GPU 6 实验记录](stage1/T0_SEED1_GPU6_20260908.md)。20260907 的 GPU 3 准备方案未启动，现已替代。
