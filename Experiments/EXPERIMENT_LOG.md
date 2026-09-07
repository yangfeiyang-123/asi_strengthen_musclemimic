# Forehand Clear 实验记录表

维护规则：append-only，一行一个 run；结果必须同时记 commit、config hash（resolved config 的 `run_manifest.json`）、seed、checkpoint 路径与 W&B run；失败 run 不删行，状态改"失败"并在备注写原因。状态取值：`未开始 / 训练中 / 完成 / 失败 / 已晋级`。

---

## Stage 1 — 身体轨迹模仿（PEASD 对照，3 seed × 5 组 = 15 runs）

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

已知线索：离线 `Experiments/stage1/` 脚本（`extract_stage1_320m_metrics.py`、`compare_arms_s2.py`）显示 T2/T3/T4 存在 320M 步 run，但其数据组（80/20 还是 40/10）待回传核对——若是 80/20 则把对应行补成实际状态，不要当新实验重跑。

| # | 组 | seed | run_id | config（80/20 版） | 状态 | 服务器/GPU | 步数 | val tracking err | no-fall | PEASD 诊断 | checkpoint | W&B | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | T0 | s0 | forehand_clear_aug100_stage1_peasd_lite_v1_t0_s0 | （待创建，照 t1 派生） | 未开始 | | | | | | | | |
| 2 | T0 | s1 | …_t0_s1 | 同上 + seed override | 未开始 | | | | | | | | |
| 3 | T0 | s2 | …_t0_s2 | 同上 | 未开始 | | | | | | | | |
| 4 | T1 | s0 | …_t1_s0 | peasd_lite_v1/conf_fullbody_forehand_clear_aug100_peasd_t1.yaml | 未开始 | | | | | | | | |
| 5 | T1 | s1 | …_t1_s1 | 同上 + seed override | 未开始 | | | | | | | | |
| 6 | T1 | s2 | …_t1_s2 | 同上 | 未开始 | | | | | | | | |
| 7 | T2 | s0 | …_t2_s0 | （待创建，照 t1 派生） | 待核对（离线 320M run 数据组未明） | | | | | | | | |
| 8 | T2 | s1 | …_t2_s1 | 同上 | 未开始 | | | | | | | | |
| 9 | T2 | s2 | …_t2_s2 | 同上 | 未开始 | | | | | | | | |
| 10 | T3 | s0 | …_t3_s0 | （待创建，照 t1 派生） | 待核对（同上） | | | | | | | | |
| 11 | T3 | s1 | …_t3_s1 | 同上 | 未开始 | | | | | | | | |
| 12 | T3 | s2 | …_t3_s2 | 同上 | 未开始 | | | | | | | | |
| 13 | T4 | s0 | …_t4_s0 | （待创建，照 t1 派生） | 待核对（同上） | | | | | | | | |
| 14 | T4 | s1 | …_t4_s1 | 同上 | 未开始 | | | | | | | | |
| 15 | T4 | s2 | …_t4_s2 | 同上 | 未开始 | | | | | | | | |

结论规则（预注册，避免事后挑指标）：
- 主指标：held-out 20 条 validation 的 tracking 指标 + no-fall + promotion gate 通过与否（80/20 配置要求 20 条视觉验收 clip）；
- PEASD 是否"有效"：T3 相对 T0 的改善必须显著大于 T4 相对 T0 的改善（负对照校准），且跨 3 seed 方向一致；
- T1/T2 用于归因（anchor vs synergy 各自贡献）；
- 每组报 3-seed 均值±区间，不以单 seed 下结论。

---

## Stage 2 — 球拍质量课程（对照消融）

目标：从 Stage-1 promoted checkpoint 起，25%→50%→75%→100% 逐档适应球拍惯量，保持挥拍质量；每档新 run_id + fresh optimizer + parent lineage。

消融轴（候选，来自仓库既有设计，**开跑前圈定实际组合，全因子 = 2×2 太贵则选主对角）**：

| 轴 | 取值 A | 取值 B | 依据 |
|---|---|---|---|
| 起点 | formal（promoted Stage-1） | early_start（未满训 Stage-1 早期 ckpt） | `Experiments/stage2/forehand_clear_formal_early_start_*` 既有对比线 |
| Stage-1 父 arm | 胜出 PEASD arm（如 T3） | T0 基线 arm | 检验 PEASD 增益是否传递到带拍阶段 |

配置：`stage2_racket_v2/conf_fullbody_forehand_clear_racket_mass_{025,050,075,100}.yaml`（synergy_v4 变体暂不进本轮）。
注意：`Experiments/stage2/forehand_clear_formal_early_start_remote7_gpu1_20260901_v1/` 的脚本引用 `experiment.env`、`server_local9.env`、`server_remote7.env`，**这三个 env 文件未随快照回传**——已加入迁移清单意识范围，回传后这套 pipeline 才能直接跑。

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

## Stage 3 — 来球击打（未正式开始）

现状（2026-09-05）：离线线已迭代到 v45（spec 待回传对账）；本机已完成 aero 出球目标校准（12→22 m/s @ up 0.84，见 `outputs/stage3_calibration/`）和 v46 统一 spec 草案（现名 `experiments/posttrain/incoming_shuttle_hit_selected_correction_v31.yaml`，待改名对账）。

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
