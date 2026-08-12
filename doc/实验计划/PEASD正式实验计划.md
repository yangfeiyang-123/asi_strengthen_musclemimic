# PEASD 正式实验计划

**研究主题**：Partial-EMG Anchored Synergy Distillation（PEASD）用于视频驱动肌肉骨骼羽毛球技能学习

**版本日期**：2026-08-11

**执行规范**：[`docs/peasd_implementation_guide.md`](../../docs/peasd_implementation_guide.md)

**运行合同**：仓库根目录 `AGENTS.md` 及 `scripts/run_fullbody_training.sh`

**状态原则**：代码、配置、dry-run、旧 checkpoint 和 plan 均不是实验结果；只有 immutable evidence、promotion、family gate 与 formal release 才是正式证据。

## 1. 研究目标与论文主张

### 1.1 问题锚点

人体视频能提供外部运动学参考，但不能唯一决定 354 个肌肉执行器的内部募集方式；reference-conditioned tracking teacher 又依赖未来动作，不能直接处理开放来球；直接在 354 维动作空间学习击球则探索困难。

本项目的核心路线是：

```text
视频/人体运动
  → full-354 生理锚定动作 teacher
  → 无未来 reference、无在线 EMG 的低维技能
  → frozen prior/decoder + LAB 自适应击球
```

### 1.2 Claim map

| ID | 类型 | 主张 | 最低可信证据 | 对应实验 |
|---|---|---|---|---|
| C1 | 主要主张 | 少量 sEMG 作为 measured-subspace anchor，能在不明显损害动作跟踪的前提下改善可测肌肉的生理一致性 | T0–T4、3 seeds；T3 相对 T0 的 M-channel 指标改善；T3 每个 seed 相对 T4 的真实 synergy loss 至少改善 5%；tracking/safety 不退化 | Stage1 matched family |
| C2 | 主要主张 | EMG-derived synergy 可作为训练期 privileged context 内化到 reference-free latent skill，部署时不需要未来 reference 或在线 EMG | 完整 S2-A；S2-B/C/D/E 共享同一 collection 和 architecture；S2-C 每 seed 优于 shuffled S2-D；prior-only closed loop 通过 | Stage2 direct + latent family |
| C3 | 支撑主张 | PEASD latent 对下游自适应击球具有价值，bounded right-arm residual 只提供受限补偿 | H1/H2/H3 各 3 seeds、九条独立 reachability lineage；H2 每 seed 优于 H1；H3 与 H2 分离比较 | Stage3 family |
| A1 | 反主张 | 改善仅来自普通正则，而非真实生理结构 | T4 phase-shift、S2-D shuffled context 不得与真实 treatment 等效 | Stage1/Stage2 负对照 |
| A2 | 反主张 | 改善只是模型容量、数据或架构选择不同 | 相同 source snapshot、split、预算、seeds、shared inputs 和 architecture lock | 全链路 provenance |

### 1.3 不允许写出的结论

- 不声称 15 路表面肌电恢复了 354 维真实肌肉激活。
- 不声称 NMF basis 是真实神经控制模块。
- 不把单 subject/session 的 P002 数据写成 population-level 生理结论。
- 不用 reward、frame 或 feed 数量冒充独立训练 seed。
- 不在 Stage3 真实 hit、正向出球、过网和落点证据产生前声称“已学会回球”。

## 2. 动作范围与优先级

| 优先级 | 动作 | Registry slug | 论文角色 | 正式终点 | 当前决定 |
|---:|---|---|---|---|---|
| P0 | 正手高远球 | `forehand_clear` | 主结果 | Stage1 → 完整 Stage2 → H1/H2/H3 Stage3 → formal release | 当前唯一主训练队列 |
| P1 | 中国跳 | `chinajump` | body-only/phase-free 泛化 | Stage1 → body-only Stage2 context family → formal release；Stage1R/S2-A/racket/Stage3 N/A | 等独占 GPU；旧 GPU 2 诊断不可替代正式 T0–T4 |
| P2 | 正手挑球 | `forehand_lift` | 全链路泛化候选 | 理论上同 Clear | Stage1 可规划；event bank、四级 racket-mass 和 Stage3 专属资产补齐前不进入下游 |

主论文先完成 Clear 的整条证据链。ChinaJump 用于证明框架可扩展到 body-only 动作，但不能复制 Clear 的 racket/Stage3 资产。Lift 在动作专属资产齐备前不占用主线训练预算。

## 3. 已冻结的数据与 EMG 合同

### 3.1 Retarget release

| 动作 | Train/Val motions | Cache 状态 | Phase 语义 |
|---|---:|---|---|
| Clear | 22/5 | 已落盘并通过数值 QC | software movement cue 到 recording stop 的 exploratory 归一化，不声称独立 impact 对齐 |
| Lift | 12/4 | 已落盘并通过数值 QC | duration-normalized |
| ChinaJump | 8/2 | 已落盘并通过数值 QC；release 仍含 legacy evidence limitation | duration-normalized / phase-free latent |

所有 split 都以 motion/trial 为单位，不按 frame 随机划分。WHAM/AMASS、GMR、训练与可视化的真实时间/FPS 必须保持一致；新 smoothing 或 mapping 不覆盖 baseline namespace。

### 3.2 EMG 观测范围

- 原始采集保留 16 通道。
- S1 右斜方肌上束没有经核验的同源 MyoFullBody actuator，因此正式比较、tube、NMF 和 privileged context 使用 15 个 comparable channels。
- 数学合同是名称安全的 `M←354` observation projection，不是 `M→354` 恢复。
- 用户已逐项完成人工 mapping、trial/channel 和 S9 审查，并对决定负责；正式 bundle 仍以文件哈希和 gate 为准。

### 3.3 Verified tube

| 动作 | Verified manifest | Gate |
|---|---|---|
| Clear | `artifacts/emg_human_review_v2/verified_tubes/forehand_high_clear/emg_reference_manifest.json` | `artifacts/forehand_clear_peasd_v1/stage1_family/stage1_peasd/verified_tube_gate.json` |
| Lift | `artifacts/emg_human_review_v2/verified_tubes/forehand_lift_footwork/emg_reference_manifest.json` | `artifacts/forehand_lift_peasd_v1/stage1_family/stage1_peasd/verified_tube_gate.json` |
| ChinaJump | `artifacts/emg_human_review_v2/verified_tubes/china_jump_high_clear/emg_reference_manifest.json` | `artifacts/chinajump_peasd_v1/stage1_family/stage1_peasd/verified_tube_gate.json` |

超过 MVC 不构成阻塞。严格遵循 `doc/MVC小于动作信号时如何处理.md`：

1. audit track 永久保存未截断的 `%MVC`，允许大于 1；
2. model track 使用仅由 clean training trials 估计并冻结的 per-channel train-P99；
3. 不 clip 到 `[0,1]`，不因 super-MVC 自动删除 trial/channel；
4. NaN/Inf、负包络、平线、零 train-P99、功率线污染等质量失败仍 fail-closed；
5. 每个动作独立拟合 basis，不共享 Clear 的尺度或 basis。

## 4. 正式运行合同

### 4.1 一张物理 GPU 一个正式训练进程

正式实验不在同一 GPU 并发多个 PPO 任务。每个训练 step 必须：

- 从仓库根目录启动；
- 使用 `scripts/run_fullbody_training.sh`；
- 显式设置一个物理 `CUDA_VISIBLE_DEVICES`；
- 使用独立 `MUSCLEMIMIC_JAX_CACHE_KEY`；
- 使用独立 append-only `MUSCLEMIMIC_TRAIN_LOG`；
- Orbax save/restore 并发上限保持 4 GB；
- 使用独立 run id、W&B run 和 checkpoint root。

启动前必须通过 focused tests、Hydra resolved config、GPU 独占检查；启动后必须确认本地 retarget cache、manifest、W&B URL、正确物理 GPU PID、`Starting training...` 和无 fatal traceback。

### 4.2 当前 Stage1 source snapshot 冻结

当前 Clear T0 seed 0 已绑定：

| 字段 | 值 |
|---|---|
| git SHA | `103f0b1538ff` |
| source-tree fingerprint | `5882a7b35f663911281419435922b5577efdb0546171687a6dcc432c2d37c45a` |
| config hash | `263024238126` |

在 Clear 的 15 份 Stage1 evidence index 封存前：

- 禁止修改 `fullbody/`、`musclemimic/`、`scripts/`、`configs/`、`analysis/latent_synergy/`、`environment/overall_environment/src/`、`experiments/`、`pyproject.toml` 或 `uv.lock`；
- 文档与运行跟踪表可以更新，但不得改变训练源快照；
- 如果 fingerprint 发生变化，当前 T0 seed 0 不能与后续 run 组成 matched family，必须在新快照下重跑完整 Stage1 family。

### 4.3 Fixed endpoint

所有 Stage1 T0–T4 run：

- `auto_resume=false`；
- 不恢复旧 checkpoint；
- fresh optimizer；
- `promotion.auto_stop=false`；
- 不用训练曲线中的“最佳 validation”替代 fixed-budget endpoint；
- endpoint、checkpoint identity、validation evidence 和 source snapshot 必须封存。

## 5. 当前运行盘点（2026-08-11 12:30 CST）

### 5.1 正式主线

| 项目 | 状态 |
|---|---|
| Run | `forehand_clear_stage1_peasd_lite_v1_t0_s0` |
| 任务 | Clear Stage1 T0 seed 0，tube-free tracking baseline |
| GPU | 物理 GPU 1，PID `612097` |
| W&B | `vsqhlyf9` |
| 预算 | 320M timesteps，15,625 updates |
| 最近封存进度 | checkpoint 3904，79,953,920 timesteps，约 25.0%（2026-08-11 10:56） |
| 预计总时长 | 约 50–55 GPU-hours；不中断时预计 2026-08-13 凌晨附近结束 |
| 状态 | RUNNING；不是完成结果 |

### 5.2 历史诊断任务

GPU 2 上的 `chinajump_root_control_v2_b0cd_early_synergy_bootstrap_contdiag_excitation_v3` 是旧的 experimental/bootstrap-only fixed-synergy continuity diagnostics：

- 640M timesteps；最近封存 339,804,160，约 53.1%；
- continuity coefficient 为 0，只记录 provisional fascicle diagnostics；
- 启动参数包含 `allow_nonproduction_runtime=true`；
- 不属于 ChinaJump PEASD T0–T4，不能用于正式 promotion 或 formal release；
- 最近一次 validation video 有 primitive source model hash mismatch；PPO 仍在运行。

是否继续它是资源决策，不改变正式计划。若停止，必须通过其 tmux pane 发送一次 Ctrl-C，等待 PID/CUDA context 消失并保留 finalized checkpoint。

## 6. 实验块

### Block 0：数据、EMG 与训练 ABI

- **Claim tested**：所有后续差异来自 treatment，而不是错 split、错 FPS、错动作、错 mapping 或错 tube。
- **数据/任务**：Clear 22/5、Lift 12/4、ChinaJump 8/2；15 comparable EMG channels。
- **比较**：不比较方法，只验证 release、QC、tube、source binding。
- **指标**：motion 数、FPS、cache existence、hard errors、mapping/review/tube hashes、P99/MVC audit、train-P99 model scale。
- **成功条件**：action-specific QC 和 tube gate 通过；所有 consumer 可重建哈希。
- **失败解释**：停止训练；修复数据或证据，不调 PPO。
- **论文位置**：Methods/Data 与 Appendix provenance。
- **优先级**：MUST-RUN；当前已完成核心人工 review/tube，运行前仍逐次 revalidate。

### Block 1：Stage1 PEASD-Lite matched family

- **Claim tested**：C1、A1、A2。
- **动作**：Clear 主结果；随后 ChinaJump/Lift 做动作泛化。
- **系统**：

| Arm | Activation anchor | Synergy anchor | Treatment |
|---|---:|---:|---|
| T0 | 0 | 0 | tube-free baseline；endpoint post-hoc physiology |
| T1 | 0.02 | 0 | activation-only |
| T2 | 0 | 0.05 | real synergy-only |
| T3 | 0.02 | 0.05 | real PEASD-Lite |
| T4 | 0.02 | 0.05 | synergy phase 循环平移 10/20 bins |

- **统一设置**：seeds `0/1/2`；active treatment update 1000 开始、4000 updates ramp；同 source snapshot、split、预算和 validation schedule。
- **决定性指标**：M-channel anchor loss/correlation、peak phase、onset/offset、co-contraction；joint/root/site tracking；frame coverage；fall/early termination；action/activation rate、energy 和 saturation；real-reference synergy loss。
- **成功条件**：
  - 每个 seed 的 T3 real-synergy loss 相对 T4 至少改善 5%；
  - 每个 seed 及 aggregate 的 T3 anchor loss 严格优于 T0；
  - 五项 measured-activation 指标 non-degraded；
  - tracking/safety/effort 通过绝对门和相对 guardrail；
  - T3 seed 0 的 opaque blind review 通过。
- **失败解释**：T3 不优于 T4 时停止 Stage2，检查 phase、mapping、tube 与 reward delivery；不得事后改阈值。
- **论文目标**：主表 1（tracking + physiology）、图 1（T0/T3/T4 M-channel 时序）、附录 T1/T2 decomposition。
- **优先级**：MUST-RUN。

### Block 2：Stage2 reference-free direct + latent family

- **Claim tested**：C2、A1、A2。
- **前置**：Stage1 T3 seed 0 promotion 已通过。
- **Clear shared build**：T3 teacher → Stage1R 003/005 → event bank → racket mass `025→050→075→100` → train/val physical collection 一次 → physical QC/gate → common direct comparator → basis/decoder seal。
- **S2-A**：每 seed 独立 train 派生目录；BC → 3 轮 DAgger → fresh-optimizer PPO → held-out compare → seal；seeds 0/1/2；family promotion 后才能进入 latent family。
- **Latent systems**：

| Arm | 作用 |
|---|---|
| S2-B | non-EMG latent baseline；只在这里选择一次 architecture |
| S2-C | real privileged EMG synergy context |
| S2-D | shuffled context 负对照 |
| S2-E | real context、无 context dropout |

- **公平性**：B/C/D/E 绑定同一 `stage2_shared_inputs_v1`、同一 S2-A promotion、同一 architecture lock、seeds 0/1/2；C/D 只能差 shuffle treatment。
- **指标**：BC/DAgger/PPO action MSE 与 closed-loop；prior/posterior gap；active dims；sigma clamp；decoder saturation；synergy-head loss/correlation；blank/shuffled-context response；physiology degradation；prior-only tracking。
- **成功条件**：S2-A 三 seed promotion；C/E 每 seed 对 blank context 有正响应；paired seed 的 `S2-D head loss - S2-C head loss > 0` 对 3/3 seeds 成立；family gate 通过。
- **失败解释**：S2-C 不优于 S2-D 时，停止 Stage3 PEASD downstream claim；先检查 context delivery、head、dropout 与 dataset lineage。
- **论文目标**：主表 2（direct/latent）、图 2（prior/posterior 与 context response）、附录 latent dimension/decoder 选择。
- **优先级**：MUST-RUN。

### Block 3：Stage3 reachability-proven H1/H2/H3 family

- **Claim tested**：C3、A2。
- **前置**：Stage2 context-family gate 通过；Stage3 spec/scene/feed/target 均是 Clear action-specific 资产。
- **系统**：

| Arm | Low-level skill | Task adjustment |
|---|---|---|
| H1 | S2-B non-EMG latent | LAB，residual disabled |
| H2 | S2-C PEASD latent | LAB，residual disabled |
| H3 | S2-C PEASD latent | LAB + grouped right-arm bounded residual，alpha ≤ 0.10 |

- **九个独立叶节点**：H1/H2/H3 × seeds 0/1/2。每个叶节点独立保存 source、CEM、CPU audit、cross-backend seal、correction dataset、short BC、reachability release、C3/C7 checkpoint 与 128-feed report。
- **严格顺序**：single-feed CEM → CPU audit → cross-backend seal → successful correction dataset → zero-PPO short BC → release → C3 → C4–C7 → held-out evaluation。
- **指标**：hit、positive outgoing-z、cross-net、legal landing、opponent-back landing、impact-position error、net clearance、recovery-complete、no-fall、control energy、完整 128-feed coverage。
- **成功条件**：
  - reachability 有真实 stringbed contact 与正 outgoing-z；
  - H2 相对 H1 的 opponent-back landing rate 每 seed 与均值均严格提高，hit/no-fall 不退化；
  - H3 相对 H2 的 impact-position error 每 seed与均值均严格降低，hit/no-fall/opponent-back 不退化。
- **失败解释**：reachability 失败时不增加 PPO 步数；H2 不优于 H1 时不声称 PEASD downstream utility；H3 不优于 H2 时保留 H2，不调大 residual alpha。
- **论文目标**：主表 3（H1/H2/H3）、图 3（真实接触/轨迹/落点）、失败案例图。
- **优先级**：MUST-RUN for Clear；ChinaJump N/A。

### Block 4：Formal release 与论文完整性

- **Claim tested**：整条结果是否可从 immutable source 重建，而不是手工汇总。
- **输入**：Stage1 promotion、Stage2 family gate、适用时 Stage3 family gate、complete evaluation evidence。
- **成功条件**：`peasd_formal_release build` 与 `validate` 均通过；所有 required metric source-bound、自绑定、有限值、非 dry-run/placeholder；action/lineage/hash 全部一致。
- **附加论文实验**：random context、leave-channel-out、更多 qualitative/failure analysis 属于论文完整性或 appendix；在核心 formal family 未通过前不抢占主线 GPU。
- **优先级**：Formal release MUST-RUN；额外 appendix 分析 NICE-TO-HAVE，但若论文声称跨通道迁移则 leave-channel-out 升为 MUST-RUN。

## 7. 执行顺序与里程碑

| Milestone | 目标 | 运行 | Go/Stop gate | 预计成本 | 状态 |
|---|---|---|---|---:|---|
| M0 | 数据、review、tube、dry-run | 三动作 data QC/release、verified tube gate、focused tests | 所有 hard errors 为 0；bundle 可重建 | CPU | 核心完成 |
| M1 | Clear Stage1 baseline | T0 seeds 0/1/2 + 三个 post-hoc physiology | 三个 fixed endpoint/evidence 完整且 source fingerprint 一致 | 约 150–165 GPU-h | RUNNING |
| M2 | Clear Stage1 treatment family | T1/T2/T3/T4 × seeds 0/1/2 | 15 evidence + blind review + pairwise gate | 约 600–660 GPU-h | TODO |
| M3 | Clear Stage2 shared + S2-A | 只收集一次 shared；BC→3×DAgger→fresh PPO × 3 | shared seal 与 direct family promotion | 首次运行后校准 | BLOCKED by M2 |
| M4 | Clear latent family | S2-B lock；S2-C/D/E × 3 | context-family gate | 首次运行后校准 | BLOCKED by M3 |
| M5 | Clear Stage3 family | 九个 reachability/C3–C7/eval 叶节点 | Stage3 family gate | 9 个 30M PPO 叶 + reachability；首次叶后校准 | BLOCKED by M4 |
| M6 | Clear formal release | complete evidence + build/validate | 可从源 artifact 全量重建 | CPU | BLOCKED by M5 |
| M7 | ChinaJump 泛化 | Stage1 15 runs → body-only latent B/C/D/E | China Stage1/Stage2 family gates；Stage3 显式 N/A | 首个正式 T0 后校准 | QUEUED |
| M8 | Lift 泛化候选 | Stage1；补 event/mass/Stage3 资产后再继续 | 动作专属资产 gate | 未估 | ASSET-BLOCKED |

### 7.1 Clear Stage1 的不可变顺序

```text
T0 seed 0（当前）
→ T0 seed 1
→ T0 seed 2
→ 复核 verified tube gate
→ T0 seed 0/1/2 post-hoc physiology
→ T1 seeds 0/1/2
→ T2 seeds 0/1/2
→ T3 seeds 0/1/2
→ T4 seeds 0/1/2
→ 15-leaf evidence index
→ T3 seed-0 opaque blind review
→ pairwise gate
→ T3 seed-0 teacher promotion
```

不能在 T0 seed 0 后直接跳到 T3，也不能根据结果挑选 deployment seed。预注册 teacher 固定为 T3 seed 0。

### 7.2 当前 run 完成后的五个动作

1. 确认 update 15,625 / 320M endpoint、最终 independent validation、checkpoint completion 与 sealed evidence 全部存在。
2. 记录 exact immutable endpoint leaf；不得把 `policy_latest.json` 或中途最佳 checkpoint 当 endpoint。
3. 不修改 source snapshot，使用新 run id、fresh optimizer 启动 `stage1_peasd_t0_s1_train`。
4. T0 seed 1 完成后同样启动 seed 2。
5. 三个 T0 完成后，用同一 verified tube 分别运行三个 post-hoc physiology evaluator。

## 8. GPU 与时间预算

### 8.1 资源原则

- 一个物理 GPU 同时只运行一个正式 PPO 任务。
- seeds/arms 可以分配到不同的空闲独占 GPU 并行，但不能与其他用户任务共享。
- 每个并行 run 保持独立 tmux socket/session、cache key、log、run id 和 checkpoint root。
- GPU 是否空闲以启动前 `nvidia-smi` 为准；显存能装下不等于适合并发。

### 8.2 Clear Stage1 实测预算

当前 T0 seed 0 的首段吞吐给出约 50–55 小时/320M run：

- 15 个 Stage1 run 总计约 750–825 GPU-hours；
- 按 2026-08-11 中午进度，当前 run 尚余约 37–41 小时，另外 14 个完整 run 尚余约 700–770 GPU-hours；
- 单 GPU 串行约还需 31–34 天；
- 3 张真正空闲且独占的同型 GPU 理论上约 10–12 天，实际取决于编译、validation video 和 checkpoint 开销。

这只是 Stage1 预算。Stage2 与 Stage3 在第一个正式 run/leaf 完成前不编造时长；届时用真实 update throughput 和 gate 开销更新 tracker。

### 8.3 当前 GPU 决策

- GPU 1：Clear T0 seed 0 独占运行。
- GPU 2：旧 ChinaJump diagnostics，且存在其他用户进程；不是正式 PEASD 资源。
- GPU 0/3：状态动态；只有空闲、显存占用低且无其他用户进程时才能加入正式队列。
- 不在 GPU 1 同时启动四个 320M 任务；这通常不会缩短总 makespan，并会破坏正式资源合同。

## 9. 指标与统计协议

### 9.1 必报指标

- **动作**：joint/keypoint/root/racket error、frame coverage、完整动作成功、fall/early termination。
- **控制与肌肉**：activation energy、action/activation saturation、action/activation rate、M-channel anchor loss/correlation、peak phase、onset/offset、co-contraction。
- **协同**：held-out/per-channel VAF、W cosine、subspace angle、H correlation（仅数据设计允许时）、bootstrap stability。
- **Direct distill**：train/val action MSE、三轮 DAgger convergence、BC/DAgger/PPO closed loop、teacher/student return、fall rate、physiology degradation。
- **Latent**：action reconstruction、prior/posterior gap、active dims、sigma clamp、decoder saturation、synergy-head loss/correlation、blank/shuffled response、prior-only closed loop。
- **Stage3**：hit、positive outgoing-z、cross-net、legal return、net clearance、opponent-back landing、impact-position error、no-fall、recovery、control energy、held-out coverage。

### 9.2 统计单位

- RL：independent training seed，`n=3`。
- 人体：trial/subject/session；P002 不能提供 population-level 结论。
- frame、environment、episode 和 feed 都是 repeated measurements，不增加独立 `n`。
- paired 设计只在同 split、同 architecture、同 source/lineage 下成立。
- 报告每 seed 原始值、均值、sample SD、failure count、Cohen's dz 和 df=2 描述区间；不声称显著性或 population effect。

## 10. 人工审查节点

| 节点 | 审查对象 | 审查人必须看到 | 审查人不能看到/不能做 |
|---|---|---|---|
| EMG mapping/trial QC | mapping、波形、S9 chronology、trial/channel evidence | 完整波形、SHA-bound QC、通道与 trial 决策 | 不得让脚本/Codex代签；已由用户完成 |
| Stage1 T3 seed-0 blind review | opaque video package | 匿名 clip id 和视频；逐项完整动作、root/hand spike、pass、notes | 不看 private arm/seed/checkpoint mapping；不以训练曲线代替视频 |
| Racket mass 025/050/075/100 | 每一 rung 的真人视频 | 当前 rung 的 tracking、手/拍连接、稳定性 | 不复用上一 rung review；不跳过中间质量 |
| Final result audit | complete evidence 与源报告 | 每项指标 provenance、失败/缺失状态 | 不手写 placeholder 或把 dry-run 当结果 |

## 11. Stop/Go 决策树

```text
Stage1 T3 未优于 T4，或 measured physiology 未优于 T0
  → STOP Stage2
  → 查 phase/mapping/tube/reward delivery

Stage2 S2-C 未优于 S2-D
  → STOP PEASD Stage3 claim
  → 查 context/head/dropout/dataset lineage

Stage3 reachability 无真实 contact 或正 outgoing-z
  → STOP long PPO
  → 查 feed、grip、authority、拍面/惯量

C3 gate 未通过
  → 禁止 C4–C7

H2 未优于 H1
  → 不声称 PEASD latent downstream utility

H3 未优于 H2
  → 保留 H2，不事后增大 residual alpha
```

任何 gate 失败都保留负结果和 immutable evidence；不得看到结果后调整冻结阈值再重算同一实验。

## 12. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| Dirty worktree 在 15 个 run 之间变化 | matched source 失效，已有 run 必须重跑 | 维持当前 source fingerprint；文档更新不触碰受控源码范围 |
| 多任务共享 GPU | 训练变慢、OOM、资源干扰 | 一 GPU 一正式进程；只在独占卡并行 |
| 旧 checkpoint 混入新 lineage | 无法审计或错误 promotion | 只接受 exact immutable leaf 与 content hash；旧 China diagnostics 明确隔离 |
| T3/T4 没有差异 | 生理主张不成立 | 先停在 Stage1，查数据与 treatment delivery，不把它解释为“训练不够” |
| Stage2 各臂重复收集数据 | C/D 不再 matched | shared inputs 只 seal 一次；B/C/D/E 消费同一 hash |
| Stage3 reward 上升但无真实事件 | 伪成功 | reachability 和最终 gate 只认 contact/outgoing-z/cross-net/landing |
| 人工 review 泄露身份 | blind evidence 无效 | reviewer package 与 private mapping 分离，reviewer 不访问后者 |
| Lift 复用 Clear 资产 | 动作身份错误 | registry exact path/action checks；资产未齐就 fail-closed |

## 13. 论文结果组织

### Main paper

1. Table 1：Clear Stage1 T0/T3/T4 tracking、M-channel physiology 与 safety。
2. Figure 1：真实 vs shifted EMG anchor/synergy 时序和代表性动作。
3. Table 2：S2-A、S2-B/C/D/E 的 reference-free tracking、context response、prior-only closed loop。
4. Figure 2：latent active dimensions、prior/posterior gap、blank/shuffled response。
5. Table 3：Stage3 H1/H2/H3 seed-level真实事件结果。
6. Figure 3：single-feed reachability、击球轨迹、落点与失败案例。

### Appendix

- T1/T2 decomposition；
- per-channel physiology；
- synergy basis stability；
- DAgger 三轮收敛；
- architecture selection 的预注册候选；
- ChinaJump body-only 泛化；
- random context、leave-channel-out 和更多失败分析。

### 暂不进入主故事

- fixed-synergy action space 主方法化；
- Graph-NMF/continuity 多组合矩阵；
- 大量旧 Stage3 reward/config 探索；
- GPU 2 的 ChinaJump bootstrap diagnostics。

## 14. Definition of Done

### Stage1 Done

- [ ] T0–T4 × seeds 0/1/2 全部 fixed endpoint 完成。
- [ ] 15 份 validation evidence 可重建且 source snapshot 一致。
- [ ] T3 seed-0 opaque blind review 通过。
- [ ] T3-vs-T4 与 T3-vs-T0 gate 通过。
- [ ] T3 seed 0 teacher promotion 生成并验证。

### Stage2 Done

- [ ] 只存在一份 shared physical train/val lineage。
- [ ] S2-A BC、3×DAgger、fresh PPO × 3 seeds 完成并 promotion。
- [ ] S2-B architecture 只选择并锁定一次。
- [ ] S2-C/D/E 使用同一 shared/lock/seeds。
- [ ] reference-free prior-only runtime 不读取 future reference 或在线 EMG。
- [ ] Stage2 context-family gate 通过。

### Stage3 Done（Clear）

- [ ] 九份独立 reachability release 全部通过。
- [ ] 九份 C3→C7 training lineage 完整。
- [ ] 九份 128-feed evaluation 完整。
- [ ] H2-vs-H1 与 H3-vs-H2 gate 通过。

### Formal release Done

- [ ] complete evidence 的每个指标绑定真实 evaluator source。
- [ ] 无 dry-run、placeholder、failed 或 incomplete 标记。
- [ ] `peasd_formal_release build` 成功。
- [ ] `peasd_formal_release validate --expected-action` 成功。
- [ ] 论文结论不超过 formal release 支持的证据边界。

## 15. 立即执行清单

1. 继续监控当前 Clear T0 seed 0；不要修改受控源码/配置。
2. endpoint 完成后封存 exact leaf、manifest、W&B、validation evidence 和实际总时长。
3. 在空闲独占 GPU 上启动 Clear T0 seed 1；随后 seed 2。
4. 三个 T0 完成后运行 post-hoc physiology；不要提前跳到 T3。
5. 每次状态变化更新 [`实验运行跟踪表.md`](实验运行跟踪表.md)，但不要手写实验指标。
6. 所有 production step 的精确 CLI、依赖参数和人工暂停点以实施指南相应章节为准；本计划不复制内部 trainer 命令来绕过 canonical launcher。
