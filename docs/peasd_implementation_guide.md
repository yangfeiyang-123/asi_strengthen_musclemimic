# PEASD 实施指南（以证据门为中心的正式执行顺序）

> 本文是 `doc/整体故事框架与思路/` 三份文档在当前仓库中的可执行落地版。主线固定为：
>
> **Full-354 physiological tracking teacher → reference-free PEASD latent skill → LAB hitting**。
>
> 本文严格区分“代码/合同已经实现”和“正式实验已经完成”。截至 2026-08-10，仓库已具备
> 下述 fail-closed 接口和测试，但没有在本轮启动正式 GPU 训练，也没有产生可写入论文的
> Stage1/Stage2/Stage3 新结果。任何人工 review、promotion 或实验数字都不得由脚本或 Codex 代签。

## 0. 唯一正式路线

正手高远球主结果必须按以下顺序推进，不能跳步：

```text
动作 release / 数值与视频 QC
  → 人工 verified 的 action-specific EMG tube
  → Stage1 T0/T1/T2/T3/T4 × seeds 0/1/2
  → T3-vs-T4 paired gate + T3/seed-0 opaque blind review
  → T3 teacher promotion
  → 一次且仅一次的 Stage2 physical train/val collection + basis/decoder seal
  → S2-A：BC → 3 轮 DAgger → fresh-optimizer PPO，seeds 0/1/2 → family promotion
  → S2-B：选择一次 latent architecture 并锁定
  → S2-C/S2-D/S2-E：同一 shared inputs、同一锁定 architecture、seeds 0/1/2
  → Stage2 context-family gate
  → H1/H2/H3 各自 seeds 0/1/2：
      独立 reachability source → CEM → CPU audit → cross-backend seal
      → successful-correction dataset → zero-PPO short BC → C3 → C4–C7
      → 128-feed held-out evaluation
  → Stage3 H1/H2/H3 family gate
  → complete evaluation evidence → formal release build + rebuild/validate
```

这里的 `S2-A` 是完整 direct lifecycle，不是旧的单 seed BC comparator；`S2-B/C/D/E` 是
latent context family。两者都完成才构成 Stage2。Stage3 的九个叶节点
`H1/H2/H3 × seed 0/1/2` 必须有九份互不混用的 reachability release 和训练根目录。

### 0.1 动作定位与适用终点

| 动作 | registry slug | 论文定位 | 正确终点 |
|---|---|---|---|
| 正手高远球 | `forehand_clear` | 主结果 | Stage1 → 完整 Stage2 → H1/H2/H3 Stage3 |
| 正手挑球 | `forehand_lift` | 全链路泛化候选 | 资产补齐后可同 Clear；当前止于 event/mass 校准前 |
| 中国跳 | `chinajump` | body-only 泛化 | Stage1 → phase-free latent family；S2-A/racket/Stage3 均为 N/A |

三动作共用 `musclemimic/badminton/action_registry.py`、同一 tube builder、同一 Stage1
profile 和同一 latent trainer。所谓 matched 只表示某个预注册对照家族内除 treatment 外的
数据、architecture、seeds、预算和 gate 一致；它不表示把 Clear 的动作资产、训练预算或
Stage3 spec 复制给 Lift/ChinaJump。动作专属配置与真实校准必须由 registry 显式提供。

### 0.2 当前真实就绪度

| 项目 | Clear | Lift | ChinaJump |
|---|---|---|---|
| train/val retarget cache | 22/5，已落盘 | 12/4，已落盘 | 8/2，已落盘 |
| data QC plan | 可通过 | 可通过 | 可通过 |
| acquired/comparable EMG | 16/15 | 16/15 | 16/15 |
| v2 tube 审计 | 可产 provisional；超 MVC 保留并分级 | 可产 provisional；超 MVC 保留并分级 | 可产 provisional；超 MVC 保留并分级 |
| Stage1 T0–T4 工程 | 已实现，未正式训练 | 已实现，未正式训练 | 已实现，未正式训练 |
| Stage2 完整 family 工程 | 已实现，未正式训练 | 接口存在；event/mass 资产阻塞 | body-only 接口存在；S2-A N/A |
| Stage3 family 工程 | 已实现，未正式训练 | 缺动作专属 spec/target/feed | N/A |

“plan 可生成”“单测通过”“provisional tube 可加载”都不是正式结果。旧 checkpoint、旧
promotion 或旧 Stage3 诊断也不能冒充这条新 lineage 的证据。

## 1. 不可绕过的运行与证据合同

### 1.1 production trainer 的唯一入口

所有 production training 都必须从仓库根目录经 `scripts/run_fullbody_training.sh` 启动。
pipeline 的 `--execute_step` 会把 FullBody、latent、BC、DAgger 和 incoming-hit trainer
自动改写到 canonical launcher；禁止从 `pipeline_plan.json` 复制内部 Python trainer 命令
直接运行。

每个训练 step 都要有独立、稳定的物理 GPU、cache key 和 append-only log：

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic
source configs/env.sh

export CUDA_VISIBLE_DEVICES=<one_physical_gpu_index>
export MUSCLEMIMIC_JAX_CACHE_KEY=<stable_task_specific_key>
export MUSCLEMIMIC_TRAIN_LOG=<append_only_log_path>
export MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB=4
export MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB=4
```

`CUDA_VISIBLE_DEVICES` 只能是一个物理 GPU 的非负整数。进程内部显示的 CUDA device 0
是该物理卡的可见索引，不代表物理 GPU 0。

`MUSCLEMIMIC_DRY_RUN=1` 只允许检查环境和解析命令。dry-run、plan、preflight、测试输出
均不得被写成 release、promotion、训练完成或实验结果。

每次生产启动前后都执行根目录 `AGENTS.md` 的检查：focused tests、Hydra resolved config、
唯一 run id、固定预算、reward/termination/promotion 合同、`nvidia-smi`；启动后还要看到所有
轨迹来自本地 retarget cache、run manifest、W&B id/URL、目标物理 GPU PID、
`Starting training...` 且无 fatal traceback。若出现 Hugging Face 下载，说明环境没有正确
绑定，必须停止该 run。

### 1.2 每次 `--execute_step` 都重建 plan

`run_forehand_clear_pipeline` 不保存上一次 CLI 参数。执行后续 step 时必须再次完整传入：

- `--profile`、`--action`、`--output_dir`；
- 该 arm 的 treatment；
- 所有已经产生且该 step 依赖的 checkpoint、metrics、review、promotion、fingerprint；
- shared inputs、architecture lock、family promotion 等跨根 lineage。

不能第一次 plan 时传 tube，后续只写 `--execute_step`；那会重建出另一条 plan。实际 step
名称以本次生成的 `pipeline_plan.json` 为准。本文只列当前代码中真实存在的 step 名。

### 1.3 不兼容合同必须 fresh optimizer

Stage1 T0–T4 都是新 run：`auto_resume=false`、不 resume 旧 checkpoint、fresh optimizer、
固定预算、`promotion.auto_stop=false`。改变 reward、termination、tube、mapping、split、
architecture 或 treatment 后必须使用新 run id；不得恢复不兼容 optimizer。

## 2. 数据 release、人工 QC 与 verified tube

### 2.1 先核对动作 release 和 retarget 数据

```bash
source configs/env.sh
for action in forehand_clear forehand_lift chinajump; do
  uv run --locked python -m musclemimic.badminton.data_qc \
    --action "$action" --require-clean
  uv run --locked python -m musclemimic.badminton.action_release \
    --action "$action" --require-pass
done
```

这只证明 registry 声明的 source/cache namespace 与 ordered train/val clip 存在。它不证明
EMG 合法、Stage1 已训练或 action release 的证据等级相同。ChinaJump 当前仍是 legacy
release evidence：`formal_release_manifest=false`，不得声称与 Clear 的 structured release
同等级。

数据 release 遵循真实时间：WHAM/AMASS、GMR、训练 config 和渲染 FPS 必须一致；train/val
按 motion/trial 划分，不按 frame 随机切。新 smoothing 或 mapping 使用新 namespace，禁止
覆盖已知 baseline cache。

### 2.2 16 acquired / 15 comparable

P002 原始采集保留全部 16 通道。S1 右斜方肌上束在当前 MyoFullBody 354 actuator 中没有
经核验的同源肌，因此正式比较、tube、NMF 和 privileged context 使用 15 个 comparable
channels；S1 原始数据不删除，也不伪映射到 DELT/LAT。

数学主张始终是名称安全的 `M←354` observation projection：少量 sEMG 是 measured-subspace
anchor，不是 354 维肌肉真值，不证明未测肌肉正确，也不证明 NMF 是真实神经模块。

### 2.3 provisional 审计构建

下面命令只写临时审计目录，不解冻训练：

```bash
AUDIT_TUBE_ROOT=$(mktemp -d)

uv run --locked python scripts/build_emg_reference_tube.py \
  --action forehand_high_clear --output-dir "$AUDIT_TUBE_ROOT"
uv run --locked python scripts/build_emg_reference_tube.py \
  --action forehand_lift_footwork --output-dir "$AUDIT_TUBE_ROOT"
uv run --locked python scripts/build_emg_reference_tube.py \
  --action china_jump_high_clear --output-dir "$AUDIT_TUBE_ROOT"
```

v2 tube 严格执行 `doc/MVC小于动作信号时如何处理.md` 的双轨合同：

1. audit track 永久保存未截断的 `percent_mvc_unclipped`；允许数值大于 1；
2. provisional 审计只在其明确绑定的 candidate training cohort 上估计 P99，仍禁止训练；
   verified tube 必须先完成人工 trial/channel QC，只用纳入的 clean training trial，并在
   phase binning 之前逐通道估计、冻结 P99；
3. model/synergy track 使用 `train_p99_per_channel`，不得把它再称为 `%MVC`；
4. `P99(task)/MVC` 按 `≤1.20 / 1.20–1.50 / 1.50–2.00 / >2.00` 标为
   `good / questionable / unreliable / invalid_for_absolute_amplitude`，幅值监督置信度分别为
   `1.0 / 0.7 / 0.4 / 0.2`；该等级不删除 trial/channel；
5. NaN/Inf、负包络、零/近零 train-P99、平线、功率线污染等信号质量失败仍 fail-closed。

2026-08-10 对当前 P002 三动作真实数据的临时 v2 重建结果均成功，且都为
`15 channels / 20 phase bins / rank 3 / training_enabled=false`：

- Clear：13 good、1 questionable、1 invalid-for-absolute-amplitude；最大 `R99=6.928058`
  （S2）；未截断 phase-tube 最大中心为 `3.639803×MVC`；
- Lift：12 good、2 questionable、1 unreliable；最大 `R99=1.538736`（S12）；
- ChinaJump：10 good、4 questionable、1 invalid-for-absolute-amplitude；最大 `R99=3.406098`
  （S2）；未截断 phase-tube 最大中心为 `1.841532×MVC`。

这些值表示“相对当前 MVC reference 的比例”，不能解释为人体真实最大激活的 693% 或 341%。
不得 clip 到 `[0,1]`、仅因超 MVC 删除 trial/channel，或把 train-P99 偷换名称为 MVC。旧
`emg_phase_reference_tube_v1` 产物不得作为 v2 证据；默认 builder 写入新的
`emg_reference_v2` namespace，不覆盖旧 tube。

Clear 的 phase 是 software `movement_cue → recording_stop` 的 exploratory 时间归一化，
不是独立视频/硬件确认的 impact。Lift/ChinaJump 是 duration-normalized phase。论文中必须
按 unpaired/action-cohort 限定表述，不能报告不存在的逐 trial impact timing 或 H correlation。

### 2.4 人工解冻条件

`--verified` 是校验开关，不是“把 provisional 改名为 verified”的开关。调用它之前必须真实
完成：

不熟悉 JSON schema 时，先按 `docs/emg_human_review_wizard.md` 运行
`scripts/review_emg_for_training.py prepare`，查看全部波形与 S9 session chronology，再用
`wizard` 逐项填写。向导允许中断续填，但不会替 reviewer 作决定或启动训练。

1. 解剖专家复核
   `configs/physiology/emg_badminton_synergy_16_v2_myofullbody_observation_v1.json`：左右侧、
   多 compartment 聚合、biceps/triceps、腕屈伸、躯干/下肢、权重与 exclusion；
2. mapping 写入真实 reviewer/evidence，所有 mapped channel 的 confidence 有明确等级，
   `review_status=verified`、`training_enabled=true`；
3. 每个动作单独完成 `emg_trial_channel_qc_review_v1`，trial 集合与磁盘精确相等；每个
   trial 决策都绑定 `mvc_normalized_emg.npz` 和 `preprocessing_qc.json` 的 SHA-256；
4. review 覆盖全部 15 comparable channels，且显式裁决
   `s9_progressive_near_flatline`；S9 不得通过批量阈值修改变成 valid。`super_mvc` 不再是
   人工 waiver 或训练阻塞项，其 P99/MVC 等级、train-P99 尺度与 amplitude confidence 由 v2
   manifest 自动计算并绑定；
5. reviewer id、时间、理由和 evidence 都来自真实人工复核，Codex 不得填写。

正式构建唯一入口：

```bash
uv run --locked python scripts/build_emg_reference_tube.py \
  --action <emg_trial_action> \
  --mapping <reviewed_mapping.json> \
  --verified \
  --trial-qc-review <action_specific_review.json> \
  --output-dir <fresh_release_root>
```

每个动作必须在自己的 train-P99 尺度上独立拟合 basis，不能共用 Clear basis。核心三产物
必须作为一个 bundle 保存：

1. `emg_reference_manifest.json`；
2. `emg_reference_tube.npz`（同时含 unclipped MVC audit 与 train-P99 model arrays）；
3. `emg_observation_mapping.json`。

verified bundle 还必须包含第四个审计文件 `emg_trial_qc_review.json`。manifest 精确绑定
NPZ、mapping 和 review 的字节哈希；只移动 manifest 或重新格式化 JSON 都会使 consumer
fail-closed。正式 Stage1 前再执行：

```bash
uv run --locked python -m musclemimic.badminton.stage1_peasd_gate tube \
  --action <registry_slug> \
  --tube <emg_reference_manifest.json> \
  --output <verified_tube_gate.json> \
  --require-pass
```

## 3. Stage1：T0–T4 matched PEASD-Lite family

### 3.1 冻结实验合同

| arm | activation anchor | synergy anchor | 额外 treatment |
|---|---:|---:|---|
| T0 | 0 | 0 | tube-free training baseline；只在 endpoint 做 post-hoc physiology |
| T1 | 0.02 | 0 | activation-only |
| T2 | 0 | 0.05 | real synergy-only |
| T3 | 0.02 | 0.05 | real PEASD-Lite |
| T4 | 0.02 | 0.05 | synergy phase 固定循环平移 10/20 bins |

所有 active arms 在 update 1000 开始、用 4000 updates ramp；seeds 固定为 0/1/2。T4 只平移
synergy lookup，不伪造 impact/event，也不打乱 activation anchor。T0–T4 必须使用同一个
source-tree snapshot、ordered split、release/QC、预算 endpoint 和 validation schedule。
开始 15 个 run 后直到 evidence index 封存前禁止改代码/config。

### 3.2 plan 与执行

先定义不会丢失关键参数的执行函数：

```bash
ACTION=forehand_clear
S1_ROOT=artifacts/forehand_clear_peasd_v1/stage1_family
TUBE=<absolute_verified_emg_reference_manifest.json>
GPU=<physical_gpu_index>

run_s1_step () {
  local step="$1"
  shift
  export CUDA_VISIBLE_DEVICES="$GPU"
  export MUSCLEMIMIC_JAX_CACHE_KEY="${ACTION}_${step}_v1"
  export MUSCLEMIMIC_TRAIN_LOG="${S1_ROOT}/logs/${step}.log"
  export MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB=4
  export MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB=4
  uv run --locked python -m fullbody.run_forehand_clear_pipeline \
    --profile stage1_peasd \
    --action "$ACTION" \
    --output_dir "$S1_ROOT" \
    "$@" \
    --execute_step "$step"
}

uv run --locked python -m fullbody.run_forehand_clear_pipeline \
  --profile stage1_peasd \
  --action "$ACTION" \
  --output_dir "$S1_ROOT"
```

先执行 `data_release_validate`、`data_qc`，再按以下真实 step 名运行：

```bash
run_s1_step data_release_validate
run_s1_step data_qc

for seed in 0 1 2; do
  run_s1_step "stage1_peasd_t0_s${seed}_train"
done

run_s1_step stage1_peasd_tube_gate \
  --emg_reference_manifest "$TUBE"

# 把三个变量设为各自 fixed-budget 的精确 immutable endpoint leaf。
T0_S0=<checkpoint_leaf>; T0_S1=<checkpoint_leaf>; T0_S2=<checkpoint_leaf>
run_s1_step stage1_peasd_t0_s0_posthoc_physiology \
  --emg_reference_manifest "$TUBE" \
  --stage1_peasd_t0_s0_checkpoint "$T0_S0"
run_s1_step stage1_peasd_t0_s1_posthoc_physiology \
  --emg_reference_manifest "$TUBE" \
  --stage1_peasd_t0_s1_checkpoint "$T0_S1"
run_s1_step stage1_peasd_t0_s2_posthoc_physiology \
  --emg_reference_manifest "$TUBE" \
  --stage1_peasd_t0_s2_checkpoint "$T0_S2"

for arm in t1 t2 t3 t4; do
  for seed in 0 1 2; do
    run_s1_step "stage1_peasd_${arm}_s${seed}_train" \
      --emg_reference_manifest "$TUBE"
  done
done
```

不要用训练曲线中“最好的一次 validation”替代 fixed-budget endpoint。runner/evaluator 会为
每个 endpoint 生成封存的 `stage1_peasd_validation_evidence_v1`；其中包含实际 delivery
treatment、最终 reward floor、checkpoint/config/source-tree/split/runtime/tube 绑定。

### 3.3 opaque blind review 与 promotion

T3/seed-0 是预注册 deployment teacher；不能根据 15 个结果再挑 seed。其 deterministic
held-out validation 会生成：

- reviewer-visible `stage1_blind_review_package/`：只含 opaque clip id、匿名视频与
  `review.json`；
- private `stage1_blind_private_mapping.json`：保存真实 checkpoint/motion 映射，严禁给 reviewer。

reviewer 只观看匿名 package，逐 clip 填
`major_swing_complete`、`root_tracking_spike_free`、`right_hand_tracking_spike_free`、
`passed`、notes，并填写真实 reviewer id。不得先看 private mapping，也不得由训练人员/Codex
自评。

收齐 15 份 evidence 后，完整列出所有 artifact。下面的数组不能缺项、不能用聚合 scalar
JSON 代替 sealed evidence：

```bash
S1_EVIDENCE_ARGS=(
  --stage1_peasd_t0_s0_validation_evidence "$T0_S0_EVIDENCE"
  --stage1_peasd_t0_s1_validation_evidence "$T0_S1_EVIDENCE"
  --stage1_peasd_t0_s2_validation_evidence "$T0_S2_EVIDENCE"
  --stage1_peasd_t1_s0_validation_evidence "$T1_S0_EVIDENCE"
  --stage1_peasd_t1_s1_validation_evidence "$T1_S1_EVIDENCE"
  --stage1_peasd_t1_s2_validation_evidence "$T1_S2_EVIDENCE"
  --stage1_peasd_t2_s0_validation_evidence "$T2_S0_EVIDENCE"
  --stage1_peasd_t2_s1_validation_evidence "$T2_S1_EVIDENCE"
  --stage1_peasd_t2_s2_validation_evidence "$T2_S2_EVIDENCE"
  --stage1_peasd_t3_s0_validation_evidence "$T3_S0_EVIDENCE"
  --stage1_peasd_t3_s1_validation_evidence "$T3_S1_EVIDENCE"
  --stage1_peasd_t3_s2_validation_evidence "$T3_S2_EVIDENCE"
  --stage1_peasd_t4_s0_validation_evidence "$T4_S0_EVIDENCE"
  --stage1_peasd_t4_s1_validation_evidence "$T4_S1_EVIDENCE"
  --stage1_peasd_t4_s2_validation_evidence "$T4_S2_EVIDENCE"
)

S1_PAIRWISE="$S1_ROOT/stage1_peasd/pairwise_evidence_index.json"
S1_PROMOTION="$S1_ROOT/stage1_peasd/stage1_peasd_teacher_promotion.json"

run_s1_step stage1_peasd_evidence_index \
  --emg_reference_manifest "$TUBE" \
  --stage1_peasd_pairwise_metrics "$S1_PAIRWISE" \
  "${S1_EVIDENCE_ARGS[@]}"

run_s1_step stage1_peasd_pairwise_gate \
  --emg_reference_manifest "$TUBE" \
  --stage1_peasd_pairwise_metrics "$S1_PAIRWISE" \
  --stage1_peasd_blind_review "$T3_BLIND_PACKAGE/review.json" \
  --stage1_peasd_blind_private_mapping "$T3_PRIVATE_MAPPING" \
  --stage1_peasd_promotion_manifest "$S1_PROMOTION" \
  "${S1_EVIDENCE_ARGS[@]}"
```

pairwise gate 要求：

- 每个 seed 的 T3 real-synergy loss 相对 T4 至少改善 5%，且三 seed 均赢、均值方向为正；
- 每个 seed 的 T3 measured activation anchor loss 严格优于 T0，aggregate mean 也严格改善；
- 对 measured activation 的 anchor loss、violation fraction、mean/max absolute deviation 和
  correlation，T3 相对 T0 还必须逐 seed 与 aggregate 全部 non-degraded：前四项越低越好，
  correlation 越高越好。任何一项下降方向错误都拒绝 promotion；
- tracking、coverage、early termination、saturation、effort 不超过预注册退化界；
- 所有 arms 通过绝对 Stage1 safety/tracking thresholds；
- T1/T2 只作 decomposition diagnostics，不可替代 T3-vs-T4 主 gate。

统计单位是 paired training seed，`n=3`。均值、样本标准差、effect size 和 df=2 interval
只能作描述；不能把 frame、episode 或 P002 trial 当作额外独立 seed，也不声称显著性或
population effect。gate 不通过就停止 Stage2，不调阈值追结果。

## 4. Stage2：一个 shared lineage，先完整 S2-A，再 B/C/D/E

### 4.1 只收集一次 shared physical inputs

Clear/Lift 的 PEASD T3 teacher 先经过动作专属 Stage1R、event bank 和四级 racket-mass
`025→050→075→100` curriculum，再从最终 100% mass teacher 各收集一次 immutable train/val
physical dataset。ChinaJump 使用 T3 body teacher，明确跳过 Stage1R/event/racket/S2-A。

ChinaJump 的 initial `synergy_v3 + disabled` 是另一条明确的 body-only/phase-free 顺序：
`data_release_validate → data_qc → physical_rollout_collect[_val] → physical_rollout_qc/gate
→ synergy_fit/gate → stage2_shared_inputs_seal → latent_dimension_sweep/execute
→ latent_synergy_analysis/gate → stage2_s2b_architecture_lock`。它不出现 event、mass、旧 direct
comparator、完整 S2-A 或 causal evaluate/finalize；C/D/E 只在消费该 body-only shared/lock 时运行，
也不传 `stage2_direct_family_promotion`。ChinaJump v2 provisional tube 已可构建，super-MVC
不再阻塞；但 mapping、trial/S9 人工 review 尚未完成，所以它仍不是 training-enabled 或已完成
实验。Lift 则必须先补齐并校准自己的 event/mass
资产；planner 的缺字段报错不能用 Clear 资产绕过。

Clear 当前 `synergy_v3 + stage1_peasd_latent_arm=disabled` 会在
`stage2_shared_inputs_seal` 后主动截断。执行顺序中的真实 step 名为：

```text
data_release_validate → data_qc
→ stage1r_train → stage1r_eval → stage1r_gate
→ stage1r005_train → stage1r005_eval → stage1r005_gate
→ racket_mass_curriculum_plan → event_reference_qc → event_reference_gate
→ racket_mass_{025,050,075,100}_{physics,train,gate,visual_gate,promote}
→ physical_rollout_collect → physical_rollout_collect_val
→ physical_rollout_qc → physical_rollout_gate
→ direct_baseline_train → direct_baseline_evaluate
→ synergy_fit → synergy_gate → stage2_shared_inputs_seal
```

这里的 `direct_baseline_*` 是用于绑定共同 collection/teacher 的旧 BC comparator，
`stage2_shared_inputs.direct_s2a_evidence.claim_limit` 明确禁止把它称为完整 S2-A。

下面给出 fail-closed invocation。首次 build **不得**传
`--stage2_shared_inputs_manifest`：这个文件尚不存在；一旦传入，planner 会把它当作已有 shared
lineage 严格解析并切到 context-arm 路径，因而在 seal 前必然失败。只有
`stage2_shared_inputs_seal` 成功后，§4.2--§4.5 的 consumer 才显式传 `SHARED`。

先固定 build root 和所有已有 lineage；尖括号变量必须在其第一个 consumer 执行前替换为真实
值。`S2_PROGRESS_ARGS` 会在每次 `--execute_step` 时完整复述目前已经产生的全部 artifact：

```bash
ACTION=forehand_clear
S2_SHARED_ROOT=artifacts/forehand_clear_peasd_v1/stage2_shared_build
SHARED="$S2_SHARED_ROOT/synergy_v3/stage2_shared_inputs.json"
S2_BUILD_GPU=<physical_gpu_index>

T3_SEED0_CHECKPOINT=<exact_promoted_T3_seed0_checkpoint_leaf>
FROZEN_BODY_DECODER=<exact_frozen_body_decoder>
FROZEN_BODY_DECODER_SHA=<sha256>
BODY_SYNERGY_CONTRACT_SHA=<sha256>
BODY_SYNERGY_PORTABLE_CORE_SHA=<sha256>

S2_BUILD_ARGS=(
  --profile synergy_v3
  --action "$ACTION"
  --output_dir "$S2_SHARED_ROOT"
  --stage1_checkpoint "$T3_SEED0_CHECKPOINT"
  --stage1_peasd_promotion_manifest "$S1_PROMOTION"
  --emg_reference_manifest "$TUBE"
  --stage1_peasd_latent_arm disabled
  --train_event_reference_manifest_list "$TRAIN_EVENT_MANIFEST_LIST"
  --val_event_reference_manifest_list "$VAL_EVENT_MANIFEST_LIST"
  --train_event_reference_bank "$TRAIN_EVENT_BANK"
  --val_event_reference_bank "$VAL_EVENT_BANK"
  --frozen_body_decoder "$FROZEN_BODY_DECODER"
  --frozen_body_decoder_fingerprint "$FROZEN_BODY_DECODER_SHA"
  --body_synergy_contract_fingerprint "$BODY_SYNERGY_CONTRACT_SHA"
  --body_synergy_portable_core_fingerprint "$BODY_SYNERGY_PORTABLE_CORE_SHA"
)
S2_PROGRESS_ARGS=()

run_s2_build_step () {
  local step="$1"
  export CUDA_VISIBLE_DEVICES="$S2_BUILD_GPU"
  export MUSCLEMIMIC_JAX_CACHE_KEY="${ACTION}_s2build_${step}_v1"
  export MUSCLEMIMIC_TRAIN_LOG="${S2_SHARED_ROOT}/logs/${step}.log"
  export MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB=4
  export MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB=4
  uv run --locked python -m fullbody.run_forehand_clear_pipeline \
    "${S2_BUILD_ARGS[@]}" \
    "${S2_PROGRESS_ARGS[@]}" \
    --execute_step "$step"
}

# 规划；不训练。
uv run --locked python -m fullbody.run_forehand_clear_pipeline \
  "${S2_BUILD_ARGS[@]}"

# release/QC 和两个 Stage1R rung。
run_s2_build_step data_release_validate
run_s2_build_step data_qc
run_s2_build_step stage1r_train

STAGE1R_CHECKPOINT=<exact_completed_stage1r_003_checkpoint_leaf>
STAGE1R_METRICS="$S2_SHARED_ROOT/stage1r_003/paired_robustness.json"
S2_PROGRESS_ARGS+=(
  --stage1r_checkpoint "$STAGE1R_CHECKPOINT"
  --stage1r_metrics "$STAGE1R_METRICS"
)
run_s2_build_step stage1r_eval
run_s2_build_step stage1r_gate
run_s2_build_step stage1r005_train

STAGE1R005_CHECKPOINT=<exact_completed_stage1r_005_checkpoint_leaf>
STAGE1R005_METRICS="$S2_SHARED_ROOT/stage1r_005/paired_robustness.json"
S2_PROGRESS_ARGS+=(
  --stage1r005_checkpoint "$STAGE1R005_CHECKPOINT"
  --stage1r005_metrics "$STAGE1R005_METRICS"
)
run_s2_build_step stage1r005_eval
run_s2_build_step stage1r005_gate

# event reference 只做一次。
EVENT_METRICS="$S2_SHARED_ROOT/synergy_v3/event_reference/promotion_metrics.json"
S2_PROGRESS_ARGS+=(--event_reference_metrics "$EVENT_METRICS")
run_s2_build_step racket_mass_curriculum_plan
run_s2_build_step event_reference_qc
run_s2_build_step event_reference_gate

# 每个变量都指向该 rung 自己的未来/已完成 artifact；不得复用上一 rung。
add_mass_artifacts () {
  local scale="$1" checkpoint="$2" metrics="$3" review="$4"
  local physics="$5" promotion="$6"
  S2_PROGRESS_ARGS+=(
    "--racket_mass_${scale}_checkpoint" "$checkpoint"
    "--racket_mass_${scale}_metrics" "$metrics"
    "--racket_mass_${scale}_visual_review" "$review"
    "--racket_mass_${scale}_physics_manifest" "$physics"
    "--racket_mass_${scale}_promotion_manifest" "$promotion"
  )
}

M025_CHECKPOINT=<exact_mass_025_checkpoint_leaf>
M025_METRICS=<exact_mass_025_training_progress_json>
M025_REVIEW=<human_signed_mass_025_visual_review_json>
M025_PHYSICS="$S2_SHARED_ROOT/synergy_v3/racket_mass_v2/mass_025_physics_manifest.json"
M025_PROMOTION="$S2_SHARED_ROOT/synergy_v3/racket_mass_v2/mass_025_promotion_manifest.json"
add_mass_artifacts 025 "$M025_CHECKPOINT" "$M025_METRICS" "$M025_REVIEW" \
  "$M025_PHYSICS" "$M025_PROMOTION"
run_s2_build_step racket_mass_025_physics
run_s2_build_step racket_mass_025_train
run_s2_build_step racket_mass_025_gate
# 此处暂停，渲染并由真人填写 M025_REVIEW；文件不存在/未通过时下一步必须失败。
run_s2_build_step racket_mass_025_visual_gate
run_s2_build_step racket_mass_025_promote

M050_CHECKPOINT=<exact_mass_050_checkpoint_leaf>
M050_METRICS=<exact_mass_050_training_progress_json>
M050_REVIEW=<human_signed_mass_050_visual_review_json>
M050_PHYSICS="$S2_SHARED_ROOT/synergy_v3/racket_mass_v2/mass_050_physics_manifest.json"
M050_PROMOTION="$S2_SHARED_ROOT/synergy_v3/racket_mass_v2/mass_050_promotion_manifest.json"
add_mass_artifacts 050 "$M050_CHECKPOINT" "$M050_METRICS" "$M050_REVIEW" \
  "$M050_PHYSICS" "$M050_PROMOTION"
run_s2_build_step racket_mass_050_physics
run_s2_build_step racket_mass_050_train
run_s2_build_step racket_mass_050_gate
# 同样先完成人工 review。
run_s2_build_step racket_mass_050_visual_gate
run_s2_build_step racket_mass_050_promote

M075_CHECKPOINT=<exact_mass_075_checkpoint_leaf>
M075_METRICS=<exact_mass_075_training_progress_json>
M075_REVIEW=<human_signed_mass_075_visual_review_json>
M075_PHYSICS="$S2_SHARED_ROOT/synergy_v3/racket_mass_v2/mass_075_physics_manifest.json"
M075_PROMOTION="$S2_SHARED_ROOT/synergy_v3/racket_mass_v2/mass_075_promotion_manifest.json"
add_mass_artifacts 075 "$M075_CHECKPOINT" "$M075_METRICS" "$M075_REVIEW" \
  "$M075_PHYSICS" "$M075_PROMOTION"
run_s2_build_step racket_mass_075_physics
run_s2_build_step racket_mass_075_train
run_s2_build_step racket_mass_075_gate
# 同样先完成人工 review。
run_s2_build_step racket_mass_075_visual_gate
run_s2_build_step racket_mass_075_promote

M100_CHECKPOINT=<exact_mass_100_checkpoint_leaf>
M100_METRICS=<exact_mass_100_training_progress_json>
M100_REVIEW=<human_signed_mass_100_visual_review_json>
M100_PHYSICS="$S2_SHARED_ROOT/synergy_v3/racket_mass_v2/mass_100_physics_manifest.json"
M100_PROMOTION="$S2_SHARED_ROOT/synergy_v3/racket_mass_v2/mass_100_promotion_manifest.json"
add_mass_artifacts 100 "$M100_CHECKPOINT" "$M100_METRICS" "$M100_REVIEW" \
  "$M100_PHYSICS" "$M100_PROMOTION"
run_s2_build_step racket_mass_100_physics
run_s2_build_step racket_mass_100_train
run_s2_build_step racket_mass_100_gate
# 同样先完成人工 review。
run_s2_build_step racket_mass_100_visual_gate
run_s2_build_step racket_mass_100_promote

# promotion 后从 immutable 100% checkpoint 计算并填写真实 fingerprint。
M100_CHECKPOINT_SHA=<sha256>
S2_PROGRESS_ARGS+=(--racket_mass_100_checkpoint_fingerprint "$M100_CHECKPOINT_SHA")

# train/val physical collection、QC、旧 BC comparator 与 basis 仍都只发生一次。
PHYSICAL_METRICS="$S2_SHARED_ROOT/synergy_v3/physical_rollout/promotion_metrics.json"
S2_PROGRESS_ARGS+=(--physical_rollout_metrics "$PHYSICAL_METRICS")
run_s2_build_step physical_rollout_collect
run_s2_build_step physical_rollout_collect_val
run_s2_build_step physical_rollout_qc
run_s2_build_step physical_rollout_gate

DIRECT_BC="$S2_SHARED_ROOT/synergy_v3/direct_baseline/bc/distill_metadata.json"
DIRECT_ROLLOUT="$S2_SHARED_ROOT/synergy_v3/direct_baseline/compare/comparison_metrics.json"
DIRECT_ACCEPTANCE="$S2_SHARED_ROOT/synergy_v3/direct_baseline/compare/direct_promotion_evidence.json"
S2_PROGRESS_ARGS+=(
  --direct_bc_metrics "$DIRECT_BC"
  --direct_rollout_metrics "$DIRECT_ROLLOUT"
  --direct_acceptance "$DIRECT_ACCEPTANCE"
)
run_s2_build_step direct_baseline_train
run_s2_build_step direct_baseline_evaluate

SYNERGY_METRICS="$S2_SHARED_ROOT/synergy_v3/synergy/promotion_metrics.json"
S2_PROGRESS_ARGS+=(--synergy_metrics "$SYNERGY_METRICS")
run_s2_build_step synergy_fit
run_s2_build_step synergy_gate

# fit/gate 后才读取实际 basis 目录及其 byte/content fingerprint。
SYNERGY_BASIS="$S2_SHARED_ROOT/synergy_v3/synergy/physical_excitation_unit/regional_composite"
SYNERGY_BASIS_SHA=<sha256_from_the_fitted_basis_contract>
S2_PROGRESS_ARGS+=(
  --synergy_basis "$SYNERGY_BASIS"
  --synergy_basis_fingerprint "$SYNERGY_BASIS_SHA"
)
run_s2_build_step stage2_shared_inputs_seal

# seal 是本 root 最后一步；此检查通过前绝不能启动 §4.2。
test -f "$SHARED"
```

上面没有名为 `<one_exact_step_name>` 的伪 step；每一次调用都使用当前 CLI 的真实 step 名，且
helper 会重传 action/profile/output 与完整 artifact 数组。尖括号只表示操作者必须解析的真实
路径或哈希，不是可传给程序的字符串。不得让 B/C/D/E 各自重新 collect、重新 fit basis 或
另选 teacher。

`stage2_shared_inputs_v1` 必须绑定同一 T3 promotion、verified tube、train/val dataset、
physical QC gate、synergy basis 和 frozen decoder。它生成后视为 immutable；完整 S2-A
promotion 是它的 sibling/downstream 证据，绝不能回写 shared JSON 制造哈希环。

### 4.2 完整 S2-A：三 seed 的 BC → 3×DAgger → fresh PPO

```bash
S2A_ROOT=artifacts/forehand_clear_peasd_v1/stage2_s2a
S2A_GPU=<physical_gpu_index>
S2A_CACHE=fc_s2a_v1

run_s2a_step () {
  local step="$1"
  export CUDA_VISIBLE_DEVICES="$S2A_GPU"
  export MUSCLEMIMIC_JAX_CACHE_KEY="${S2A_CACHE}_${step}"
  export MUSCLEMIMIC_TRAIN_LOG="${S2A_ROOT}/logs/${step}.log"
  export MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB=4
  export MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB=4
  uv run --locked python -m fullbody.run_forehand_clear_pipeline \
    --profile stage2_direct \
    --action "$ACTION" \
    --output_dir "$S2A_ROOT" \
    --stage2_shared_inputs_manifest "$SHARED" \
    --stage2_direct_physical_gpu "$S2A_GPU" \
    --stage2_direct_cache_key_prefix "$S2A_CACHE" \
    --execute_step "$step"
}

uv run --locked python -m fullbody.run_forehand_clear_pipeline \
  --profile stage2_direct \
  --action "$ACTION" \
  --output_dir "$S2A_ROOT" \
  --stage2_shared_inputs_manifest "$SHARED" \
  --stage2_direct_physical_gpu "$S2A_GPU" \
  --stage2_direct_cache_key_prefix "$S2A_CACHE"

run_s2a_step stage2_direct_plan
for seed in 0 1 2; do
  run_s2a_step "s2a_seed${seed}_derive_direct_dataset"
  run_s2a_step "s2a_seed${seed}_bc"
  run_s2a_step "s2a_seed${seed}_dagger_3round"
  run_s2a_step "s2a_seed${seed}_fresh_ppo"
  run_s2a_step "s2a_seed${seed}_heldout_compare"
  run_s2a_step "s2a_seed${seed}_seal"
done
run_s2a_step s2a_family_promotion
```

每 seed 的 train dataset 是 shared train dataset 的字节级派生副本；shared validation 始终
read-only。DAgger 恰好三轮；PPO 从 DAgger checkpoint 初始化 actor，但 reset optimizer 和
LR schedule。三个 seed 都封存后才允许生成
`stage2_direct_family_promotion_v1`，预注册 deployment seed 仍是 0，不按结果挑 seed。

### 4.3 S2-B 选择并锁 architecture

S2-B 与 C/D/E 共用 T3 teacher、tube、shared inputs 和完整 S2-A promotion。S2-B 可以扫预注册
latent dimensions/decoder，但只允许选择一次 architecture；C/D/E 随后只跑该 architecture
的 seeds 0/1/2。

```bash
S2A_PROMOTION="$S2A_ROOT/stage2_direct/stage2_direct_family_promotion.json"
S2B_ROOT=artifacts/forehand_clear_peasd_v1/stage2_s2b
S2_LOCK=artifacts/forehand_clear_peasd_v1/stage2_family/s2b_architecture_lock.json
CAUSAL_CONFIG=<shared_stage2_causal_adapter_config>
S2B_GPU=<physical_gpu_index>
S2B_CACHE=fc_s2b_v1
S2B_METRICS="$S2B_ROOT/synergy_v3/latent_synergy/promotion_metrics.json"

run_s2b_step () {
  local step="$1"
  export CUDA_VISIBLE_DEVICES="$S2B_GPU"
  export MUSCLEMIMIC_JAX_CACHE_KEY="${S2B_CACHE}_${step}"
  export MUSCLEMIMIC_TRAIN_LOG="${S2B_ROOT}/logs/${step}.log"
  export MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB=4
  export MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB=4
  uv run --locked python -m fullbody.run_forehand_clear_pipeline \
    --profile synergy_v3 \
    --action "$ACTION" \
    --output_dir "$S2B_ROOT" \
    --stage1_checkpoint "$T3_SEED0_CHECKPOINT" \
    --stage1_peasd_promotion_manifest "$S1_PROMOTION" \
    --emg_reference_manifest "$TUBE" \
    --stage1_peasd_latent_arm disabled \
    --stage2_shared_inputs_manifest "$SHARED" \
    --stage2_architecture_lock_manifest "$S2_LOCK" \
    --stage2_direct_family_promotion "$S2A_PROMOTION" \
    --latent_causal_adapter_config "$CAUSAL_CONFIG" \
    --latent_synergy_metrics "$S2B_METRICS" \
    --execute_step "$step"
}

run_s2b_step data_release_validate
run_s2b_step data_qc
for step in latent_dimension_sweep latent_dimension_execute \
  latent_causal_evaluate latent_causal_finalize \
  latent_synergy_analysis latent_synergy_gate stage2_s2b_architecture_lock; do
  run_s2b_step "$step"
done
```

对 ChinaJump 的 phase-free family，不存在 `latent_causal_evaluate` 和
`latent_causal_finalize`，不得调用这两个 step；其余 B/lock/C/D/E family 逻辑相同，且不传
`stage2_direct_family_promotion`。

### 4.4 锁定 architecture 后跑 S2-C/D/E

| arm | `stage1_peasd_latent_arm` | 额外参数 | 含义 |
|---|---|---|---|
| S2-C | `real` | `--emg_synergy_dim 3` | real privileged context |
| S2-D | `shuffled` | `--emg_synergy_dim 3` | 只增加 context shuffle |
| S2-E | `real_no_dropout` | `--emg_synergy_dim 3` | real context，dropout=0 |

每臂使用独立 output root；以下函数完整复述 action/profile/root/lineage/treatment：

```bash
run_context_step () {
  local arm="$1" mode="$2" root="$3" step="$4"
  export CUDA_VISIBLE_DEVICES="$S2_CONTEXT_GPU"
  export MUSCLEMIMIC_JAX_CACHE_KEY="${ACTION}_${arm}_${step}_v1"
  export MUSCLEMIMIC_TRAIN_LOG="${root}/logs/${step}.log"
  export MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB=4
  export MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB=4
  uv run --locked python -m fullbody.run_forehand_clear_pipeline \
    --profile synergy_v3 \
    --action "$ACTION" \
    --output_dir "$root" \
    --stage1_checkpoint "$T3_SEED0_CHECKPOINT" \
    --stage1_peasd_promotion_manifest "$S1_PROMOTION" \
    --emg_reference_manifest "$TUBE" \
    --stage1_peasd_latent_arm "$mode" \
    --emg_synergy_dim 3 \
    --stage2_shared_inputs_manifest "$SHARED" \
    --stage2_architecture_lock_manifest "$S2_LOCK" \
    --stage2_direct_family_promotion "$S2A_PROMOTION" \
    --latent_causal_adapter_config "$CAUSAL_CONFIG" \
    --latent_synergy_metrics "$root/synergy_v3/latent_synergy/promotion_metrics.json" \
    --execute_step "$step"
}

S2C_ROOT=artifacts/forehand_clear_peasd_v1/stage2_s2c
S2D_ROOT=artifacts/forehand_clear_peasd_v1/stage2_s2d
S2E_ROOT=artifacts/forehand_clear_peasd_v1/stage2_s2e
S2_CONTEXT_GPU=<physical_gpu_index>

for spec in "S2-C real $S2C_ROOT" "S2-D shuffled $S2D_ROOT" \
            "S2-E real_no_dropout $S2E_ROOT"; do
  set -- $spec
  for step in data_release_validate data_qc \
    latent_dimension_sweep latent_dimension_execute \
    latent_causal_evaluate latent_causal_finalize \
    latent_synergy_analysis latent_synergy_gate; do
    run_context_step "$1" "$2" "$3" "$step"
  done
done
```

S2-C 与 S2-D 的 command、shared inputs、architecture、seed 和超参只能相差 output identity
及 `emg_shuffle_context_ablation`。C/E 每 seed 都必须有正的 blank-context posterior response；
C/D/E 都必须提供 finite synergy-head loss/correlation、blank-context posterior/action diagnostics。

### 4.5 封存 Stage2 context family

family profile 接受的是每臂真正包含 `sweep_plan.json` 的 latent output 目录，而不是上层
pipeline root：

```bash
S2_FAMILY_ROOT=artifacts/forehand_clear_peasd_v1/stage2_family
S2_FAMILY_INDEX="$S2_FAMILY_ROOT/stage2_context_family/family_index.json"
S2_FAMILY_GATE="$S2_FAMILY_ROOT/stage2_context_family/family_gate.json"

run_s2_family_step () {
  local step="$1"
  uv run --locked python -m fullbody.run_forehand_clear_pipeline \
    --profile stage2_context_family \
    --action "$ACTION" \
    --output_dir "$S2_FAMILY_ROOT" \
    --stage2_shared_inputs_manifest "$SHARED" \
    --stage2_architecture_lock_manifest "$S2_LOCK" \
    --stage2_s2b_output_dir "$S2B_ROOT/synergy_v3/latent_synergy" \
    --stage2_s2c_output_dir "$S2C_ROOT/synergy_v3/latent_synergy" \
    --stage2_s2d_output_dir "$S2D_ROOT/synergy_v3/latent_synergy" \
    --stage2_s2e_output_dir "$S2E_ROOT/synergy_v3/latent_synergy" \
    --stage2_context_family_index "$S2_FAMILY_INDEX" \
    --stage2_context_family_gate "$S2_FAMILY_GATE" \
    --execute_step "$step"
}

run_s2_family_step stage2_context_family_index
run_s2_family_step stage2_context_family_gate
```

主 gate 是 paired seed 的 `S2-D emg_synergy_head_loss - S2-C loss > 0`，要求 seeds 0/1/2
三次均为正且均值为正。`n=3`、one-sided exact sign-test `p=0.125`，因此只报告 mean、sample
SD、Cohen's dz、df=2 interval 和 failure count，不声称统计显著性。action/closed-loop 指标
必须报告，但不得看到结果后追加 acceptance threshold。

## 5. Stage3：九个独立 H1/H2/H3 叶节点

Stage3 只对 `stage3_applicable=true` 的动作执行。Clear 使用已通过的 Stage2 family：

| arm | Stage2 source | residual |
|---|---|---|
| H1 | S2-B selected latent | disabled |
| H2 | S2-C selected latent | disabled |
| H3 | S2-C selected latent | 必须是非空 grouped right-arm residual，且每组 `alpha≤0.10` |

H3 不能放开 full-354 residual。一个诚实的最小配置可以只启用已有 actuator roster 的
`wrist_forearm`，例如 `{"wrist_forearm": {"alpha": 0.05}}`；空的 shoulder/elbow roster
不能写成“已启用”。H1/H2 严禁带任何 bounded residual。

Stage3 task spec 不是自由输入：single-leaf profile 只使用 action registry 中该动作的
`stage3_v2_spec`，reachability manifest 还要求其**解析后路径精确相等**并绑定 spec/scene hash。
因此 Lift 当前 `stage3_v2_spec=None` 必须 fail-closed；把 Clear spec 复制、软包装或仅改 action
字段都不能解冻 Lift。

### 5.1 单叶 profile 与不可自选的 latent

正式单叶入口是 `--profile stage3_peasd_arm`。它从通过的 Stage2 family index 自动解析
latent：H1 固定取 S2-B `best_synergy`，H2/H3 固定取 S2-C `best_synergy`。用户不能传另一份
latent 来替换选择；可选的 `--stage3_expected_latent_fingerprint` 只做一致性断言。

对每个 `H∈{H1,H2,H3}`、`seed∈{0,1,2}` 创建独立 `LEAF_OUTPUT`。九份 source、CEM、
CPU audit、cross-backend seal、correction dataset、short-BC、release 与 PPO root 不得跨叶
复用。先定义每次都会完整复述的 base args：

```bash
H=H1                         # H1 / H2 / H3
SEED=0                       # 0 / 1 / 2
LEAF_OUTPUT="artifacts/forehand_clear_peasd_v1/stage3/${H}_s${SEED}"
RUN_ROOT="$LEAF_OUTPUT/stage3_peasd_arm"
GPU=<physical_gpu_index>
S3_CACHE_PREFIX=fc_stage3_v1
# 直接复用 §4.5 已定义且通过的 S2_FAMILY_GATE，不重指到另一份 gate。

SOURCE_CHECKPOINT=<exact_reachability_source_checkpoint_for_this_H_and_seed>
SINGLE_FEED_FINGERPRINT=<sha256_of_the_frozen_single_feed>
SOURCE_CONTROL_HASH=<sha256_of_the_source_control_manifest>
TRAIN_TARGET_BANK=<exact_action_owned_training_target_bank>
EVAL_TARGET_BANK=<exact_action_owned_evaluation_target_bank>
TRAIN_FEED_BANK=<exact_action_owned_training_feed_bank>
HELDOUT_128_FEED_BANK=<exact_action_owned_128_feed_evaluation_bank>
H3_GROUPS=<reviewed_nonempty_grouped_right_arm_residual_json>

S3_LEAF_ARGS=(
  --profile stage3_peasd_arm
  --action forehand_clear
  --output_dir "$LEAF_OUTPUT"
  --stage3_peasd_arm "$H"
  --stage3_training_seed "$SEED"
  --stage3_physical_gpu "$GPU"
  --stage3_cache_key_prefix "$S3_CACHE_PREFIX"
  --stage2_context_family_gate "$S2_FAMILY_GATE"
  --stage3_reachability_source_checkpoint "$SOURCE_CHECKPOINT"
  --stage3_expected_feed_fingerprint "$SINGLE_FEED_FINGERPRINT"
  --stage3_expected_control_hash "$SOURCE_CONTROL_HASH"
  --recovery_target_bank "$TRAIN_TARGET_BANK"
  --recovery_eval_target_bank "$EVAL_TARGET_BANK"
  --recovery_train_feed_bank "$TRAIN_FEED_BANK"
  --recovery_eval_feed_bank "$HELDOUT_128_FEED_BANK"
)

# 可选，只用于检查 Stage2 自动解析结果，不能改变选择；不知道时就完全省略。
# EXPECTED_LATENT_SHA=<sha256_read_from_the_sealed_S2_selection>
if test -n "${EXPECTED_LATENT_SHA:-}"; then
  S3_LEAF_ARGS+=(--stage3_expected_latent_fingerprint "$EXPECTED_LATENT_SHA")
fi

# 只对 H3 加；H1/H2 加这个字段会 fail-closed。
if test "$H" = H3; then
  S3_LEAF_ARGS+=(--stage3_bounded_residual_groups_json "$H3_GROUPS")
fi

# 只生成 plan，不启动训练。
uv run --locked python -m fullbody.run_forehand_clear_pipeline \
  "${S3_LEAF_ARGS[@]}"
```

`SOURCE_CHECKPOINT` 必须是与该叶的 selected latent、control manifest、单 feed 和 residual
treatment 一致的 exact Stage3 source checkpoint。它不是让用户绕过 Stage2 selection 的
另一个 latent 入口。H3 groups 必须非空且每组 `alpha≤0.10`；H1/H2 必须完全无 residual。
source 的 `stage3_lab_control_v1` 会把 residual 归一化成完整 identity：enabled/dimension、
schema SHA、ordered groups，以及每组精确的 name、非空且互斥的 actuator names、dim 和 alpha。
H1/H2 必须是 dimension=0 且 schema/groups=null；H3 的 group dim 必须等于 actuator 数，总 dim
必须相加一致。correction manifest、short-BC runtime metadata 和 reachability release 必须逐字段
等于 source identity；改 schema、group、actuator roster 或 alpha 中任何一个都会 fail-closed。

### 5.2 reachability → short BC → C3 → C4–C7

当前单叶 profile 的真实 step 顺序是：

```text
stage3_v2_preflight
→ stage3_v2_feed_check
→ stage3_v2_base_only
→ stage3_single_feed_cem
→ stage3_candidate_cpu_audit
→ stage3_cross_backend_seal
→ stage3_correction_dataset_seal
→ stage3_short_bc
→ stage3_reachability_release
→ stage3_static_target_train
→ stage3_static_target_evaluate
→ stage3_static_target_gate
→ stage3_v2_train
→ stage3_v2_evaluate
→ stage3_v2_gate
```

执行函数始终复述 §5.1 的完整 base args；producer 完成后，再把其 exact artifacts 加入同一
命令。不能仅写 profile/root/step：

```bash
run_s3_leaf_step () {
  local step="$1"
  shift
  export MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB=4
  export MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB=4
  uv run --locked python -m fullbody.run_forehand_clear_pipeline \
    "${S3_LEAF_ARGS[@]}" \
    "$@" \
    --execute_step "$step"
}

run_s3_leaf_step stage3_v2_preflight
run_s3_leaf_step stage3_v2_feed_check
run_s3_leaf_step stage3_v2_base_only
run_s3_leaf_step stage3_single_feed_cem

CEM_DIR="$RUN_ROOT/reachability/single_feed_cem"
CEM_CONTRACT="$CEM_DIR/cem_contract.json"
CEM_REPORT="$CEM_DIR/cem_report.json"
CEM_CANDIDATE="$CEM_DIR/best_teacher.json"
CEM_ARGS=(
  --stage3_cem_contract "$CEM_CONTRACT"
  --stage3_cem_report "$CEM_REPORT"
  --stage3_cem_candidate "$CEM_CANDIDATE"
)
run_s3_leaf_step stage3_candidate_cpu_audit "${CEM_ARGS[@]}"

CPU_TRACE="$RUN_ROOT/reachability/cpu_audit_trace.npz"
CPU_REPORT="$RUN_ROOT/reachability/cpu_audit_trace.json"
CPU_ARGS=(
  "${CEM_ARGS[@]}"
  --stage3_cpu_audit_trace "$CPU_TRACE"
  --stage3_cpu_audit_report "$CPU_REPORT"
)
run_s3_leaf_step stage3_cross_backend_seal "${CPU_ARGS[@]}"

SEAL_REPORT="$RUN_ROOT/reachability/cross_backend_seal/cem_report.json"
CORRECTION="$RUN_ROOT/reachability/cross_backend_seal/teacher_trajectory_cpu_quality.npz"
SEAL_ARGS=(
  "${CPU_ARGS[@]}"
  --stage3_cross_backend_seal_report "$SEAL_REPORT"
  --stage3_correction_dataset "$CORRECTION"
)
run_s3_leaf_step stage3_correction_dataset_seal "${SEAL_ARGS[@]}"

CORRECTION_MANIFEST="$RUN_ROOT/reachability/correction_dataset_manifest.json"
CORRECTION_ARGS=(
  "${SEAL_ARGS[@]}"
  --stage3_correction_dataset_manifest "$CORRECTION_MANIFEST"
)
run_s3_leaf_step stage3_short_bc "${CORRECTION_ARGS[@]}"

# 必须解析为 immutable post_teacher_bc_pre_ppo versioned leaf，不是 policy_latest.json。
SHORT_BC_CHECKPOINT=<exact_immutable_short_bc_checkpoint_leaf>
SHORT_BC_METRICS="$RUN_ROOT/teacher_bc_pretrain_report.json"
SHORT_BC_TRAIN_REPORT="$RUN_ROOT/train_report.json"
SHORT_BC_ARGS=(
  "${CORRECTION_ARGS[@]}"
  --stage3_short_bc_checkpoint "$SHORT_BC_CHECKPOINT"
  --stage3_short_bc_metrics "$SHORT_BC_METRICS"
  --stage3_short_bc_train_report "$SHORT_BC_TRAIN_REPORT"
)
run_s3_leaf_step stage3_reachability_release "${SHORT_BC_ARGS[@]}"

REACHABILITY_RELEASE="$RUN_ROOT/reachability/reachability_release.json"
RELEASE_ARGS=(
  "${SHORT_BC_ARGS[@]}"
  --stage3_reachability_release "$REACHABILITY_RELEASE"
)
run_s3_leaf_step stage3_static_target_train "${RELEASE_ARGS[@]}"

C3_CHECKPOINT=<exact_completed_C3_checkpoint_leaf>
STATIC_METRICS="$RUN_ROOT/evaluate_static/evaluate_report.json"
C3_ARGS=(
  "${RELEASE_ARGS[@]}"
  --static_target_checkpoint "$C3_CHECKPOINT"
  --static_target_metrics "$STATIC_METRICS"
)
run_s3_leaf_step stage3_static_target_evaluate "${C3_ARGS[@]}"
run_s3_leaf_step stage3_static_target_gate "${C3_ARGS[@]}"
run_s3_leaf_step stage3_v2_train "${C3_ARGS[@]}"

FINAL_CHECKPOINT=<exact_completed_C7_checkpoint_leaf>
FINAL_METRICS="$RUN_ROOT/evaluate/evaluate_report.json"
FINAL_ARGS=(
  "${C3_ARGS[@]}"
  --stage3_v2_checkpoint "$FINAL_CHECKPOINT"
  --stage3_v2_metrics "$FINAL_METRICS"
)
run_s3_leaf_step stage3_v2_evaluate "${FINAL_ARGS[@]}"
run_s3_leaf_step stage3_v2_gate "${FINAL_ARGS[@]}"
```

profile 把 CEM、short BC、C3 和 C7 production trainer 全部路由到 canonical launcher，并由
`stage3_physical_gpu`、`stage3_cache_key_prefix` 和 append-only `training.log` 提供稳定环境。
只有原生 CEM、独立 CPU replay 和 cross-backend seal 全部通过，correction manifest 才授权
`short_bc_only`。short BC 必须是 zero-PPO；release 固化 immutable payload/metadata/completion、
BC metrics 和 zero-step train-report snapshot。

C3 必须从 release 中的 short-BC immutable leaf resume；C4–C7 必须从已完成且
release-bound 的 C3 leaf resume，并继续使用同一 correction dataset、release 与 `RUN_ROOT`。
禁止 fresh-start C3、直接从 short BC 跳 C7、替换 dataset、用 `--initialize-policy-from`
代替 resume，或在高阶段另开 root。

最终 evaluation 固定 seed 123，完整覆盖 128 个 held-out feeds。每个 feed/episode/frame 都是
repeated measurement，不是额外独立 `n`。

### 5.3 Stage3 family index 与 gate

九个 evaluate report 和九个 reachability release 完成后：

```bash
S3_FAMILY_ROOT=artifacts/forehand_clear_peasd_v1/stage3_family
S3_COMPARISON_CONTRACT=configs/public/stage3_peasd_family_comparison_contract_v1.json
S3_FAMILY_INDEX="$S3_FAMILY_ROOT/stage3_peasd_family/family_index.json"
S3_FAMILY_GATE="$S3_FAMILY_ROOT/stage3_peasd_family/family_gate.json"

H1_S0_REPORT=artifacts/forehand_clear_peasd_v1/stage3/H1_s0/stage3_peasd_arm/evaluate/evaluate_report.json
H1_S1_REPORT=artifacts/forehand_clear_peasd_v1/stage3/H1_s1/stage3_peasd_arm/evaluate/evaluate_report.json
H1_S2_REPORT=artifacts/forehand_clear_peasd_v1/stage3/H1_s2/stage3_peasd_arm/evaluate/evaluate_report.json
H2_S0_REPORT=artifacts/forehand_clear_peasd_v1/stage3/H2_s0/stage3_peasd_arm/evaluate/evaluate_report.json
H2_S1_REPORT=artifacts/forehand_clear_peasd_v1/stage3/H2_s1/stage3_peasd_arm/evaluate/evaluate_report.json
H2_S2_REPORT=artifacts/forehand_clear_peasd_v1/stage3/H2_s2/stage3_peasd_arm/evaluate/evaluate_report.json
H3_S0_REPORT=artifacts/forehand_clear_peasd_v1/stage3/H3_s0/stage3_peasd_arm/evaluate/evaluate_report.json
H3_S1_REPORT=artifacts/forehand_clear_peasd_v1/stage3/H3_s1/stage3_peasd_arm/evaluate/evaluate_report.json
H3_S2_REPORT=artifacts/forehand_clear_peasd_v1/stage3/H3_s2/stage3_peasd_arm/evaluate/evaluate_report.json

H1_S0_RELEASE=artifacts/forehand_clear_peasd_v1/stage3/H1_s0/stage3_peasd_arm/reachability/reachability_release.json
H1_S1_RELEASE=artifacts/forehand_clear_peasd_v1/stage3/H1_s1/stage3_peasd_arm/reachability/reachability_release.json
H1_S2_RELEASE=artifacts/forehand_clear_peasd_v1/stage3/H1_s2/stage3_peasd_arm/reachability/reachability_release.json
H2_S0_RELEASE=artifacts/forehand_clear_peasd_v1/stage3/H2_s0/stage3_peasd_arm/reachability/reachability_release.json
H2_S1_RELEASE=artifacts/forehand_clear_peasd_v1/stage3/H2_s1/stage3_peasd_arm/reachability/reachability_release.json
H2_S2_RELEASE=artifacts/forehand_clear_peasd_v1/stage3/H2_s2/stage3_peasd_arm/reachability/reachability_release.json
H3_S0_RELEASE=artifacts/forehand_clear_peasd_v1/stage3/H3_s0/stage3_peasd_arm/reachability/reachability_release.json
H3_S1_RELEASE=artifacts/forehand_clear_peasd_v1/stage3/H3_s1/stage3_peasd_arm/reachability/reachability_release.json
H3_S2_RELEASE=artifacts/forehand_clear_peasd_v1/stage3/H3_s2/stage3_peasd_arm/reachability/reachability_release.json

run_s3_family_step () {
  local step="$1"
  uv run --locked python -m fullbody.run_forehand_clear_pipeline \
    --profile stage3_peasd_family \
    --action forehand_clear \
    --output_dir "$S3_FAMILY_ROOT" \
    --stage2_context_family_gate "$S2_FAMILY_ROOT/stage2_context_family/family_gate.json" \
    --stage3_peasd_comparison_contract "$S3_COMPARISON_CONTRACT" \
    --stage3_peasd_family_index "$S3_FAMILY_INDEX" \
    --stage3_peasd_family_gate "$S3_FAMILY_GATE" \
    --stage3_h1_s0_report "$H1_S0_REPORT" \
    --stage3_h1_s1_report "$H1_S1_REPORT" \
    --stage3_h1_s2_report "$H1_S2_REPORT" \
    --stage3_h2_s0_report "$H2_S0_REPORT" \
    --stage3_h2_s1_report "$H2_S1_REPORT" \
    --stage3_h2_s2_report "$H2_S2_REPORT" \
    --stage3_h3_s0_report "$H3_S0_REPORT" \
    --stage3_h3_s1_report "$H3_S1_REPORT" \
    --stage3_h3_s2_report "$H3_S2_REPORT" \
    --stage3_h1_s0_reachability_release "$H1_S0_RELEASE" \
    --stage3_h1_s1_reachability_release "$H1_S1_RELEASE" \
    --stage3_h1_s2_reachability_release "$H1_S2_RELEASE" \
    --stage3_h2_s0_reachability_release "$H2_S0_RELEASE" \
    --stage3_h2_s1_reachability_release "$H2_S1_RELEASE" \
    --stage3_h2_s2_reachability_release "$H2_S2_RELEASE" \
    --stage3_h3_s0_reachability_release "$H3_S0_RELEASE" \
    --stage3_h3_s1_reachability_release "$H3_S1_RELEASE" \
    --stage3_h3_s2_reachability_release "$H3_S2_RELEASE" \
    --execute_step "$step"
}

run_s3_family_step stage3_peasd_family_index
run_s3_family_step stage3_peasd_family_gate
```

比较合同在看到结果前冻结于
`configs/public/stage3_peasd_family_comparison_contract_v1.json`：

- H2 vs H1 主指标：`opponent_back_landing_rate`，每 seed 与均值均严格提高；
- H2 vs H1 guardrails：`hit_rate`、`no_fall_rate` 每 seed 与均值均不退化；
- H3 vs H2 主指标：`impact_position_error_m`，每 seed 与均值均严格降低；
- H3 vs H2 guardrails：`hit_rate`、`no_fall_rate`、`opponent_back_landing_rate` 不退化。

统计单位仍是 independent training seed，`n=3, df=2`。报告 mean、sample SD、Cohen's dz、
df=2 interval 和 failure count；不声称 null-hypothesis significance 或 population effect。
family gate 未过不得修改冻结合同后重算。

## 6. 指标、证据边界与失败解释

### 6.1 必报指标

- 动作：joint/root/racket error、frame coverage、完整动作成功、fall/early termination；
- 控制与肌肉：activation energy、action/activation saturation、action/activation rate、
  M-channel anchor loss/correlation、peak phase、onset/offset、co-contraction；
- 协同：held-out/per-channel VAF、W cosine、subspace angle、H correlation（只有数据设计允许时）、
  bootstrap stability；
- direct distill：train/val action MSE、三轮 DAgger convergence、BC/DAgger/PPO held-out
  closed-loop、teacher/student return、fall rate、physiology degradation；
- latent：action reconstruction、prior/posterior gap、active dimensions、sigma clamp、decoder
  saturation、synergy-head loss/correlation、blank/shuffled-context response、prior-only closed loop；
- Stage3：hit、positive outgoing-z、cross-net、legal return、net clearance、opponent-back landing、
  impact-position error、no-fall、完整 held-out-feed coverage。

不能用 episode return 代替真实接触/出球/过网/落点事件，也不能用 reward proxy 当 ground truth。

### 6.2 统计单位

- 人体：trial/subject/session；P002 单 subject/session 不能形成 population CI；
- RL family：independent training seed；frame、environment、episode、feed 都不是额外独立 `n`；
- paired 与 unpaired 设计分开；没有相同 reference/impact evidence 时只能做 cohort/basis geometry；
- `n=3` 的 t interval、Cohen's dz 与 sign test 全部为描述性证据，不宣称显著性。

### 6.3 失败时停止在哪里

- T3 不优于 T4：先查 tube/mapping/phase/reward delivery，停止 Stage2；
- S2-C 不优于 S2-D：说明 privileged context 主张不成立，停止 Stage3 PEASD claim；
- reachability 无真实 contact/正 outgoing-z：先查 grip、feed workspace、authority、拍面/惯量，
  不增加 PPO 步数；
- C3 未完成：不能启动 C4–C7；
- H2 不优于 H1：不能声称 PEASD latent 有 downstream utility；
- H3 不优于 H2：保留 H2，不能事后增大 residual alpha 或改指标阈值。

## 7. 最终 formal release / Definition of Done

所有阶段通过后仍不能靠手写结论宣布完成。唯一最终入口会重新验证 Stage1 promotion、Stage2
context-family gate、适用时的 Stage3 family gate、它们的 action/lineage/content hash，以及一份
source-bound 且 self-bound 的 `peasd_complete_evaluation_evidence_v1`。release 本身不增加看到结果后才设定的
数值阈值；acceptance 只来自前述预注册 gates，complete evidence 负责完整、有限值的描述性报告。

complete evidence 至少必须满足：

- `action` 精确等于 registry 的 `slug/action_id`；`execution` 精确为
  `mode=formal, completed=true, passed=true, dry_run=false, placeholder=false`；
- 整棵 JSON 禁止 dry-run、placeholder、failed、failure、incomplete 标记，并以移除
  `binding_sha256` 后的 canonical JSON SHA-256 自绑定；
- `source_artifacts` 必须精确列出互不相同的 `physiology`、`stage1`、`stage2`，Stage3 适用时
  还必须列 `stage3`；每项绑定真实 evaluator JSON 的绝对/可解析路径、文件 SHA-256 和其
  `schema_version`（源若有内部 binding 还要逐字匹配），且 evidence 严禁把自己列为来源；
- `metric_provenance` 必须逐项覆盖下面每个 required metric，记录规定的 evaluator layer、源内
  JSON path 与源值 canonical SHA-256；validator 会重新读取源值并与汇总值逐字比较。仅修改
  汇总数值并重算 self-hash、替换源报告、把 Stage2 数值指向 Stage1 source 都必须失败；
- upstream bindings 精确绑定本次 Stage1/Stage2/Stage3 gates；统计域固定为 seeds 0/1/2，
  RL unit 是 independent training seed，episode/frame/feed 不是独立 `n`，不声明 significance、
  population policy effect 或 population physiology；
- M-channel 必须列出非空且不重复的 `channel_ids`、匹配的 `channel_count`，并逐 channel 报告
  anchor loss、correlation、peak-phase error、onset error，另有至少一个 co-contraction pair；
- 必须有 action/activation rate、energy、两种 saturation，fall/early termination、joint/keypoint/
  root/tracking error，以及 context/blank/shuffled response、synergy-head loss/correlation；
- Stage3 适用动作还必须有 hit、no-fall、opponent-back landing、impact-position error、legal
  landing、recovery-complete 和 normalized control energy。上述是 release validator 的最低精确
  路径；论文与完整实验报告仍须覆盖本文 §6.1（对应研究方法 §27）及完整 S2-A §4.2 指标，
  不能把 validator 的最小集合误写成报告上限。complete evidence 必须由这些 immutable
  evaluator outputs 汇编；手写一个只有自哈希、没有 `source_artifacts/metric_provenance` 的 JSON
  不再是合法输入。

Clear 的最终命令形状如下；这里的每个输入都必须是本指南前面生成的真实 immutable artifact：

```bash
ACTION=forehand_clear
COMPLETE_EVIDENCE=<peasd_complete_evaluation_evidence_v1.json>
FORMAL_RELEASE=artifacts/forehand_clear_peasd_v1/formal_release/peasd_formal_release.json

uv run --locked python -m musclemimic.badminton.peasd_formal_release build \
  --action "$ACTION" \
  --stage1-peasd-promotion "$S1_PROMOTION" \
  --stage2-context-family-gate "$S2_FAMILY_GATE" \
  --stage3-peasd-family-gate "$S3_FAMILY_GATE" \
  --complete-evaluation-evidence "$COMPLETE_EVIDENCE" \
  --output "$FORMAL_RELEASE"

uv run --locked python -m musclemimic.badminton.peasd_formal_release validate \
  --release "$FORMAL_RELEASE" \
  --expected-action "$ACTION"
```

Lift 的 `stage3_applicable=true` 是科学适用性，不因当前缺资产而改变；它未来只能用自己的 passed
Stage3 family gate 走同一命令，**不允许** `--stage3-not-applicable`。Clear 同样不允许 N/A。
ChinaJump 是唯一当前可显式 N/A 的动作，且 complete evidence 中不得出现 Stage3 hitting block：

```bash
ACTION=chinajump
COMPLETE_EVIDENCE=<chinajump_peasd_complete_evaluation_evidence_v1.json>
FORMAL_RELEASE=artifacts/chinajump_peasd_v1/formal_release/peasd_formal_release.json

uv run --locked python -m musclemimic.badminton.peasd_formal_release build \
  --action "$ACTION" \
  --stage1-peasd-promotion "$CHINA_S1_PROMOTION" \
  --stage2-context-family-gate "$CHINA_S2_FAMILY_GATE" \
  --stage3-not-applicable \
  --complete-evaluation-evidence "$COMPLETE_EVIDENCE" \
  --output "$FORMAL_RELEASE"

uv run --locked python -m musclemimic.badminton.peasd_formal_release validate \
  --release "$FORMAL_RELEASE" \
  --expected-action "$ACTION"
```

`--stage3-peasd-family-gate` 与 `--stage3-not-applicable` 互斥且必须二选一；“文件缺失”永远不
等于 N/A 或 pass。builder 对不同内容的既有 release 拒绝覆盖；validator 会从所有绑定源重建
并逐字比较。只有 build 与 validate 都成功、§9 回归真实通过、且人工/资产阻塞全部解除，才达到
DoD。当前仓库没有满足这些条件的 formal release，以上命令不得提前执行来制造结果。

## 8. 当前必须由人或新资产解除的阻塞

1. 三动作共用 mapping 尚未完成真实解剖 review；
2. 三动作都缺 action-specific trial/channel/S9 的人工签核；super-MVC 已由 v2 双轨规范处理，
   不再需要人工 waiver，也不再阻塞构建；
3. 三动作 v2 provisional tube 均可构建，但 mapping 未 verified、trial/S9 review 未完成，因此
   都仍是 `training_enabled=false`，不能作为正式结果；
4. Clear 尚无这条新 lineage 的 T0–T4 × 3、opaque review、T3 promotion、Stage2/Stage3
   formal results；
5. Lift 缺动作专属 event bank、四级 racket-mass-v2 校准和 Stage3 spec/target/feed；不能借
   Clear 资产；
6. ChinaJump release 仍有 legacy evidence limitation；它只做 body-only/phase-free Stage2，
   Stage1R、S2-A、racket 和 Stage3 都是 N/A；
7. Stage3 每个 H/seed 仍需真实 source checkpoint、feed/target/control/latent identity、CEM、
   CPU/cross-backend evidence 和 30M curriculum 训练。

这些阻塞未解除前，最多只能写“接口与证据合同已实现”，并另附当时真实的回归通过/失败
状态；不能写“PEASD 提升了 tracking/latent/击球”或给出正式数值。

## 9. 回归与审计

代码或合同改动后，至少运行：

```bash
source configs/env.sh
uv run --locked pytest -q \
  tests/unit/test_emg_reference_tube.py \
  tests/unit/test_emg_anchor_loss.py \
  tests/unit/test_build_emg_reference_tube.py \
  tests/unit/test_stage1_peasd_runtime.py \
  tests/unit/test_stage1_peasd_configs.py \
  tests/unit/test_stage1_peasd_gate.py \
  tests/unit/test_stage2_context_family.py \
  tests/unit/test_stage2_direct_lifecycle.py \
  tests/unit/test_stage3_reachability_release.py \
  tests/unit/test_stage3_peasd_pipeline.py \
  tests/unit/test_stage3_peasd_family.py \
  tests/unit/test_peasd_formal_release.py \
  tests/unit/test_forehand_clear_pipeline.py

(cd jidian_measurement && ../.venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_preprocessing_v2.py)
```

同时审计每个 plan：

- 无 `<required:...>` 残留进入 production step；
- 每个训练 step 经 canonical launcher；
- T0 command 内没有 tube/reward token；T1–T4 tube hash 相同；
- 15 个 Stage1 run id/config hash 唯一，但 matched core/source snapshot 相同；
- Stage1 五项 measured-activation 指标逐 seed/aggregate 均检查正确方向，且 degraded
  correlation 的负测必须被拒绝；
- Stage2 只有一份 shared collection，B/C/D/E 绑定同一个 shared hash 与 S2-A promotion；
- C/D 只差 shuffle，C/E 只差 dropout treatment；
- Stage3 九个 root/release/report 路径唯一，H1/H2 无 residual，H3 residual 非空且有界；
- Stage3 spec 路径等于 action registry 的 exact asset，source→correction→short BC→release 的
  residual schema/group/actuator/alpha identity 完全相等；
- 所有 release/promotion 都能从原始 immutable source 重新验证。

## 10. 关键入口

- 研究叙事：`doc/整体故事框架与思路/01_研究故事与论文叙事主线.md`
- 三阶段方法：`doc/整体故事框架与思路/02_三阶段方法与肌电参与机制.md`
- 路线图：`doc/整体故事框架与思路/03_仓库改进与待办路线图.md`
- 动作 registry：`musclemimic/badminton/action_registry.py`
- tube：`scripts/build_emg_reference_tube.py`、`musclemimic/physiology/emg_reference.py`
- Stage1 family：`fullbody/run_forehand_clear_pipeline.py --profile stage1_peasd`、
  `musclemimic/badminton/stage1_peasd_gate.py`
- Stage2 direct：`fullbody/stage2_direct_lifecycle.py`、
  `musclemimic/distill/stage2_direct_lifecycle.py`
- Stage2 context family：`musclemimic/badminton/stage2_context_family.py`
- Stage3 reachability：`musclemimic/badminton/stage3_reachability_release.py`
- Stage3 family：`musclemimic/badminton/stage3_peasd_family.py`
- final release：`musclemimic/badminton/peasd_formal_release.py`
- canonical production launcher：`scripts/run_fullbody_training.sh`
