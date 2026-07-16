# ChinaJump Stage-1 早期 Primitive 肌肉协同

## 1. 目标与不变量

这是一条和现有 354 维纯轨迹跟踪并行的实验线，不替换
`conf_fullbody_chinajump_root_control_v2`。六组实验使用完全相同的：

- ChinaJump train8/val2 轨迹；
- observation、tracking reward 和 terminal contract；
- 640M environment steps；
- promotion 规则与验证轨迹；
- MyoFullBody 354 个非手指 muscle actuator 的顺序和 control range。

唯一的受控变量是动作表示，以及是否打开现有 ASI frame-bucket curriculum。
因此结果可以回答“低维固定协同是否改善复杂动作的探索”，而不会把奖励、终止或
训练集变化误当成协同收益。

## 2. 新动作接口

策略仍输出无界对角高斯 raw action，PPO 的 log-probability、KL 和 loss 全部在
`K` 或 `K+R` 维计算。动作进入原环境前，由冻结 wrapper 解码：

```text
c_i = cmax_i * sigmoid(z_c_i / temperature_i + logit(q50_i / cmax_i))
u   = clip(b0 + W c + alpha * R tanh(z_r), 0, 1)
a   = physical_excitation_to_existing_normalized_body_action(u)
```

- `W ∈ R_+^(354×K)`：只由 primitive train split 拟合的固定非负 basis；
- `cmax=1.2×train_q99`，中心为 `train_q50`，两者绑定最终 physical `W`；
- `b0=0`：第一阶段禁止 learned 354-D baseline 绕过低维约束；
- `R ∈ R^(354×r)`：可选的 4–12 维 signed structured residual；
- `alpha=0.03`：Phase-A 固定值，当前没有假装实现 update-level schedule；
- 初始 policy std 由 decoder 零点 Jacobian 逐维标定，使线性化 physical
  excitation RMS 为 0.08，residual std 再乘 0.25。

固定解码器对应数学说明中的条件结论：当最优控制接近可表达集合时，降维可减少
高维独立探索方差；若 coverage 不足，则会产生不可消除的表示偏差。因此 basis、
held-out gate 和 Full-354D 对照都是方法的一部分，不是可选的工程细节。

## 3. Wrapper 与训练路径

动作 wrapper 顺序为：

```text
optional finger isolation
→ fixed early-synergy action decoder
→ optional student/history/vector/log wrappers
→ original tracking environment
```

CPU validation、MJX validation、sequential MJX inference、MuJoCo viewer 和 Viser 都使用
同一解码接口。旧 checkpoint 与旧配置没有新增默认 action 字段，F0 行为和配置 hash
不因新方法改变。

当前普通 teacher collector/DAgger 尚未升级为 `K+R policy action + 354D applied action`
双 schema；对 early-synergy 配置会明确 fail closed，避免把低维 logits 错当成 354D
normalized muscle action。Stage-2 双动作数据集属于后续改造。

## 4. Primitive shard 的生产合同

不能把 ChinaJump target rollout、失败 rollout 或 validation motion 混入 `W`。
输入必须来自独立 primitive controller、轨迹优化器或其他已验证的可行控制生产流程。
当前仓库提供 exact-runtime `.mjb` 封存、单 trial physical-control writer、P01–P12
catalog、严格入库与 transactional pipeline；它不会替用户训练 primitive controller，
但采集器只要提供实际 `data.ctrl` 和事件 phase，后续处理已经闭环。完整操作见
`docs/chinajump_primitive_synergy_runbook.md`。

每个 train/validation shard 至少包含：

```text
teacher_ctrl_physical  [T,354]
muscle_excitation      [T,354]  # physical_excitation_unit
phase_id               [T]      # integer event phase, 禁止 float 截断
motion_uid              [T]      # stable motion identity
task_id                 [T]
trial_id                [T]      # 一个 trial 只能绑定一个 task 和一个 motion_uid
source_kind             [T]      # Phase-A 固定为 "primitive"
success                 [T]      # 必须全部成功
quality_weight          [T]      # finite and > 0
```

当且仅当还要拟合 activation basis 时，shard 另需
`muscle_activation [T,354]`，metadata 另需 `physical_capture`；本阶段命令固定
`--signals physical_excitation`，因此这两项不是 excitation-only collection 的硬要求。
若 train/validation 共用一个目录，文件必须分别命名为 `train_*.npz` 和 `val_*.npz`；
不要只放 `shard_*.npz`，否则两个 split 会读到同一组文件并因 overlap 正确拒绝。

train 和 validation 的 `metadata.json` 必须同时给出并完全一致：

```text
actuator_names / actuator_ctrlrange / ctrlrange_schema_hash
physical_signal_semantics
model_hash
source_checkpoint_fingerprints             # task -> content SHA256
source_checkpoint_contents                 # task -> 完整文件级 checkpoint audit
primitive_required_phase_ids               # task -> sorted phase ids
primitive_phase_schema_fingerprints         # task -> phase schema SHA256
```

`physical_capture` 仍只在请求 activation fitting 时加入 metadata。

每个 task 在 train 和 validation 都必须覆盖其全部 required phases；train 至少两个
trial，validation 至少一个 trial；两个 split 的 motion/trial 不得重叠。一个 task 对应
一个 primitive policy checkpoint 是当前 v2 source contract 的边界。多 seed expert、
无 checkpoint 的逆动力学来源和 optimization-only source 需要后续 schema 扩展。

## 5. Artifact 构建顺序

下面的路径是示例，所有 SHA256 都必须从实际产物读取，不能手填占位符。

### 5.1 封存 primitive source manifest

`source_checkpoints.json` 是 `task_id -> checkpoint content SHA256`；完整文件清单来自
train/validation metadata，并会再次核对。

```bash
uv run musclemimic-synergy-build-primitive-manifest \
  --train artifacts/primitive_rollouts \
  --val artifacts/primitive_rollouts \
  --output artifacts/primitive_synergy/chinajump_v1/source_manifest.json \
  --target-skill-id ChinaJump \
  --excluded-target-motion-path ChinaJump/muscle_trajectory/optimized/forehandJump-1 \
  --excluded-target-motion-path ChinaJump/muscle_trajectory/optimized/forehandJump-2 \
  --excluded-target-motion-path ChinaJump/muscle_trajectory/optimized/forehandJump-4 \
  --excluded-target-motion-path ChinaJump/muscle_trajectory/optimized/forehandJump-6 \
  --excluded-target-motion-path ChinaJump/muscle_trajectory/optimized/forehandJump-7 \
  --excluded-target-motion-path ChinaJump/muscle_trajectory/optimized/forehandJump-13 \
  --excluded-target-motion-path ChinaJump/muscle_trajectory/optimized/forehandJump-16 \
  --excluded-target-motion-path ChinaJump/muscle_trajectory/optimized/forehandJump-18 \
  --excluded-target-motion-path ChinaJump/muscle_trajectory/optimized/forehandJump-8 \
  --excluded-target-motion-path ChinaJump/muscle_trajectory/optimized/forehandJump-17 \
  --source-checkpoints-json artifacts/primitive_rollouts/source_checkpoints.json \
  --nmf-seeds 0 1 2 3 4
```

生产配置固定排除当前 train8/val2 的全部十条 ChinaJump path；实际命令也必须列出
同一清单。builder 会把 path 转成 stable motion UID，并拒绝任何 target overlap。

### 5.2 拟合 formal primitive basis

manifest 的 `NMF_seeds` 必须与 fit 的 `--seeds` 完全相同。primitive 版本至少需要
两个不同 seed；正式拟合建议 5 个。
`normalization`、`near_zero_threshold` 和自定义 `phase_weights` 也会进入 source
fingerprint；若修改，它们在 manifest builder 和 fit 两条命令中必须逐项一致。

```bash
uv run musclemimic-synergy-fit \
  --train artifacts/primitive_rollouts \
  --val artifacts/primitive_rollouts \
  --output-dir artifacts/primitive_synergy/chinajump_v1/fit \
  --signals physical_excitation \
  --mode both \
  --grouping-json artifacts/primitive_rollouts/regional_grouping.json \
  --primitive-source-manifest artifacts/primitive_synergy/chinajump_v1/source_manifest.json \
  --seeds 0 1 2 3 4
```

primitive fitting 先做 equal-task，再按配置的 phase weight 分配；每个 phase 内按
trial 的 mean quality 分配，最后乘 frame quality。每个候选 rank
都会重算 held-out balanced VAF、per-task/per-phase/per-trial VAF、initialization、
split-half、bootstrap 和 task-conditioned cross-trial stability。没有 rank 同时通过全部
门时，不写可用于 early control 的 fallback。

每个最终 decoder-ready physical basis 同目录会生成绑定其 fingerprint 的
`coefficient_stats.npz`。生产配置固定选择最低的合格 rank，并限制：

```text
policy_action_dim <= 64
condition_number <= 100
effective_rank_fraction >= 0.80
val global VAF >= 0.90
val local-VAF q10 >= 0.70
initialization/split-half/bootstrap >= 0.80
cross-trial >= 0.75
```

Phase-A 的 primary artifact 是严格 block-diagonal regional composite；whole-body global
basis 作为 comparator 保留。修改意见中的“global + regional + jump/landing/rotation
专项列联合字典、列聚类去重与 held-out usage selection”尚未实现，不能把当前 artifact
描述成完整 hybrid dictionary。

### 5.3 可选：拟合 structured residual

SR0/SR1 不再接受一个无来源的手工矩阵。先准备 mask：

```json
{
  "schema_version": "early_synergy_residual_mask_v1",
  "actuator_names": ["exact ordered 354 names"],
  "groups": [
    {
      "name": "jump_takeoff",
      "task_phase_selectors": {"jump": [1]},
      "allowed_muscle_names": ["ordered allowed muscle names"],
      "rank": 1
    }
  ]
}
```

所有 group 的 selector 必须无重叠地覆盖 manifest 中每个 task 的全部 required phase，
总 rank 必须在 4–12。每组 column 在 allowed muscles 外严格为零。

```bash
uv run musclemimic-synergy-fit-structured-residual \
  --train artifacts/primitive_rollouts \
  --val artifacts/primitive_rollouts \
  --primary-basis artifacts/primitive_synergy/chinajump_v1/fit/physical_excitation_unit/regional_composite \
  --coefficient-stats artifacts/primitive_synergy/chinajump_v1/fit/physical_excitation_unit/regional_composite/coefficient_stats.npz \
  --primitive-source-manifest artifacts/primitive_synergy/chinajump_v1/source_manifest.json \
  --expected-primitive-source-fingerprint <64_hex> \
  --residual-mask artifacts/primitive_synergy/chinajump_v1/residual_mask.json \
  --output artifacts/primitive_synergy/chinajump_v1/residual_basis \
  --alpha 0.03 \
  --min-dimension 4 \
  --max-dimension 12 \
  --max-row-l1-norm 2.0 \
  --min-validation-residual-energy-reduction 0.01 \
  --min-group-validation-residual-energy-reduction 0.01 \
  --max-validation-coordinate-saturation-fraction 0.75
```

builder 用运行时相同的 `c∈[0,1.2×q99]` 有界 NNLS 解码 primary `W`，只在 train
unexplained excitation 上按 task/phase mask 做 signed SVD，再用 validation 评估
residual-energy reduction 和 coordinate saturation。held-out gate 未过时不会写 runtime
artifact。residual v3 manifest 绑定 mask、primary basis、stats、source dataset、primitive
manifest、alpha、solver tolerance/max-iterations/energy epsilon 和 train/validation content
fingerprints。全局 validation 与每个 task/phase group 都必须达到 improvement 门槛且
coordinate saturation 不超过 0.75；primary residual energy 为零的 group 记为 0 改善，
不能用 `0/0` 空证据通过。

### 5.4 构建 ChinaJump static coverage gate

proxy NPZ 至少包含：

```text
physical_excitation  [T,354]
phase_id             [T]
actuator_names       [354]
```

本实验配置把 ChinaJump coverage schema 固定在
`fullbody/config_specific_task/stage1_body/chinajump_coverage_phase_schema_v1.json`：

```text
1 = takeoff_propulsion
2 = rotation_flight_adjustment
3 = landing_impact_absorption
4 = post_landing_balance
```

该 JSON 同时给出了每段的 contact-event 边界定义。proxy producer 必须明确采用这一
schema；它与各 primitive 自己的 task-specific phase schema 不是同一个东西。gate 会把
完整编号/名称/定义及其 SHA256 写入 proxy fingerprint，四个 S/SR runtime 配置固定期望
fingerprint；即使 1–4 都存在，交换 landing/flight 的语义也会在训练前失败。示例命令与
runtime 配置都使用 1–4：

```bash
uv run musclemimic-synergy-static-coverage \
  --basis-artifact <formal_basis_dir> \
  --coefficient-stats <formal_basis_dir>/coefficient_stats.npz \
  --proxy-manifest artifacts/chinajump_proxy/proxy_manifest.json \
  --phase-schema fullbody/config_specific_task/stage1_body/chinajump_coverage_phase_schema_v1.json \
  --output artifacts/chinajump_proxy/static_coverage_gate.json \
  --required-phase-id 1 \
  --required-phase-id 2 \
  --required-phase-id 3 \
  --required-phase-id 4
```

每个 required phase 不仅要存在，其 target RMS 还必须非零。gate 同时检查全局/逐阶段
relative L2 NRMSE、active-muscle fraction、decoded saturation、basis condition number 和
effective rank，并绑定 proxy 内容与 coefficient upper bounds。

这是静态 excitation reconstruction proxy，不是短时 dynamics oracle。仓库现已提供
严格的 target-control proxy producer：它只封存成功 full-354D teacher 或经 forward
replay 的 full-action optimizer 控制，并把 source/QC/phase/model provenance 绑定到 gate。
它不会从失败的 ChinaJump、WHAM、SMPL 或 qpos/qvel 自动猜 excitation；在独立目标控制
和后续 dynamics oracle 产出可信证据前，不能声称跳跃/转体/落地的动力学 coverage 已被证明。

## 6. 对照矩阵

| ID | 配置 | policy action | ASI |
|---|---|---|---|
| F0 | `conf_fullbody_chinajump_root_control_v2` | Full-354D | 关 |
| F1 | `conf_fullbody_chinajump_full_asi` | Full-354D | 开 |
| S0 | `conf_fullbody_chinajump_early_synergy` | fixed `W`, K-D | 关 |
| S1 | `conf_fullbody_chinajump_early_synergy_asi` | fixed `W`, K-D | 开 |
| SR0 | `conf_fullbody_chinajump_early_synergy_residual` | fixed `W+R`, K+r-D | 关 |
| SR1 | `conf_fullbody_chinajump_early_synergy_residual_asi` | fixed `W+R`, K+r-D | 开 |

F1 只打开现有 ASI，不声称包含 hard-trajectory mining。每组使用新 run id 和 fresh
optimizer，不允许从动作 ABI、reward 或 terminal contract 不兼容的 checkpoint resume。

只完成 primitive 数据、尚无独立 ChinaJump target-control proxy 时，使用隔离的
`conf_fullbody_chinajump_early_synergy_bootstrap{,_asi}`（B0/B1）。二者不要求 coverage
gate，但 checkpoint action manifest 会显式记录
`not_evaluated_primitive_bootstrap`；不能把 B0/B1 改名报告为正式 S0/S1。

## 7. 训练前环境绑定

S0/S1/SR0/SR1 必须设置：

```bash
export MUSCLEMIMIC_CHINAJUMP_SYNERGY_BASIS=<formal_basis_dir>
export MUSCLEMIMIC_CHINAJUMP_SYNERGY_BASIS_FINGERPRINT=<64_hex>
export MUSCLEMIMIC_CHINAJUMP_PRIMITIVE_SOURCE_MANIFEST=<source_manifest.json>
export MUSCLEMIMIC_CHINAJUMP_PRIMITIVE_SOURCE_FINGERPRINT=<64_hex>
export MUSCLEMIMIC_CHINAJUMP_SYNERGY_COEFFICIENT_STATS_FINGERPRINT=<64_hex>
export MUSCLEMIMIC_CHINAJUMP_SYNERGY_COVERAGE_GATE=<static_coverage_gate.json>
export MUSCLEMIMIC_CHINAJUMP_SYNERGY_COVERAGE_GATE_FINGERPRINT=<64_hex>
export MUSCLEMIMIC_CHINAJUMP_SYNERGY_PROXY_FINGERPRINT=<64_hex>
```

SR0/SR1 另需：

```bash
export MUSCLEMIMIC_CHINAJUMP_SYNERGY_RESIDUAL_BASIS=<residual_basis_dir>
export MUSCLEMIMIC_CHINAJUMP_SYNERGY_RESIDUAL_FINGERPRINT=<64_hex>
```

任何空路径、fingerprint 漂移、target-motion exclusion 不一致、model/ctrlrange/354D
actuator ABI 不一致、rank gate/coverage gate 未通过、stats bounds 不一致、residual alpha
或 mask/fit contract 不一致，都会在 PPO 开始前失败。

B0/B1 只绑定 basis、primitive source 和 coefficient statistics；它们由
`musclemimic-chinajump-synergy-pipeline apply --readiness bootstrap` 发布独立 bindings，
明确不包含 coverage/proxy 环境变量。正式 S/SR 配置仍保持上面的严格要求。

这些配置仍标记为 experimental。先通过 canonical launcher 做非启动检查：

```bash
export CUDA_VISIBLE_DEVICES=<physical_gpu_index>
export MUSCLEMIMIC_JAX_CACHE_KEY=chinajump_stage1_s0_early_synergy
export MUSCLEMIMIC_TRAIN_LOG=datasets/ChinaJump/training/logs/chinajump_stage1_s0_early_synergy.log
export MUSCLEMIMIC_DRY_RUN=1
scripts/run_fullbody_training.sh \
  --config-name=config_specific_task/stage1_body/conf_fullbody_chinajump_early_synergy \
  config_status.allow_nonproduction_runtime=true \
  wandb.mode=online
```

实际训练只能遵守根目录 `AGENTS.md`，通过 `scripts/run_fullbody_training.sh`、显式物理
GPU、独立 JAX cache、append-only log 和命名 tmux 启动，并完成其中全部 pre-flight 与
post-launch 验证。

## 8. 比较指标与结论规则

主指标仍是相同 validation set 上的：

```text
episode return / full-motion success / frame coverage / early termination
root, joint, site tracking error
activation energy / full physical-action saturation
wall-clock time / environment steps to a fixed success threshold
```

新增解释指标：

```text
synergy/coefficient_mean
synergy/coefficient_max
synergy/coefficient_saturation_fraction
synergy/coefficient_effective_dimension
synergy/decoded_excitation_mean
synergy/decoded_excitation_rms
synergy/decoded_excitation_saturation_fraction
synergy/residual_l1
synergy/residual_l2
synergy/residual_energy_fraction
```

验证侧使用对应 `val_synergy_*`。第一轮先看 F0/F1/S0/S1；SR0/SR1 用来判断小范围
结构化表达补偿是否必要，不能把 residual 当作 354D bypass。

正式方法比较至少使用 3–5 个相同 RL seeds，并报告均值、方差和失败率。NMF 的多个
initialization seeds 是 basis 稳定性证据，不能替代 PPO 多 seed。当前代码和测试只建立
实验基础设施，没有真实 basis、coverage proxy 或六组训练结果，因此不能预先声称新方法
优于纯轨迹跟踪。

低维组的 0.08 physical RMS 标定是动作表示设计的一部分，并不保证与 F0 的初始
354D physical perturbation RMS 完全相同。如果主结论对早期 exploration amplitude 敏感，
还应增加一个 Full-354D physical-RMS-matched ablation，不能把幅度效应全部归因于降维。

## 9. 当前 Phase-A 边界

- 已实现：opt-in 低维 Stage-1 action、固定 W、bounded coefficient transform、物理
  exploration 标定、strict primitive manifest v2、P01–P12 recording/ingest、formal rank
  gates、sealed target-control static coverage、transactional plan/apply/preflight、bootstrap/
  formal readiness、structured residual 自动拟合、CPU/MJX/Viser 解码和公平配置。
- 未实现：自动训练 primitive controller、global+regional+phase/task columns 的完整 hybrid
  composer、短时 dynamics oracle、phase-conditioned W bank、residual alpha curriculum、
  Stage-2 双动作 collector/schema。首轮 B0/B1 使用当前严格 regional composite；这些
  后续项不阻断 primitive 数据后的 bootstrap，但会限制可声称的方法范围。

这些边界必须随实验结果一起报告，避免把“representation plumbing 可运行”误写成
“复杂 ChinaJump 已经学会”或“数学上无条件优于 Full-354D”。
