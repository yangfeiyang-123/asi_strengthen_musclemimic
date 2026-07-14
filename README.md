# MuscleMimic Badminton

面向羽毛球全身动作学习的肌肉骨骼强化学习与模仿学习代码库。

本项目基于 [MuscleMimic](https://github.com/amathislab/musclemimic)，围绕 MyoFullBody 肌肉驱动模型扩展了羽毛球动作重定向、来球击打环境、分阶段策略训练、教师—学生蒸馏、DAgger、latent muscle policy、训练恢复以及发布前质量门控。

> 本仓库发布源码、锁文件以及能够在干净 clone 中复现流水线结构的少量 canonical YAML。数据集、授权模型、checkpoint、训练输出、视频和机器相关实验配置不会提交到 Git；根目录 `README.md` 是唯一跟踪的 Markdown 文件。

## 项目目标

项目希望建立从人体运动参考到可训练羽毛球策略的完整代码链路：

1. 将 WHAM/SMPL/AMASS 动作转换并重定向到 MyoFullBody。
2. 检查帧率、轨迹连续性、手部速度、关节跳变和重定向误差。
3. 在 MuJoCo/MJX 中构建球场、球拍、羽毛球和来球击打任务。
4. 通过模仿学习、PPO、BC、DAgger 和 latent policy 学习全身击球控制。
5. 使用数据契约、训练门控、checkpoint 恢复和鲁棒性评估保证实验可复现。

## 主要能力

- **肌肉骨骼控制**：支持 MyoFullBody 等高维肌肉驱动模型。
- **GPU 并行训练**：基于 JAX、MJX、Flax、Optax 和 Warp。
- **羽毛球环境**：包含球拍握持、来球生成、碰撞、击球目标与分层控制逻辑。
- **动作重定向**：提供 WHAM/SMPL/AMASS 到 GMR/MyoFullBody 的处理脚本。
- **轨迹质量控制**：检查 FPS 一致性、根节点与手部运动连续性、缓存质量及训练准入条件。
- **三阶段训练**：身体动作、球拍控制和击球技能可按阶段训练与验收。
- **策略蒸馏**：支持 teacher rollout、行为克隆、DAgger、student evaluation 与 provenance 记录。
- **Latent muscle policy**：支持 latent action、归一化、闭环评估和高层策略接口。
- **可靠训练**：支持自动恢复、checkpoint 限额、验证指标、视频记录接口和 early stop。

## 代码结构

```text
musclemimic/
├── musclemimic/
│   ├── algorithms/          # PPO、推理、checkpoint 与通用训练组件
│   ├── badminton/           # 羽毛球数据 QC、训练门控、重定向和评估脚本
│   ├── core/                # 奖励、wrapper 与环境公共逻辑
│   ├── distill/             # BC/DAgger、数据契约、动作与观测 schema
│   ├── environments/        # 肌肉骨骼环境
│   ├── latent_muscle/       # latent policy 训练与运行时
│   └── runner/              # 训练引擎、恢复、日志和验证
├── environment/
│   ├── overall_environment/ # 羽毛球综合场景与 MJX 来球训练环境
│   ├── court/               # 球场模型
│   ├── racket/              # 球拍模型
│   └── shuttlecock/         # 羽毛球动力学与碰撞测试
├── fullbody/                # 全身训练、评估、蒸馏和三阶段流程入口
├── bimanual/                # 双臂模型训练与评估入口
├── loco_mujoco/             # 模型、数据加载与 SMPL/GMR 支撑代码
├── scripts/                 # 数据与动作扫描工具
├── tests/                   # 单元测试和集成测试
└── pyproject.toml           # Python 包与命令行入口
```

## 环境要求

- Python 3.11+
- Linux（训练推荐）或 macOS（CPU 推理/部分评估）
- 使用 GPU 训练时需要兼容的 NVIDIA 驱动和 CUDA 12 环境
- 推荐使用 [uv](https://docs.astral.sh/uv/) 管理依赖

## 安装

```bash
git clone https://github.com/yangfeiyang-123/asi_strengthen_musclemimic.git
cd asi_strengthen_musclemimic

# 基础依赖与开发测试工具
uv sync --extra dev
```

Linux GPU 环境：

```bash
uv sync --extra dev --extra cuda
```

如果需要 SMPL/GMR 动作重定向：

```bash
uv sync --extra dev --extra smpl --extra gmr
```

安装完成后可先检查包导入和命令行入口：

```bash
uv run python -c "import musclemimic; print('musclemimic import OK')"
uv run forehand-clear-three-stage --help
uv run forehand-clear-data-qc --help
```

## 本地配置与资产

为了避免把授权数据和机器相关配置提交到代码仓库，以下内容需要在本地准备：

- 非 canonical 的个人 Hydra/YAML 覆盖配置；
- WHAM、SMPL、AMASS 或其他动作数据；
- Myo/MuJoCo、SMPL-H、MANO 等模型资产；
- GMR 重定向缓存；
- teacher/student checkpoint；
- W&B 日志、评估结果、渲染图片与视频。

常用本地目录包括：

```text
data/
datasets/
caches/
checkpoints/
ckpts/
outputs/
wandb/
smpl_models/
```

这些文件受 `.gitignore` 保护。Stage 1/2、蒸馏、Stage 3 以及新研究流水线所需的 portable canonical YAML 随源码发布；其中只包含相对路径或待注入的 artifact 参数，不包含数据与 checkpoint。运行资产依赖步骤前仍需按配置准备本地数据。

## 常用入口

### 来球击打环境

```bash
uv run python environment/overall_environment/src/train_incoming_hit_mjx.py --help
uv run python environment/overall_environment/src/overall_env.py --help
```

### 羽毛球三阶段流程

```bash
uv run forehand-clear-three-stage --help
uv run forehand-clear-promotion-gate --help
uv run forehand-clear-finger-robustness --help
```

### 教师—学生蒸馏

```bash
uv run musclemimic-distill-collect-teacher --help
uv run musclemimic-distill-train-bc --help
uv run musclemimic-distill-run-dagger --help
uv run musclemimic-distill-compare --help
```

### Latent muscle policy

```bash
uv run musclemimic-latent-train --help
uv run musclemimic-latent-eval --help
uv run musclemimic-latent-closed-loop-eval --help
```

### 重定向与数据质量检查

```bash
uv run python -m musclemimic.badminton.scripts.run_retarget --help
uv run python -m musclemimic.badminton.scripts.render_retarget_cache --help
uv run forehand-clear-data-qc --help
uv run forehand-clear-visual-review --help
```

### Synergy v3 研究流水线（显式 opt-in）

这一研究 profile 的核心问题是：在全身肌肉驱动的正手高远球中，控制 latent
的**有效复杂度**是否会随神经肌肉协同复杂度共同变化；无约束 direct latent
是否会自然对齐物理 excitation 的协同子空间；固定生理协同先验能否在保持
击球能力的同时改善控制紧凑性、可解释性和生理合理性。

为避免把“结构约束”误写成“自然发现”，主比较固定为：state-only、direct
latent、`latent -> fixed Wc`，以及 `latent -> fixed Wc + 10-D distal residual`。
NMF 只接收 simulator 的非负物理 excitation 或 activation，绝不接收归一化的
有符号 action；excitation 与 activation 分开拟合、分开报告。维度筛选覆盖
`2/4/8/16/32` 与多个预注册 seed，随后把 sealed best-direct 和 best-synergy
在完全相同的 held-out feed、target bank、seed 和 Stage-3 指标下成对比较。

完整验证链依次约束：measured/fused racket 与六阶段 event reference、真实球拍
质量 `0 -> 25% -> 50% -> 75% -> 100%` 课程、物理肌肉 rollout、held-out NMF、
latent 子空间/Jacobian/干预、impact 状态、落点与 recovery，最后才是右上肢局部
sEMG 和全身生理指标。离线 decoder 扰动只能称为 excitation-level intervention；
只有绑定 checkpoint、样本、方向和 epsilon 的环境 rollout 才能支持关节、球拍、
impact 或落点的因果表述。

默认 profile 始终是 `legacy_v2`，其 Stage 1/2→蒸馏→Stage 3 顺序不变。新研究路径只写入调用者指定目录下的 `synergy_v3/` namespace；先生成命令计划，不会启动训练：

```bash
uv run python -m fullbody.run_forehand_clear_pipeline \
  --profile synergy_v3 \
  --output_dir outputs/research_plans/forehand_clear_synergy_v3
```

数据与 checkpoint 准备后，各框架可独立运行。下面的变量均指向你后续生成的本地
artifact；命令不会复用旧 run 目录，且所有训练/评估动作都需要显式执行。
`TEACHER_CONTROL_DT` 必须取 teacher checkpoint 环境的真实控制周期
`timestep * n_substeps`（MyoFullBody 默认是 `0.002 * 5 = 0.01 s`），不是原始
reference 视频的帧周期；event bank、collector 和 QC 会对此做指纹绑定并拒绝不一致。

```bash
set -euo pipefail

# 1. 为每条已审核 v2 reference 生成 tracking cache，再分别建立 train/val bank
uv run badminton-build-tracking-reference-cache \
  --manifest "$REFERENCE_BUNDLE" --out-dir "$TRACKING_CACHE_DIR" \
  --control-dt "$TEACHER_CONTROL_DT"
uv run python -m musclemimic.badminton.data.event_lookup \
  --entries-json "$TRAIN_EVENT_ENTRIES" --output "$TRAIN_EVENT_BANK"
uv run python -m musclemimic.badminton.data.event_lookup \
  --entries-json "$VAL_EVENT_ENTRIES" --output "$VAL_EVENT_BANK"
uv run forehand-clear-event-reference-qc \
  --train-manifests-json "$TRAIN_REFERENCE_MANIFESTS" \
  --val-manifests-json "$VAL_REFERENCE_MANIFESTS" \
  --train-event-bank "$TRAIN_EVENT_BANK" --val-event-bank "$VAL_EVENT_BANK" \
  --output "$EVENT_REFERENCE_METRICS"
uv run forehand-clear-promotion-gate \
  --stage event_reference_v2 --metrics "$EVENT_REFERENCE_METRICS" \
  --output outputs/synergy_v3/event_reference_gate.json --require_pass

# 2. 使用已晋级的 100% 球拍质量 teacher，分开采集 train/val 物理 rollout
uv run python -m fullbody.distill_collect --teacher_ckpt "$MASS100_TEACHER_CKPT" \
  --output_dir outputs/synergy_v3/physical/train --num_transitions 1000000 \
  --teacher-promotion-manifest "$MASS100_PROMOTION_MANIFEST" \
  --event-reference-bank "$TRAIN_EVENT_BANK" --motion_path $TRAIN_MOTION_PATHS \
  --save-physical-muscle-state --save-event-features --save_reference_features \
  --physical-racket-site-name racket_stringbed_center_site --split train
uv run python -m fullbody.distill_collect --teacher_ckpt "$MASS100_TEACHER_CKPT" \
  --output_dir outputs/synergy_v3/physical/val --num_transitions 200000 \
  --teacher-promotion-manifest "$MASS100_PROMOTION_MANIFEST" \
  --event-reference-bank "$VAL_EVENT_BANK" --motion_path $VAL_MOTION_PATHS \
  --save-physical-muscle-state --save-event-features --save_reference_features \
  --physical-racket-site-name racket_stringbed_center_site --split val
uv run musclemimic-physical-rollout-qc \
  --train outputs/synergy_v3/physical/train --val outputs/synergy_v3/physical/val \
  --teacher-checkpoint-fingerprint "$MASS100_TEACHER_SHA256" \
  --event-reference-metrics "$EVENT_REFERENCE_METRICS" \
  --output outputs/synergy_v3/physical/physical_rollout_qc.json
uv run forehand-clear-promotion-gate \
  --stage physical_rollout_v2 \
  --metrics outputs/synergy_v3/physical/physical_rollout_qc.json \
  --output outputs/synergy_v3/physical_rollout_gate.json --require_pass

# 3. 先建立同一物理 rollout 上的 direct-BC 对照与晋级证据
uv run python -m fullbody.distill_train_bc \
  --dataset_dir outputs/synergy_v3/physical/train \
  --student_config \
    fullbody/config_specific_task/distill/conf_fullbody_forehandclear_racket_student_phase_bc.yaml \
  --output_dir outputs/synergy_v3/direct_baseline/bc \
  --batch_size 4096 --num_steps 200000 --lr 0.0003 --seed 0 \
  --require_dataset_manifest
uv run python -m fullbody.distill_compare \
  --teacher_ckpt "$MASS100_TEACHER_CKPT" \
  --student_ckpt outputs/synergy_v3/direct_baseline/bc/checkpoints/checkpoint_200000 \
  --output_dir outputs/synergy_v3/direct_baseline/compare \
  --dataset_dir outputs/synergy_v3/physical/val \
  --convergence_metrics outputs/synergy_v3/direct_baseline/bc/distill_metadata.json \
  --motion_path $VAL_MOTION_PATHS --metrics_envs 20 --metrics_steps 500 \
  --eval_seed 0 --deterministic --promotion_policy student_bc --require_pass
export DIRECT_BC_METRICS=outputs/synergy_v3/direct_baseline/bc/distill_metadata.json
export DIRECT_ROLLOUT_METRICS=outputs/synergy_v3/direct_baseline/compare/comparison_metrics.json
export DIRECT_PROMOTION_EVIDENCE=outputs/synergy_v3/direct_baseline/compare/direct_promotion_evidence.json

# 4. excitation/activation 分开做 held-out NMF；regional/both 使用固定 354-D 分组
uv run musclemimic-synergy-fit \
  --train outputs/synergy_v3/physical/train --val outputs/synergy_v3/physical/val \
  --output-dir outputs/synergy_v3/synergy --mode both \
  --signals excitation activation --ranks 1 2 3 4 5 6 7 8 9 10 \
  --grouping-json experiments/synergy/forehand_clear_myofullbody_354_regions_v1.json
uv run forehand-clear-promotion-gate \
  --stage synergy_v2 --metrics outputs/synergy_v3/synergy/promotion_metrics.json \
  --output outputs/synergy_v3/synergy_gate.json --require_pass

# 5. 2/4/8/16/32 × direct/fixed-synergy/synergy-residual × seed 生命周期
# plan 只写 manifest；execute 才训练；evaluate 产生 closed-loop 与因果 bootstrap 输入。
uv run musclemimic-latent-synergy-sweep plan \
  --dataset-dir outputs/synergy_v3/physical/train \
  --val-dataset-dir outputs/synergy_v3/physical/val \
  --teacher-ckpt "$MASS100_TEACHER_CKPT" \
  --teacher-promotion-manifest "$MASS100_PROMOTION_MANIFEST" \
  --direct-bc-metrics "$DIRECT_BC_METRICS" \
  --direct-rollout-metrics "$DIRECT_ROLLOUT_METRICS" \
  --direct-promotion-evidence "$DIRECT_PROMOTION_EVIDENCE" \
  --synergy-basis "$SYNERGY_BASIS" --synergy-basis-fingerprint "$SYNERGY_BASIS_SHA256" \
  --output-dir outputs/synergy_v3/latent \
  --dimensions 2 4 8 16 32 --seeds 0 1 2 \
  --require-causal-interventions
uv run musclemimic-latent-synergy-sweep execute \
  --output-dir outputs/synergy_v3/latent --stage train
uv run musclemimic-latent-synergy-sweep evaluate \
  --output-dir outputs/synergy_v3/latent

# 6. 自动对 sweep_plan.json 的每一个注册 run 做 Stage-2 真实环境级诊断并封存；
# 不允许只跑晋级候选。该前置诊断强制 excitation/activation/joint/trunk/racket，
# 但没有 shuttle task 的 impact/landing，不能据此声称任务级因果效果。
# 共享 JSON 只放 horizon 等公共参数，checkpoint、teacher、train/val dataset 和
# analysis_inputs 均由 fingerprinted plan 逐 run 强制注入。
export CAUSAL_ADAPTER_CONFIG=configs/public/latent_causal_adapter_shared_config_template.json
uv run musclemimic-latent-synergy-sweep causal-evaluate \
  --output-dir outputs/synergy_v3/latent \
  --shared-config "$CAUSAL_ADAPTER_CONFIG"
# 全部注册 run 的 driver + paired artifact 完成后，第二遍封存再统计/独立选优。
uv run musclemimic-latent-synergy-sweep finalize-causal \
  --output-dir outputs/synergy_v3/latent
uv run musclemimic-latent-synergy-sweep analyze \
  --output-dir outputs/synergy_v3/latent \
  --require-all-phases --require-causal-interventions
uv run forehand-clear-promotion-gate \
  --stage latent_synergy_v2 --metrics outputs/synergy_v3/latent/promotion_metrics.json \
  --output outputs/synergy_v3/latent_synergy_gate.json --require_pass

# 7. Stage 3 v2：建立 train/eval 目标库，再对 direct/synergy 做完全对称的生命周期。
uv run python -m environment.overall_environment.src.stage3_target_bank_v2 --dry-run
export STAGE3_TRAIN_FEED_BANK=outputs/synergy_v3/stage3_feeds/train.npz
export STAGE3_EVAL_FEED_BANK=outputs/synergy_v3/stage3_feeds/eval.npz
# 目标库必须绑定 runner 后续加载的同一 feed 样本和顺序；先确定性物化两个 bank。
uv run python -m musclemimic.badminton.scripts.run_incoming_shuttle_hit \
  --spec experiments/posttrain/incoming_shuttle_hit_impact_recovery_v2.yaml \
  --stage feed-check --feed-bank "$STAGE3_TRAIN_FEED_BANK" \
  --eval-feed-bank "$STAGE3_EVAL_FEED_BANK" \
  --out-dir outputs/synergy_v3/stage3_feed_contract
uv run python -m environment.overall_environment.src.stage3_target_bank_v2 \
  --input-jsonl "$STAGE3_TRAIN_IMPACT_TARGETS" \
  --event-reference-metrics "$EVENT_REFERENCE_METRICS" --reference-split train \
  --feed-bank-path "$STAGE3_TRAIN_FEED_BANK" --consumer-order difficulty_sorted \
  --output outputs/synergy_v3/stage3_targets/train_targets_v2.json
uv run python -m environment.overall_environment.src.stage3_target_bank_v2 \
  --input-jsonl "$STAGE3_EVAL_IMPACT_TARGETS" \
  --event-reference-metrics "$EVENT_REFERENCE_METRICS" --reference-split validation \
  --feed-bank-path "$STAGE3_EVAL_FEED_BANK" --consumer-order stored \
  --output outputs/synergy_v3/stage3_targets/eval_targets_v2.json
export STAGE3_TRAIN_TARGET_BANK=outputs/synergy_v3/stage3_targets/train_targets_v2.json
export STAGE3_EVAL_TARGET_BANK=outputs/synergy_v3/stage3_targets/eval_targets_v2.json

run_stage3_branch() (
  set -euo pipefail
  local latent_checkpoint="$1"
  local branch_dir="$2"
  local runner=(uv run python -m musclemimic.badminton.scripts.run_incoming_shuttle_hit)
  local common=(
    --spec experiments/posttrain/incoming_shuttle_hit_impact_recovery_v2.yaml
    --feed-bank "$STAGE3_TRAIN_FEED_BANK"
    --eval-feed-bank "$STAGE3_EVAL_FEED_BANK"
    --target-bank "$STAGE3_TRAIN_TARGET_BANK"
    --eval-target-bank "$STAGE3_EVAL_TARGET_BANK"
  )

  "${runner[@]}" "${common[@]}" --stage preflight --out-dir "$branch_dir"
  "${runner[@]}" "${common[@]}" --stage feed-check --out-dir "$branch_dir"
  "${runner[@]}" "${common[@]}" --stage base-only-check \
    --latent-checkpoint "$latent_checkpoint" --out-dir "$branch_dir"

  # C0--C3 静态速度目标：总计 6M env steps，先评估并通过门槛。
  "${runner[@]}" "${common[@]}" --stage train-gpu \
    --latent-checkpoint "$latent_checkpoint" --total-env-steps 6000000 \
    --curriculum-max-stage C3_static_velocity --seed 0 --out-dir "$branch_dir"
  "${runner[@]}" "${common[@]}" --stage evaluate \
    --checkpoint "$branch_dir/policy_latest.json" --episodes 128 \
    --out-dir "$branch_dir/evaluate_static"
  uv run forehand-clear-promotion-gate --stage static_target_v2 \
    --metrics "$branch_dir/evaluate_static/evaluate_report.json" \
    --output "$branch_dir/static_target_gate.json" --require_pass

  # 从 C3 封存 checkpoint 继续 C4--C7；30M 是恢复训练的总 env-step 目标。
  "${runner[@]}" "${common[@]}" --stage train-gpu \
    --latent-checkpoint "$latent_checkpoint" --total-env-steps 30000000 \
    --curriculum-max-stage C7_recovery --resume-from "$branch_dir/policy_latest.json" \
    --seed 0 --out-dir "$branch_dir"
  "${runner[@]}" "${common[@]}" --stage evaluate \
    --checkpoint "$branch_dir/policy_latest.json" --episodes 128 \
    --out-dir "$branch_dir/evaluate"
  uv run forehand-clear-promotion-gate --stage stage3_v2 \
    --metrics "$branch_dir/evaluate/evaluate_report.json" \
    --output "$branch_dir/stage3_v2_gate.json" --require_pass
)

# analyze 会在同一预注册矩阵内独立封存 best_direct/best_synergy。
run_stage3_branch outputs/synergy_v3/latent/selected/best_direct \
  outputs/synergy_v3/stage3_direct
run_stage3_branch outputs/synergy_v3/latent/selected/best_synergy \
  outputs/synergy_v3/stage3_synergy

# 两分支使用相同 seed/feed/target，最后做 episode-level bootstrap 配对比较。
uv run python -m musclemimic.badminton.stage3_paired_comparison \
  --direct-report outputs/synergy_v3/stage3_direct/evaluate/evaluate_report.json \
  --synergy-report outputs/synergy_v3/stage3_synergy/evaluate/evaluate_report.json \
  --selection-manifest outputs/synergy_v3/latent/selected/selection_manifest.json \
  --output outputs/synergy_v3/stage3_paired/paired_comparison.json

# 只有 C7 direct/synergy 成对评估后，才能在真实 shuttle task 上做完整因果验证。
# 先复制 public template，再填写 paired comparison 和 direct/synergy 各自的
# analysis_inputs/manifest。sample point 必须严格位于首次真实击球之前；horizon 必须
# 覆盖 snapshot 到 episode 上限的全部剩余步数（canonical template 为 420）。
export STAGE3_TASK_CAUSAL_CONFIG=configs/public/latent_task_causal_v1_template.json
uv run python -m musclemimic.badminton.stage3_task_causal \
  --config "$STAGE3_TASK_CAUSAL_CONFIG"
uv run forehand-clear-promotion-gate \
  --stage latent_task_causal_v1 \
  --metrics outputs/synergy_v3/stage3_task_causal/promotion_metrics.json \
  --output outputs/synergy_v3/stage3_task_causal_gate.json --require_pass

# 8. 用 paired comparison 选定的 synergy C7 checkpoint 重放最终策略并导出物理信号。
# identity JSON 只列出已在前一次确定性评估中确认成功且有真实受试者/试次身份的
# held-out feeds；runner 仍完整跑 128 个 held-out feeds，但只采集清单中的窗口。
export STAGE3_SIGNAL_IDENTITY=configs/public/stage3_signal_trial_identity_template.json
export STAGE3_SIGNAL_NPZ=outputs/synergy_v3/stage3_signal/simulation_signals.npz
uv run python -m musclemimic.badminton.scripts.run_incoming_shuttle_hit \
  --spec experiments/posttrain/incoming_shuttle_hit_impact_recovery_v2.yaml \
  --stage evaluate --checkpoint outputs/synergy_v3/stage3_synergy/policy_latest.json \
  --episodes 128 --feed-bank "$STAGE3_TRAIN_FEED_BANK" \
  --eval-feed-bank "$STAGE3_EVAL_FEED_BANK" \
  --target-bank "$STAGE3_TRAIN_TARGET_BANK" \
  --eval-target-bank "$STAGE3_EVAL_TARGET_BANK" \
  --out-dir outputs/synergy_v3/stage3_signal/evaluate \
  --export-simulation-npz "$STAGE3_SIGNAL_NPZ" \
  --signal-identity-json "$STAGE3_SIGNAL_IDENTITY" \
  --policy-evidence-json outputs/synergy_v3/stage3_paired/paired_comparison.json

# 有真实 sEMG 后，两个 evaluator 直接从 paired report/identity manifest 推导并核验
# checkpoint、promotion、formal W、event、decoder 和 session 绑定，无需手抄哈希。
uv run musclemimic-emg-eval \
  --simulation-npz "$STAGE3_SIGNAL_NPZ" --emg-npz "$HELDOUT_EMG_NPZ" \
  --mapping-json configs/public/emg_right_upper_limb_mapping_template.json \
  --policy-evidence-json outputs/synergy_v3/stage3_paired/paired_comparison.json \
  --output-json outputs/synergy_v3/emg/report.json
uv run musclemimic-physiology-eval \
  --input-npz "$STAGE3_SIGNAL_NPZ" \
  --evaluation-config-json configs/public/physiology_evaluation_template.json \
  --policy-evidence-json outputs/synergy_v3/stage3_paired/paired_comparison.json \
  --signal-identity-json "$STAGE3_SIGNAL_IDENTITY" \
  --output-json outputs/synergy_v3/physiology/report.json

# 尚无 EMG/生理数据时只检查输入契约，不伪造实验结果。
uv run python -m musclemimic.evaluation.emg_eval --dry-run
uv run python -m musclemimic.evaluation.physiology --dry-run

# 9. 实验完成后生成含失败 run、均值/标准差、bootstrap CI 与 paired dz 的消融表
uv run python -m musclemimic.badminton.scripts.build_forehand_clear_ablation_report \
  --input-jsonl "$ABLATION_RUNS_JSONL" --output-dir outputs/synergy_v3/ablation
```

`causal-evaluate` 默认使用仓库内的 Stage-2 adapter，并为每个 run 单独启动真实
`causal_rollout_driver`、验证完整 simulator snapshot/共同随机数，再调用 paired artifact
封存器。前置 latent gate 只接受六类 Stage-2 diagnostic outcome；impact/landing 为空时会
显式标记 unavailable，而不是用零或 NaN 冒充测量值。最终任务级因果结论必须再通过
post-C7 `latent_task_causal_v1`，其中八类 outcome、impact/landing mask、全干预矩阵和
direct/synergy checkpoint 绑定均须完整。共享 config 禁止覆盖 plan 绑定的 checkpoint、
数据集或 analysis 输入；`replay-record` 也被批处理入口明确拒绝。以上命令均为新入口；不会复用或自动恢复当前
`legacy_v2` run directory。

科学结论边界：当前 Stage 3 的 strong weld、禁用 hand–racket contact 和 constant grip 只能称为 **rigidly attached racket baseline**。它不支持真实握力、拍柄滑移或手指协同方面的结论。当前 6-channel sEMG 也只提供右上肢局部验证，不能外推为对全身 354 个肌肉执行器的实验验证；全身结论必须等待覆盖相应肌群、受试者和 held-out session 的独立数据。

## 测试

干净 clone（不含数据、checkpoint 和 SMPL 资产）先运行源码发布契约：

```bash
make source-only
```

该检查覆盖包导入、严格 JSON、canonical Hydra 配置组合、流水线计划和纯 CPU 契约测试。完整本地模型与数据准备好后运行更广的测试：

```bash
make asset-test
```

CI 的 `source-only` job 对 pull request 和 main 分支 push 自动运行；资产依赖 job 仅能通过手动 workflow dispatch 在预先配置的 runner 上启动。部分集成测试会根据 GPU、Warp、模型文件或数据是否存在自动跳过。

## 数据与提交策略

仓库只跟踪源代码、测试、锁文件、必要的 JSON/XML/TOML 代码资产、portable canonical YAML 和根目录 README。下列内容不会跟踪：

- `*.md`（根目录 `README.md` 除外）
- 非 allowlist 的 `*.yaml`、`*.yml`（尤其是含本机路径的实验覆盖）
- `*.gif`、本地生成的 PNG 和视频
- `*.pt`、`*.pth`、`*.ckpt`、`*.pkl`、`*.npz`
- 数据集、缓存、checkpoint、日志和训练输出目录

提交前建议检查：

```bash
git status --short
git diff --cached --check
git ls-files | rg -i '\.(yaml|yml|gif|pt|pth|ckpt|npz|mp4)$'
```

## 上游项目与许可

本项目基于 MuscleMimic 进行羽毛球方向扩展。原始项目、模型和第三方数据分别受其各自许可约束。

本仓库代码遵循 [Apache License 2.0](LICENSE)。
